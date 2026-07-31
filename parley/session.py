"""The session engine: append-only log, derived state, presence, forking."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .events import AGENT, SYSTEM, Event, EventType, Participant, ToolCall

PALETTE = [
    "#7c5cff", "#00c2a8", "#ff8a3d", "#ff5c8a",
    "#3da5ff", "#ffd23d", "#9d6bff", "#4ade80",
]


class ApprovalPolicy:
    """Who has to say yes before an agent may touch the real world.

    The whole point of putting several humans in one agent session is that you
    can then *require* several humans. `quorum` is the number of distinct human
    approvals a high-risk tool call needs before it executes.
    """

    def __init__(
        self,
        quorum: int = 1,
        auto_approve: Iterable[str] = (),
        never_approve: Iterable[str] = (),
        timeout_s: float = 300.0,
    ) -> None:
        self.quorum = quorum
        self.auto_approve = set(auto_approve)
        self.never_approve = set(never_approve)
        self.timeout_s = timeout_s

    def required_for(self, tool_name: str, risk: str) -> int:
        if tool_name in self.never_approve:
            return -1  # hard deny
        if tool_name in self.auto_approve or risk == "low":
            return 0
        if self.quorum <= 0:  # quorum 0 == trust the agent, run unattended
            return 0
        return self.quorum


class Session:
    """A live, multiplayer agent session.

    State is derived by folding the event log. `apply()` is the *only* way state
    changes, which is what makes replay and forking exact rather than
    approximate.
    """

    def __init__(
        self,
        title: str = "untitled session",
        session_id: str | None = None,
        policy: ApprovalPolicy | None = None,
    ) -> None:
        self.id = session_id or uuid.uuid4().hex[:12]
        self.title = title
        self.policy = policy or ApprovalPolicy()
        self.log: list[Event] = []
        self.participants: dict[str, Participant] = {}
        self.transcript: list[dict[str, Any]] = []
        self.tool_calls: dict[str, ToolCall] = {}
        self.wheel: str | None = None  # participant holding the wheel
        self.running = False
        self.forked_from: tuple[str, int] | None = None
        self.children: list[str] = []

        self._subscribers: set[asyncio.Queue] = set()
        self._steering: list[dict[str, str]] = []
        self._approval_waiters: dict[str, asyncio.Event] = {}
        self._agent_task: asyncio.Task | None = None

        self.append(Event(type=EventType.SESSION_CREATED, payload={"title": title}))

    # ------------------------------------------------------------------ log

    def append(self, event: Event) -> Event:
        """Seal an event into the log and fold it into derived state."""
        event.session_id = self.id
        if event.ephemeral:
            self._broadcast(event)
            return event
        event.seq = len(self.log)
        self.log.append(event)
        self.apply(event)
        self._broadcast(event)
        return event

    def apply(self, e: Event) -> None:
        """Fold a single event into derived state. Pure w.r.t. the log."""
        p = e.payload
        if e.type == EventType.PARTICIPANT_JOINED:
            self.participants[p["id"]] = Participant(**p)
        elif e.type == EventType.PARTICIPANT_LEFT:
            if p["id"] in self.participants:
                self.participants[p["id"]].online = False
        elif e.type in (EventType.MESSAGE, EventType.STEER):
            who = self.participants.get(e.actor)
            self.transcript.append(
                {
                    "role": "user",
                    "name": who.name if who else e.actor,
                    "content": p.get("text", ""),
                    "steer": e.type == EventType.STEER,
                }
            )
            if e.type == EventType.STEER:
                self._steering.append(
                    {"from": who.name if who else e.actor, "text": p.get("text", "")}
                )
        elif e.type == EventType.AGENT_MESSAGE:
            self.transcript.append({"role": "assistant", "name": AGENT, "content": p.get("text", "")})
        elif e.type == EventType.WHEEL_TAKEN:
            self.wheel = e.actor
        elif e.type == EventType.WHEEL_RELEASED:
            self.wheel = None
        elif e.type == EventType.RUN_STARTED:
            self.running = True
        elif e.type in (EventType.RUN_FINISHED, EventType.RUN_FAILED):
            self.running = False
        elif e.type == EventType.TOOL_REQUESTED:
            self.tool_calls[p["id"]] = ToolCall(**p)
        elif e.type == EventType.TOOL_VOTE:
            call = self.tool_calls.get(p["id"])
            if call:
                call.votes[e.actor] = bool(p["approve"])
        elif e.type in (EventType.TOOL_APPROVED, EventType.TOOL_DENIED):
            call = self.tool_calls.get(p["id"])
            if call:
                call.status = "approved" if e.type == EventType.TOOL_APPROVED else "denied"
        elif e.type == EventType.TOOL_RESULT:
            call = self.tool_calls.get(p["id"])
            if call:
                if call.status != "denied":  # a veto is permanent, not overwritten by the result frame
                    call.status = "done"
                call.result = p.get("result")
            self.transcript.append(
                {"role": "tool", "name": p.get("name", "tool"), "content": json.dumps(p.get("result"))[:4000]}
            )
        elif e.type == EventType.SESSION_FORKED:
            self.forked_from = (p["parent"], p["at_seq"])

    # ------------------------------------------------------------ presence

    def join(self, name: str, role: str = "member") -> Participant:
        color = PALETTE[len(self.participants) % len(PALETTE)]
        p = Participant(name=name, role=role, color=color)
        self.append(Event(type=EventType.PARTICIPANT_JOINED, actor=p.id, payload=p.model_dump()))
        return p

    def leave(self, participant_id: str) -> None:
        if participant_id in self.participants:
            self.append(
                Event(
                    type=EventType.PARTICIPANT_LEFT,
                    actor=participant_id,
                    payload={"id": participant_id},
                )
            )
        if self.wheel == participant_id:
            self.append(Event(type=EventType.WHEEL_RELEASED, actor=participant_id))

    @property
    def online(self) -> list[Participant]:
        return [p for p in self.participants.values() if p.online]

    def humans_online(self) -> int:
        return sum(1 for p in self.online if p.role != "observer")

    # ------------------------------------------------------------- streams

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def _broadcast(self, event: Event) -> None:
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - slow client
                dead.append(q)
        for q in dead:
            self._subscribers.discard(q)

    # ------------------------------------------------------------ steering

    def drain_steering(self) -> list[dict[str, str]]:
        """Pull any mid-flight human nudges. The agent calls this between steps."""
        out, self._steering = self._steering, []
        return out

    # ------------------------------------------------------------ approvals

    async def request_approval(self, call: ToolCall) -> ToolCall:
        required = self.policy.required_for(call.name, call.risk)
        call.quorum = max(required, 0)
        self.append(Event(type=EventType.TOOL_REQUESTED, actor=AGENT, payload=call.model_dump()))
        live = self.tool_calls[call.id]

        if required < 0:
            self.append(
                Event(type=EventType.TOOL_DENIED, actor=SYSTEM, payload={"id": call.id, "reason": "policy"})
            )
            return live
        if required == 0:
            self.append(Event(type=EventType.TOOL_APPROVED, actor=SYSTEM, payload={"id": call.id}))
            return live

        gate = asyncio.Event()
        self._approval_waiters[call.id] = gate
        try:
            await asyncio.wait_for(gate.wait(), timeout=self.policy.timeout_s)
        except asyncio.TimeoutError:
            self.append(
                Event(type=EventType.TOOL_DENIED, actor=SYSTEM, payload={"id": call.id, "reason": "timeout"})
            )
        finally:
            self._approval_waiters.pop(call.id, None)
        return self.tool_calls[call.id]

    def vote(self, call_id: str, participant_id: str, approve: bool) -> ToolCall | None:
        call = self.tool_calls.get(call_id)
        if not call or call.status != "pending":
            return call
        self.append(
            Event(
                type=EventType.TOOL_VOTE,
                actor=participant_id,
                payload={"id": call_id, "approve": approve},
            )
        )
        yes, no = call.tally()
        if no > 0:
            self.append(
                Event(type=EventType.TOOL_DENIED, actor=participant_id, payload={"id": call_id, "reason": "vetoed"})
            )
        elif yes >= call.quorum:
            self.append(Event(type=EventType.TOOL_APPROVED, actor=participant_id, payload={"id": call_id}))
        if call.status != "pending":
            gate = self._approval_waiters.get(call_id)
            if gate:
                gate.set()
        return call

    # -------------------------------------------------------------- forking

    def fork(self, at_seq: int | None = None, title: str | None = None) -> Session:
        """Branch this session into a parallel timeline at `at_seq`.

        The child replays the parent's events up to that point, so it is a bit
        for bit identical world — and then diverges. Run three forks of the same
        decision point side by side and compare what the agent does.
        """
        at_seq = len(self.log) - 1 if at_seq is None else at_seq
        child = Session(
            title=title or f"{self.title} @{at_seq}",
            policy=self.policy,
        )
        child.log.clear()
        child.transcript.clear()
        child.participants.clear()
        child.tool_calls.clear()
        for e in self.log[: at_seq + 1]:
            copy = Event(**e.model_dump())
            copy.session_id = child.id
            copy.seq = len(child.log)
            child.log.append(copy)
            child.apply(copy)
        child.running = False
        child.append(
            Event(
                type=EventType.SESSION_FORKED,
                actor=SYSTEM,
                payload={"parent": self.id, "at_seq": at_seq},
            )
        )
        self.children.append(child.id)
        return child

    # ------------------------------------------------------ persist / replay

    def dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "forked_from": self.forked_from,
            "log": [e.model_dump(mode="json") for e in self.log],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.dump(), indent=2))
        return path

    @classmethod
    def replay(cls, data: dict[str, Any] | str | Path) -> Session:
        """Rebuild a session exactly from its log. No network, no model, no luck."""
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text())
        s = cls.__new__(cls)
        s.id = data["id"]
        s.title = data["title"]
        s.policy = ApprovalPolicy()
        s.log = []
        s.participants = {}
        s.transcript = []
        s.tool_calls = {}
        s.wheel = None
        s.running = False
        s.forked_from = tuple(data["forked_from"]) if data.get("forked_from") else None
        s.children = []
        s._subscribers = set()
        s._steering = []
        s._approval_waiters = {}
        s._agent_task = None
        for raw in data["log"]:
            e = Event(**raw)
            s.log.append(e)
            s.apply(e)
        s._steering = []  # steering is consumed at runtime, not part of replayed state
        return s

    def diff(self, other: Session) -> dict[str, Any]:
        """Compare two timelines from their common ancestor forward."""
        common = 0
        for a, b in zip(self.log, other.log):
            if a.type != b.type or a.payload != b.payload:
                break
            common += 1
        return {
            "common_prefix": common,
            "left": [e.model_dump(mode="json") for e in self.log[common:]],
            "right": [e.model_dump(mode="json") for e in other.log[common:]],
        }


class SessionStore:
    """In-memory registry. Swap for Postgres/Redis without touching the engine."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, title: str, policy: ApprovalPolicy | None = None) -> Session:
        s = Session(title=title, policy=policy)
        self._sessions[s.id] = s
        return s

    def add(self, session: Session) -> Session:
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def list(self) -> list[dict[str, Any]]:
        return [
            {
                "id": s.id,
                "title": s.title,
                "participants": len(s.online),
                "events": len(s.log),
                "running": s.running,
                "forked_from": s.forked_from,
                "created_at": s.log[0].ts if s.log else time.time(),
            }
            for s in self._sessions.values()
        ]
