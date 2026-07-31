"""Agent runtime: the bridge between an agent's loop and a multiplayer session.

An agent never touches the session directly. It gets a `Turn` handle, and every
side effect it can have — speaking, thinking out loud, touching the world — goes
through that handle so it lands in the log, reaches every human in the room, and
can be paused, steered or vetoed mid-flight.
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Any, Protocol

from .events import AGENT, Event, EventType, ToolCall
from .session import Session
from .tools import ToolRegistry
from .tools import registry as default_registry


class ToolDenied(Exception):
    """Raised into the agent when humans veto a tool call."""


class Turn:
    """What an agent is handed when it runs inside a Parley session."""

    def __init__(self, session: Session, tools: ToolRegistry) -> None:
        self.session = session
        self.tools = tools
        self._paused = asyncio.Event()
        self._paused.set()
        self._cancelled = False

    # -- speaking -----------------------------------------------------------

    async def token(self, text: str) -> None:
        """Stream a partial thought. Everyone in the room sees it as it lands."""
        await self._checkpoint()
        self.session.append(Event(type=EventType.AGENT_TOKEN, actor=AGENT, payload={"text": text}))

    async def say(self, text: str) -> None:
        """Commit a complete message to the shared transcript."""
        await self._checkpoint()
        self.session.append(Event(type=EventType.AGENT_MESSAGE, actor=AGENT, payload={"text": text}))

    # -- listening ----------------------------------------------------------

    def steering(self) -> list[dict[str, str]]:
        """Human nudges that arrived since the last step. Fold these into context."""
        return self.session.drain_steering()

    @property
    def transcript(self) -> list[dict[str, Any]]:
        return self.session.transcript

    # -- acting -------------------------------------------------------------

    async def call(self, name: str, **args: Any) -> Any:
        """Invoke a tool. High-risk tools block here until humans reach quorum."""
        await self._checkpoint()
        tool = self.tools.get(name)
        risk = tool.risk if tool else "high"
        call = ToolCall(name=name, args=args, risk=risk)
        decided = await self.session.request_approval(call)

        if decided.status != "approved":
            self.session.append(
                Event(
                    type=EventType.TOOL_RESULT,
                    actor=AGENT,
                    payload={"id": call.id, "name": name, "result": {"denied": True}},
                )
            )
            raise ToolDenied(f"humans denied {name}")

        if tool is None:
            result: Any = {"error": f"unknown tool {name!r}"}
        else:
            try:
                result = await tool(**args)
            except Exception as exc:  # pragma: no cover - defensive
                result = {"error": f"{type(exc).__name__}: {exc}"}

        self.session.append(
            Event(
                type=EventType.TOOL_RESULT,
                actor=AGENT,
                payload={"id": call.id, "name": name, "result": result},
            )
        )
        return result

    # -- flow control -------------------------------------------------------

    def pause(self) -> None:
        self._paused.clear()

    def resume(self) -> None:
        self._paused.set()

    def cancel(self) -> None:
        self._cancelled = True
        self._paused.set()

    async def _checkpoint(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError()
        await self._paused.wait()
        if self._cancelled:
            raise asyncio.CancelledError()


class Agent(Protocol):
    """Any object with this shape can run in a Parley room."""

    async def run(self, turn: Turn) -> None: ...


class Runner:
    """Owns the one running agent per session."""

    def __init__(self, session: Session, agent: Agent, tools: ToolRegistry | None = None) -> None:
        self.session = session
        self.agent = agent
        self.tools = tools or default_registry
        self.turn: Turn | None = None
        self.task: asyncio.Task | None = None

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def start(self, prompt: str | None = None) -> asyncio.Task:
        if self.busy:
            raise RuntimeError("agent already running in this session")
        self.turn = Turn(self.session, self.tools)
        self.session.append(Event(type=EventType.RUN_STARTED, actor=AGENT, payload={"prompt": prompt or ""}))
        self.task = asyncio.create_task(self._run())
        return self.task

    async def _run(self) -> None:
        assert self.turn is not None
        try:
            await self.agent.run(self.turn)
            self.session.append(Event(type=EventType.RUN_FINISHED, actor=AGENT))
        except (asyncio.CancelledError, ToolDenied) as exc:
            self.session.append(
                Event(type=EventType.RUN_FINISHED, actor=AGENT, payload={"halted": type(exc).__name__})
            )
        except Exception as exc:  # pragma: no cover - surfaced to the room
            self.session.append(
                Event(
                    type=EventType.RUN_FAILED,
                    actor=AGENT,
                    payload={"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]},
                )
            )

    def pause(self) -> None:
        if self.turn:
            self.turn.pause()

    def resume(self) -> None:
        if self.turn:
            self.turn.resume()

    def stop(self) -> None:
        if self.turn:
            self.turn.cancel()
        if self.task and not self.task.done():
            self.task.cancel()
