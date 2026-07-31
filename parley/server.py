"""FastAPI + WebSocket transport for Parley rooms."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .events import SYSTEM, Event, EventType
from .runtime import Runner
from .session import ApprovalPolicy, Session, SessionStore
from .tools import registry

WEB = Path(__file__).resolve().parent.parent / "web"

store = SessionStore()
runners: dict[str, Runner] = {}

app = FastAPI(title="Parley", version="0.1.0", description="Multiplayer sessions for AI agents")


def build_agent():
    """Real model if a key is around, scripted demo otherwise. Never blocks the demo."""
    choice = os.getenv("PARLEY_AGENT", "auto").lower()
    if choice == "demo":
        from .agents.demo import DemoAgent

        return DemoAgent()
    if choice in ("llm", "auto") and (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        from .agents.llm import LLMAgent

        return LLMAgent(model=os.getenv("PARLEY_MODEL"))
    from .agents.demo import DemoAgent

    return DemoAgent()


def runner_for(session: Session) -> Runner:
    r = runners.get(session.id)
    if r is None:
        r = Runner(session, build_agent(), registry)
        runners[session.id] = r
    return r


def default_policy() -> ApprovalPolicy:
    return ApprovalPolicy(
        quorum=int(os.getenv("PARLEY_QUORUM", "1")),
        never_approve={"drop_table"},
    )


# ---------------------------------------------------------------------- REST


class NewSession(BaseModel):
    title: str = "untitled session"
    quorum: int | None = None


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (WEB / "index.html").read_text()


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "sessions": len(store.list()), "tools": len(registry), "agent": type(build_agent()).__name__}


@app.get("/api/tools")
async def tools() -> list[dict[str, Any]]:
    return registry.specs()


@app.get("/api/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    return store.list()


@app.post("/api/sessions")
async def create_session(body: NewSession) -> dict[str, Any]:
    policy = default_policy()
    if body.quorum is not None:
        policy.quorum = body.quorum
    s = store.create(body.title, policy)
    return {"id": s.id, "title": s.title, "quorum": policy.quorum}


@app.get("/api/sessions/{sid}")
async def get_session(sid: str) -> dict[str, Any]:
    s = store.get(sid)
    if not s:
        raise HTTPException(404, "no such session")
    return snapshot(s)


@app.post("/api/sessions/{sid}/fork")
async def fork_session(sid: str, at_seq: int | None = None, title: str | None = None) -> dict[str, Any]:
    s = store.get(sid)
    if not s:
        raise HTTPException(404, "no such session")
    child = store.add(s.fork(at_seq=at_seq, title=title))
    return {"id": child.id, "title": child.title, "forked_from": child.forked_from}


@app.get("/api/sessions/{sid}/export")
async def export_session(sid: str) -> JSONResponse:
    s = store.get(sid)
    if not s:
        raise HTTPException(404, "no such session")
    return JSONResponse(s.dump(), headers={"Content-Disposition": f'attachment; filename="{sid}.parley.json"'})


@app.get("/api/sessions/{a}/diff/{b}")
async def diff_sessions(a: str, b: str) -> dict[str, Any]:
    left, right = store.get(a), store.get(b)
    if not left or not right:
        raise HTTPException(404, "no such session")
    return left.diff(right)


def snapshot(s: Session) -> dict[str, Any]:
    return {
        "kind": "snapshot",
        "id": s.id,
        "title": s.title,
        "running": s.running,
        "wheel": s.wheel,
        "quorum": s.policy.quorum,
        "forked_from": s.forked_from,
        "participants": [p.model_dump() for p in s.participants.values()],
        "tool_calls": {k: v.model_dump() for k, v in s.tool_calls.items()},
        "log": [e.model_dump(mode="json") for e in s.log],
    }


# ----------------------------------------------------------------- WebSocket


@app.websocket("/ws/{sid}")
async def ws(sock: WebSocket, sid: str, name: str = "anon", role: str = "member") -> None:
    await sock.accept()
    s = store.get(sid)
    if not s:
        await sock.send_json({"kind": "error", "error": "no such session"})
        await sock.close()
        return

    queue = s.subscribe()
    me = s.join(name, role=role)
    await sock.send_json({"kind": "hello", "you": me.model_dump()})
    await sock.send_json(snapshot(s))

    async def pump() -> None:
        while True:
            e: Event = await queue.get()
            await sock.send_json({"kind": "event", **e.model_dump(mode="json")})

    pumper = asyncio.create_task(pump())
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            await handle(s, me.id, msg)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - transport noise
        await _safe(sock, {"kind": "error", "error": str(exc)})
    finally:
        pumper.cancel()
        s.unsubscribe(queue)
        s.leave(me.id)


async def _safe(sock: WebSocket, payload: dict) -> None:  # pragma: no cover
    try:
        await sock.send_json(payload)
    except Exception:
        pass


async def handle(s: Session, pid: str, msg: dict[str, Any]) -> None:
    op = msg.get("op")
    runner = runner_for(s)

    if op == "message":
        s.append(Event(type=EventType.MESSAGE, actor=pid, payload={"text": msg.get("text", "")}))
        if msg.get("run") and not runner.busy:
            runner.start(msg.get("text", ""))

    elif op == "steer":
        # The headline move: inject into a *running* agent without restarting it.
        s.append(Event(type=EventType.STEER, actor=pid, payload={"text": msg.get("text", "")}))

    elif op == "run":
        if not runner.busy:
            runner.start(msg.get("prompt"))

    elif op == "pause":
        runner.pause()
    elif op == "resume":
        runner.resume()
    elif op == "stop":
        runner.stop()

    elif op == "vote":
        s.vote(msg["call_id"], pid, bool(msg.get("approve")))

    elif op == "wheel":
        if msg.get("take"):
            if s.wheel in (None, pid):
                s.append(Event(type=EventType.WHEEL_TAKEN, actor=pid))
        elif s.wheel == pid:
            s.append(Event(type=EventType.WHEEL_RELEASED, actor=pid))

    elif op == "fork":
        child = store.add(s.fork(at_seq=msg.get("at_seq")))
        s.append(
            Event(
                type=EventType.AGENT_MESSAGE,
                actor=SYSTEM,
                payload={"text": f"↳ forked to {child.id} at #{msg.get('at_seq')}", "system": True},
            )
        )

    elif op in ("cursor", "typing"):
        s.append(
            Event(
                type=EventType.CURSOR if op == "cursor" else EventType.TYPING,
                actor=pid,
                payload=msg.get("payload", {}),
            )
        )
