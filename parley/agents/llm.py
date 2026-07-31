"""A real tool-calling agent, provider-agnostic.

Works with Anthropic or OpenAI depending on which key is in the environment.
The interesting part is not the loop — it is that between every step it drains
human steering out of the room and injects it into context, so five people can
redirect one agent mid-run without restarting it.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from ..runtime import ToolDenied, Turn

SYSTEM = (
    "You are running inside a Parley session: a live room with several humans "
    "watching and able to interrupt you. Think out loud briefly. When humans "
    "steer you mid-task, acknowledge it and adjust. High-risk tools require "
    "human approval and may be denied — if that happens, adapt rather than retry."
)


class LLMAgent:
    """Bring your own key. `provider` is inferred if not given."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        max_steps: int = 12,
        system: str = SYSTEM,
    ) -> None:
        self.provider = provider or ("anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai")
        self.model = model or (
            "claude-sonnet-4-5" if self.provider == "anthropic" else "gpt-4.1-mini"
        )
        self.max_steps = max_steps
        self.system = system

    # ------------------------------------------------------------------ run

    async def run(self, turn: Turn) -> None:
        messages = self._seed(turn)
        specs = turn.tools.specs()

        async with httpx.AsyncClient(timeout=120) as client:
            for _ in range(self.max_steps):
                for nudge in turn.steering():
                    messages.append(
                        {"role": "user", "content": f"[live steer from {nudge['from']}] {nudge['text']}"}
                    )

                text, calls = await self._step(client, messages, specs)
                if text:
                    await turn.say(text)
                if not calls:
                    return

                messages.append({"role": "assistant", "content": text or "", "tool_calls": calls})
                for call in calls:
                    try:
                        result = await turn.call(call["name"], **call["args"])
                    except ToolDenied:
                        result = {"denied_by_humans": True}
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)[:6000]}
                    )

    def _seed(self, turn: Turn) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in turn.transcript:
            if m["role"] == "user":
                prefix = "[steer] " if m.get("steer") else ""
                out.append({"role": "user", "content": f"{prefix}{m['name']}: {m['content']}"})
            elif m["role"] == "assistant":
                out.append({"role": "assistant", "content": m["content"]})
        return out or [{"role": "user", "content": "Introduce yourself and ask what the room needs."}]

    # -------------------------------------------------------------- providers

    async def _step(self, client, messages, specs) -> tuple[str, list[dict[str, Any]]]:
        if self.provider == "anthropic":
            return await self._anthropic(client, messages, specs)
        return await self._openai(client, messages, specs)

    async def _anthropic(self, client, messages, specs):
        key = os.environ["ANTHROPIC_API_KEY"]
        payload = {
            "model": self.model,
            "max_tokens": 1500,
            "system": self.system,
            "messages": [self._to_anthropic(m) for m in messages],
            "tools": [
                {"name": s["name"], "description": s["description"], "input_schema": s["input_schema"]}
                for s in specs
            ],
        }
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
        text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        calls = [
            {"id": b["id"], "name": b["name"], "args": b["input"]}
            for b in data["content"]
            if b["type"] == "tool_use"
        ]
        return text.strip(), calls

    @staticmethod
    def _to_anthropic(m: dict[str, Any]) -> dict[str, Any]:
        if m["role"] == "tool":
            return {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                ],
            }
        if m["role"] == "assistant" and m.get("tool_calls"):
            blocks: list[dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for c in m["tool_calls"]:
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]})
            return {"role": "assistant", "content": blocks}
        return {"role": m["role"], "content": m["content"] or "..."}

    async def _openai(self, client, messages, specs):
        key = os.environ["OPENAI_API_KEY"]
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system}] + [self._to_openai(m) for m in messages],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": s["name"],
                        "description": s["description"],
                        "parameters": s["input_schema"],
                    },
                }
                for s in specs
            ],
        }
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        calls = [
            {"id": c["id"], "name": c["function"]["name"], "args": json.loads(c["function"]["arguments"] or "{}")}
            for c in (msg.get("tool_calls") or [])
        ]
        return (msg.get("content") or "").strip(), calls

    @staticmethod
    def _to_openai(m: dict[str, Any]) -> dict[str, Any]:
        if m["role"] == "tool":
            return {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
        if m["role"] == "assistant" and m.get("tool_calls"):
            return {
                "role": "assistant",
                "content": m.get("content") or None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                    }
                    for c in m["tool_calls"]
                ],
            }
        return {"role": m["role"], "content": m["content"]}
