"""A scripted agent so `parley demo` works with zero API keys.

It is deliberately theatrical: it streams, it checks for human steering between
every step, and it reaches for a high-risk tool so you can watch the quorum gate
do its job. Swap in `LLMAgent` for the real thing.
"""

from __future__ import annotations

import asyncio
import re

from ..runtime import ToolDenied, Turn


class DemoAgent:
    """Handles a support escalation, badly enough that a human should intervene."""

    def __init__(self, speed: float = 0.045) -> None:
        self.speed = speed

    async def _stream(self, turn: Turn, text: str) -> None:
        for word in text.split(" "):
            await turn.token(word + " ")
            await asyncio.sleep(self.speed)

    async def _absorb(self, turn: Turn) -> list[str]:
        nudges = turn.steering()
        if nudges:
            for n in nudges:
                await self._stream(turn, f"\n[heard {n['from']}: \"{n['text']}\"] adjusting.")
            await turn.say("Folding that into the plan.")
        return [n["text"] for n in nudges]

    async def run(self, turn: Turn) -> None:
        ask = next(
            (m["content"] for m in reversed(turn.transcript) if m["role"] == "user"),
            "Customer c_8812 is angry about a double charge.",
        )

        heard: list[str] = []

        await self._stream(turn, f"Working on: {ask}")
        await turn.say("Plan: 1) check the docs, 2) pull the numbers, 3) act.")
        heard += await self._absorb(turn)

        docs = await turn.call("search_docs", query="refund")
        await turn.say(f"Policy says: {docs['hits'][0]['text']}")
        heard += await self._absorb(turn)

        await self._stream(turn, "Pulling retention numbers so I know what this account is worth.")
        rows = await turn.call("query_analytics", sql="select cohort, retained_d30 from cohorts limit 2")
        await turn.say(f"Latest cohort retention is {rows['rows'][-1]['retained_d30']:.0%}.")
        heard += await self._absorb(turn)

        # Deliberate beat before touching the real world: last chance for anyone
        # in the room to redirect this without restarting the run.
        await self._stream(turn, "About to move money — holding a beat in case anyone objects.")
        await asyncio.sleep(max(0.35, self.speed * 40))
        heard += await self._absorb(turn)

        amount = 240.0
        for n in heard:  # a human said a number mid-run — that number wins
            numbers = re.findall(r"\d+(?:\.\d+)?", n)
            if numbers:
                amount = float(numbers[-1])

        await self._stream(turn, f"I want to refund ${amount:,.2f} to c_8812. That moves real money, so I need sign-off.")
        try:
            receipt = await turn.call("issue_refund", customer_id="c_8812", amount_usd=amount)
            await turn.say(f"Refund issued — reference {receipt['reference']}.")
        except ToolDenied:
            await turn.say("Refund vetoed by the room. Escalating to a human owner instead, no money moved.")
            return

        try:
            await turn.call(
                "send_email",
                to="c_8812@example.com",
                subject="Your refund is on the way",
                body=f"We've refunded ${amount:,.2f}. Sorry for the double charge.",
            )
            await turn.say("Customer notified. Done.")
        except ToolDenied:
            await turn.say("Email held. Refund went through but nothing was sent.")
