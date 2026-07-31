"""Tool registry with a risk label baked in.

Risk is not decoration: `risk="high"` is what routes a call into the human
quorum gate before it executes.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    description: str = ""
    risk: str = "low"
    schema: dict[str, Any] = field(default_factory=dict)

    async def __call__(self, **kwargs: Any) -> Any:
        result = self.fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(
        self,
        name: str | None = None,
        description: str = "",
        risk: str = "low",
        schema: dict[str, Any] | None = None,
    ):
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            t = Tool(
                name=name or fn.__name__,
                fn=fn,
                description=description or (fn.__doc__ or "").strip(),
                risk=risk,
                schema=schema or {},
            )
            self._tools[t.name] = t
            return fn

        return deco

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "risk": t.risk, "input_schema": t.schema or {"type": "object", "properties": {}}}
            for t in self._tools.values()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


registry = ToolRegistry()


# ---------------------------------------------------------------- demo tools


@registry.tool(
    description="Search an internal knowledge base. Read-only.",
    risk="low",
    schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)
def search_docs(query: str) -> dict[str, Any]:
    corpus = {
        "refund": "Refunds are issued to the original payment method within 5 business days.",
        "churn": "Q2 churn was 4.1%, up from 3.2% in Q1. Driver: onboarding drop-off at step 3.",
        "pricing": "Team plan is $20/seat/mo. Enterprise is custom, minimum 25 seats.",
    }
    hits = [{"key": k, "text": v} for k, v in corpus.items() if k in query.lower()]
    return {"query": query, "hits": hits or [{"key": "none", "text": "No exact match."}]}


@registry.tool(
    description="Run a read-only SQL query against the analytics replica.",
    risk="low",
    schema={"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
)
def query_analytics(sql: str) -> dict[str, Any]:
    return {"sql": sql, "rows": [{"cohort": "2026-05", "retained_d30": 0.61}, {"cohort": "2026-06", "retained_d30": 0.58}]}


@registry.tool(
    description="Issue a refund to a customer. Moves real money.",
    risk="high",
    schema={
        "type": "object",
        "properties": {"customer_id": {"type": "string"}, "amount_usd": {"type": "number"}},
        "required": ["customer_id", "amount_usd"],
    },
)
def issue_refund(customer_id: str, amount_usd: float) -> dict[str, Any]:
    return {"ok": True, "customer_id": customer_id, "amount_usd": amount_usd, "reference": "rf_9f21c"}


@registry.tool(
    description="Send an email on behalf of the company. Irreversible.",
    risk="high",
    schema={
        "type": "object",
        "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
        "required": ["to", "subject", "body"],
    },
)
def send_email(to: str, subject: str, body: str) -> dict[str, Any]:
    return {"ok": True, "to": to, "subject": subject, "chars": len(body)}


@registry.tool(
    description="Delete a production database table. Never allowed.",
    risk="high",
    schema={"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]},
)
def drop_table(table: str) -> dict[str, Any]:  # pragma: no cover - blocked by policy
    return {"ok": True, "dropped": table}
