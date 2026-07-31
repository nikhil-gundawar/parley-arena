"""Parley — multiplayer sessions for AI agents.

    from parley import Session, Runner, ApprovalPolicy
    from parley.agents import DemoAgent

    s = Session("incident 421", policy=ApprovalPolicy(quorum=2))
    s.join("alice"); s.join("bob")
    await Runner(s, DemoAgent()).start("the payments queue is backing up")
"""

from .events import Event, EventType, Participant, ToolCall
from .runtime import Runner, ToolDenied, Turn
from .session import ApprovalPolicy, Session, SessionStore
from .tools import ToolRegistry, registry

__version__ = "0.1.0"
__all__ = [
    "ApprovalPolicy",
    "Event",
    "EventType",
    "Participant",
    "Runner",
    "Session",
    "SessionStore",
    "ToolCall",
    "ToolDenied",
    "ToolRegistry",
    "Turn",
    "registry",
]
