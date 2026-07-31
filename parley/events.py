"""Event schema for Parley sessions.

Everything that happens in a session — a human speaking, an agent thinking, a
tool being approved — is an immutable, sequenced Event appended to a log.
Session state is *derived* from that log, never mutated directly. That single
constraint is what buys us the three things nobody else has:

  * late joiners get the full history and converge to identical state
  * any session can be replayed deterministically, offline, forever
  * any session can be forked at any point into a parallel timeline
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    # lifecycle
    SESSION_CREATED = "session.created"
    SESSION_FORKED = "session.forked"

    # presence
    PARTICIPANT_JOINED = "participant.joined"
    PARTICIPANT_LEFT = "participant.left"

    # human input
    MESSAGE = "message"
    STEER = "steer"

    # turn control
    WHEEL_TAKEN = "wheel.taken"
    WHEEL_RELEASED = "wheel.released"

    # agent output
    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    RUN_FAILED = "run.failed"
    AGENT_TOKEN = "agent.token"
    AGENT_MESSAGE = "agent.message"

    # tools + governance
    TOOL_REQUESTED = "tool.requested"
    TOOL_VOTE = "tool.vote"
    TOOL_APPROVED = "tool.approved"
    TOOL_DENIED = "tool.denied"
    TOOL_RESULT = "tool.result"

    # ephemeral (broadcast but not persisted)
    CURSOR = "cursor"
    TYPING = "typing"


EPHEMERAL = {EventType.CURSOR, EventType.TYPING}

SYSTEM = "system"
AGENT = "agent"


class Event(BaseModel):
    """One immutable fact about a session."""

    seq: int = 0
    session_id: str = ""
    type: EventType
    actor: str = SYSTEM
    ts: float = Field(default_factory=time.time)
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def ephemeral(self) -> bool:
        return self.type in EPHEMERAL


class Participant(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    role: str = "member"  # member | owner | observer
    color: str = "#7c5cff"
    online: bool = True


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    risk: str = "low"  # low | high
    quorum: int = 0  # humans required to approve; 0 == auto
    votes: dict[str, bool] = Field(default_factory=dict)  # participant_id -> approve?
    status: str = "pending"  # pending | approved | denied | done
    result: Any = None

    def tally(self) -> tuple[int, int]:
        yes = sum(1 for v in self.votes.values() if v)
        no = sum(1 for v in self.votes.values() if not v)
        return yes, no
