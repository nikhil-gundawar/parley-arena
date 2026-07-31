import asyncio
import json

import pytest

from parley import ApprovalPolicy, Runner, Session, ToolDenied
from parley.agents import DemoAgent
from parley.events import Event, EventType
from parley.tools import ToolRegistry


def test_state_is_derived_from_log():
    s = Session("t")
    a = s.join("alice")
    s.append(Event(type=EventType.MESSAGE, actor=a.id, payload={"text": "hi"}))
    assert s.transcript[-1]["content"] == "hi"
    assert s.participants[a.id].name == "alice"
    # replaying the log alone reproduces identical state
    clone = Session.replay(s.dump())
    assert clone.transcript == s.transcript
    assert [p.name for p in clone.participants.values()] == ["alice"]


def test_late_joiner_converges():
    s = Session("t")
    a = s.join("alice")
    for i in range(5):
        s.append(Event(type=EventType.MESSAGE, actor=a.id, payload={"text": str(i)}))
    late = Session.replay(json.loads(json.dumps(s.dump())))
    assert len(late.log) == len(s.log)
    assert late.transcript == s.transcript


def test_fork_creates_identical_prefix_then_diverges():
    s = Session("t")
    a = s.join("alice")
    for i in range(4):
        s.append(Event(type=EventType.MESSAGE, actor=a.id, payload={"text": f"m{i}"}))
    child = s.fork(at_seq=3)

    assert [e.type for e in child.log[:4]] == [e.type for e in s.log[:4]]
    assert child.forked_from == (s.id, 3)
    assert child.id != s.id

    s.append(Event(type=EventType.MESSAGE, actor=a.id, payload={"text": "parent path"}))
    ca = next(iter(child.participants))
    child.append(Event(type=EventType.MESSAGE, actor=ca, payload={"text": "child path"}))

    d = s.diff(child)
    assert d["common_prefix"] == 4
    assert d["left"][-1]["payload"]["text"] == "parent path"
    assert d["right"][-1]["payload"]["text"] == "child path"


async def test_quorum_blocks_until_enough_humans_approve():
    s = Session("t", policy=ApprovalPolicy(quorum=2))
    a, b = s.join("alice"), s.join("bob")
    tools = ToolRegistry()

    calls = []

    @tools.tool(risk="high")
    def wire_money(amount: float):
        calls.append(amount)
        return {"sent": amount}

    from parley.runtime import Turn

    turn = Turn(s, tools)
    task = asyncio.create_task(turn.call("wire_money", amount=100.0))
    await asyncio.sleep(0.05)

    call_id = next(iter(s.tool_calls))
    assert calls == []  # still blocked on humans

    s.vote(call_id, a.id, True)
    await asyncio.sleep(0.02)
    assert calls == []  # one approval is not quorum

    s.vote(call_id, b.id, True)
    result = await task
    assert result == {"sent": 100.0}
    assert calls == [100.0]


async def test_single_veto_kills_a_call():
    s = Session("t", policy=ApprovalPolicy(quorum=3))
    a, b = s.join("alice"), s.join("bob")
    tools = ToolRegistry()

    @tools.tool(risk="high")
    def nuke():
        return "boom"

    from parley.runtime import Turn

    turn = Turn(s, tools)
    task = asyncio.create_task(turn.call("nuke"))
    await asyncio.sleep(0.05)
    call_id = next(iter(s.tool_calls))
    s.vote(call_id, a.id, True)
    s.vote(call_id, b.id, False)  # one veto beats any number of approvals

    with pytest.raises(ToolDenied):
        await task
    assert s.tool_calls[call_id].status == "denied"


async def test_policy_can_hard_deny():
    s = Session("t", policy=ApprovalPolicy(quorum=1, never_approve={"drop_table"}))
    tools = ToolRegistry()

    @tools.tool(risk="high")
    def drop_table(table: str):
        raise AssertionError("must never execute")

    from parley.runtime import Turn

    with pytest.raises(ToolDenied):
        await Turn(s, tools).call("drop_table", table="users")


async def test_steering_reaches_a_running_agent():
    s = Session("t", policy=ApprovalPolicy(quorum=0))
    a = s.join("alice")
    s.append(Event(type=EventType.MESSAGE, actor=a.id, payload={"text": "handle it"}))

    runner = Runner(s, DemoAgent(speed=0.002))
    task = runner.start("handle it")
    await asyncio.sleep(0.01)
    s.append(Event(type=EventType.STEER, actor=a.id, payload={"text": "cap it at 50"}))
    await asyncio.wait_for(task, timeout=10)

    texts = " ".join(e.payload.get("text", "") for e in s.log)
    assert "cap it at 50" in texts
    refunds = [c for c in s.tool_calls.values() if c.name == "issue_refund"]
    assert refunds and refunds[0].args["amount_usd"] == 50.0  # the nudge changed the action


async def test_pause_and_resume():
    s = Session("t", policy=ApprovalPolicy(quorum=0))
    runner = Runner(s, DemoAgent(speed=0.02))
    task = runner.start("go")
    await asyncio.sleep(0.05)
    runner.pause()
    n = len(s.log)
    await asyncio.sleep(0.15)
    assert len(s.log) == n  # frozen mid-run
    runner.resume()
    await task
    assert len(s.log) > n


async def test_broadcast_reaches_every_subscriber():
    s = Session("t")
    q1, q2 = s.subscribe(), s.subscribe()
    s.join("alice")
    assert q1.qsize() == q2.qsize() >= 1


async def test_full_run_replays_byte_identical(tmp_path):
    s = Session("t", policy=ApprovalPolicy(quorum=1))
    a = s.join("alice")
    runner = Runner(s, DemoAgent(speed=0.0))
    task = runner.start("double charge on c_8812")

    async def approve_everything():
        while not task.done():
            await asyncio.sleep(0.01)
            for c in list(s.tool_calls.values()):
                if c.status == "pending":
                    s.vote(c.id, a.id, True)

    voter = asyncio.create_task(approve_everything())
    await task
    voter.cancel()

    path = s.save(tmp_path / "s.json")
    back = Session.replay(path)
    assert back.dump() == s.dump()
    assert back.transcript == s.transcript
