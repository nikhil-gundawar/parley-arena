"""Two humans join one room, the agent starts working, one of them steers it
mid-run, and the refund only goes through once both say yes.

    python examples/two_humans_one_agent.py
"""

import asyncio

from parley import ApprovalPolicy, Runner, Session
from parley.agents import DemoAgent
from parley.events import Event, EventType

INTERESTING = {
    EventType.MESSAGE,
    EventType.STEER,
    EventType.AGENT_MESSAGE,
    EventType.TOOL_REQUESTED,
    EventType.TOOL_APPROVED,
    EventType.TOOL_DENIED,
    EventType.TOOL_RESULT,
    EventType.RUN_FINISHED,
}


async def watch(session: Session) -> None:
    """Anyone can subscribe to the room and see exactly what everyone else sees."""
    q = session.subscribe()
    while True:
        e: Event = await q.get()
        if e.type in INTERESTING:
            who = session.participants.get(e.actor)
            print(f"  [{(who.name if who else e.actor):>6}] {e.type.value:<16} {str(e.payload)[:110]}")


async def main() -> None:
    session = Session("refund escalation", policy=ApprovalPolicy(quorum=2))
    alice = session.join("alice")
    bob = session.join("bob")
    watcher = asyncio.create_task(watch(session))

    session.append(
        Event(type=EventType.MESSAGE, actor=alice.id, payload={"text": "c_8812 got double charged, fix it"})
    )

    runner = Runner(session, DemoAgent(speed=0.01))
    run = runner.start()

    # bob changes his mind about the amount while the agent is already working
    await asyncio.sleep(0.35)
    session.append(Event(type=EventType.STEER, actor=bob.id, payload={"text": "make it 120 not the full amount"}))

    # both humans have to sign off before real money moves
    async def approvals() -> None:
        while not run.done():
            await asyncio.sleep(0.05)
            for call in list(session.tool_calls.values()):
                if call.status == "pending":
                    print(f"  >>> {call.name}{call.args} needs {call.quorum} humans")
                    session.vote(call.id, alice.id, True)
                    if call.status == "pending":
                        session.vote(call.id, bob.id, True)

    voter = asyncio.create_task(approvals())
    await run
    voter.cancel()
    watcher.cancel()

    print(f"\n{len(session.log)} events · {len(session.tool_calls)} tool calls")
    path = session.save("session.parley.json")
    print(f"saved → {path}   (replay it any time: parley replay {path})")

    # and fork the timeline at the moment of the refund to try the other path
    fork_point = next(
        (e.seq for e in session.log if e.type == EventType.TOOL_REQUESTED and e.payload["name"] == "issue_refund"),
        len(session.log) - 1,
    )
    child = session.fork(at_seq=fork_point - 1, title="what if we said no")
    print(f"forked at #{fork_point - 1} → {child.id} ({len(child.log)} events carried over)")


if __name__ == "__main__":
    asyncio.run(main())
