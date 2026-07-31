"""parley — command line entry points."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .events import EventType
from .session import Session


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    print(f"\n  parley → http://{args.host}:{args.port}   (share that link, that's the whole point)\n")
    uvicorn.run("parley.server:app", host=args.host, port=args.port, reload=args.reload, log_level="warning")


def _replay(args: argparse.Namespace) -> None:
    s = Session.replay(args.file)
    print(f"session {s.id} · {s.title} · {len(s.log)} events")
    if s.forked_from:
        print(f"forked from {s.forked_from[0]} at #{s.forked_from[1]}")
    print("-" * 72)
    for e in s.log:
        if e.type == EventType.AGENT_TOKEN and not args.verbose:
            continue
        who = s.participants.get(e.actor)
        label = who.name if who else e.actor
        body = json.dumps(e.payload)[: args.width]
        print(f"#{e.seq:<4} {e.type.value:<20} {label:<12} {body}")
    print("-" * 72)
    print(f"{len(s.transcript)} transcript turns · {len(s.tool_calls)} tool calls")


def _diff(args: argparse.Namespace) -> None:
    a, b = Session.replay(args.left), Session.replay(args.right)
    d = a.diff(b)
    print(f"common prefix: {d['common_prefix']} events")
    print(f"\n── {a.id} diverges ──")
    for e in d["left"]:
        print(f"  #{e['seq']:<4} {e['type']:<20} {json.dumps(e['payload'])[:100]}")
    print(f"\n── {b.id} diverges ──")
    for e in d["right"]:
        print(f"  #{e['seq']:<4} {e['type']:<20} {json.dumps(e['payload'])[:100]}")


def _headless(args: argparse.Namespace) -> None:
    """Run one agent turn with no browser — handy for CI and for screenshots."""
    from .agents.demo import DemoAgent
    from .runtime import Runner
    from .session import ApprovalPolicy
    from .tools import registry

    async def main() -> None:
        s = Session(title="headless", policy=ApprovalPolicy(quorum=args.quorum))
        alice = s.join("alice")
        bob = s.join("bob")
        from .events import Event

        s.append(Event(type=EventType.MESSAGE, actor=alice.id, payload={"text": args.prompt}))
        runner = Runner(s, DemoAgent(speed=0.0), registry)
        task = runner.start(args.prompt)

        async def auto_vote() -> None:
            while not task.done():
                await asyncio.sleep(0.05)
                for call in list(s.tool_calls.values()):
                    if call.status == "pending":
                        s.vote(call.id, alice.id, args.approve)
                        if call.status == "pending":
                            s.vote(call.id, bob.id, args.approve)

        voter = asyncio.create_task(auto_vote())
        await task
        voter.cancel()

        for e in s.log:
            if e.type == EventType.AGENT_TOKEN:
                continue
            print(f"#{e.seq:<3} {e.type.value:<20} {json.dumps(e.payload)[:120]}")
        if args.out:
            print(f"\nsaved → {s.save(args.out)}")

    asyncio.run(main())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="parley", description="Multiplayer sessions for AI agents")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="start the room server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(fn=_serve)

    p = sub.add_parser("replay", help="deterministically replay a saved session log")
    p.add_argument("file", type=Path)
    p.add_argument("--verbose", "-v", action="store_true", help="include token stream")
    p.add_argument("--width", type=int, default=90)
    p.set_defaults(fn=_replay)

    p = sub.add_parser("diff", help="diff two forked timelines")
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.set_defaults(fn=_diff)

    p = sub.add_parser("headless", help="run one agent turn in the terminal")
    p.add_argument("--prompt", default="Customer c_8812 was double charged. Sort it out.")
    p.add_argument("--quorum", type=int, default=2)
    p.add_argument("--approve", action="store_true", help="humans approve instead of veto")
    p.add_argument("--out", type=Path)
    p.set_defaults(fn=_headless)

    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
