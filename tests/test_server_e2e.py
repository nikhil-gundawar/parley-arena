"""Boots the real server and drives it with two real WebSocket clients."""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import websockets

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = free_port()
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PARLEY_AGENT": "demo", "PARLEY_QUORUM": "2"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "parley.server:app", "--port", str(port), "--log-level", "error"],
        cwd=ROOT,
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(base + "/api/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:  # pragma: no cover
        proc.kill()
        pytest.fail("server did not start")
    yield base, port
    proc.terminate()
    proc.wait(timeout=10)


async def recv_until(sock, predicate, timeout=15):
    seen = []
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        raw = await asyncio.wait_for(sock.recv(), timeout=deadline - asyncio.get_event_loop().time())
        msg = json.loads(raw)
        seen.append(msg)
        if predicate(msg):
            return msg, seen
    raise AssertionError(f"never saw it; last: {seen[-3:]}")


async def test_two_clients_share_one_timeline_and_gate_a_tool(server):
    base, port = server
    r = httpx.post(base + "/api/sessions", json={"title": "e2e", "quorum": 2}, timeout=10)
    sid = r.json()["id"]
    url = f"ws://127.0.0.1:{port}/ws/{sid}"

    async with websockets.connect(url + "?name=alice") as alice, websockets.connect(url + "?name=bob") as bob:
        await recv_until(alice, lambda m: m.get("kind") == "snapshot")
        await recv_until(bob, lambda m: m.get("kind") == "snapshot")

        # alice speaks; bob sees it in his stream — one shared timeline
        await alice.send(json.dumps({"op": "message", "text": "c_8812 double charge", "run": True}))
        msg, _ = await recv_until(bob, lambda m: m.get("type") == "message")
        assert msg["payload"]["text"] == "c_8812 double charge"

        # bob steers the running agent without restarting it
        await bob.send(json.dumps({"op": "steer", "text": "cap it at 75"}))
        await recv_until(alice, lambda m: m.get("type") == "steer")

        # the high-risk call blocks on a 2-human quorum
        req, _ = await recv_until(
            alice, lambda m: m.get("type") == "tool.requested" and m["payload"]["name"] == "issue_refund"
        )
        assert req["payload"]["quorum"] == 2
        assert req["payload"]["args"]["amount_usd"] == 75.0  # the live steer landed

        call_id = req["payload"]["id"]
        await alice.send(json.dumps({"op": "vote", "call_id": call_id, "approve": True}))
        await bob.send(json.dumps({"op": "vote", "call_id": call_id, "approve": True}))
        res, _ = await recv_until(alice, lambda m: m.get("type") == "tool.result" and m["payload"]["id"] == call_id)
        assert res["payload"]["result"]["amount_usd"] == 75.0

    # the room's history survives everyone leaving
    snap = httpx.get(f"{base}/api/sessions/{sid}", timeout=10).json()
    assert len(snap["log"]) > 10


async def test_fork_endpoint_carries_the_prefix(server):
    base, _ = server
    sid = httpx.post(base + "/api/sessions", json={"title": "forkme"}, timeout=10).json()["id"]
    child = httpx.post(f"{base}/api/sessions/{sid}/fork", timeout=10).json()
    assert child["forked_from"][0] == sid
    diff = httpx.get(f"{base}/api/sessions/{sid}/diff/{child['id']}", timeout=10).json()
    assert diff["common_prefix"] >= 1


async def test_index_and_tools_served(server):
    base, _ = server
    assert "parley" in httpx.get(base + "/", timeout=10).text.lower()
    tools = httpx.get(base + "/api/tools", timeout=10).json()
    assert any(t["risk"] == "high" for t in tools)
