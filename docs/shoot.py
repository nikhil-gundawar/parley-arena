"""Drives two real browser clients through a room and screenshots it.

    python docs/shoot.py
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def enter(page, base, sid, name):
    await page.goto(f"{base}/?s={sid}&name={name}")
    await page.fill("#nm", name)
    await page.fill("#rid", sid)
    await page.click("button.primary")
    await page.wait_for_timeout(600)


async def main():
    port = free_port()
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PARLEY_AGENT": "demo", "PARLEY_QUORUM": "2"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "parley.server:app", "--port", str(port), "--log-level", "error"],
        cwd=ROOT, env=env,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(base + "/api/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)

    sid = httpx.post(base + "/api/sessions", json={"title": "refund escalation", "quorum": 2}).json()["id"]

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=["--no-sandbox"], executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
            alice_ctx = await browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
            bob_ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
            alice, bob = await alice_ctx.new_page(), await bob_ctx.new_page()

            await enter(alice, base, sid, "alice")
            await enter(bob, base, sid, "bob")
            await enter(await (await browser.new_context(viewport={"width": 900, "height": 700})).new_page(), base, sid, "carol")

            await alice.fill("#box", "c_8812 was double charged, sort it out")
            await alice.click("button.primary:has-text('Send')")
            await alice.wait_for_timeout(1200)

            await bob.fill("#box", "cap it at 75, we're not refunding the whole thing")
            await bob.click("button:has-text('Steer')")

            # wait for the high-risk call to land in the approvals rail
            for _ in range(80):
                if await alice.locator(".card.pending").count():
                    break
                await alice.wait_for_timeout(150)
            await alice.wait_for_timeout(400)

            out = ROOT / "docs" / "screenshot.png"
            await alice.screenshot(path=str(out))
            print("wrote", out)

            # a second frame: after both approve
            await alice.locator(".card.pending .yes").first.click()
            await bob.locator(".card.pending .yes").first.click()
            await alice.wait_for_timeout(1500)
            out2 = ROOT / "docs" / "screenshot-approved.png"
            await alice.screenshot(path=str(out2))
            print("wrote", out2)

            await browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


asyncio.run(main())
