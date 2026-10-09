"""Capture screenshots of the running console from the real runs in runs/ and results in results/.

Usage: uv run python scripts/screenshots.py [--out docs/screenshots]
Needs the built console (npm --prefix web run build), Chromium for Playwright, and E1's runs.
"""

from __future__ import annotations

import argparse
import json
import socket
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn
from playwright.sync_api import ViewportSize, sync_playwright

from discoverylab.api import create_app

ROOT = Path(__file__).resolve().parents[1]
VIEWPORTS: dict[str, ViewportSize] = {
    "desktop": {"width": 1440, "height": 900},
    "mobile": {"width": 390, "height": 844},
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "screenshots")
    args = ap.parse_args()
    latest = (ROOT / "results" / "e1_runs" / "LATEST").read_text(encoding="utf-8").strip()
    rows = json.loads((ROOT / "results" / "e1_runs" / latest / "runs.json").read_text(encoding="utf-8"))
    run_ids = [r["run_id"] for r in rows if r["status"] == "complete"]
    if not run_ids:
        raise SystemExit("no completed E1 run to show")
    # Show only the runs of the latest E1 result; older runs in runs/ are superseded (DECISIONS D13).
    shown = Path(tempfile.mkdtemp(prefix="lattice-runs-"))
    for rid in run_ids:
        (shown / rid).symlink_to(ROOT / "runs" / rid, target_is_directory=True)
    app = create_app(shown, results=ROOT / "results")
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/api/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    first = run_ids[0]
    shots = [("runs", "/runs"), ("run-network", f"/runs/{first}"), ("run-process", f"/runs/{first}/process"),
             ("run-citations", f"/runs/{run_ids[-1]}/citations"), ("run-corrections", f"/runs/{first}/corrections"),
             ("evaluation", "/evaluation"), ("method", "/method")]  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for vp, size in VIEWPORTS.items():
            for scheme in ("light", "dark"):
                if vp == "mobile" and scheme == "dark":
                    continue
                ctx = browser.new_context(viewport=size, color_scheme=scheme, device_scale_factor=1)
                page = ctx.new_page()
                for name, path in shots:
                    page.goto(base + path)
                    page.locator("h1").first.wait_for(timeout=30000)
                    page.wait_for_load_state("networkidle")
                    if page.locator(".skeleton").count():
                        page.locator(".skeleton").first.wait_for(state="detached", timeout=30000)
                    out = args.out / f"{name}-{vp}{'-dark' if scheme == 'dark' else ''}.png"
                    page.screenshot(path=str(out), full_page=vp == "desktop")
                    print(out.relative_to(ROOT))
                ctx.close()
        browser.close()
    server.should_exit = True


if __name__ == "__main__":
    main()
