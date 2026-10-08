"""Browser tests for the Lattice console.

They serve two runs made from the invented fixture literature in ``tests/conftest.py`` (one
complete, one paused for review), so they test the console, not any finding. Run with
``uv run pytest -m ui`` after ``npm --prefix web run build``.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import ALL_FIXTURE, PAPERS, FixtureSource  # type: ignore[import-not-found]

from discoverylab.agents import RunConfig
from discoverylab.human import CorrectionsFile, NoReviewer
from discoverylab.literature.registry import SourceRegistry
from discoverylab.models import RunState
from discoverylab.reasoners.rule import RuleReasoner
from discoverylab.run import execute, load_question

ROOT = Path(__file__).resolve().parents[2]
AXE = ROOT / "web" / "node_modules" / "axe-core" / "axe.min.js"
pytestmark = pytest.mark.ui
playwright_api = pytest.importorskip("playwright.sync_api")

VIEWPORTS = {"desktop": {"width": 1440, "height": 900}, "mobile": {"width": 390, "height": 844}}
DONE, PAUSED = "fixture-complete", "fixture-paused"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def base_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    if not (ROOT / "web" / "dist" / "index.html").exists():
        pytest.skip("frontend not built (npm --prefix web run build)")
    import uvicorn

    from discoverylab.api import create_app

    tmp = tmp_path_factory.mktemp("lab")
    registry = SourceRegistry([FixtureSource(PAPERS)], resolvers=ALL_FIXTURE)
    q, op = load_question(ROOT / "configs" / "questions" / "q1_calibration.yaml")
    assert op is not None
    op.datasets = ["wine_binary"]
    cfg = RunConfig(max_papers=10)
    execute(tmp / "runs" / DONE, RunState(run_id=DONE, question=q, reasoner="rule"), registry, RuleReasoner(min_score=0.5), op,
            NoReviewer(), cfg=cfg)  # fmt: skip
    review = tmp / "review.json"
    review.write_text(json.dumps({"evidence": []}), encoding="utf-8")
    execute(tmp / "runs" / PAUSED, RunState(run_id=PAUSED, question=q, reasoner="rule"), registry, RuleReasoner(min_score=0.5),
            op, CorrectionsFile(review), cfg=cfg)  # fmt: skip

    mp = pytest.MonkeyPatch()
    mp.delenv("DISCOVERYLAB_API_TOKEN", raising=False)
    app = create_app(tmp / "runs", results=tmp / "results")
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"{url}/api/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    yield url
    server.should_exit = True
    thread.join(timeout=5)
    mp.undo()


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with playwright_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


PAGES = [
    ("runs", "/runs", "Citations verified"),
    ("network", f"/runs/{DONE}", "verified quote"),
    ("process", f"/runs/{DONE}/process", "Critique cycle"),
    ("citations", f"/runs/{DONE}/citations", "Every emitted citation"),
    ("corrections", f"/runs/{DONE}/corrections", "No human review"),
    ("paused", f"/runs/{PAUSED}/corrections", "Paused at"),
    ("evaluation", "/evaluation", "No experiment has been run"),
    ("method", "/method", "Limits"),
    ("missing-run", "/runs/no-such-run", "Run not found"),
    ("not-found", "/nowhere", "Page not found"),
]


def _open(browser: Any, base_url: str, viewport: str, path: str, marker: str, scheme: str = "light") -> Any:
    ctx = browser.new_context(viewport=VIEWPORTS[viewport], color_scheme=scheme)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "404" not in m.text else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(base_url + path)
    page.get_by_text(marker, exact=False).first.wait_for(timeout=30000)
    page.wait_for_load_state("networkidle")
    page.errors = errors
    return page


@pytest.mark.parametrize("viewport", list(VIEWPORTS))
@pytest.mark.parametrize(("name", "path", "marker"), PAGES)
def test_page_renders(browser: Any, base_url: str, viewport: str, name: str, path: str, marker: str) -> None:
    page = _open(browser, base_url, viewport, path, marker)
    assert page.locator("h1").count() == 1
    assert not page.locator(".skeleton").count(), "loading skeleton still visible"
    if name not in ("missing-run", "not-found"):
        assert not page.locator("[role=alert]").count(), "error state rendered"
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 1, f"page scrolls horizontally by {overflow}px"
    assert not page.errors, page.errors
    page.context.close()


def test_graph_is_keyboard_navigable(browser: Any, base_url: str) -> None:
    page = _open(browser, base_url, "desktop", f"/runs/{DONE}", "verified quote")
    nodes = page.locator("g.node")
    assert nodes.count() > 5
    question = page.locator("g.node.k-question")
    question.focus()
    page.keyboard.press("ArrowRight")
    focused = page.evaluate("document.activeElement.getAttribute('aria-label')")
    assert focused.startswith("Paper"), focused
    page.keyboard.press("ArrowRight")
    assert page.evaluate("document.activeElement.getAttribute('aria-label')").startswith("Evidence")
    page.keyboard.press("Enter")
    inspector = page.locator("aside.inspector")
    inspector.wait_for()
    assert "Verified" in inspector.inner_text() and "Resolved by" in inspector.inner_text()
    page.keyboard.press("Escape")
    inspector.wait_for(state="detached")
    page.context.close()


def test_mobile_starts_in_list_view_and_opens_the_inspector(browser: Any, base_url: str) -> None:
    page = _open(browser, base_url, "mobile", f"/runs/{DONE}", "verified quote")
    assert not page.locator("svg.graph").count()
    page.locator(".node-list .link-btn").first.click()
    page.locator("aside.inspector").wait_for()
    page.context.close()


def test_paused_run_tells_the_reviewer_how_to_resume(browser: Any, base_url: str) -> None:
    page = _open(browser, base_url, "desktop", f"/runs/{PAUSED}/corrections", "Paused at")
    text = page.locator("main").inner_text()
    assert (
        "Waiting for review" in text and f"discoverylab resume runs/{PAUSED}" in text and "reviewed by a person" in text
    )
    page.context.close()


@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.parametrize("viewport", list(VIEWPORTS))
def test_no_wcag_violations(browser: Any, base_url: str, viewport: str, scheme: str) -> None:
    """axe-core audit (WCAG 2.1 A/AA and best practices) of every view in both themes."""
    if not AXE.exists():
        pytest.skip("axe-core not installed (npm --prefix web ci)")
    ctx = browser.new_context(viewport=VIEWPORTS[viewport], color_scheme=scheme, bypass_csp=True)
    page = ctx.new_page()
    found: dict[str, list[str]] = {}
    for _, path, marker in PAGES:
        page.goto(base_url + path)
        page.get_by_text(marker, exact=False).first.wait_for(timeout=30000)
        page.wait_for_load_state("networkidle")
        page.add_script_tag(path=str(AXE))
        violations = page.evaluate(
            """async () => (await axe.run(document, {runOnly: ['wcag2a', 'wcag2aa', 'wcag21aa',
               'best-practice']})).violations.map(v => v.id + ': ' + v.nodes.map(n => n.target.join(' ')).join(', '))"""
        )
        if violations:
            found[path] = violations
    ctx.close()
    assert not found, found
