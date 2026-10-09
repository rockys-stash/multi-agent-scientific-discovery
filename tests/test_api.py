"""API tests against runs built from the invented fixture literature (see conftest)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from discoverylab.agents import RunConfig
from discoverylab.api import create_app
from discoverylab.human import CorrectionsFile, NoReviewer
from discoverylab.models import RunState
from discoverylab.reasoners.rule import RuleReasoner
from discoverylab.run import execute


@pytest.fixture
def runs(tmp_path: Path, registry, question, op) -> Path:  # type: ignore[no-untyped-def]
    root = tmp_path / "runs"
    cfg = RunConfig(max_papers=10)
    execute(root / "done", RunState(run_id="done", question=question, reasoner="rule"), registry,
            RuleReasoner(min_score=0.5), op, NoReviewer(), cfg=cfg)  # fmt: skip
    execute(root / "paused", RunState(run_id="paused", question=question, reasoner="rule"), registry,
            RuleReasoner(min_score=0.5), op, CorrectionsFile(tmp_path / "none.json"), cfg=cfg)  # fmt: skip
    return root


def _client(runs: Path, tmp_path: Path) -> TestClient:
    return TestClient(create_app(runs, results=tmp_path / "results", static_dir=tmp_path / "dist"))


def test_runs_graph_and_log(runs: Path, tmp_path: Path) -> None:
    c = _client(runs, tmp_path)
    assert c.get("/api/health").json() == {"status": "ok"}
    items = {r["run_id"]: r for r in c.get("/api/runs").json()["items"]}
    assert items["done"]["status"] == "complete" and items["paused"]["status"] == "awaiting_review"
    assert items["done"]["citation_accuracy"]["rate"] == 1.0
    assert items["done"]["corrections"]["evidence"] is None  # no human reviewed: not reported as zero
    run = c.get("/api/runs/done").json()
    kinds = {n["kind"] for n in run["graph"]["nodes"]}
    assert {"question", "paper", "evidence", "gap", "hypothesis", "design", "result", "conclusion"} <= kinds
    ids = {n["id"] for n in run["graph"]["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in run["graph"]["edges"])
    assert {e["verification"] for e in run["graph"]["edges"] if e["rel"] == "quotes"} == {"verified"}
    assert run["log_integrity"]["valid"] and run["metrics"]["design_validity"]["D1"]["passed"] >= 7
    log = c.get("/api/runs/done/log").json()
    assert log["integrity"]["valid"] and log["events"][0]["kind"] == "stage"
    r = c.get("/api/runs/done")
    assert r.headers["x-frame-options"] == "DENY" and "script-src 'self'" in r.headers["content-security-policy"]


def test_input_validation_and_not_found(runs: Path, tmp_path: Path) -> None:
    c = _client(runs, tmp_path)
    assert c.get("/api/runs/nope").status_code == 404
    assert c.get("/api/runs/..%2F..%2Fetc").status_code in (400, 404)
    assert c.get("/api/runs/bad id!").status_code == 400
    assert c.get("/api/experiments/e1_runs").status_code == 404
    assert {e["status"] for e in c.get("/api/experiments").json()["items"]} == {"pending"}
    assert c.get("/api/whatever").status_code == 404


def test_corrections_need_a_token_and_a_paused_run(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"stage": "evidence", "actor": "reviewer", "corrections": [{"stage": "evidence", "target_id": "E1", "action": "reject",
                                                                         "reason": "off topic", "actor": "x"}]}  # fmt: skip
    assert _client(runs, tmp_path).post("/api/runs/paused/corrections", json=body).status_code == 401
    monkeypatch.setenv("DISCOVERYLAB_API_TOKEN", "t" * 32)
    c = _client(runs, tmp_path)
    h = {"Authorization": "Bearer " + "t" * 32}
    assert c.get("/api/runs").status_code == 401 and c.get("/api/health").status_code == 200
    assert c.post("/api/runs/done/corrections", json=body, headers=h).status_code == 409
    r = c.post("/api/runs/paused/corrections", json=body, headers=h)
    assert r.status_code == 200 and "discoverylab resume" in r.json()["resume"]
    saved = json.loads((runs / "paused" / "corrections.json").read_text())
    assert saved["evidence"][0]["actor"] == "reviewer" and saved["evidence"][0]["action"] == "reject"


def test_rate_limit(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCOVERYLAB_RATE_PER_SEC", "0.001")
    c = _client(runs, tmp_path)
    codes = [c.get("/api/health").status_code for _ in range(125)]
    assert codes.count(429) >= 4


def test_serve_refuses_a_public_host_without_a_token(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from discoverylab.cli import main

    monkeypatch.delenv("DISCOVERYLAB_API_TOKEN", raising=False)
    assert main(["serve", "--host", "0.0.0.0"]) == 2
    assert "DISCOVERYLAB_API_TOKEN" in capsys.readouterr().err


def test_experiment_with_a_pending_part_is_partial(runs: Path, tmp_path: Path) -> None:
    for exp, notes in (("e2_verifier", []), ("e4_fluency", ["Convincingness: Status pending."])):
        d = tmp_path / "results" / exp / "r1"
        d.mkdir(parents=True)
        (tmp_path / "results" / exp / "LATEST").write_text("r1\n")
        (d / "provenance.json").write_text(json.dumps({"commit": "abc", "dirty": False}))
        (d / "summary.json").write_text(
            json.dumps({"title": exp, "question": "", "findings": [], "tables": [], "notes": notes})
        )
    items = {i["experiment"]: i["status"] for i in _client(runs, tmp_path).get("/api/experiments").json()["items"]}
    assert items["e2_verifier"] == "complete" and items["e4_fluency"] == "partial" and items["e1_runs"] == "pending"
