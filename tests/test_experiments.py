"""Experiment harness tests on the invented fixture literature (see conftest). They check the
harness's mechanics: provenance, corruption and fault injection, replay. Nothing here is a result."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from discoverylab.agents import RunConfig
from discoverylab.experiments import e1_runs, e2_verifier, e3_critic, e4_fluency, e5_reproducibility
from discoverylab.experiments.common import load_run
from discoverylab.judges import RUBRIC, ModelJudge
from discoverylab.reasoners.rule import RuleReasoner
from discoverylab.report import flesch_reading_ease, render_report


@pytest.fixture
def lab(tmp_path: Path, question, op) -> Path:  # type: ignore[no-untyped-def]
    qdir = tmp_path / "configs" / "questions"
    qdir.mkdir(parents=True)
    raw = {**question.model_dump(), "operationalisation": op.__dict__}
    (qdir / f"{question.id}.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return tmp_path


def _rule(name: str, client: Any = None) -> RuleReasoner:
    assert name == "rule"
    return RuleReasoner(min_score=0.5)


@pytest.fixture
def e1(lab: Path, registry, monkeypatch: pytest.MonkeyPatch) -> Path:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = {"questions": ["configs/questions/qt.yaml"], "reasoners": [{"reasoner": "rule"}, {"reasoner": "claude", "repeats": 2}],
           "run_config": {"max_papers": 10}}  # fmt: skip
    return e1_runs.run(cfg, lab / "runs", registry, _rule, results=lab / "results", root=lab)


def test_e1_writes_provenance_and_records_skipped_reasoners(e1: Path, lab: Path) -> None:
    prov = json.loads((e1 / "provenance.json").read_text())
    assert prov["experiment"] == "e1_runs" and len(prov["config_sha256"]) == 64 and prov["runtime_seconds"] >= 0
    assert (lab / "results" / "e1_runs" / "LATEST").read_text().strip() == e1.name
    rows = json.loads((e1 / "runs.json").read_text())
    assert [r["status"] for r in rows] == ["complete", "skipped", "skipped"]
    assert rows[1]["reason"] == "ANTHROPIC_API_KEY is not set" and rows[1]["run_id"] is None
    summary = json.loads((e1 / "summary.json").read_text())
    [runs_table, not_run] = summary["tables"]
    assert runs_table["rows"][0]["citation_accuracy"] == 1.0 and len(not_run["rows"]) == 2


def test_failed_experiment_leaves_no_latest_pointer(lab: Path, registry) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError):
        e1_runs.run({"questions": ["configs/questions/missing.yaml"], "reasoners": [{"reasoner": "rule"}]}, lab / "runs", registry, _rule,
                    results=lab / "results", root=lab)  # fmt: skip
    [d] = (lab / "results" / "e1_runs").iterdir()
    assert (d / "FAILED").is_file() and not (lab / "results" / "e1_runs" / "LATEST").exists()


def test_e2_rejects_every_corruption_and_accepts_benign_variants(e1: Path, lab: Path, registry) -> None:  # type: ignore[no-untyped-def]
    out = e2_verifier.run({"seed": 0}, lab / "runs", registry, results=lab / "results")
    trials = json.loads((out / "trials.json").read_text())
    by = {(t["variant"], t["accepted"]) for t in trials}
    assert {v for v, _ in by} >= set(e2_verifier.CORRUPTIONS) | {"unchanged", "reformatted_title", "preprint_year"}
    assert not [t for t in trials if t["group"] == "corruption" and t["accepted"]]
    assert all(t["accepted"] for t in trials if t["group"] == "benign")


def test_e3_catches_structural_faults_and_reports_semantic_ones_as_missed(e1: Path, lab: Path) -> None:
    out = e3_critic.run({"seed": 1, "max_targets_per_fault": 2}, lab / "runs", results=lab / "results")
    trials = json.loads((out / "trials.json").read_text())
    structural = [t for t in trials if t["kind"] == "structural"]
    assert structural and all(t["caught"] for t in structural), [t for t in structural if not t["caught"]]
    assert not any(t["caught"] for t in trials if t["kind"] == "semantic")
    clean = json.loads((out / "clean.json").read_text())
    assert all(not v for c in clean for v in c["false_alarms"].values())


class _Judge:
    name = "stub"

    def rate(self, report: str) -> dict[str, Any]:
        return {"convincingness": 7 + ("confirms" in report), "clarity": 7, "rationale": "stub"}


def test_e4_process_score_falls_while_the_report_stays_readable(e1: Path, lab: Path, registry) -> None:  # type: ignore[no-untyped-def]
    out = e4_fluency.run({"seed": 0}, lab / "runs", registry, judge=_Judge(), results=lab / "results")
    rows = {r["variant"]: r for r in json.loads((out / "variants.json").read_text())}
    assert rows["original"]["process"] == 1.0
    assert rows["fabricated_citations"]["evidence_correctness"] == 0.0
    assert rows["all_three"]["process"] < rows["original"]["process"]
    assert rows["original"]["readability"] is not None and rows["original"]["convincingness"] == 7


def test_e5_replays_a_rule_run_identically_offline(e1: Path, lab: Path, registry) -> None:  # type: ignore[no-untyped-def]
    out = e5_reproducibility.run({"run_config": {"max_papers": 10}}, lab / "runs", registry, root=lab, results=lab / "results",
                                 reasoner_factory=_rule)  # fmt: skip
    [row] = json.loads((out / "replays.json").read_text())
    assert row["replayed"] and row["different"] == [] and row["verification_identical"]


def test_e5_detects_a_run_that_does_not_reproduce(e1: Path, lab: Path, registry) -> None:  # type: ignore[no-untyped-def]
    [run_dir] = [p for p in (lab / "runs").iterdir()]
    state, _ = load_run(run_dir)
    state.evidence[0].claim = "edited after the fact"
    (run_dir / "state.json").write_text(state.model_dump_json(), encoding="utf-8")
    row = e5_reproducibility.replay(run_dir, lab / "replays", registry, lab, RunConfig(max_papers=10), _rule)
    assert row["different"] == ["evidence"]


def test_replay_client_serves_the_transcript_and_refuses_a_changed_prompt() -> None:
    from discoverylab.reasoners.model import _Queries

    client = e5_reproducibility.ReplayClient([{"step": "plan_queries", "prompt": "p1", "output": {"queries": ["a"]}}])
    resp = client.messages.parse(
        model="m", max_tokens=1, system="s", messages=[{"role": "user", "content": "p1"}], output_format=_Queries
    )
    assert resp.parsed_output.queries == ["a"]
    with pytest.raises(e5_reproducibility.TranscriptMismatchError):
        client.messages.parse(
            model="m", max_tokens=1, system="s", messages=[{"role": "user", "content": "p1"}], output_format=_Queries
        )
    other = e5_reproducibility.ReplayClient([{"step": "plan_queries", "prompt": "p1", "output": {"queries": ["a"]}}])
    with pytest.raises(e5_reproducibility.TranscriptMismatchError):
        other.messages.parse(
            model="m", max_tokens=1, system="s", messages=[{"role": "user", "content": "p2"}], output_format=_Queries
        )


def test_report_and_readability(e1: Path, lab: Path) -> None:
    [run_dir] = list((lab / "runs").iterdir())
    state, _ = load_run(run_dir)
    text = render_report(state)
    assert state.question.text in text and state.evidence[0].quote in text
    assert flesch_reading_ease("The cat sat on the mat. It was warm.") > flesch_reading_ease(  # type: ignore[operator]
        "Comprehensive probabilistic recalibration methodologies substantially ameliorate miscalibration."
    )
    assert flesch_reading_ease("") is None
    shutil.rmtree(run_dir)


class _StubJudgeClient:
    """Answers the judge's two prompt types with fixed, schema-valid outputs."""

    class messages:  # noqa: N801 - mirrors the SDK's attribute name
        @staticmethod
        def parse(**kw: Any) -> Any:
            schema = kw["output_format"]
            if "hypothesis" in kw["messages"][0]["content"]:
                data: dict[str, Any] = {
                    "scores": [
                        {"criterion": c, "score": 2 if c != "grounding" else 1, "reason": "stub"} for c in RUBRIC
                    ]
                }
            else:
                data = {"verdict": "supports", "reason": "stub"}
            return type("R", (), {"parsed_output": schema.model_validate(data)})()


def test_e1_reports_judged_metrics_when_a_judge_is_available(lab: Path, registry) -> None:  # type: ignore[no-untyped-def]
    cfg = {
        "questions": ["configs/questions/qt.yaml"],
        "reasoners": [{"reasoner": "rule"}],
        "run_config": {"max_papers": 10},
    }
    out = e1_runs.run(cfg, lab / "runs", registry, _rule, results=lab / "results", root=lab,
                      judge_factory=lambda: ModelJudge(client=_StubJudgeClient(), model="stub"))  # fmt: skip
    [m] = json.loads((out / "metrics.json").read_text())
    assert m["hypothesis_quality"]["mean_of_10"] == 9 and m["evidence_support"]["supports"] == m["evidence"]
    [row] = json.loads((out / "summary.json").read_text())["tables"][0]["rows"]
    assert row["hypothesis_quality"] == 9


def test_results_markdown_renders_latest_and_marks_missing(tmp_path: Path) -> None:
    from discoverylab.experiments.results_md import render

    d = tmp_path / "e2_verifier" / "20260101T000000Z-abc1234"
    d.mkdir(parents=True)
    (tmp_path / "e2_verifier" / "LATEST").write_text(d.name + "\n")
    summary = {"title": "E2 Citation verifier", "question": "Q?", "findings": ["Corruptions rejected: 3/3."],
               "tables": [{"id": "v", "caption": "By variant", "columns": [{"key": "variant", "label": "Variant", "align": "l"},
                          {"key": "rejected", "label": "Rejected", "align": "r"}], "rows": [{"variant": "a|b", "rejected": 1.0}]}],
               "notes": []}  # fmt: skip
    (d / "summary.json").write_text(json.dumps(summary))
    (d / "provenance.json").write_text(
        json.dumps({"commit": "abc1234def", "dirty": False, "created_at": "t", "runtime_seconds": 1})
    )
    md = render(tmp_path)
    assert "| a\\|b | 1.000 |" in md and "Corruptions rejected: 3/3." in md and "commit `abc1234`" in md
    assert "## e1_runs" in md and "Status: pending" in md
