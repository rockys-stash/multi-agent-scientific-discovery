"""Workflow tests: toolbox, critic, the Director with the rule reasoner, human checkpoints, and
the language-model reasoner against a stub client. Literature here is the invented fixture set."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from discoverylab import toolbox
from discoverylab.agents import Director, RunConfig
from discoverylab.critic import rule_critique
from discoverylab.human import CorrectionsFile, NoReviewer
from discoverylab.log import ProcessLog, verify
from discoverylab.models import Citation, Evidence, ExperimentDesign, RunState, Stage
from discoverylab.reasoners.model import ModelReasoner
from discoverylab.reasoners.rule import RuleReasoner
from discoverylab.run import execute, load_question


def _design(**kw: Any) -> ExperimentDesign:
    base = dict(id="D1", hypothesis_id="H1", datasets=["wine_binary"], model="random_forest", control="random_forest",
                treatments=["random_forest+isotonic"], metrics=["brier", "roc_auc"], test="paired_wilcoxon",
                folds=3, seeds=[0, 1])  # fmt: skip
    return ExperimentDesign(**{**base, **kw})


def test_toolbox_runs_designs_and_reports_consistent_numbers() -> None:
    r = toolbox.run_design(_design(), "R1")
    assert {(c.condition, c.metric) for c in r.conditions} >= {
        ("random_forest", "brier"),
        ("random_forest+isotonic", "roc_auc"),
    }
    for c in r.conditions:
        assert len(c.values) == 6 and c.lo <= c.mean <= c.hi
    [primary] = [c for c in r.comparisons if c["metric"] == "brier"]
    assert primary["p_holm"] is not None and primary["n"] == 6
    assert toolbox.run_design(_design(), "R1").comparisons == r.comparisons  # seeded: reproducible
    assert toolbox.validate(_design(datasets=["mnist"], treatments=["svm"], test="anova"))
    _X, y = toolbox.DATASETS["digits_binary_imb10"].loader()
    counts = sorted(int(n) for n in __import__("numpy").bincount(y))
    assert counts[0] <= 0.11 * counts[1]


def _run(registry, question, op, tmp_path: Path, human=None) -> tuple[RunState, ProcessLog]:  # type: ignore[no-untyped-def]
    log = ProcessLog(tmp_path / "log.jsonl")
    state = RunState(run_id="t", question=question, reasoner="rule")
    d = Director(registry, RuleReasoner(min_score=0.5), log, human or NoReviewer(), op, cfg=RunConfig(max_papers=10))
    return d.run(state), log


def test_rule_run_completes_end_to_end_and_logs_every_stage(registry, question, op, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    state, log = _run(registry, question, op, tmp_path)
    assert state.status == "complete", state.error
    assert state.papers and state.evidence and state.hypotheses and state.results and state.conclusions
    assert all(e.quote in e.claim for e in state.evidence)
    assert [s.value for s in state.completed] == ["literature", "evidence", "gaps", "hypotheses", "design",
                                                  "analysis", "conclusion", "critique"]  # fmt: skip
    events = list(log.events())
    stages = {e.stage for e in events if e.kind == "stage"}
    assert {"literature", "evidence", "gaps", "hypotheses", "design", "analysis", "conclusion", "critique"} <= stages
    assert verify(log.path).valid
    assert any(e.kind == "message" and e.data.get("checkpoint") == "no human reviewer configured" for e in events)
    assert not [c for c in state.critiques if c.severity == "blocking" and c.resolution == "open"]


def test_checkpoint_pauses_then_resumes_with_real_corrections(registry, question, op, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "corrections.json"
    state, log = _run(registry, question, op, tmp_path, CorrectionsFile(path))
    assert state.status == "awaiting_review" and state.stage == Stage.EVIDENCE
    first = state.evidence[0].id
    path.write_text(
        json.dumps({"evidence": [{"target_id": first, "action": "reject", "reason": "off topic", "actor": "tester"}]})
    )
    d = Director(registry, RuleReasoner(min_score=0.5), log, CorrectionsFile(path), op, cfg=RunConfig(max_papers=10))
    state = d.run(state)
    assert state.status == "awaiting_review" and state.stage == Stage.HYPOTHESES
    assert first not in {e.id for e in state.evidence}
    assert [c.action for c in state.corrections] == ["reject"] and state.corrections[0].before
    data = json.loads(path.read_text())
    data.update({"hypotheses": [], "design": [], "conclusion": []})
    path.write_text(json.dumps(data))
    state = d.run(state)
    assert state.status == "complete" and state.reviewed == {
        s: "file" for s in ("evidence", "hypotheses", "design", "conclusion")
    }
    assert sum(e.kind == "correction" for e in log.events()) == 1 and verify(log.path).valid


def test_rule_critic_catches_each_injected_fault(registry, question, op, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    clean, _ = _run(registry, question, op, tmp_path)
    for stage in Stage:
        assert not [c for c in rule_critique(clean, stage, 1, 1) if c.severity == "blocking"], stage

    def faults(state: RunState) -> dict[str, tuple[Stage, RunState]]:
        out = {}
        s = state.model_copy(deep=True)
        s.evidence.append(
            Evidence(id="E99", claim="x", quote="anything", citation=Citation(identifier="10.5555/never.retrieved"))
        )
        out["citation_not_retrieved"] = (Stage.EVIDENCE, s)
        s = state.model_copy(deep=True)
        s.evidence[0].quote = (
            s.evidence[0].quote.replace("reduces", "increases").replace("improved", "worsened") + " always"
        )
        out["quote_not_in_source"] = (Stage.EVIDENCE, s)
        s = state.model_copy(deep=True)
        s.hypotheses[0].gap_ids = []
        out["ungrounded_hypothesis"] = (Stage.HYPOTHESES, s)
        s = state.model_copy(deep=True)
        s.designs[0].treatments = [s.designs[0].control]
        out["no_control"] = (Stage.DESIGN, s)
        s = state.model_copy(deep=True)
        s.designs[0].metrics = ["accuracy", "brier"]
        out["metric_mismatch"] = (Stage.DESIGN, s)
        s = state.model_copy(deep=True)
        s.results[0].conditions[0].mean += 0.05
        out["arithmetic_error"] = (Stage.ANALYSIS, s)
        s = state.model_copy(deep=True)
        s.conclusions[0].verdict = "supported" if s.conclusions[0].verdict != "supported" else "not_supported"
        out["overreach" if s.conclusions[0].verdict == "supported" else "verdict_mismatch"] = (Stage.CONCLUSION, s)
        return out

    for issue, (stage, broken) in faults(clean).items():
        assert issue in {c.issue for c in rule_critique(broken, stage, 1, 1)}, issue


class _StubMessages:
    """Answers ``messages.parse`` from canned outputs keyed by schema name."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    def parse(self, **kw: Any) -> Any:
        schema = kw["output_format"]
        self.calls.append(schema.__name__)
        data = self.answers[schema.__name__]
        return SimpleNamespace(parsed_output=schema.model_validate(data), stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))  # fmt: skip


def test_claude_reasoner_maps_schema_outputs_and_the_verifier_catches_a_bad_copy(registry, question) -> None:  # type: ignore[no-untyped-def]
    answers = {
        "_Queries": {"queries": ["random forest calibration", "isotonic regression brier"]},
        "_EvidenceList": {"items": [
            {"paper_id": "doi:10.5555/test.1", "title": "Calibration of random forest probabilities with isotonic regression",
             "first_author": "Ada Example", "year": 2015, "quote": "isotonic regression reduces the Brier score of random forest classifiers",
             "claim": "Isotonic regression lowers RF Brier score.", "concepts": ["calibration"], "stance": "supports"},
            {"paper_id": "doi:10.5555/test.2", "title": "Platt scaling for tree ensembles", "first_author": "Chen Placeholder",
             "year": 2011, "quote": "Platt scaling always improves ranking quality", "claim": "Platt improves AUC.",
             "concepts": ["platt scaling"], "stance": "supports"}]},
    }  # fmt: skip
    stub = _StubMessages(answers)
    r = ModelReasoner(client=SimpleNamespace(messages=stub))
    assert r.plan_queries(question) == ["random forest calibration", "isotonic regression brier"]
    papers = registry.search("calibration", 10)
    ev = r.extract_evidence(question, papers, 1)
    assert [e.id for e in ev] == ["E1", "E2"] and r.usage.calls == 2 and r.usage.input_tokens == 200
    assert len(r.transcript) == 2 and r.transcript[1]["output"]["items"][0]["stance"] == "supports"
    from discoverylab.verify import Verifier

    checks = [Verifier(registry).check_evidence(e) for e in ev]
    assert checks[0].correct
    assert checks[1].citation.status == "metadata_mismatch" and checks[1].quote == "not_found"


def test_execute_writes_state_log_and_verification(registry, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    q, op = load_question(Path("configs/questions/q1_calibration.yaml"))
    op.datasets = ["wine_binary"]  # type: ignore[union-attr]
    state = RunState(run_id="x", question=q, reasoner="rule")
    out = execute(
        tmp_path / "x", state, registry, RuleReasoner(min_score=0.5), op, NoReviewer(), cfg=RunConfig(max_papers=10)
    )
    assert out.status == "complete"
    ver = json.loads((tmp_path / "x" / "verification.json").read_text())
    assert ver["evidence"] and all(c["correct"] for c in ver["evidence"])
    assert RunState.model_validate_json((tmp_path / "x" / "state.json").read_text()).status == "complete"


@pytest.mark.parametrize("name", ["q1_calibration", "q2_scaling_knn", "q3_imbalance"])
def test_question_configs_are_runnable(name: str) -> None:
    q, op = load_question(Path(f"configs/questions/{name}.yaml"))
    assert op is not None and q.concepts
    d = ExperimentDesign(id="D", hypothesis_id="H", datasets=op.datasets, model=op.control.split("+")[0], control=op.control,
                         treatments=op.treatments, metrics=op.metrics, test="paired_wilcoxon")  # fmt: skip
    assert toolbox.validate(d) == []
