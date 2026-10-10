"""The local open-weights path without weights: a fake llama.cpp model stands in, so these check the
adapter's contract (schema, stop reasons, retries, availability) and the E3 model-critic arm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from discoverylab.experiments import e1_runs, e3_critic
from discoverylab.models import Critique, RunState, Stage
from discoverylab.reasoners import local
from discoverylab.reasoners.model import ModelReasoner, _Queries
from discoverylab.reasoners.rule import RuleReasoner
from discoverylab.run import make_reasoner, unavailable


class _FakeLlama:
    """Returns queued completions in order and records what it was asked."""

    def __init__(self, contents: list[tuple[str, str]]) -> None:
        self.contents = contents
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        content, finish = self.contents.pop(0)
        return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}  # fmt: skip


@pytest.fixture
def weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "fake.gguf"
    p.write_bytes(b"GGUF")
    monkeypatch.setenv("DISCOVERYLAB_LOCAL_MODEL", str(p))
    return p


def _client(weights: Path, monkeypatch: pytest.MonkeyPatch, contents: list[tuple[str, str]]) -> tuple[Any, _FakeLlama]:
    fake = _FakeLlama(contents)
    monkeypatch.setattr(local, "_load", lambda path, n_ctx=0: fake)
    return local.LocalClient(weights, seed=3), fake


def test_unavailable_names_what_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISCOVERYLAB_LOCAL_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert unavailable("local") == "DISCOVERYLAB_LOCAL_MODEL is not set to a GGUF file"
    assert unavailable("claude") == "ANTHROPIC_API_KEY is not set"
    assert unavailable("rule") is None


def test_local_client_constrains_to_the_schema_and_logs_the_seed(
    weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, fake = _client(weights, monkeypatch, [('{"queries": ["a b", "c"]}', "stop")])
    r = make_reasoner("local", client)
    assert r.name == "local" and r.plan_queries(_question()) == ["a b", "c"]
    [call] = fake.calls
    assert call["response_format"]["schema"] == _Queries.model_json_schema()
    assert call["messages"][0]["role"] == "system"
    [rec] = r.transcript  # type: ignore[attr-defined]
    assert rec["model"] == "fake" and isinstance(rec["seed"], int)


def test_unusable_answer_is_retried_once_then_fails_the_step(weights: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _client(weights, monkeypatch, [("{", "length"), ('{"queries": ["x"]}', "stop")])
    r = ModelReasoner(client=client, model="fake", name="local")
    assert r.plan_queries(_question()) == ["x"]
    assert r.usage.calls == 2 and r.usage.by_step["retry:plan_queries"] == 1

    client, _ = _client(weights, monkeypatch, [('{"wrong": 1}', "stop"), ('{"wrong": 2}', "stop")])
    r = ModelReasoner(client=client, model="fake", name="local")
    with pytest.raises(RuntimeError, match="no usable output"):
        r.plan_queries(_question())


def test_e1_records_local_arm_as_skipped_without_weights(tmp_path: Path, registry, question, op,  # type: ignore[no-untyped-def]
                                                          monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    monkeypatch.delenv("DISCOVERYLAB_LOCAL_MODEL", raising=False)
    q = tmp_path / "configs" / "questions"
    q.mkdir(parents=True)
    (q / "qt.yaml").write_text(yaml.safe_dump({**question.model_dump(), "operationalisation": op.__dict__}))
    cfg = {"questions": ["configs/questions/qt.yaml"], "reasoners": [{"reasoner": "local", "model_critic": "local"}],
           "run_config": {"max_papers": 10}}  # fmt: skip
    out = e1_runs.run(cfg, tmp_path / "runs", registry, results=tmp_path / "results", root=tmp_path)
    [row] = json.loads((out / "runs.json").read_text())
    assert row["status"] == "skipped" and row["reasoner"] == "local+mc" and "DISCOVERYLAB_LOCAL_MODEL" in row["reason"]


class _FlagEverything:
    """A model critic that flags the first artefact of whatever stage it reviews."""

    name, model = "stub", "stub"

    def critique(self, state: RunState, stage: Stage, round_: int, start: int) -> list[Critique]:
        ids = {Stage.EVIDENCE: [e.id for e in state.evidence], Stage.HYPOTHESES: [h.id for h in state.hypotheses],
               Stage.DESIGN: [d.id for d in state.designs], Stage.CONCLUSION: [k.id for k in state.conclusions],
               Stage.GAPS: [g.id for g in state.gaps]}.get(stage, [])  # fmt: skip
        return [Critique(id=f"C{start}", round=round_, stage=stage, reviewer="model", target_id=ids[0],
                         issue="stub", severity="major", message="stub")] if ids else []  # fmt: skip


def test_e3_model_critic_reviews_the_same_faulted_states_and_leaves_rule_trials_unchanged(  # type: ignore[no-untyped-def]
    tmp_path: Path, registry, question, op
) -> None:
    from discoverylab.human import NoReviewer
    from discoverylab.run import execute

    state = execute(tmp_path / "r1", RunState(run_id="r1", question=question, reasoner="rule"), registry,
                    RuleReasoner(min_score=0.5), op, NoReviewer())  # fmt: skip
    plain, _ = e3_critic.evaluate([state], 1, 2)
    with_model, clean = e3_critic.evaluate([state], 1, 2, _FlagEverything(), 2)  # type: ignore[arg-type]
    strip = [{k: v for k, v in t.items() if not k.startswith("model_")} for t in with_model]
    assert strip == plain
    judged = [t for t in with_model if "model_flagged_target" in t]
    assert judged and all(t["fault"] != "misreported_mean" for t in judged)  # analysis is not a model stage
    assert all("model_critiques" in c for c in clean)


def _question() -> Any:
    from discoverylab.models import Question

    return Question(id="q", text="Does X improve Y?", keywords=["x"], concepts=["x", "y"])


def test_batched_evidence_extraction_splits_records_and_numbers_continuously(
    weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conftest import PAPERS

    from discoverylab.run import split_name

    assert split_name("local-b5") == ("local", 5) and split_name("local") == ("local", None)
    papers = [p for p in PAPERS if p.abstract][:3]
    item = {"paper_id": papers[0].id, "title": papers[0].title, "first_author": "A", "year": 2020, "quote": "q",
            "claim": "c", "concepts": [], "stance": "context"}  # fmt: skip
    client, fake = _client(weights, monkeypatch, [(json.dumps({"items": [item]}), "stop")] * 2)
    r = make_reasoner("local-b2", client)
    ev = r.extract_evidence(_question(), papers, 1)
    assert r.name == "local-b2" and len(fake.calls) == 2 and [e.id for e in ev] == ["E1", "E2"]


def test_e1_reuse_refuses_a_changed_configuration(tmp_path: Path) -> None:
    prior = tmp_path / "results" / "e1_runs" / "old"
    prior.mkdir(parents=True)
    (prior / "config.yaml").write_text(yaml.safe_dump({"run_config": {"max_papers": 25}}))
    (prior / "runs.json").write_text(
        json.dumps([{"question": "q", "reasoner": "rule", "repeat": 0, "status": "complete"}])
    )
    assert e1_runs._reusable({"reuse_from": "old", "run_config": {"max_papers": 25}}, tmp_path / "results")
    with pytest.raises(ValueError, match="run_config"):
        e1_runs._reusable({"reuse_from": "old", "run_config": {"max_papers": 10}}, tmp_path / "results")
