"""Judged metrics: the hypothesis-quality rubric and whether a quote supports its claim.

These need judgement, so they are scored by a language-model judge (and, for a pilot, by the
researcher through a rubric file). Without a judge they are not estimated: ``run_metrics``
reports them as missing and the experiments mark them pending.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from discoverylab.models import Evidence, Hypothesis, RunState

RUBRIC: dict[str, str] = {
    "specificity": "Names the intervention, the outcome and the population or setting precisely.",
    "testability": "Can be tested with data and methods that exist.",
    "grounding": "Follows from a gap the run actually identified in the retrieved evidence.",
    "falsifiability": "States an outcome that would show it wrong.",
    "consistency": "Does not contradict the evidence the run collected.",
}
# Each criterion is scored 0 (absent), 1 (partly met) or 2 (met); the total is out of 10.


class _Score(BaseModel):
    criterion: Literal["specificity", "testability", "grounding", "falsifiability", "consistency"]
    score: int = Field(ge=0, le=2)
    reason: str


class _HypothesisRating(BaseModel):
    scores: list[_Score]


class _Support(BaseModel):
    verdict: Literal["supports", "partly", "does_not_support"]
    reason: str


def judge_client(name: str) -> tuple[Any, str]:
    """The client and model id for a judge named in a config: ``claude`` or ``local``."""
    if name == "claude":
        import anthropic

        return anthropic.Anthropic(), os.environ.get("DISCOVERYLAB_MODEL", "claude-opus-5-5")
    if name == "local":
        from discoverylab.reasoners.local import JUDGE_CTX, LocalClient, model_id, model_path

        path = model_path("judge")
        if path is None:
            raise ValueError("DISCOVERYLAB_LOCAL_JUDGE is not set to a GGUF file")
        # judging is scoring, not generation: sample near-greedily so a re-judge agrees with itself
        return LocalClient(path, seed=0, temperature=0.0, n_ctx=JUDGE_CTX), model_id(path)
    raise ValueError(f"unknown judge {name!r}")


def judge_unavailable(name: str | None) -> str | None:
    """Why the configured judge cannot run here (None when it can, or when no judge is configured)."""
    if not name:
        return None
    if name == "claude":
        return None if os.environ.get("ANTHROPIC_API_KEY") else "ANTHROPIC_API_KEY is not set"
    if name == "local":
        from discoverylab.reasoners.local import missing

        return missing("judge")
    return f"unknown judge {name!r}"


class ModelJudge:
    """A language-model judge; ``name`` records which family judged (claude or local)."""

    def __init__(self, client: Any = None, model: str | None = None, name: str = "claude") -> None:
        self.name = name
        if client is None:
            client, default = judge_client(name)
            model = model or default
        self.client = client
        self.model = model or os.environ.get("DISCOVERYLAB_MODEL", "claude-opus-5-5")

    def _ask(self, prompt: str, schema: type[BaseModel]) -> Any:
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=4000,
            system="You are a careful research reviewer. Score strictly against the stated criteria.",
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        if resp.parsed_output is None:
            raise RuntimeError("judge returned no output")
        return resp.parsed_output

    def hypothesis(self, h: Hypothesis, state: RunState) -> dict[str, int]:
        gaps = [g.description for g in state.gaps if g.id in h.gap_ids]
        ev = [f"{e.id}: {e.claim} (quote: {e.quote})" for e in state.evidence]
        criteria = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
        out = self._ask(
            f"Score this hypothesis 0, 1 or 2 on each criterion.\n\nCriteria:\n{criteria}\n\n"
            f"Hypothesis: {h.statement}\nGaps it claims to address: {json.dumps(gaps)}\nEvidence collected:\n"
            + "\n".join(ev),
            _HypothesisRating,
        )
        return {s.criterion: s.score for s in out.scores}

    def support(self, e: Evidence) -> str:
        out = self._ask(
            f"Does the quote support the claim?\n\nClaim: {e.claim}\nQuote: {e.quote}\n\n"
            "Answer supports, partly or does_not_support, judging only from the quote.",
            _Support,
        )
        return str(out.verdict)


def judge_run(state: RunState, judge: ModelJudge) -> dict[str, Any]:
    """Judgements for every hypothesis and evidence item, written as ``judgements.json``."""
    return {
        "judge": judge.name,
        "judge_model": judge.model,
        "hypotheses": {h.id: judge.hypothesis(h, state) for h in state.hypotheses},
        "support": {e.id: judge.support(e) for e in state.evidence},
    }


def load_judgements(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "judgements.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def summarise_judgements(j: dict[str, Any] | None) -> dict[str, Any]:
    if not j:
        return {"hypothesis_quality": None, "evidence_support": None, "judge": None, "judge_model": None}
    totals = {h: sum(s.values()) for h, s in j["hypotheses"].items()}
    sup = list(j["support"].values())
    return {
        "judge": j["judge"],
        "judge_model": j.get("judge_model"),
        "hypothesis_quality": {
            "per_hypothesis": totals,
            "mean_of_10": sum(totals.values()) / len(totals) if totals else None,
        },
        "evidence_support": {
            "supports": sup.count("supports"),
            "partly": sup.count("partly"),
            "does_not_support": sup.count("does_not_support"),
            "n": len(sup),
        },
    }
