"""Process metrics for a run (definitions in docs/METRICS.md).

Everything here is computed from the run's artefacts, its process log and the independent
verification. Scores that need a judge (hypothesis rubric, evidence support) are reported only
when judgements exist; they are never estimated.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from discoverylab.models import ExperimentDesign, RunState
from discoverylab.text import tfidf_cosine
from discoverylab.toolbox import validate

VALIDITY_CHECKS = (
    "runnable",
    "has_control",
    "has_treatment",
    "held_out_evaluation",
    "primary_metric_matches_hypothesis",
    "paired_test",
    "at_least_5_paired_measurements",
    "multiple_seeds",
    "more_than_one_dataset",
)


def design_validity(d: ExperimentDesign, dependent_variable: str | None) -> dict[str, bool]:
    n = len(d.seeds) * (d.folds if d.split == "stratified_kfold" else 1)
    return {
        "runnable": not validate(d),
        "has_control": bool(d.control) and d.control not in d.treatments,
        "has_treatment": bool(d.treatments),
        "held_out_evaluation": d.split
        in ("stratified_kfold", "stratified_holdout"),  # the toolbox only evaluates out of fold
        "primary_metric_matches_hypothesis": bool(d.metrics) and d.metrics[0] == dependent_variable,
        "paired_test": d.test in ("paired_wilcoxon", "paired_t"),
        "at_least_5_paired_measurements": n >= 5,
        "multiple_seeds": len(d.seeds) > 1,
        "more_than_one_dataset": len(d.datasets) > 1,
    }


def _rate(k: int, n: int) -> dict[str, float | int | None]:
    return {"k": k, "n": n, "rate": (k / n) if n else None}


def run_metrics(state: RunState, verification: dict[str, Any] | None) -> dict[str, Any]:
    checks = (verification or {}).get("evidence", [])
    by_id: dict[str, dict[str, Any]] = {}
    for c in checks:
        by_id.setdefault(c["citation"]["identifier"], c["citation"])
    cit_status = Counter(c["status"] for c in by_id.values())
    quote_status = Counter(c["quote"] for c in checks)
    unchecked = [c for c in checks if c["citation"]["status"] == "unverifiable" or c["quote"] == "unchecked"]
    hyp = {h.id: h for h in state.hypotheses}
    validity = {d.id: design_validity(d, hyp[d.hypothesis_id].dependent_variable if d.hypothesis_id in hyp else None)
                for d in state.designs}  # fmt: skip
    abstracts = [p.abstract for p in state.papers if p.abstract]
    novelty = (
        {h.id: max(tfidf_cosine(h.statement, abstracts), default=None) for h in state.hypotheses} if abstracts else {}
    )
    crit = Counter((c.stage.value, c.reviewer, c.severity) for c in state.critiques)
    resolved = Counter(c.resolution for c in state.critiques)
    corrections: dict[str, Any] = {}
    for stage, reviewer in state.reviewed.items():
        n = sum(c.stage.value == stage and c.action != "approve" for c in state.corrections)
        corrections[stage] = None if reviewer == "none" else n  # None = no human reviewed this stage
    return {
        "run_id": state.run_id,
        "reasoner": state.reasoner,
        "status": state.status,
        "papers": len(state.papers),
        "papers_with_abstract": len(abstracts),
        "evidence": len(state.evidence),
        # Over citations that could be checked: an index outage is neither a pass nor a fabrication.
        "citation_accuracy": _rate(cit_status.get("verified", 0), len(by_id) - cit_status.get("unverifiable", 0)),
        "citations_unverifiable": cit_status.get("unverifiable", 0),
        "citation_status": dict(cit_status),
        # Likewise over evidence that could be checked (citation and quote not blocked by an outage).
        "evidence_correctness": _rate(sum(c["correct"] for c in checks), len(checks) - len(unchecked)),
        "evidence_unchecked": len(unchecked),
        "quote_status": dict(quote_status),
        "gaps": len(state.gaps),
        "hypotheses": len(state.hypotheses),
        "design_validity": {k: {"passed": sum(v.values()), "of": len(v), "checks": v} for k, v in validity.items()},
        "novelty_max_tfidf_similarity": novelty,
        "verdicts": {k.id: k.verdict for k in state.conclusions},
        "critiques": [{"stage": s, "reviewer": r, "severity": sev, "n": n} for (s, r, sev), n in sorted(crit.items())],
        "critique_resolution": dict(resolved),
        "human_corrections": corrections,
        "human_reviewed": any(v != "none" for v in state.reviewed.values()),
    }
