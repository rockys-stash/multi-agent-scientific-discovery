"""E1 arm comparison: the reasoners side by side on the same questions, from E1's runs.

Offline and model-free: it reads the latest E1's runs, their independent verification and their
judgements, and reports each arm's process metrics as mean and standard deviation over repeats.
The rule baseline is deterministic (one run per question), so its spread is undefined, not zero.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from discoverylab.critic import expected_verdict
from discoverylab.evaluate import run_metrics
from discoverylab.experiments.common import Summary, load_run, result_dir, table
from discoverylab.judges import load_judgements, summarise_judgements
from discoverylab.models import RunState, Stage

DESCRIPTION = "E1 arm comparison: process metrics per reasoner, mean and spread over repeats."

LABELS = {"rule": "Rule baseline", "local": "Local model", "local+mc": "Local model + model critic",
          "claude": "Claude", "claude+mc": "Claude + model critic"}  # fmt: skip

METRICS: list[tuple[str, str]] = [
    ("evidence", "Evidence items"),
    ("citation_accuracy", "Citations verified"),
    ("evidence_correctness", "Evidence correct"),
    ("quote_supports", "Quotes judged to support claim"),
    ("hypotheses", "Hypotheses"),
    ("hypothesis_rubric", "Hypothesis rubric (of 10)"),
    ("design_validity", "Design checks passed"),
    ("verdict_valid", "Verdicts matching the statistics"),
    ("critiques", "Critiques raised"),
    ("evidence_removed", "Evidence removed by critique"),
    ("minutes", "Run time (min)"),
]


def verdict_validity(s: RunState) -> float | None:
    """Share of conclusions whose verdict equals the fixed rule on the computed comparisons."""
    hyps, designs, results = {h.id: h for h in s.hypotheses}, {d.id: d for d in s.designs}, {r.id: r for r in s.results}
    ok = []
    for k in s.conclusions:
        r, h = results.get(k.result_id), hyps.get(k.hypothesis_id)
        if r and h and r.design_id in designs:
            ok.append(k.verdict == expected_verdict(h, designs[r.design_id], r))
    return sum(ok) / len(ok) if ok else None


def _minutes(run_dir: Path) -> float | None:
    lines = (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return None
    a, b = (datetime.fromisoformat(json.loads(x)["ts"]) for x in (lines[0], lines[-1]))
    return (b - a).total_seconds() / 60


def per_run(run_dir: Path) -> dict[str, Any]:
    s, ver = load_run(run_dir)
    m = run_metrics(s, ver)
    j = summarise_judgements(load_judgements(run_dir))
    val = list(m["design_validity"].values())
    sup = j["evidence_support"]
    return {
        "run_id": s.run_id,
        "question": s.question.id,
        "arm": s.reasoner,
        "evidence": m["evidence"],
        "citation_accuracy": m["citation_accuracy"]["rate"],
        "evidence_correctness": m["evidence_correctness"]["rate"],
        "quote_supports": (sup["supports"] / sup["n"]) if sup and sup["n"] else None,
        "hypotheses": m["hypotheses"],
        "hypothesis_rubric": j["hypothesis_quality"]["mean_of_10"] if j["hypothesis_quality"] else None,
        "design_validity": (sum(v["passed"] for v in val) / sum(v["of"] for v in val)) if val else None,
        "verdict_valid": verdict_validity(s),
        "critiques": len(s.critiques),
        "evidence_removed": sum(
            c.stage == Stage.EVIDENCE and c.severity == "blocking" and c.resolution == "fixed" for c in s.critiques
        ),
        "critiques_by_reviewer": {r: sum(c.reviewer == r for c in s.critiques) for r in ("rule", "model")},
        "verdicts": sorted(k.verdict for k in s.conclusions),
        "novelty_max": max((v for v in m["novelty_max_tfidf_similarity"].values() if v is not None), default=None),
        "minutes": _minutes(run_dir),
        "judge_model": j.get("judge_model"),
    }


def _agg(values: list[Any]) -> dict[str, float | int | None]:
    xs = [float(v) for v in values if v is not None]
    return {"n": len(xs), "mean": statistics.fmean(xs) if xs else None,
            "sd": statistics.stdev(xs) if len(xs) > 1 else None}  # fmt: skip


def _num(x: float, pct: bool) -> str:
    return f"{x:.0%}" if pct else f"{x:.1f}"


def _fmt(a: dict[str, Any], pct: bool) -> str:
    if a["mean"] is None:
        return "n/a"
    return _num(a["mean"], pct) + (f" ± {_num(a['sd'], pct)}" if a["sd"] is not None else "")


PCT = {"citation_accuracy", "evidence_correctness", "quote_supports", "design_validity", "verdict_valid"}


def compare(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    arms = sorted({r["arm"] for r in rows}, key=lambda a: list(LABELS).index(a) if a in LABELS else 99)
    return {a: {k: _agg([r[k] for r in rows if r["arm"] == a]) for k, _ in METRICS} for a in arms}


def summarise(s: Summary, rows: list[dict[str, Any]]) -> None:
    s.title = "E1 Arm comparison"
    s.question = "How do the reasoners compare on the same questions when only the process is judged?"
    cmp = compare(rows)
    arms = list(cmp)
    cols = [("metric", "Metric", "l")] + [(a, f"{LABELS.get(a, a)} (n={cmp[a]['evidence']['n']})", "r") for a in arms]
    trows = [{"metric": label, **{a: _fmt(cmp[a][k], k in PCT) for a in arms}} for k, label in METRICS]
    s.tables.append(table("arms", "All questions: mean ± standard deviation over runs", cols, trows))
    for q in sorted({r["question"] for r in rows}):
        qc = compare([r for r in rows if r["question"] == q])
        qa = list(qc)
        s.tables.append(table(f"arms_{q}", f"{q}: mean ± standard deviation over repeats",
                              [("metric", "Metric", "l")] + [(a, LABELS.get(a, a), "r") for a in qa],
                              [{"metric": label, **{a: _fmt(qc[a][k], k in PCT) for a in qa}} for k, label in METRICS]))  # fmt: skip
    for a in arms:
        c = cmp[a]
        if c["evidence"]["n"]:
            s.findings.append(
                f"{LABELS.get(a, a)}: {_fmt(c['evidence'], False)} evidence items per run, citations verified "
                f"{_fmt(c['citation_accuracy'], True)}, evidence correct {_fmt(c['evidence_correctness'], True)}, "
                f"verdicts matching the statistics {_fmt(c['verdict_valid'], True)}, design checks {_fmt(c['design_validity'], True)}."
            )
    s.notes.append(
        "The rule baseline runs once per question (it is deterministic), so its spread is undefined; model arms ran 3 repeats per question with different sampling seeds."
    )
    s.notes.append(
        "The rule baseline's hypotheses and designs come from a person-written operationalisation and its verdicts are the fixed rule itself (D4), so its design and verdict scores are not its own reasoning."
    )
    s.notes.append(
        "Quote support and the hypothesis rubric come from a small local judge (D17); citations, quotes, design checks and verdicts are checked independently and carry more weight."
    )


def run(cfg: dict[str, Any], runs: Path, results: Path | None = None, e1_results: Path | None = None) -> Path:
    from discoverylab.experiments.common import ROOT

    base = (e1_results or results or ROOT / "results") / "e1_runs"
    latest = (base / "LATEST").read_text().strip()
    e1rows = json.loads((base / latest / "runs.json").read_text(encoding="utf-8"))
    with result_dir("e1_compare", {**cfg, "e1_result": latest}, DESCRIPTION, results) as (rd, s):
        rows = [per_run(runs / r["run_id"]) for r in e1rows if r["status"] == "complete"]
        usage = {r["run_id"]: {k: r.get(k) for k in ("model_calls", "input_tokens", "output_tokens")} for r in e1rows}
        for r in rows:
            r.update(usage.get(r["run_id"], {}))
        rd.write_json("per_run.json", rows)
        rd.write_json("comparison.json", compare(rows))
        summarise(s, rows)
        skipped = [r for r in e1rows if r["status"] != "complete"]
        if skipped:
            s.notes.append(
                "Not run in E1: " + "; ".join(sorted({f"{r['reasoner']} ({r['reason']})" for r in skipped})) + "."
            )
    return rd.path
