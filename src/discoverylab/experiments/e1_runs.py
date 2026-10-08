"""E1: end-to-end runs on real research questions.

Every configured question goes through the full workflow with every configured reasoner. The
process metrics come from the run's artefacts and the independent verification only. A reasoner
that cannot run here (no API key) is recorded as skipped with the reason; it is never estimated.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from discoverylab.agents import RunConfig
from discoverylab.evaluate import run_metrics
from discoverylab.experiments.common import Summary, load_run, result_dir, table
from discoverylab.human import NoReviewer
from discoverylab.judges import ClaudeJudge, judge_run, load_judgements, summarise_judgements
from discoverylab.literature.registry import SourceRegistry
from discoverylab.models import RunState
from discoverylab.reasoners.base import Reasoner
from discoverylab.run import execute, load_question, make_reasoner, new_run_id

DESCRIPTION = "End-to-end runs: every question with every reasoner, verified independently."


def _rate(r: dict[str, Any]) -> float | None:
    return r["rate"]


def _missing_key(reasoner: str) -> str | None:
    if reasoner.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY is not set"
    return None


def run(
    cfg: dict[str, Any],
    runs: Path,
    registry: SourceRegistry,
    reasoner_factory: Callable[[str], Reasoner] = make_reasoner,
    results: Path | None = None,
    root: Path = Path("."),
    judge_factory: Callable[[], ClaudeJudge] | None = None,
) -> Path:
    rc = RunConfig(**cfg.get("run_config", {}))
    judge = (
        judge_factory()
        if judge_factory
        else (ClaudeJudge() if cfg.get("judge") == "claude" and not _missing_key("claude") else None)
    )
    with result_dir("e1_runs", cfg, DESCRIPTION, results) as (rd, s):
        rows: list[dict[str, Any]] = []
        for qpath in cfg["questions"]:
            q, op = load_question(root / qpath)
            for spec in cfg["reasoners"]:
                name, critic_name = spec["reasoner"], spec.get("model_critic")
                label = name + ("+mc" if critic_name else "")
                for rep in range(int(spec.get("repeats", 1))):
                    why = _missing_key(name) or (_missing_key(critic_name) if critic_name else None)
                    if why:
                        rows.append(
                            {
                                "question": q.id,
                                "reasoner": label,
                                "repeat": rep,
                                "run_id": None,
                                "status": "skipped",
                                "reason": why,
                            }
                        )
                        continue
                    reasoner = reasoner_factory(name)
                    critic = reasoner_factory(critic_name) if critic_name else None
                    state = RunState(run_id=f"{new_run_id(q.id, label)}-r{rep}", question=q, reasoner=label)
                    try:
                        state = execute(runs / state.run_id, state, registry, reasoner, op, NoReviewer(), critic, rc)
                    except Exception as exc:  # recorded, not hidden: a failed run is a result too
                        rows.append({"question": q.id, "reasoner": label, "repeat": rep, "run_id": state.run_id,
                                     "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})  # fmt: skip
                        continue
                    if judge is not None and state.status == "complete":
                        j = judge_run(state, judge)
                        (runs / state.run_id / "judgements.json").write_text(json.dumps(j, indent=1), encoding="utf-8")
                    usage = getattr(reasoner, "usage", None)
                    rows.append({"question": q.id, "reasoner": label, "repeat": rep, "run_id": state.run_id, "status": state.status,
                                 "reason": state.error or "", "model_calls": getattr(usage, "calls", 0),
                                 "input_tokens": getattr(usage, "input_tokens", 0), "output_tokens": getattr(usage, "output_tokens", 0)})  # fmt: skip
        rd.write_json("runs.json", rows)
        rd.write_json("source_unavailable.json", registry.unavailable)
        metrics = []
        for r in rows:
            if r["status"] != "complete":
                continue
            state, ver = load_run(runs / r["run_id"])
            judged = summarise_judgements(load_judgements(runs / r["run_id"]))
            metrics.append({**run_metrics(state, ver), **judged, "question": r["question"], "label": r["reasoner"]})
        rd.write_json("metrics.json", metrics)
        _summarise(s, rows, metrics)
        if registry.unavailable:
            s.notes.append("Index requests answered 'try later' after retries (rate limit or quota), by source: "
                           + ", ".join(f"{k} {v}" for k, v in sorted(registry.unavailable.items()))
                           + ". Their citations are reported as 'could not check', not as failures.")  # fmt: skip
    return rd.path


def _summarise(s: Summary, rows: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
    s.title = "E1 End-to-end runs"
    s.question = (
        "Do runs on real research questions cite real sources, quote them correctly and design valid experiments?"
    )
    per_run = []
    for m in metrics:
        validity = list(m["design_validity"].values())
        per_run.append({
            "question": m["question"], "reasoner": m["label"], "run_id": m["run_id"], "papers": m["papers"],
            "evidence": m["evidence"],
            "citation_accuracy": _rate(m["citation_accuracy"]),
            "evidence_correctness": _rate(m["evidence_correctness"]),
            "hypotheses": m["hypotheses"],
            "design_validity": (sum(v["passed"] for v in validity) / sum(v["of"] for v in validity)) if validity else None,
            "critiques": sum(c["n"] for c in m["critiques"]),
            "verdicts": ", ".join(sorted(m["verdicts"].values())) or "none",
            "hypothesis_quality": m["hypothesis_quality"]["mean_of_10"] if m["hypothesis_quality"] else "pending",
        })  # fmt: skip
    s.tables.append(table("runs", "Per run", [("question", "Question", "l"), ("reasoner", "Reasoner", "l"), ("papers", "Papers", "r"),
        ("evidence", "Evidence", "r"), ("citation_accuracy", "Citations verified", "r"), ("evidence_correctness", "Evidence correct", "r"),
        ("hypotheses", "Hypotheses", "r"), ("design_validity", "Design checks passed", "r"), ("critiques", "Critiques", "r"),
        ("hypothesis_quality", "Hypothesis rubric (of 10)", "r"), ("verdicts", "Verdicts", "l")], per_run))  # fmt: skip
    not_run = [r for r in rows if r["status"] != "complete"]
    if not_run:
        s.tables.append(table("not_run", "Runs that did not complete", [("question", "Question", "l"), ("reasoner", "Reasoner", "l"),
            ("repeat", "Repeat", "r"), ("status", "Status", "l"), ("reason", "Reason", "l")], not_run))  # fmt: skip
    for label in sorted({r["reasoner"] for r in per_run}):
        mine = [r for r in per_run if r["reasoner"] == label]
        acc = [r["citation_accuracy"] for r in mine if r["citation_accuracy"] is not None]
        cor = [r["evidence_correctness"] for r in mine if r["evidence_correctness"] is not None]
        if acc and cor:
            s.findings.append(f"{label}: {len(mine)} completed runs; citations verified {min(acc):.0%} to {max(acc):.0%}, "
                              f"evidence correct {min(cor):.0%} to {max(cor):.0%} (range over runs).")  # fmt: skip
    s.notes.append(
        "Citation accuracy counts distinct cited identifiers; evidence correctness counts evidence items whose citation verified and whose quote was found in the source text."
    )
    s.notes.append("Runs had no human reviewer, so no correction count is reported.")
    if any(m["hypothesis_quality"] is None for m in metrics):
        s.notes.append(
            "Hypothesis rubric and quote-supports-claim judgements: Status pending for runs without a language-model judge (no ANTHROPIC_API_KEY)."
        )
