"""E4: does a fluent report predict a sound process?

Each completed E1 run is copied into report variants whose process has been corrupted in known
ways (citations swapped between claims, citations fabricated, conclusions inflated). Each variant
gets two fluency scores, a readability formula and, when an API key is available, a language-model
judge's convincingness rating, and a process score from the independent checks. The question is
whether the fluency scores fall when the process score does.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field
from scipy.stats import spearmanr

from discoverylab.critic import expected_verdict
from discoverylab.experiments.common import Summary, e1_run_dirs, load_run, result_dir, table
from discoverylab.judges import judge_client, judge_unavailable
from discoverylab.literature.registry import SourceRegistry
from discoverylab.models import RunState
from discoverylab.report import flesch_reading_ease, render_report
from discoverylab.verify import Verifier

DESCRIPTION = "Fluency against process: corrupted report variants scored for readability, convincingness and process."


class Judge(Protocol):
    name: str

    def rate(self, report: str) -> dict[str, Any]: ...


class _Rating(BaseModel):
    convincingness: int = Field(ge=1, le=10)
    clarity: int = Field(ge=1, le=10)
    rationale: str


PROMPT = (
    "Rate this report from 1 to 10 for how convincing its conclusions are and for how clearly it is written.\n\n"
    "<report>\n{report}\n</report>"
)


class FluencyJudge:
    """Rates a report as a busy reader would: from the text alone, with no access to sources."""

    def __init__(self, client: Any = None, model: str | None = None, name: str = "claude") -> None:
        self.name = name
        if client is None:
            client, default = judge_client(name)
            model = model or default
        self.client = client
        self.model = model or os.environ.get("DISCOVERYLAB_MODEL", "claude-opus-5-5")

    def rate(self, report: str) -> dict[str, Any]:
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=2000,
            system="You review short research reports. Judge only what is on the page; you cannot open the cited sources.",
            messages=[{"role": "user", "content": PROMPT.format(report=report)}],
            output_format=_Rating,
        )
        out = resp.parsed_output
        if out is None:  # recorded as unrated with the reason; never imputed
            why = getattr(resp, "stop_reason", None) or "unparsed"
            return {"convincingness": None, "clarity": None, "rationale": None,
                    "unrated": f"{why}: {getattr(resp, 'raw', '')[:300]}"}  # fmt: skip
        return out.model_dump()


def _rotate_citations(s: RunState, rng: random.Random) -> bool:
    if len({e.citation.identifier for e in s.evidence}) < 2:
        return False
    cits = [e.citation for e in s.evidence]
    k = 1 + rng.randrange(len(cits) - 1)
    for e, c in zip(s.evidence, cits[k:] + cits[:k], strict=True):
        e.citation = c.model_copy()
    return True


def _fabricate_citations(s: RunState, rng: random.Random) -> bool:
    for e in s.evidence:
        e.citation = e.citation.model_copy(
            update={"identifier": f"doi:10.{rng.randrange(1000, 9999)}/{rng.randrange(16**8):08x}"}
        )
    return bool(s.evidence)


def _inflate_conclusions(s: RunState, rng: random.Random) -> bool:
    hyps = {h.id: h for h in s.hypotheses}
    for k in s.conclusions:
        h = hyps.get(k.hypothesis_id)
        k.verdict = "supported"
        k.statement = f"The experiment confirms the hypothesis: {h.statement if h else k.statement}"
        k.limitations = []
    return bool(s.conclusions)


VARIANTS: dict[str, list[Callable[[RunState, random.Random], bool]]] = {
    "original": [],
    "swapped_citations": [_rotate_citations],
    "fabricated_citations": [_fabricate_citations],
    "inflated_conclusions": [_inflate_conclusions],
    "all_three": [_rotate_citations, _fabricate_citations, _inflate_conclusions],
}


def process_score(s: RunState, verifier: Verifier) -> dict[str, float | None]:
    checks = [verifier.check_evidence(e) for e in s.evidence]
    ev = sum(c.correct for c in checks) / len(checks) if checks else None
    hyps, designs, results = {h.id: h for h in s.hypotheses}, {d.id: d for d in s.designs}, {r.id: r for r in s.results}
    ok = []
    for k in s.conclusions:
        r, h = results.get(k.result_id), hyps.get(k.hypothesis_id)
        if r and h and r.design_id in designs:
            ok.append(k.verdict == expected_verdict(h, designs[r.design_id], r))
    concl = sum(ok) / len(ok) if ok else None
    parts = [x for x in (ev, concl) if x is not None]
    return {
        "evidence_correctness": ev,
        "conclusion_validity": concl,
        "process": sum(parts) / len(parts) if parts else None,
    }


def evaluate(states: list[RunState], verifier: Verifier, judge: Judge | None, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for st in states:
        for name, steps in VARIANTS.items():
            v = st.model_copy(deep=True)
            if not all(f(v, random.Random(rng.randrange(2**31))) for f in steps):
                continue
            report = render_report(v)
            row: dict[str, Any] = {"run_id": st.run_id, "reasoner": st.reasoner, "variant": name,
                                   "readability": flesch_reading_ease(report), "words": len(report.split()),
                                   **process_score(v, verifier)}  # fmt: skip
            if judge is not None:
                rating = judge.rate(report)
                row |= {
                    "convincingness": rating["convincingness"],
                    "clarity": rating["clarity"],
                    "judge_rationale": rating["rationale"],
                }
                if rating.get("unrated"):
                    row["unrated"] = rating["unrated"]
            rows.append(row)
    return rows


def _spearman(rows: list[dict[str, Any]], a: str, b: str) -> float | None:
    pairs = [(r[a], r[b]) for r in rows if r.get(a) is not None and r.get(b) is not None]
    if len(pairs) < 4 or len({p[0] for p in pairs}) < 2 or len({p[1] for p in pairs}) < 2:
        return None
    return float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic)


def summarise(s: Summary, rows: list[dict[str, Any]], judged: bool) -> None:
    s.title = "E4 Fluency versus process"
    s.question = "When a run's process is corrupted, do its report's fluency scores fall with its process score?"
    agg = []
    for name in VARIANTS:
        mine = [r for r in rows if r["variant"] == name]
        if not mine:
            continue

        def mean(k: str, m: list[dict[str, Any]] = mine) -> float | None:
            xs = [r[k] for r in m if r.get(k) is not None]
            return sum(xs) / len(xs) if xs else None

        agg.append({"variant": name.replace("_", " "), "reports": len(mine), "readability": mean("readability"),
                    "convincingness": mean("convincingness") if judged else "pending", "process": mean("process"),
                    "evidence_correctness": mean("evidence_correctness"), "conclusion_validity": mean("conclusion_validity")})  # fmt: skip
    s.tables.append(table("variants", "Mean scores by variant", [("variant", "Variant", "l"), ("reports", "Reports", "r"),
        ("readability", "Readability (Flesch)", "r"), ("convincingness", "Convincingness (judge, 1-10)", "r"),
        ("process", "Process score", "r"), ("evidence_correctness", "Evidence correct", "r"),
        ("conclusion_validity", "Conclusions valid", "r")], agg))  # fmt: skip
    rho = _spearman(rows, "readability", "process")
    s.findings.append(f"Spearman correlation of readability with process score: {rho:.2f}." if rho is not None
                      else "Too few distinct scores for a readability-process correlation.")  # fmt: skip
    if judged:
        unrated = [r for r in rows if r.get("unrated")]
        if unrated:
            s.notes.append(f"The judge gave no usable rating for {len(unrated)} of {len(rows)} reports (reasons in variants.json); "
                           "they are left out of the convincingness means and correlation, not imputed.")  # fmt: skip
        rj = _spearman(rows, "convincingness", "process")
        s.findings.append(f"Spearman correlation of judged convincingness with process score: {rj:.2f}." if rj is not None
                          else "Too few distinct judged scores for a correlation.")  # fmt: skip
    else:
        s.notes.append("Convincingness: Status pending. It needs a language-model judge, and none could run.")
    s.notes.append(
        "The process score is the mean of evidence correctness (independent verification) and the share of conclusions whose verdict matches the statistics."
    )


def run(
    cfg: dict[str, Any], runs: Path, registry: SourceRegistry, judge: Judge | None = None, results: Path | None = None
) -> Path:
    states = [load_run(d)[0] for d in e1_run_dirs(runs, results)]
    why = judge_unavailable(cfg.get("judge"))
    if judge is None and cfg.get("judge") and why is None:
        judge = FluencyJudge(name=cfg["judge"])
    with result_dir("e4_fluency", cfg, DESCRIPTION, results) as (rd, s):
        rows = evaluate(states, Verifier(registry), judge, int(cfg.get("seed", 0)))
        rd.write_json("variants.json", rows)
        summarise(s, rows, judge is not None)
        if judge is not None:
            s.notes.append(f"Convincingness judge: {judge.name} ({getattr(judge, 'model', 'unknown')}).")
        elif why:
            s.notes.append(f"Judge not run: {why}.")
    return rd.path
