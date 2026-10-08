"""Rule-based critique: structural checks any reviewer would make, applied to every stage.

These checks need no language model. They catch faults that are visible in the artefacts'
structure (a citation to a paper that was never retrieved, a design with no control, a
conclusion the statistics do not support). Semantic faults (a quote that does not support its
claim, a hypothesis that misreads a gap) need the model critic or a person.
"""

from __future__ import annotations

import math

from discoverylab.models import (
    AnalysisResult,
    Critique,
    ExperimentDesign,
    Hypothesis,
    RunState,
    Stage,
)
from discoverylab.text import contains_quote
from discoverylab.toolbox import validate

ALPHA = 0.05


def expected_verdict(h: Hypothesis, design: ExperimentDesign, result: AnalysisResult) -> str:
    """What the primary-metric comparisons support, by a fixed rule.

    Supported: every dataset shows a Holm-significant difference in the hypothesised direction.
    For ``no_difference``, no unadjusted test is significant (a weak criterion: absence of
    evidence; the report shows the intervals). Not supported: any Holm-significant difference
    against the hypothesis. Otherwise inconclusive.
    """
    primary = design.metrics[0]
    comps = [c for c in result.comparisons if c["metric"] == primary]
    if not comps:
        return "inconclusive"
    sig_for = sig_against = 0
    for c in comps:
        p = c.get("p_holm")
        if p is None or float(p) >= ALPHA:
            continue
        # "increase" and "decrease" refer to the dependent variable itself, not to "better".
        raised = float(c["mean_diff"]) > 0  # type: ignore[arg-type]
        if h.expected_direction != "no_difference" and raised == (h.expected_direction == "increase"):
            sig_for += 1
        else:
            sig_against += 1
    if sig_against:
        return "not_supported"
    if h.expected_direction == "no_difference":
        return "supported" if all(float(c["p"]) >= ALPHA for c in comps) else "inconclusive"  # type: ignore[arg-type]
    return "supported" if sig_for == len(comps) else "inconclusive"


def _c(
    state: RunState, stage: Stage, round_: int, n: int, target: str, issue: str, severity: str, msg: str
) -> Critique:
    return Critique(id=f"C{n}", round=round_, target_id=target, stage=stage, issue=issue,
                    severity=severity, message=msg, reviewer="rule")  # fmt: skip


def rule_critique(state: RunState, stage: Stage, round_: int, start: int) -> list[Critique]:
    out: list[Critique] = []

    def add(target: str, issue: str, severity: str, msg: str) -> None:
        out.append(_c(state, stage, round_, start + len(out), target, issue, severity, msg))

    papers = {p.id: p for p in state.papers}
    ev_ids = {e.id for e in state.evidence}
    gap_ids = {g.id for g in state.gaps}
    if stage == Stage.LITERATURE and not state.papers:
        add("literature", "no_sources", "blocking", "The search returned no papers.")
    if stage == Stage.EVIDENCE:
        if not state.evidence:
            add("evidence", "no_evidence", "blocking", "No evidence was extracted.")
        for e in state.evidence:
            src = papers.get(_norm(e.citation.identifier))
            if src is None:
                add(e.id, "citation_not_retrieved", "blocking",
                    f"{e.citation.identifier} is not among the papers the literature search returned.")  # fmt: skip
            elif not contains_quote(f"{src.title}\n{src.abstract}", e.quote):
                add(
                    e.id,
                    "quote_not_in_source",
                    "blocking",
                    "The quoted text does not appear in the cited paper's record.",
                )
            if not e.quote.strip():
                add(e.id, "missing_quote", "blocking", "The claim has no supporting quote.")
    if stage == Stage.GAPS:
        if not state.gaps:
            add("gaps", "no_gaps", "major", "No gap was identified.")
        for g in state.gaps:
            missing = [i for i in g.evidence_ids if i not in ev_ids]
            if missing:
                add(
                    g.id,
                    "unknown_reference",
                    "blocking",
                    f"Refers to evidence that does not exist: {', '.join(missing)}.",
                )
    if stage == Stage.HYPOTHESES:
        for h in state.hypotheses:
            if not h.gap_ids:
                add(h.id, "ungrounded_hypothesis", "major", "The hypothesis is not linked to any identified gap.")
            missing = [i for i in [*h.gap_ids, *h.evidence_ids] if i not in gap_ids | ev_ids]
            if missing:
                add(
                    h.id,
                    "unknown_reference",
                    "blocking",
                    f"Refers to artefacts that do not exist: {', '.join(missing)}.",
                )
    if stage == Stage.DESIGN:
        hyp = {h.id: h for h in state.hypotheses}
        for d in state.designs:
            for p in validate(d):
                add(d.id, "unrunnable_design", "blocking", p)
            if not d.control:
                add(d.id, "no_control", "blocking", "The design has no control condition.")
            if d.control in d.treatments:
                add(d.id, "no_control", "blocking", "The control is also listed as a treatment.")
            if not d.treatments:
                add(d.id, "no_treatment", "blocking", "The design has no treatment condition.")
            if len(d.seeds) * (d.folds if d.split == "stratified_kfold" else 1) < 5:
                add(d.id, "insufficient_repeats", "major",
                    "Fewer than 5 paired measurements per condition; the test has almost no power.")  # fmt: skip
            tested = hyp.get(d.hypothesis_id)
            if tested is None:
                add(d.id, "unknown_reference", "blocking", f"Tests hypothesis {d.hypothesis_id}, which does not exist.")
            elif d.metrics and d.metrics[0] != tested.dependent_variable:
                add(d.id, "metric_mismatch", "major",
                    f"Primary metric {d.metrics[0]!r} is not the hypothesis's dependent variable {tested.dependent_variable!r}.")  # fmt: skip
    if stage == Stage.ANALYSIS:
        for r in state.results:
            for cr in r.conditions:
                if cr.values and not math.isclose(sum(cr.values) / len(cr.values), cr.mean, rel_tol=1e-6, abs_tol=1e-9):
                    add(r.id, "arithmetic_error", "blocking",
                        f"Reported mean of {cr.condition} on {cr.dataset} ({cr.metric}) does not equal the mean of its values.")  # fmt: skip
                if not (cr.lo - 1e-9 <= cr.mean <= cr.hi + 1e-9):
                    add(
                        r.id,
                        "arithmetic_error",
                        "blocking",
                        f"Interval for {cr.condition} on {cr.dataset} does not contain its mean.",
                    )
    if stage == Stage.CONCLUSION:
        hyps = {h.id: h for h in state.hypotheses}
        designs = {d.id: d for d in state.designs}
        results = {r.id: r for r in state.results}
        for k in state.conclusions:
            res = results.get(k.result_id)
            hy = hyps.get(k.hypothesis_id)
            if res is None or hy is None or res.design_id not in designs:
                add(
                    k.id,
                    "unknown_reference",
                    "blocking",
                    "The conclusion refers to a result or hypothesis that does not exist.",
                )
                continue
            exp = expected_verdict(hy, designs[res.design_id], res)
            if k.verdict != exp:
                add(k.id, "overreach" if k.verdict == "supported" else "verdict_mismatch", "blocking",
                    f"The statistics support {exp!r}, not {k.verdict!r}.")  # fmt: skip
            missing = [i for i in k.evidence_ids if i not in ev_ids]
            if missing:
                add(k.id, "unknown_reference", "major", f"Cites evidence that does not exist: {', '.join(missing)}.")
    return out


def _norm(identifier: str) -> str:
    from discoverylab.literature.sources import normalise_identifier

    return normalise_identifier(identifier) or identifier
