"""A transparent, deterministic reasoner with no language model.

Every step is a fixed procedure, so its outputs can be inspected and tested exactly:

- queries: the question's keywords, plus each pair of its concepts;
- evidence: for each paper, the abstract sentences that best match the question (BM25) and
  mention at least one question concept, quoted verbatim;
- gaps: pairs of question concepts that no evidence (or only one paper) covers together;
- hypotheses and designs: the question's written operationalisation, linked to the gaps;
- conclusions: the fixed verdict rule applied to the primary-metric comparisons.

It can only find what is lexically present, and it cannot read a quote for meaning, so it has
no stance detection (every quote is "context") and no semantic critique.
"""

from __future__ import annotations

from itertools import combinations

from discoverylab.critic import expected_verdict
from discoverylab.models import (
    AnalysisResult,
    Citation,
    Conclusion,
    Critique,
    Evidence,
    ExperimentDesign,
    Gap,
    Hypothesis,
    Paper,
    Question,
    RunState,
    Stage,
)
from discoverylab.reasoners.base import Operationalisation, Usage
from discoverylab.text import BM25, sentences, tokens


def _mentions(text: str, concept: str) -> bool:
    words = tokens(concept)
    t = set(tokens(text))
    return bool(words) and all(w in t for w in words)


class RuleReasoner:
    name = "rule"

    def __init__(self, max_quotes_per_paper: int = 1, min_score: float = 2.0) -> None:
        self.max_quotes_per_paper = max_quotes_per_paper
        self.min_score = min_score
        self.usage = Usage()

    def plan_queries(self, q: Question) -> list[str]:
        qs = [" ".join(q.keywords)] if q.keywords else [q.text]
        qs += [f"{a} {b}" for a, b in combinations(q.concepts, 2)]
        return list(dict.fromkeys(qs))

    def extract_evidence(self, q: Question, papers: list[Paper], start: int) -> list[Evidence]:
        query = tokens(" ".join([q.text, *q.keywords, *q.concepts]))
        cands: list[tuple[float, int, str, Paper]] = []
        for p in papers:
            sents = sentences(p.abstract)
            if not sents:
                continue
            bm = BM25([tokens(s) for s in sents])
            ranked = sorted(((bm.score(query, i), i, s) for i, s in enumerate(sents)), key=lambda x: (-x[0], x[1]))
            kept = 0
            for score, i, s in ranked:
                if kept >= self.max_quotes_per_paper or score < self.min_score:
                    break
                if any(_mentions(s, c) for c in q.concepts):
                    cands.append((score, i, s, p))
                    kept += 1
        out = []
        for n, (_, _, s, p) in enumerate(cands):
            out.append(Evidence(
                id=f"E{start + n}",
                claim=s,  # extractive: the claim is the quote itself
                quote=s,
                citation=Citation(identifier=p.id, title=p.title,
                                  first_author=p.authors[0] if p.authors else "", year=p.year),
                concepts=[c for c in q.concepts if _mentions(s, c)],
                stance="context",
            ))  # fmt: skip
        return out

    def identify_gaps(self, q: Question, evidence: list[Evidence], papers: list[Paper]) -> list[Gap]:
        gaps: list[Gap] = []
        for a, b in combinations(q.concepts, 2):
            both = [e for e in evidence if a in e.concepts and b in e.concepts]
            sources = {e.citation.identifier for e in both}
            if len(sources) <= 1:
                kind = "uncovered" if not both else "narrow_scope"
                gaps.append(Gap(
                    id=f"G{len(gaps) + 1}",
                    description=(f"No retrieved evidence addresses {a} together with {b}." if not both else
                                 f"Only one source addresses {a} together with {b}."),
                    concepts=[a, b],
                    evidence_ids=[e.id for e in evidence if a in e.concepts or b in e.concepts][:6],
                    kind=kind,
                ))  # fmt: skip
        return gaps

    def generate_hypotheses(
        self, q: Question, gaps: list[Gap], evidence: list[Evidence], op: Operationalisation | None
    ) -> list[Hypothesis]:
        if op is None:
            return []
        linked = [g.id for g in gaps]
        direction = {"increase": "increases", "decrease": "decreases", "no_difference": "does not change"}[
            op.expected_direction
        ]
        return [Hypothesis(
            id="H1",
            statement=f"{op.independent_variable} {direction} {op.dependent_variable} relative to {op.control}.",
            independent_variable=op.independent_variable,
            dependent_variable=op.dependent_variable,
            expected_direction=op.expected_direction,
            gap_ids=linked,
            evidence_ids=sorted({i for g in gaps for i in g.evidence_ids}, key=lambda x: int(x[1:]))[:6],
            rationale="Template filled from the question's written operationalisation; linked to every identified gap.",
        )]  # fmt: skip

    def design_experiment(
        self, q: Question, h: Hypothesis, catalogue: dict[str, list[str]], op: Operationalisation | None, design_id: str
    ) -> ExperimentDesign:
        if op is None:
            raise ValueError("the rule reasoner needs an operationalisation to design an experiment")
        return ExperimentDesign(
            id=design_id,
            hypothesis_id=h.id,
            datasets=op.datasets,
            model=op.control.split("+")[0],
            control=op.control,
            treatments=op.treatments,
            metrics=op.metrics,
            test="paired_wilcoxon",
            split="stratified_kfold",
            folds=5,
            seeds=[0, 1, 2],
            rationale="Paired 5-fold cross-validation x 3 seeds; Wilcoxon signed-rank on paired fold scores.",
        )

    def conclude(
        self,
        q: Question,
        h: Hypothesis,
        design: ExperimentDesign,
        result: AnalysisResult,
        evidence: list[Evidence],
        cid: str,
    ) -> Conclusion:
        verdict = expected_verdict(h, design, result)
        primary = design.metrics[0]
        parts = []
        for c in result.comparisons:
            if c["metric"] == primary:
                parts.append(f"{c['treatment']} vs {c['control']} on {c['dataset']}: "
                             f"{float(c['mean_diff']):+.4f} [{float(c['lo']):+.4f}, {float(c['hi']):+.4f}], "  # type: ignore[arg-type]
                             f"Holm p = {float(c['p_holm']):.3g}")  # type: ignore[arg-type]  # fmt: skip
        return Conclusion(
            id=cid,
            hypothesis_id=h.id,
            result_id=result.id,
            verdict=verdict,
            statement=f"Hypothesis {h.id} is {verdict.replace('_', ' ')} on {primary}. " + "; ".join(parts) + ".",
            evidence_ids=h.evidence_ids,
            limitations=[
                f"Datasets: {', '.join(design.datasets)} only.",
                "Evidence selection is lexical; quotes were not read for meaning.",
            ],
        )

    def critique(self, state: RunState, stage: Stage, round_: int, start: int) -> list[Critique]:
        return []  # the rule critic in discoverylab.critic covers what rules can check

    def revise(self, state: RunState, stage: Stage, critiques: list[Critique]) -> RunState:
        """Drop the artefacts a blocking critique targets; the rule reasoner cannot rewrite them."""
        blocked = {c.target_id for c in critiques if c.severity == "blocking"}
        if stage == Stage.EVIDENCE:
            state.evidence = [e for e in state.evidence if e.id not in blocked]
        elif stage == Stage.HYPOTHESES:
            state.hypotheses = [h for h in state.hypotheses if h.id not in blocked]
        else:
            return state
        for c in critiques:
            if c.severity == "blocking":
                c.resolution, c.resolution_note = "fixed", f"{c.target_id} removed."
        return state
