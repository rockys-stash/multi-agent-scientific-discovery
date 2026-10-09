"""A language-model reasoner with schema-constrained outputs.

Every call returns a validated pydantic object, so a malformed answer fails loudly instead of
flowing into the next stage. Prompts ask for verbatim quotes and identifiers copied from the
supplied records; the verifier then checks that this is what happened.

The client is anything with the Anthropic ``messages.parse`` shape: the Anthropic client (Claude,
needs ``ANTHROPIC_API_KEY``), ``reasoners.local.LocalClient`` (an open-weights model on the CPU),
the E5 replay client, or a test stub. The prompts are the same for every model.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

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

MODEL = os.environ.get("DISCOVERYLAB_MODEL", "claude-opus-5-5")
T = TypeVar("T", bound=BaseModel)

SYSTEM = (
    "You are one agent in a research team. You work only from the records you are given. "
    "Never cite a source that is not in the records, never invent an identifier, and copy every "
    "quote character for character from the abstract it comes from. If the records do not "
    "support something, say so instead of filling the gap."
)


# ---- output schemas (kept flat: they are what the model fills in) ----
class _Queries(BaseModel):
    queries: list[str]


class _EvidenceItem(BaseModel):
    paper_id: str
    title: str
    first_author: str
    year: int | None
    quote: str
    claim: str
    concepts: list[str]
    stance: Literal["supports", "contradicts", "context"]


class _EvidenceList(BaseModel):
    items: list[_EvidenceItem]


class _GapItem(BaseModel):
    description: str
    concepts: list[str]
    evidence_ids: list[str]
    kind: Literal["uncovered", "conflicting", "narrow_scope"]


class _GapList(BaseModel):
    items: list[_GapItem]


class _HypItem(BaseModel):
    statement: str
    independent_variable: str
    dependent_variable: str
    expected_direction: Literal["increase", "decrease", "no_difference"]
    gap_ids: list[str]
    evidence_ids: list[str]
    rationale: str


class _HypList(BaseModel):
    items: list[_HypItem]


class _Design(BaseModel):
    datasets: list[str]
    control: str
    treatments: list[str]
    metrics: list[str] = Field(description="Primary metric first")
    test: str
    folds: int
    seeds: list[int]
    rationale: str


class _ConclusionOut(BaseModel):
    verdict: Literal["supported", "not_supported", "inconclusive"]
    statement: str
    evidence_ids: list[str]
    limitations: list[str]


class _CritiqueItem(BaseModel):
    target_id: str
    issue: str
    severity: Literal["blocking", "major", "minor"]
    message: str


class _CritiqueList(BaseModel):
    items: list[_CritiqueItem]


def _dump(x: Any) -> str:
    if isinstance(x, BaseModel):
        return x.model_dump_json()
    if isinstance(x, list):
        return json.dumps([i.model_dump(mode="json") if isinstance(i, BaseModel) else i for i in x], ensure_ascii=False)
    return json.dumps(x, ensure_ascii=False, default=str)


class ModelReasoner:
    def __init__(
        self,
        client: Any = None,
        model: str = MODEL,
        max_tokens: int = 16000,
        name: str = "claude",
        evidence_batch: int | None = None,
    ) -> None:
        self.name = name
        # None: every record in one call (the design). An integer: that many records per call (ablation, D18).
        self.evidence_batch = evidence_batch
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.usage = Usage()
        self.transcript: list[dict[str, Any]] = []  # prompts and parsed outputs, for the process log

    def _ask(self, step: str, prompt: str, schema: type[T], attempts: int = 2) -> T:
        """One schema-constrained call; an unusable answer is retried once, then fails the step."""
        for attempt in range(attempts):
            resp = self.client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
            )
            u = getattr(resp, "usage", None)
            self.usage.calls += 1
            self.usage.by_step[step] = self.usage.by_step.get(step, 0) + 1
            if u is not None:
                self.usage.input_tokens += int(getattr(u, "input_tokens", 0) or 0)
                self.usage.output_tokens += int(getattr(u, "output_tokens", 0) or 0)
            stop = getattr(resp, "stop_reason", None)
            if stop == "refusal":
                raise RuntimeError(f"{step}: model refused")
            out = resp.parsed_output
            if stop != "max_tokens" and out is not None:
                break
            self.usage.by_step[f"retry:{step}"] = self.usage.by_step.get(f"retry:{step}", 0) + 1
            if attempt == attempts - 1:
                raise RuntimeError(f"{step}: no usable output ({stop or 'unparsed'}) after {attempts} attempts")
        rec = {"step": step, "model": self.model, "prompt": prompt, "output": out.model_dump(mode="json")}
        if getattr(resp, "seed", None) is not None:
            rec["seed"] = resp.seed  # local models: the sampling seed of this call
        self.transcript.append(rec)
        return out

    def plan_queries(self, q: Question) -> list[str]:
        out = self._ask("plan_queries", (
            f"Research question: {q.text}\nKey concepts: {', '.join(q.concepts)}\n\n"
            "Write 3 to 6 short search queries (3-8 words each) for scholarly indexes that together "
            "would find the core prior work on this question."), _Queries)  # fmt: skip
        return [s.strip() for s in out.queries if s.strip()][:6]

    def extract_evidence(self, q: Question, papers: list[Paper], start: int) -> list[Evidence]:
        records = [{"paper_id": p.id, "title": p.title, "authors": p.authors[:3], "year": p.year, "abstract": p.abstract}
                   for p in papers if p.abstract]  # fmt: skip
        size = self.evidence_batch or max(1, len(records))
        items: list[_EvidenceItem] = []
        for i in range(0, len(records), size):
            items += self._extract(q, records[i : i + size])
        return [Evidence(
            id=f"E{start + i}", claim=it.claim, quote=it.quote,
            citation=Citation(identifier=it.paper_id, title=it.title, first_author=it.first_author, year=it.year),
            concepts=it.concepts, stance=it.stance,
        ) for i, it in enumerate(items)]  # fmt: skip

    def _extract(self, q: Question, records: list[dict[str, Any]]) -> list[_EvidenceItem]:
        out = self._ask("extract_evidence", (
            f"Research question: {q.text}\nConcepts: {', '.join(q.concepts)}\n\nRecords:\n{_dump(records)}\n\n"
            "Extract the evidence relevant to the question: at most 2 items per record, only where the "
            "abstract says something specific. For each item give the record's paper_id, title, first "
            "author and year exactly as in the record, a quote copied verbatim from the abstract (one "
            "sentence or less), the claim the quote supports in your own words, the question concepts it "
            "concerns, and whether it supports, contradicts or only gives context for the idea that the "
            "question's intervention helps."), _EvidenceList)  # fmt: skip
        return out.items

    def identify_gaps(self, q: Question, evidence: list[Evidence], papers: list[Paper]) -> list[Gap]:
        out = self._ask("identify_gaps", (
            f"Research question: {q.text}\n\nEvidence:\n{_dump(evidence)}\n\n"
            "Identify 1 to 4 gaps: what the evidence leaves untested, contested or tested only narrowly "
            "(few datasets, one model family). Refer to evidence by id."), _GapList)  # fmt: skip
        return [Gap(id=f"G{i + 1}", **it.model_dump()) for i, it in enumerate(out.items)]

    def generate_hypotheses(
        self, q: Question, gaps: list[Gap], evidence: list[Evidence], op: Operationalisation | None
    ) -> list[Hypothesis]:
        out = self._ask("generate_hypotheses", (
            f"Research question: {q.text}\n\nGaps:\n{_dump(gaps)}\n\nEvidence:\n{_dump(evidence)}\n\n"
            "Propose 1 to 3 falsifiable hypotheses that address the gaps and could be tested on public "
            "tabular classification data. Name one independent and one dependent variable each, the "
            "expected direction of the dependent variable, and the gap and evidence ids it rests on."), _HypList)  # fmt: skip
        return [Hypothesis(id=f"H{i + 1}", **it.model_dump()) for i, it in enumerate(out.items)]

    def design_experiment(
        self, q: Question, h: Hypothesis, catalogue: dict[str, list[str]], op: Operationalisation | None, design_id: str
    ) -> ExperimentDesign:
        out = self._ask("design_experiment", (
            f"Hypothesis:\n{_dump(h)}\n\nToolbox catalogue (use only these names):\n{_dump(catalogue)}\n\n"
            "A condition is a model optionally followed by '+modifier' parts, for example "
            "'random_forest+isotonic'. Design a controlled experiment: datasets, one control condition, "
            "treatment conditions, metrics (the hypothesis's dependent variable first), a paired test, "
            "number of cross-validation folds and the seeds. Explain the choice briefly."), _Design)  # fmt: skip
        return ExperimentDesign(id=design_id, hypothesis_id=h.id, model=out.control.split("+")[0],
                                split="stratified_kfold", **out.model_dump())  # fmt: skip

    def conclude(
        self,
        q: Question,
        h: Hypothesis,
        design: ExperimentDesign,
        result: AnalysisResult,
        evidence: list[Evidence],
        cid: str,
    ) -> Conclusion:
        out = self._ask("conclude", (
            f"Hypothesis:\n{_dump(h)}\n\nDesign:\n{_dump(design)}\n\nComparisons (computed, Holm-adjusted on "
            f"the primary metric):\n{_dump(result.comparisons)}\n\nEvidence:\n{_dump(evidence)}\n\n"
            "State whether the hypothesis is supported, not supported or inconclusive at alpha = 0.05 "
            "(Holm-adjusted, primary metric), in two or three sentences that quote the numbers. Do not "
            "generalise beyond the datasets tested. List the limitations."), _ConclusionOut)  # fmt: skip
        return Conclusion(id=cid, hypothesis_id=h.id, result_id=result.id, **out.model_dump())

    def critique(self, state: RunState, stage: Stage, round_: int, start: int) -> list[Critique]:
        views: dict[Stage, dict[str, list[Any]]] = {
            Stage.EVIDENCE: {"evidence": state.evidence},
            Stage.GAPS: {"gaps": state.gaps, "evidence": state.evidence},
            Stage.HYPOTHESES: {"hypotheses": state.hypotheses, "gaps": state.gaps},
            Stage.DESIGN: {"designs": state.designs, "hypotheses": state.hypotheses},
            Stage.CONCLUSION: {"conclusions": state.conclusions, "comparisons": [r.comparisons for r in state.results]},
        }
        if stage not in views:
            return []
        body = {
            k: [i.model_dump(mode="json") if isinstance(i, BaseModel) else i for i in v]
            for k, v in views[stage].items()
        }
        out = self._ask(f"critique_{stage.value}", (
            f"Research question: {state.question.text}\nStage under review: {stage.value}\n\n{_dump(body)}\n\n"
            "Act as a critical reviewer of the process, not the prose. Report only real problems: claims "
            "the quote does not support, misread evidence, hypotheses not grounded in a gap or not "
            "falsifiable, designs without a fair control or with leakage, conclusions that overreach the "
            "numbers. Use target_id to name the artefact. Return an empty list if there is nothing."), _CritiqueList)  # fmt: skip
        return [Critique(id=f"C{start + i}", round=round_, stage=stage, reviewer="model", **it.model_dump())
                for i, it in enumerate(out.items)]  # fmt: skip

    def revise(self, state: RunState, stage: Stage, critiques: list[Critique]) -> RunState:
        """Drop blocked artefacts; for designs, ask for a corrected design."""
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
