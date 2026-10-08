"""Typed artefacts that agents exchange.

Agents never pass free text to each other: each stage reads and writes these models, which is
what makes the process checkable. Identifiers are short, stable and unique within a run.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Stage(StrEnum):
    QUESTION = "question"
    LITERATURE = "literature"
    EVIDENCE = "evidence"
    GAPS = "gaps"
    HYPOTHESES = "hypotheses"
    DESIGN = "design"
    ANALYSIS = "analysis"
    CRITIQUE = "critique"
    CONCLUSION = "conclusion"


STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)


class Question(BaseModel):
    id: str
    text: str
    keywords: list[str] = Field(default_factory=list)
    # Concepts the question is about; gap identification checks which combinations the
    # literature covers.
    concepts: list[str] = Field(default_factory=list)
    domain_hint: str = ""


class Paper(BaseModel):
    """A source as retrieved from a scholarly index, never as remembered by an agent."""

    id: str  # "doi:<doi>", "arxiv:<id>" or "openalex:<W-id>"
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    abstract: str = ""
    url: str = ""
    source: str  # which index returned it
    cache_key: str = ""  # content hash of the raw response it was parsed from
    query: str = ""  # the search that found it


class Citation(BaseModel):
    """What an agent claims about a source. The verifier checks it against the source itself."""

    identifier: str
    title: str = ""
    first_author: str = ""
    year: int | None = None


class Evidence(BaseModel):
    id: str
    claim: str
    quote: str  # must appear verbatim in the cited source's text
    citation: Citation
    concepts: list[str] = Field(default_factory=list)
    stance: Literal["supports", "contradicts", "context"] = "context"


class Gap(BaseModel):
    id: str
    description: str
    concepts: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    kind: Literal["uncovered", "conflicting", "narrow_scope"] = "uncovered"


class Hypothesis(BaseModel):
    id: str
    statement: str
    independent_variable: str
    dependent_variable: str
    expected_direction: Literal["increase", "decrease", "no_difference"]
    gap_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class ExperimentDesign(BaseModel):
    id: str
    hypothesis_id: str
    datasets: list[str]
    model: str
    control: str  # condition name of the baseline
    treatments: list[str]
    metrics: list[str]  # first one is the primary metric
    test: str
    split: Literal["stratified_holdout", "stratified_kfold"] = "stratified_kfold"
    folds: int = 5
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    rationale: str = ""


class ConditionResult(BaseModel):
    condition: str
    dataset: str
    metric: str
    values: list[float]  # one per fold x seed
    mean: float
    lo: float
    hi: float


class AnalysisResult(BaseModel):
    id: str
    design_id: str
    conditions: list[ConditionResult]
    comparisons: list[dict[str, float | str | int | bool | None]]
    runtime_seconds: float
    summary: str = ""


class Critique(BaseModel):
    id: str
    round: int
    target_id: str
    stage: Stage
    issue: str  # a short machine-readable type, e.g. "unsupported_claim"
    severity: Literal["blocking", "major", "minor"]
    message: str
    reviewer: Literal["rule", "model", "human"]
    resolution: Literal["open", "fixed", "accepted", "dismissed"] = "open"
    resolution_note: str = ""


class Conclusion(BaseModel):
    id: str
    hypothesis_id: str
    result_id: str
    verdict: Literal["supported", "not_supported", "inconclusive"]
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class Correction(BaseModel):
    """A human action at a checkpoint. Only real human input is ever recorded as one."""

    stage: Stage
    target_id: str
    action: Literal["approve", "edit", "reject", "add"]
    before: dict[str, object] | None = None
    after: dict[str, object] | None = None
    reason: str = ""
    actor: str


class RunState(BaseModel):
    run_id: str
    question: Question
    reasoner: str
    stage: Stage = Stage.QUESTION
    status: Literal["running", "awaiting_review", "complete", "failed"] = "running"
    papers: list[Paper] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    designs: list[ExperimentDesign] = Field(default_factory=list)
    results: list[AnalysisResult] = Field(default_factory=list)
    critiques: list[Critique] = Field(default_factory=list)
    conclusions: list[Conclusion] = Field(default_factory=list)
    corrections: list[Correction] = Field(default_factory=list)
    completed: list[Stage] = Field(default_factory=list)  # stages produced and critiqued
    reviewed: dict[str, str] = Field(default_factory=dict)  # checkpoint stage -> reviewer name
    error: str | None = None
