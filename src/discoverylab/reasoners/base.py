"""The reasoning interface every agent uses.

Agents own the workflow (what is asked, what is checked, what is logged); a ``Reasoner``
supplies the judgement at each step. Two implementations exist:

- ``RuleReasoner``: transparent, deterministic, no language model. It extracts quotes
  verbatim, finds gaps by concept coverage and fills hypothesis templates from the question's
  operationalisation. It is the baseline and the test double.
- ``ModelReasoner``: a language model with schema-constrained outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from discoverylab.models import (
    AnalysisResult,
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


@dataclass
class Operationalisation:
    """How a question's concepts map to toolbox variables (written by a person per question).

    The rule baseline depends on it; the language-model reasoner is given the toolbox catalogue
    instead and must map concepts itself. Reported as a difference between the two.
    """

    independent_variable: str
    dependent_variable: str
    control: str
    treatments: list[str]
    metrics: list[str]
    datasets: list[str]
    expected_direction: str = "increase"


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_step: dict[str, int] = field(default_factory=dict)


class Reasoner(Protocol):
    name: str
    usage: Usage

    def plan_queries(self, q: Question) -> list[str]: ...

    def extract_evidence(self, q: Question, papers: list[Paper], start: int) -> list[Evidence]: ...

    def identify_gaps(self, q: Question, evidence: list[Evidence], papers: list[Paper]) -> list[Gap]: ...

    def generate_hypotheses(
        self, q: Question, gaps: list[Gap], evidence: list[Evidence], op: Operationalisation | None
    ) -> list[Hypothesis]: ...

    def design_experiment(
        self, q: Question, h: Hypothesis, catalogue: dict[str, list[str]], op: Operationalisation | None, design_id: str
    ) -> ExperimentDesign: ...

    def conclude(
        self,
        q: Question,
        h: Hypothesis,
        design: ExperimentDesign,
        result: AnalysisResult,
        evidence: list[Evidence],
        cid: str,
    ) -> Conclusion: ...

    def critique(self, state: RunState, stage: Stage, round_: int, start: int) -> list[Critique]: ...

    def revise(self, state: RunState, stage: Stage, critiques: list[Critique]) -> RunState: ...
