"""Test doubles. Everything here is invented test data, not literature: titles, authors and
abstracts are written for the tests and the identifiers use the reserved test DOI prefix
10.5555. No test result is ever reported as a research finding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from discoverylab.literature.registry import SourceRegistry
from discoverylab.models import Paper, Question
from discoverylab.reasoners.base import Operationalisation

PAPERS = [
    Paper(id="doi:10.5555/test.1", title="Calibration of random forest probabilities with isotonic regression",
          authors=["Ada Example", "Ben Sample"], year=2015, source="fixture",
          abstract=("Random forest probabilities are often poorly calibrated. We show that isotonic regression "
                    "reduces the Brier score of random forest classifiers on ten tabular benchmarks. "
                    "Platt scaling helped less when the calibration set was large.")),
    Paper(id="doi:10.5555/test.2", title="Platt scaling for tree ensembles",
          authors=["Chen Placeholder"], year=2018, source="fixture",
          abstract=("Platt scaling fits a sigmoid to classifier scores. For boosted trees, Platt scaling improved "
                    "calibration but did not change ranking quality measured by the area under the ROC curve.")),
    Paper(id="doi:10.5555/test.3", title="A survey of probability calibration",
          authors=["Dana Stand-In"], year=2020, source="fixture",
          abstract=("We review calibration methods including temperature scaling, Platt scaling and isotonic "
                    "regression. Most evaluations use a small number of datasets.")),
    Paper(id="arxiv:2101.00001", title="Brier score decomposition for tabular classifiers",
          authors=["Eli Mock"], year=2021, source="fixture",
          abstract="The Brier score decomposes into calibration and refinement terms for any classifier."),
]  # fmt: skip


class FixtureSource:
    """An in-memory scholarly index with the ``Source`` interface."""

    def __init__(self, papers: list[Paper], name: str = "fixture") -> None:
        self.name = name
        self.papers = {p.id: p for p in papers}

    def search(self, query: str, limit: int) -> list[Paper]:
        words = set(query.lower().split())
        hits = [p for p in self.papers.values() if words & set(f"{p.title} {p.abstract}".lower().split())]
        return [p.model_copy(update={"query": query}) for p in hits[:limit]]

    def resolve(self, identifier: str) -> Paper | None:
        return self.papers.get(identifier)


ALL_FIXTURE = {"doi": ("fixture",), "arxiv": ("fixture",), "openalex": ("fixture",)}


@pytest.fixture
def registry() -> SourceRegistry:
    return SourceRegistry([FixtureSource(PAPERS)], resolvers=ALL_FIXTURE)


@pytest.fixture
def question() -> Question:
    return Question(id="qt", text="Does isotonic regression or Platt scaling improve random forest calibration?",
                    keywords=["calibration", "random forest", "isotonic regression"],
                    concepts=["calibration", "random forest", "isotonic regression", "platt scaling", "brier score"])  # fmt: skip


@pytest.fixture
def op() -> Operationalisation:
    return Operationalisation(
        independent_variable="post-hoc calibration of a random forest",
        dependent_variable="brier",
        control="random_forest",
        treatments=["random_forest+isotonic"],
        metrics=["brier", "roc_auc"],
        datasets=["wine_binary"],
        expected_direction="decrease",
    )


@pytest.fixture
def tmp_run(tmp_path: Path) -> Path:
    return tmp_path / "run"
