"""Human-in-the-loop checkpoints.

A run stops at each checkpoint stage until a reviewer's decisions are available. Decisions come
from a corrections file written by a person, through the console or the CLI. Nothing in this
module invents a decision: with no reviewer configured the run records that no human reviewed
the stage, and the correction count for that stage is not reported as zero but as "no review".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from discoverylab.models import Correction, RunState, Stage

CHECKPOINTS = (Stage.EVIDENCE, Stage.HYPOTHESES, Stage.DESIGN, Stage.CONCLUSION)


class HumanGate(Protocol):
    name: str

    def review(self, stage: Stage, state: RunState) -> list[Correction] | None:
        """Corrections for this stage, [] for approval without changes, or None to pause."""
        ...


class NoReviewer:
    name = "none"

    def review(self, stage: Stage, state: RunState) -> list[Correction] | None:
        return []


class CorrectionsFile:
    """Reads ``{"<stage>": [Correction, ...]}`` written by a person. A missing stage pauses the run."""

    name = "file"

    def __init__(self, path: Path) -> None:
        self.path = path

    def review(self, stage: Stage, state: RunState) -> list[Correction] | None:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if stage.value not in data:
            return None
        return [Correction.model_validate({**c, "stage": stage.value}) for c in data[stage.value]]


def apply(state: RunState, corrections: list[Correction]) -> RunState:
    """Apply approve / edit / reject / add actions to the run's artefacts."""
    for c in corrections:
        if c.action == "approve":
            continue
        field = _field_for(c)
        items = getattr(state, field)
        idx = next((i for i, x in enumerate(items) if x.id == c.target_id), None)
        if c.action == "reject":
            if idx is None:
                raise ValueError(f"cannot reject {c.target_id}: not found")
            c.before = items[idx].model_dump(mode="json")
            del items[idx]
        elif c.action == "edit":
            if idx is None or not c.after:
                raise ValueError(f"cannot edit {c.target_id}")
            c.before = items[idx].model_dump(mode="json")
            items[idx] = items[idx].model_validate({**c.before, **c.after, "id": c.target_id})
        elif c.action == "add":
            if not c.after:
                raise ValueError("add needs the new artefact in 'after'")
            model: type[BaseModel] = type(items[0]) if items else _model_for(field)
            items.append(model.model_validate({**c.after, "id": c.target_id}))
    return state


def _field_for(c: Correction) -> str:
    return {Stage.EVIDENCE: "evidence", Stage.GAPS: "gaps", Stage.HYPOTHESES: "hypotheses",
            Stage.DESIGN: "designs", Stage.CONCLUSION: "conclusions"}[c.stage]  # fmt: skip


def _model_for(field: str) -> type[BaseModel]:
    from discoverylab import models

    table: dict[str, type[BaseModel]] = {"evidence": models.Evidence, "gaps": models.Gap, "hypotheses": models.Hypothesis,
            "designs": models.ExperimentDesign, "conclusions": models.Conclusion}  # fmt: skip
    return table[field]
