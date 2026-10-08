"""The research team: a Director coordinating Literature, Hypothesis, Experiment and Critic agents.

Agents are thin roles around a shared ``Reasoner`` and the tools they own. The Director runs the
fixed workflow, sends every stage's output to the Critic, asks the producing agent to revise
when a blocking problem is found, stops at human checkpoints, and logs every step.
"""

from __future__ import annotations

from dataclasses import dataclass

from discoverylab import toolbox
from discoverylab.critic import rule_critique
from discoverylab.human import CHECKPOINTS, HumanGate, apply
from discoverylab.literature.registry import SourceRegistry
from discoverylab.log import ProcessLog
from discoverylab.models import Critique, Paper, Question, RunState, Stage
from discoverylab.reasoners.base import Operationalisation, Reasoner
from discoverylab.text import BM25, tokens

WORKFLOW = (Stage.LITERATURE, Stage.EVIDENCE, Stage.GAPS, Stage.HYPOTHESES, Stage.DESIGN,
            Stage.ANALYSIS, Stage.CONCLUSION, Stage.CRITIQUE)  # fmt: skip
OWNER = {Stage.LITERATURE: "literature", Stage.EVIDENCE: "literature", Stage.GAPS: "hypothesis",
         Stage.HYPOTHESES: "hypothesis", Stage.DESIGN: "experiment", Stage.ANALYSIS: "experiment",
         Stage.CONCLUSION: "experiment", Stage.CRITIQUE: "critic"}  # fmt: skip


@dataclass
class RunConfig:
    per_source: int = 10
    max_papers: int = 25
    max_hypotheses: int = 2
    critique_rounds: int = 2


class LiteratureAgent:
    def __init__(self, registry: SourceRegistry, reasoner: Reasoner, log: ProcessLog, cfg: RunConfig) -> None:
        self.registry, self.reasoner, self.log, self.cfg = registry, reasoner, log, cfg

    def search(self, q: Question) -> list[Paper]:
        queries = self.reasoner.plan_queries(q)
        self.log.append("literature", "message", "literature", {"queries": queries})
        found: dict[str, Paper] = {}
        for query in queries:
            papers = self.registry.search(query, self.cfg.per_source)
            self.log.append("literature", "tool_call", "literature", {
                "tool": "search", "query": query, "results": len(papers),
                "by_source": {s: sum(p.source == s for p in papers) for s in self.registry.sources},
                "cache_keys": sorted({p.cache_key for p in papers}),
            })  # fmt: skip
            for p in papers:
                if p.id not in found or (not found[p.id].abstract and p.abstract):
                    found[p.id] = p
        # Rank by lexical relevance to the question; papers without an abstract cannot supply
        # quotable evidence, so they rank last.
        pool = list(found.values())
        qtok = tokens(" ".join([q.text, *q.keywords, *q.concepts]))
        bm = BM25([tokens(f"{p.title} {p.abstract}") for p in pool])
        order = sorted(range(len(pool)), key=lambda i: (not pool[i].abstract, -bm.score(qtok, i), pool[i].id))
        return [pool[i] for i in order[: self.cfg.max_papers]]


class Director:
    def __init__(
        self,
        registry: SourceRegistry,
        reasoner: Reasoner,
        log: ProcessLog,
        human: HumanGate,
        op: Operationalisation | None = None,
        model_critic: Reasoner | None = None,
        cfg: RunConfig | None = None,
    ) -> None:
        self.cfg = cfg or RunConfig()
        self.reasoner, self.log, self.human, self.op = reasoner, log, human, op
        self.model_critic = model_critic
        self.literature = LiteratureAgent(registry, reasoner, log, self.cfg)

    # ---- helpers ----
    def _critique(self, state: RunState, stage: Stage, round_: int) -> list[Critique]:
        start = len(state.critiques) + 1
        found = rule_critique(state, stage, round_, start)
        if self.model_critic is not None:
            found += self.model_critic.critique(state, stage, round_, start + len(found))
        for c in found:
            self.log.append("critic", "critique", stage.value, c.model_dump(mode="json"))
        state.critiques.extend(found)
        return found

    def _review_cycle(self, state: RunState, stage: Stage) -> None:
        for round_ in range(1, self.cfg.critique_rounds + 1):
            found = self._critique(state, stage, round_)
            blocking = [c for c in found if c.severity == "blocking"]
            if not blocking:
                return
            self.log.append("director", "message", stage.value, {
                "to": OWNER[stage], "request": "revise", "round": round_, "critiques": [c.id for c in blocking]})  # fmt: skip
            self.reasoner.revise(state, stage, blocking)
            for c in blocking:
                self.log.append(OWNER[stage], "message", stage.value, {
                    "response_to": c.id, "resolution": c.resolution, "note": c.resolution_note})  # fmt: skip
        # Whatever is still blocking after the last round stays open and is visible in the run.

    def _produce(self, state: RunState, stage: Stage) -> None:
        q = state.question
        if stage == Stage.LITERATURE:
            state.papers = self.literature.search(q)
            self.log.append("literature", "artefact", stage.value, {"papers": [p.id for p in state.papers]})
        elif stage == Stage.EVIDENCE:
            state.evidence = self.reasoner.extract_evidence(q, [p for p in state.papers if p.abstract], 1)
            for e in state.evidence:
                self.log.append("literature", "artefact", stage.value, e.model_dump(mode="json"))
        elif stage == Stage.GAPS:
            state.gaps = self.reasoner.identify_gaps(q, state.evidence, state.papers)
            for g in state.gaps:
                self.log.append("hypothesis", "artefact", stage.value, g.model_dump(mode="json"))
        elif stage == Stage.HYPOTHESES:
            hs = self.reasoner.generate_hypotheses(q, state.gaps, state.evidence, self.op)
            state.hypotheses = hs[: self.cfg.max_hypotheses]
            for h in state.hypotheses:
                self.log.append("hypothesis", "artefact", stage.value, h.model_dump(mode="json"))
        elif stage == Stage.DESIGN:
            cat = catalogue()
            state.designs = []
            for i, h in enumerate(state.hypotheses, start=1):
                d = self.reasoner.design_experiment(q, h, cat, self.op, f"D{i}")
                state.designs.append(d)
                self.log.append("experiment", "artefact", stage.value, d.model_dump(mode="json"))
        elif stage == Stage.ANALYSIS:
            state.results = []
            for i, d in enumerate(state.designs, start=1):
                if toolbox.validate(d):
                    self.log.append(
                        "experiment", "error", stage.value, {"design": d.id, "problems": toolbox.validate(d)}
                    )
                    continue
                self.log.append("experiment", "tool_call", stage.value, {"tool": "run_design", "design": d.id})
                r = toolbox.run_design(d, f"R{i}")
                state.results.append(r)
                self.log.append("experiment", "artefact", stage.value, r.model_dump(mode="json"))
        elif stage == Stage.CONCLUSION:
            hyp = {h.id: h for h in state.hypotheses}
            designs = {d.id: d for d in state.designs}
            state.conclusions = []
            for i, r in enumerate(state.results, start=1):
                d = designs[r.design_id]
                c = self.reasoner.conclude(q, hyp[d.hypothesis_id], d, r, state.evidence, f"K{i}")
                state.conclusions.append(c)
                self.log.append("experiment", "artefact", stage.value, c.model_dump(mode="json"))
        elif stage == Stage.CRITIQUE:
            # Final whole-run review: every earlier stage is checked again against the final state.
            for s in WORKFLOW[:-1]:
                self._critique(state, s, round_=99)

    # ---- the workflow ----
    def run(self, state: RunState) -> RunState:
        if not state.completed:
            self.log.append("director", "stage", Stage.QUESTION.value, {"question": state.question.model_dump(mode="json"),
                            "reasoner": self.reasoner.name, "reviewer": self.human.name})  # fmt: skip
        try:
            for stage in WORKFLOW:
                if stage not in state.completed:
                    state.stage = stage
                    self.log.append("director", "stage", stage.value, {"to": OWNER[stage], "action": "start"})
                    self._produce(state, stage)
                    if stage not in (Stage.CRITIQUE,):
                        self._review_cycle(state, stage)
                    state.completed.append(stage)
                if stage in CHECKPOINTS and stage.value not in state.reviewed:
                    corrections = self.human.review(stage, state)
                    if corrections is None:
                        state.status = "awaiting_review"
                        self.log.append("director", "message", stage.value, {"awaiting": "human review"})
                        return state
                    apply(state, corrections)
                    for c in corrections:
                        state.corrections.append(c)
                        self.log.append(f"human:{c.actor}", "correction", stage.value, c.model_dump(mode="json"))
                    state.reviewed[stage.value] = self.human.name
                    if self.human.name == "none":
                        self.log.append(
                            "system", "message", stage.value, {"checkpoint": "no human reviewer configured"}
                        )
            state.status = "complete"
            self.log.append("director", "stage", "complete", {"status": "complete"})
        except Exception as e:  # recorded, then re-raised for the caller
            state.status, state.error = "failed", f"{type(e).__name__}: {e}"
            self.log.append("system", "error", state.stage.value, {"error": state.error})
            raise
        return state


def catalogue() -> dict[str, list[str]]:
    return {
        "datasets": sorted(toolbox.DATASETS),
        "dataset_notes": [f"{k}: {v.description}" for k, v in toolbox.DATASETS.items()],
        "models": sorted(toolbox.MODELS),
        "modifiers": list(toolbox.MODIFIERS),
        "metrics": sorted(toolbox.METRICS),
        "tests": list(toolbox.TESTS),
    }
