"""Create, resume and verify runs. A run lives in ``runs/<run_id>/``:

- ``state.json``: the artefacts and stage status;
- ``log.jsonl``: the hash-chained process log;
- ``transcript.jsonl``: every language-model prompt and parsed output (language-model runs);
- ``verification.json``: the independent citation and evidence checks.

Retrieval responses are shared across runs in ``cache/`` (content-addressed).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from discoverylab.agents import Director, RunConfig
from discoverylab.human import CorrectionsFile, HumanGate, NoReviewer
from discoverylab.literature.cache import Mode, ResponseCache
from discoverylab.literature.registry import SourceRegistry
from discoverylab.literature.sources import Arxiv, Crossref, OpenAlex, SemanticScholar, Source
from discoverylab.log import ProcessLog
from discoverylab.models import Question, RunState
from discoverylab.reasoners.base import Operationalisation, Reasoner
from discoverylab.verify import Verifier


def load_question(path: Path) -> tuple[Question, Operationalisation | None]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    op = raw.pop("operationalisation", None)
    return Question.model_validate(raw), Operationalisation(**op) if op else None


def build_registry(cache_dir: Path, mode: Mode, sources: list[str] | None = None) -> SourceRegistry:
    cache = ResponseCache(cache_dir, mode=mode)
    mailto = os.environ.get("DISCOVERYLAB_MAILTO", "")
    available: dict[str, Source] = {"openalex": OpenAlex(cache, mailto), "crossref": Crossref(cache, mailto),
                 "arxiv": Arxiv(cache), "semanticscholar": SemanticScholar(cache)}  # fmt: skip
    return SourceRegistry([available[s] for s in (sources or list(available))])


def make_reasoner(name: str, client: Any = None) -> Reasoner:
    if name == "rule":
        from discoverylab.reasoners.rule import RuleReasoner

        return RuleReasoner()
    if name == "claude":
        from discoverylab.reasoners.claude import ClaudeReasoner

        return ClaudeReasoner(client=client)
    raise ValueError(f"unknown reasoner {name!r}")


def new_run_id(question_id: str, reasoner: str) -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{question_id}-{reasoner}"


def _save(run_dir: Path, state: RunState, reasoner: Reasoner, critic: Reasoner | None = None) -> None:
    (run_dir / "state.json").write_text(state.model_dump_json(indent=1), encoding="utf-8")
    for role, agent in (("reasoner", reasoner), ("critic", critic)):
        transcript = getattr(agent, "transcript", None)
        if transcript:
            with (run_dir / "transcript.jsonl").open("a", encoding="utf-8") as f:
                for t in transcript:
                    f.write(json.dumps({"role": role, **t}, ensure_ascii=False) + "\n")
            transcript.clear()


def execute(
    run_dir: Path,
    state: RunState,
    registry: SourceRegistry,
    reasoner: Reasoner,
    op: Operationalisation | None,
    human: HumanGate,
    model_critic: Reasoner | None = None,
    cfg: RunConfig | None = None,
) -> RunState:
    run_dir.mkdir(parents=True, exist_ok=True)
    log = ProcessLog(run_dir / "log.jsonl")
    director = Director(registry, reasoner, log, human, op, model_critic, cfg)
    try:
        state = director.run(state)
    finally:
        _save(run_dir, state, reasoner, model_critic)
    if state.status == "complete":
        verify_run(run_dir, state, registry)
    return state


def verify_run(run_dir: Path, state: RunState, registry: SourceRegistry) -> dict[str, Any]:
    """Independent verification of every citation the run emitted, written next to the run."""
    v = Verifier(registry)
    checks = [v.check_evidence(e).as_dict() for e in state.evidence]
    cited = {e.citation.identifier for e in state.evidence}
    out = {"run_id": state.run_id, "evidence": checks, "citations_emitted": len(cited)}
    (run_dir / "verification.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def human_gate(spec: str | None) -> HumanGate:
    if not spec or spec == "none":
        return NoReviewer()
    return CorrectionsFile(Path(spec))
