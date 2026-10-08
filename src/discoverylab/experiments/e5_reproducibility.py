"""E5: is a run reproducible?

Replay: every completed E1 run is executed again from the retrieval cache in replay mode (no
network). A language-model run is replayed from its transcript: each prompt must match the
recorded one and gets the recorded answer. Everything downstream (ranking, verification,
the experiments themselves, the statistics, the critic) is recomputed, and the artefacts are
compared hash by hash.

Live re-run (needs the network and, for model runs, a key): the same question is run again from
scratch, and the overlap of papers, evidence and verdicts with the original is measured.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from discoverylab.agents import RunConfig
from discoverylab.canonical import digest
from discoverylab.experiments.common import Summary, e1_run_dirs, load_run, result_dir, table
from discoverylab.human import NoReviewer
from discoverylab.literature.registry import SourceRegistry
from discoverylab.models import RunState
from discoverylab.reasoners.base import Reasoner
from discoverylab.run import execute, load_question, make_reasoner

DESCRIPTION = "Reproducibility: artefact-level comparison of replayed runs with the originals."

ARTEFACTS = ("papers", "evidence", "gaps", "hypotheses", "designs", "results", "conclusions", "critiques")


class TranscriptMismatchError(RuntimeError):
    pass


class _Messages:
    def __init__(self, transcript: list[dict[str, Any]]) -> None:
        self.queue = list(transcript)

    def parse(
        self, *, model: str, max_tokens: int, system: str, messages: list[dict[str, Any]], output_format: Any
    ) -> Any:
        if not self.queue:
            raise TranscriptMismatchError("the replay asked more questions than the original run")
        rec = self.queue.pop(0)
        prompt = messages[-1]["content"]
        if prompt != rec["prompt"]:
            raise TranscriptMismatchError(f"prompt for step {rec['step']!r} differs from the recorded one")

        class _Resp:
            stop_reason = "end_turn"
            usage = None
            parsed_output = output_format.model_validate(rec["output"])

        return _Resp()


class ReplayClient:
    """Stands in for the Anthropic client and answers from a run's transcript, in order."""

    def __init__(self, transcript: list[dict[str, Any]]) -> None:
        self.messages = _Messages(transcript)


def artefact_hashes(state: RunState) -> dict[str, str]:
    data = state.model_dump(mode="json")
    for r in data["results"]:
        r.pop("runtime_seconds", None)
    return {k: digest(data[k]) for k in ARTEFACTS}


def _transcript(run_dir: Path) -> list[dict[str, Any]]:
    p = run_dir / "transcript.jsonl"
    return (
        [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.is_file() else []
    )


def replay(run_dir: Path, out_dir: Path, registry: SourceRegistry, root: Path, rc: RunConfig,
           reasoner_factory: Callable[[str, Any], Reasoner] = make_reasoner) -> dict[str, Any]:  # fmt: skip
    original, ver = load_run(run_dir)
    q, op = load_question(root / "configs" / "questions" / f"{original.question.id}.yaml")
    base = original.reasoner.removesuffix("+mc")
    t = _transcript(run_dir)
    reasoner = reasoner_factory(
        base, ReplayClient([x for x in t if x.get("role", "reasoner") == "reasoner"]) if base != "rule" else None
    )
    critic = (reasoner_factory("claude", ReplayClient([x for x in t if x.get("role") == "critic"]))
              if original.reasoner.endswith("+mc") else None)  # fmt: skip
    fresh = RunState(run_id=original.run_id, question=q, reasoner=original.reasoner)
    try:
        replayed = execute(out_dir / original.run_id, fresh, registry, reasoner, op, NoReviewer(), critic, rc)
    except Exception as exc:
        return {
            "run_id": original.run_id,
            "reasoner": original.reasoner,
            "replayed": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    a, b = artefact_hashes(original), artefact_hashes(replayed)
    new_ver = (
        json.loads((out_dir / original.run_id / "verification.json").read_text(encoding="utf-8"))
        if replayed.status == "complete"
        else None
    )
    return {"run_id": original.run_id, "reasoner": original.reasoner, "replayed": True, "status": replayed.status,
            "identical": [k for k in ARTEFACTS if a[k] == b[k]], "different": [k for k in ARTEFACTS if a[k] != b[k]],
            "verification_identical": digest(ver) == digest(new_ver)}  # fmt: skip


def summarise(s: Summary, rows: list[dict[str, Any]]) -> None:
    s.title = "E5 Reproducibility"
    s.question = "Does replaying a recorded run reproduce every artefact and number?"
    out = []
    for r in rows:
        out.append({"run_id": r["run_id"], "reasoner": r["reasoner"], "replayed": "yes" if r["replayed"] else "no",
                    "identical": f"{len(r.get('identical', []))}/{len(ARTEFACTS)}" if r["replayed"] else None,
                    "different": ", ".join(r.get("different", [])) or ("none" if r["replayed"] else r.get("reason", "")),
                    "verification": ("same" if r.get("verification_identical") else "differs") if r["replayed"] else None})  # fmt: skip
    s.tables.append(table("replay", "Replay from the recorded cache and transcript", [("run_id", "Run", "l"), ("reasoner", "Reasoner", "l"),
        ("replayed", "Replayed", "l"), ("identical", "Artefact groups identical", "r"), ("different", "Differences", "l"),
        ("verification", "Verification", "l")], out))  # fmt: skip
    ok = [r for r in rows if r["replayed"] and not r["different"] and r["verification_identical"]]
    s.findings.append(f"{len(ok)}/{len(rows)} runs replayed with every artefact group and the verification identical.")
    s.notes.append(
        "Live re-run overlap (papers, evidence, verdicts) needs the scholarly APIs and, for model runs, an API key: Status pending until a run with both."
    )


def run(
    cfg: dict[str, Any],
    runs: Path,
    registry: SourceRegistry,
    root: Path = Path("."),
    results: Path | None = None,
    reasoner_factory: Callable[[str, Any], Reasoner] = make_reasoner,
) -> Path:
    dirs = e1_run_dirs(runs, results)
    rc = RunConfig(**cfg.get("run_config", {}))
    with result_dir("e5_reproducibility", cfg, DESCRIPTION, results) as (rd, s):
        rows = [replay(d, rd.path / "replays", registry, root, rc, reasoner_factory) for d in dirs]
        rd.write_json("replays.json", rows)
        summarise(s, rows)
    return rd.path
