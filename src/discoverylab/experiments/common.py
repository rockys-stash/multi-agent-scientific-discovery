"""Shared experiment plumbing: versioned result directories with provenance, and run loading."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from discoverylab.models import RunState

ROOT = Path(__file__).resolve().parents[3]


def git_commit(root: Path = ROOT) -> tuple[str, bool]:
    """The current commit and whether the work tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout.strip())  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return sha, dirty


def table(id_: str, caption: str, columns: list[tuple[str, str, str]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A summary table; ``columns`` are (key, label, align) with align "l" or "r"."""
    return {"id": id_, "caption": caption, "columns": [{"key": k, "label": lab, "align": a} for k, lab, a in columns],
            "rows": rows}  # fmt: skip


@dataclass
class Summary:
    title: str
    question: str
    findings: list[str] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ResultDir:
    experiment: str
    path: Path
    run_id: str
    config: dict[str, Any]

    def write_json(self, name: str, data: Any) -> Path:
        p = self.path / name
        p.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        return p


@contextmanager
def result_dir(
    experiment: str, config: dict[str, Any], description: str, results: Path | None = None
) -> Iterator[tuple[ResultDir, Summary]]:
    """Create the versioned directory, yield it, then write provenance and point ``LATEST`` at it.

    If the body raises, the directory is kept with ``FAILED`` in it and ``LATEST`` is not moved, so a
    failed run can never be mistaken for a result.
    """
    results = results or ROOT / "results"
    sha, dirty = git_commit()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{sha[:7]}"
    path = results / experiment / run_id
    path.mkdir(parents=True, exist_ok=False)
    (path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    rd, summary = ResultDir(experiment, path, run_id, config), Summary(title="", question="")
    t0 = time.perf_counter()
    try:
        yield rd, summary
    except BaseException as exc:
        (path / "FAILED").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
    prov = {
        "experiment": experiment,
        "run_id": run_id,
        "commit": sha,
        "dirty": dirty,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.perf_counter() - t0, 2),
        "description": description,
        "config_sha256": hashlib.sha256((path / "config.yaml").read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "packages": {p: metadata.version(p) for p in ("scikit-learn", "numpy", "scipy", "anthropic")},
    }
    rd.write_json("provenance.json", prov)
    rd.write_json("summary.json", summary.__dict__)
    (results / experiment / "LATEST").write_text(run_id + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    return data


def load_run(run_dir: Path) -> tuple[RunState, dict[str, Any] | None]:
    state = RunState.model_validate_json((run_dir / "state.json").read_text(encoding="utf-8"))
    vf = run_dir / "verification.json"
    return state, json.loads(vf.read_text(encoding="utf-8")) if vf.is_file() else None


def e1_run_dirs(runs: Path, results: Path | None = None) -> list[Path]:
    """The runs produced by the latest E1, which every later experiment builds on."""
    results = results or ROOT / "results"
    latest = results / "e1_runs" / "LATEST"
    if not latest.is_file():
        raise FileNotFoundError("E1 has not been run; later experiments use its runs (make e1)")
    summary = json.loads((results / "e1_runs" / latest.read_text().strip() / "runs.json").read_text(encoding="utf-8"))
    return [runs / r["run_id"] for r in summary if r["status"] == "complete"]
