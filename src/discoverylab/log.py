"""Append-only, hash-chained process log.

Every agent message, tool call, artefact, critique and human action in a run is one event:
``hash = sha256(prev_hash + canonical(body))``. The log is the record the evaluation reads;
the console renders it; anyone can check it has not been edited with ``verify``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from discoverylab.canonical import canonical, normalise, sha256

GENESIS = "0" * 64


def chain_hash(prev: str, body: dict[str, Any]) -> str:
    return sha256(prev + canonical(body))


@dataclass(frozen=True)
class LogEvent:
    seq: int
    ts: str
    actor: str  # "director", "literature", ..., "critic", "human:<name>", "system"
    kind: str  # "message", "tool_call", "artefact", "critique", "correction", "stage", "error"
    stage: str
    data: dict[str, Any]
    prev: str
    hash: str

    def body(self) -> dict[str, Any]:
        return {"seq": self.seq, "ts": self.ts, "actor": self.actor, "kind": self.kind,
                "stage": self.stage, "data": self.data}  # fmt: skip


class ProcessLog:
    """A JSONL file of chained events. Appends are flushed immediately."""

    def __init__(self, path: Path, clock: Any = None) -> None:
        self.path = path
        self._clock = clock or (lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._head = GENESIS
        if self.path.exists():
            for ev in self.events():
                self._seq, self._head = ev.seq, ev.hash

    @property
    def head(self) -> str:
        return self._head

    def append(self, actor: str, kind: str, stage: str, data: dict[str, Any]) -> LogEvent:
        body = {"seq": self._seq + 1, "ts": self._clock(), "actor": actor, "kind": kind,
                "stage": stage, "data": normalise(data)}  # fmt: skip
        h = chain_hash(self._head, body)
        ev = LogEvent(prev=self._head, hash=h, **body)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(canonical({**body, "prev": self._head, "hash": h}) + "\n")
        self._seq, self._head = ev.seq, h
        return ev

    def events(self) -> Iterator[LogEvent]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield LogEvent(**json.loads(line))


@dataclass
class VerifyReport:
    valid: bool
    events: int
    first_bad_seq: int | None
    problem: str | None


def verify(path: Path) -> VerifyReport:
    prev, n = GENESIS, 0
    for n, ev in enumerate(ProcessLog(path).events() if path.exists() else iter(()), start=1):
        if ev.seq != n:
            return VerifyReport(False, n, n, f"expected seq {n}, found {ev.seq}")
        if ev.prev != prev:
            return VerifyReport(False, n, ev.seq, "previous hash does not match")
        if chain_hash(prev, ev.body()) != ev.hash:
            return VerifyReport(False, n, ev.seq, "event content does not match its hash")
        prev = ev.hash
    return VerifyReport(True, n, None, None)
