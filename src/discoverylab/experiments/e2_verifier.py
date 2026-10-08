"""E2: is the citation verifier itself trustworthy?

Every evidence item from E1 that verified is copied and corrupted in seven known ways; the
verifier must reject every corruption. Benign variants (reformatted title, preprint year, the
item unchanged) must still verify. The verifier resolves through the same cached indexes as E1.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from discoverylab.experiments.common import Summary, e1_run_dirs, load_run, result_dir, table
from discoverylab.literature.registry import SourceRegistry
from discoverylab.models import Evidence
from discoverylab.verify import Verifier

DESCRIPTION = "Verifier validity: known corruptions of real citations, and benign variants that must still pass."

Corrupt = Callable[[Evidence, list[Evidence], random.Random], Evidence | None]


def _other(e: Evidence, pool: list[Evidence], rng: random.Random) -> Evidence | None:
    others = [o for o in pool if o.citation.identifier != e.citation.identifier]
    return rng.choice(others) if others else None


def _cit(e: Evidence, **kw: Any) -> Evidence:
    return e.model_copy(update={"citation": e.citation.model_copy(update=kw)}, deep=True)


def _fabricated(e: Evidence, pool: list[Evidence], rng: random.Random) -> Evidence:
    return _cit(e, identifier=f"doi:10.{rng.randrange(1000, 9999)}/{rng.randrange(16**8):08x}")


def _wrong_title(e: Evidence, pool: list[Evidence], rng: random.Random) -> Evidence | None:
    o = _other(e, pool, rng)
    return _cit(e, title=o.citation.title) if o and o.citation.title else None


def _wrong_author(e: Evidence, pool: list[Evidence], rng: random.Random) -> Evidence | None:
    o = _other(e, pool, rng)
    if not o or not o.citation.first_author:
        return None
    return _cit(e, first_author=o.citation.first_author)


def _wrong_year(e: Evidence, pool: list[Evidence], rng: random.Random) -> Evidence | None:
    if e.citation.year is None:
        return None
    return _cit(e, year=e.citation.year + rng.choice([-5, -4, -3, 3, 4, 5]))


def _swapped_identifier(e: Evidence, pool: list[Evidence], rng: random.Random) -> Evidence | None:
    o = _other(e, pool, rng)
    return _cit(e, identifier=o.citation.identifier) if o else None


def _misattributed_quote(e: Evidence, pool: list[Evidence], rng: random.Random) -> Evidence | None:
    o = _other(e, pool, rng)
    return e.model_copy(update={"quote": o.quote}) if o else None


def _altered_quote(e: Evidence, pool: list[Evidence], rng: random.Random) -> Evidence | None:
    words = e.quote.split()
    idx = [i for i, w in enumerate(words) if len(w) > 3]
    if not idx:
        return None
    i = rng.choice(idx)
    words[i] = {"not": "", "increase": "decrease", "decrease": "increase", "improves": "worsens"}.get(
        words[i].lower(), "never " + words[i]
    )
    return e.model_copy(update={"quote": " ".join(w for w in words if w)})


# Benign: the verifier must accept these.
def _reformatted_title(e: Evidence, pool: list[Evidence], rng: random.Random) -> Evidence | None:
    t = e.citation.title
    return _cit(e, title=t.upper().replace(":", " -")) if t else None


def _preprint_year(e: Evidence, pool: list[Evidence], rng: random.Random) -> Evidence | None:
    return _cit(e, year=e.citation.year - 1) if e.citation.year else None


def _shorter_quote(e: Evidence, pool: list[Evidence], rng: random.Random) -> Evidence | None:
    words = e.quote.split()
    return e.model_copy(update={"quote": " ".join(words[: max(6, len(words) // 2)])}) if len(words) >= 12 else None


CORRUPTIONS: dict[str, Corrupt] = {
    "fabricated_identifier": _fabricated,
    "wrong_title": _wrong_title,
    "wrong_first_author": _wrong_author,
    "wrong_year": _wrong_year,
    "swapped_identifier": _swapped_identifier,
    "misattributed_quote": _misattributed_quote,
    "altered_quote": _altered_quote,
}
BENIGN: dict[str, Corrupt] = {
    "unchanged": lambda e, p, r: e,
    "reformatted_title": _reformatted_title,
    "preprint_year": _preprint_year,
    "shorter_quote": _shorter_quote,
}


def evaluate(evidence: list[Evidence], verifier: Verifier, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    trials = []
    for e in evidence:
        for group, table_ in (("corruption", CORRUPTIONS), ("benign", BENIGN)):
            for name, fn in table_.items():
                v = fn(e, evidence, random.Random(rng.randrange(2**31)))
                if v is None:
                    continue
                chk = verifier.check_evidence(v)
                trials.append({"source_evidence": e.id, "identifier": e.citation.identifier, "group": group, "variant": name,
                               "accepted": chk.correct, "citation_status": chk.citation.status, "quote": chk.quote,
                               "problems": chk.citation.problems})  # fmt: skip
    return trials


def verified_evidence(run_dirs: list[Path]) -> list[Evidence]:
    """Evidence items whose citation and quote verified in E1; deduplicated by (identifier, quote)."""
    seen: set[tuple[str, str]] = set()
    out: list[Evidence] = []
    for d in run_dirs:
        state, ver = load_run(d)
        ok = {c["evidence_id"] for c in (ver or {}).get("evidence", []) if c["correct"]}
        for e in state.evidence:
            key = (e.citation.identifier, e.quote)
            if e.id in ok and key not in seen:
                seen.add(key)
                out.append(e.model_copy(update={"id": f"{state.run_id}:{e.id}"}))
    return out


def summarise(s: Summary, trials: list[dict[str, Any]], n_items: int) -> None:
    s.title = "E2 Citation verifier"
    s.question = "Does the verifier reject fabricated, altered and misattributed citations without rejecting real ones?"
    rows = []
    for group, names in (("corruption", CORRUPTIONS), ("benign", BENIGN)):
        for name in names:
            mine = [t for t in trials if t["variant"] == name]
            acc = sum(t["accepted"] for t in mine)
            rows.append({"variant": name.replace("_", " "), "group": group, "trials": len(mine),
                         "rejected": (len(mine) - acc) / len(mine) if mine else None,
                         "accepted": acc / len(mine) if mine else None})  # fmt: skip
    s.tables.append(table("variants", "Outcome by variant (share of trials)", [("variant", "Variant", "l"), ("group", "Should be", "l"),
        ("trials", "Trials", "r"), ("rejected", "Rejected", "r"), ("accepted", "Accepted", "r")], rows))  # fmt: skip
    cor = [t for t in trials if t["group"] == "corruption"]
    ben = [t for t in trials if t["group"] == "benign"]
    s.findings.append(f"{n_items} verified evidence items from E1 were varied {len(trials)} times.")
    if cor:
        s.findings.append(f"Corruptions rejected: {sum(not t['accepted'] for t in cor)}/{len(cor)}.")
    if ben:
        s.findings.append(f"Benign variants wrongly rejected: {sum(not t['accepted'] for t in ben)}/{len(ben)}.")
    s.notes.append(
        "A citation with no title, author or year is checked only for resolution and quote; leaving metadata out is not a way past the quote check."
    )


def run(cfg: dict[str, Any], runs: Path, registry: SourceRegistry, results: Path | None = None) -> Path:
    items = verified_evidence(e1_run_dirs(runs, results))
    if not items:
        raise RuntimeError("E1 produced no verified evidence to corrupt")
    with result_dir("e2_verifier", cfg, DESCRIPTION, results) as (rd, s):
        trials = evaluate(items, Verifier(registry), int(cfg.get("seed", 0)))
        rd.write_json("trials.json", trials)
        summarise(s, trials, len(items))
    return rd.path


__all__ = ["BENIGN", "CORRUPTIONS", "evaluate", "run", "verified_evidence"]
