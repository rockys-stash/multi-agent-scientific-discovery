"""Independent citation and evidence verification.

The verifier never trusts the paper record an agent worked from. It re-resolves every cited
identifier through the source registry and compares what the agent claims (title, first
author, year, quoted text) with what the authoritative source returns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from discoverylab.literature.registry import Resolved, SourceRegistry
from discoverylab.literature.sources import normalise_identifier
from discoverylab.models import Citation, Evidence
from discoverylab.text import contains_quote, surname, title_similarity

TITLE_MIN = 0.85  # token-sequence similarity; tolerates punctuation and subtitle formatting
YEAR_TOLERANCE = 1  # preprint vs. published year

# "unverifiable": no record, and at least one authoritative source was rate-limited or down, so
# absence is not evidence of fabrication. It is neither counted as verified nor as fabricated.
CitationStatus = Literal["verified", "metadata_mismatch", "unresolvable", "unverifiable", "malformed"]
QuoteStatus = Literal["found", "not_found", "no_source_text"]


@dataclass
class CitationCheck:
    identifier: str
    normalised: str | None
    status: CitationStatus
    title_similarity: float | None = None
    author_match: bool | None = None
    year_match: bool | None = None
    resolved_title: str = ""
    resolved_first_author: str = ""
    resolved_year: int | None = None
    resolved_by: str = ""
    tried: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class EvidenceCheck:
    evidence_id: str
    citation: CitationCheck
    quote: QuoteStatus
    found_in: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.citation.status == "verified" and self.quote == "found"

    def as_dict(self) -> dict[str, object]:
        return {"evidence_id": self.evidence_id, "citation": self.citation.as_dict(),
                "quote": self.quote, "found_in": self.found_in, "correct": self.correct}  # fmt: skip


class Verifier:
    def __init__(self, registry: SourceRegistry) -> None:
        self.registry = registry
        self._cache: dict[str, Resolved] = {}

    def _resolve(self, identifier: str) -> Resolved:
        if identifier not in self._cache:
            self._cache[identifier] = self.registry.resolve(identifier)
        return self._cache[identifier]

    def check_citation(self, c: Citation) -> CitationCheck:
        if normalise_identifier(c.identifier) is None:
            return CitationCheck(c.identifier, None, "malformed", problems=["not a DOI, arXiv or OpenAlex identifier"])
        r = self._resolve(c.identifier)
        if r.record is None and r.unavailable:
            return CitationCheck(c.identifier, r.identifier, "unverifiable", tried=r.tried,
                                 problems=[f"no record; unavailable: {', '.join(r.unavailable)}"])  # fmt: skip
        if r.record is None:
            return CitationCheck(c.identifier, r.identifier, "unresolvable", tried=r.tried,
                                 problems=["no source returned a record"])  # fmt: skip
        rec = r.record
        first = rec.authors[0] if rec.authors else ""
        chk = CitationCheck(
            c.identifier, r.identifier, "verified",
            resolved_title=rec.title, resolved_first_author=first, resolved_year=rec.year,
            resolved_by=rec.source, tried=r.tried,
        )  # fmt: skip
        if c.title:
            chk.title_similarity = round(title_similarity(c.title, rec.title), 4)
            if chk.title_similarity < TITLE_MIN:
                chk.problems.append("title does not match the source")
        if c.first_author:
            chk.author_match = bool(first) and surname(c.first_author) == surname(first)
            if not chk.author_match:
                chk.problems.append("first author does not match the source")
        if c.year is not None:
            chk.year_match = rec.year is not None and abs(rec.year - c.year) <= YEAR_TOLERANCE
            if not chk.year_match:
                chk.problems.append("year does not match the source")
        if chk.problems:
            chk.status = "metadata_mismatch"
        return chk

    def check_evidence(self, e: Evidence) -> EvidenceCheck:
        cit = self.check_citation(e.citation)
        if cit.status in ("malformed", "unresolvable", "unverifiable"):
            return EvidenceCheck(e.id, cit, "no_source_text")
        texts = self._resolve(e.citation.identifier).texts
        if not any(t.strip() for t in texts.values()):
            return EvidenceCheck(e.id, cit, "no_source_text")
        found = [name for name, t in texts.items() if contains_quote(t, e.quote)]
        return EvidenceCheck(e.id, cit, "found" if found else "not_found", found)
