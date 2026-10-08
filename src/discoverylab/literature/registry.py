"""Search across sources and resolve identifiers to authoritative records."""

from __future__ import annotations

from dataclasses import dataclass, field

from discoverylab.literature.cache import ResponseCache
from discoverylab.literature.sources import Source, SourceUnavailableError, normalise_identifier
from discoverylab.models import Paper

# Which sources are authoritative for each identifier scheme, in order. The first answer
# supplies the metadata; every answer contributes source text for quote checks.
RESOLVERS: dict[str, tuple[str, ...]] = {
    "doi": ("crossref", "openalex", "semanticscholar"),
    "arxiv": ("arxiv", "semanticscholar"),
    "openalex": ("openalex",),
}


@dataclass
class Resolved:
    identifier: str
    record: Paper | None
    texts: dict[str, str] = field(default_factory=dict)  # source name -> title + abstract
    tried: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)  # tried, but answered "try later"


class SourceRegistry:
    def __init__(self, sources: list[Source], resolvers: dict[str, tuple[str, ...]] | None = None) -> None:
        self.sources = {s.name: s for s in sources}
        self.resolvers = resolvers or RESOLVERS
        self.unavailable: dict[str, int] = {}  # source -> count of "try later" answers seen

    @property
    def caches(self) -> list[ResponseCache]:
        """The distinct response caches behind the sources (usually one, shared)."""
        out: dict[int, ResponseCache] = {}
        for src in self.sources.values():
            cache = getattr(src, "cache", None)
            if cache is not None:
                out[id(cache)] = cache
        return list(out.values())

    def search(self, query: str, per_source: int = 10) -> list[Paper]:
        """Union of results from every source, de-duplicated by identifier.

        When two sources return the same work, the first keeps its metadata and an empty
        abstract is filled from the other.
        """
        merged: dict[str, Paper] = {}
        for src in self.sources.values():
            try:
                found = src.search(query, per_source)
            except SourceUnavailableError:
                self.unavailable[src.name] = self.unavailable.get(src.name, 0) + 1
                continue
            for p in found:
                if p.id in merged:
                    if not merged[p.id].abstract and p.abstract:
                        merged[p.id] = merged[p.id].model_copy(update={"abstract": p.abstract})
                else:
                    merged[p.id] = p
        return list(merged.values())

    def resolve(self, raw_identifier: str) -> Resolved:
        ident = normalise_identifier(raw_identifier)
        if ident is None:
            return Resolved(raw_identifier, None)
        out = Resolved(ident, None)
        for name in self.resolvers.get(ident.split(":", 1)[0], ()):
            src = self.sources.get(name)
            if src is None:
                continue
            out.tried.append(name)
            try:
                rec = src.resolve(ident)
            except SourceUnavailableError:
                self.unavailable[name] = self.unavailable.get(name, 0) + 1
                out.unavailable.append(name)
                continue
            if rec is None:
                continue
            if out.record is None:
                out.record = rec
            out.texts[name] = f"{rec.title}\n{rec.abstract}"
        return out
