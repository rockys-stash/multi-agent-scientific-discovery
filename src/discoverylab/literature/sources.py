"""Adapters for open scholarly indexes, behind one interface.

Each adapter turns raw responses into ``Paper`` records and can resolve a single identifier.
Parsing is defensive: a field a source does not provide stays empty rather than guessed.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Protocol

from discoverylab.literature.cache import CachedResponse, ResponseCache
from discoverylab.models import Paper

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
ARXIV_RE = re.compile(r"^(\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]{2})?/\d{7})$", re.IGNORECASE)
OPENALEX_RE = re.compile(r"^W\d+$")


def normalise_identifier(raw: str) -> str | None:
    """Map a DOI, arXiv id or OpenAlex id in any common spelling to ``scheme:value``."""
    s = raw.strip()
    low = s.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:"):
        if low.startswith(prefix):
            s = s[len(prefix) :]
            return f"doi:{s.lower()}" if DOI_RE.match(s) else None
    if DOI_RE.match(s):
        return f"doi:{s.lower()}"
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv:"):
        if low.startswith(prefix):
            s = s[len(prefix) :]
            break
    s_nov = re.sub(r"v\d+$", "", s)
    if ARXIV_RE.match(s_nov):
        return f"arxiv:{s_nov}"
    for prefix in ("https://openalex.org/", "openalex:"):
        if low.startswith(prefix):
            s = s[len(prefix) :]
            break
    if OPENALEX_RE.match(s.upper()):
        return f"openalex:{s.upper()}"
    return None


class Source(Protocol):
    name: str

    def search(self, query: str, limit: int) -> list[Paper]: ...

    def resolve(self, identifier: str) -> Paper | None: ...


def _ok(resp: CachedResponse) -> bool:
    return 200 <= resp.status < 300


def _openalex_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, places in inv.items():
        for p in places:
            pos[p] = word
    return " ".join(pos[i] for i in sorted(pos))


class OpenAlex:
    name = "openalex"
    base = "https://api.openalex.org"

    def __init__(self, cache: ResponseCache, mailto: str = "") -> None:
        self.cache, self.mailto = cache, mailto

    def _params(self, **kw: str) -> dict[str, str]:
        return {**kw, **({"mailto": self.mailto} if self.mailto else {})}

    def parse(self, w: dict[str, Any], key: str, query: str = "") -> Paper:
        doi = (w.get("doi") or "").removeprefix("https://doi.org/")
        oa = (w.get("id") or "").removeprefix("https://openalex.org/")
        loc = (w.get("primary_location") or {}).get("source") or {}
        return Paper(
            id=f"doi:{doi.lower()}" if doi else f"openalex:{oa}",
            title=w.get("display_name") or w.get("title") or "",
            authors=[
                a["author"]["display_name"] for a in w.get("authorships", []) if a.get("author", {}).get("display_name")
            ],
            year=w.get("publication_year"),
            venue=loc.get("display_name") or "",
            abstract=_openalex_abstract(w.get("abstract_inverted_index")),
            url=w.get("id") or "",
            source=self.name,
            cache_key=key,
            query=query,
        )

    def search(self, query: str, limit: int) -> list[Paper]:
        r = self.cache.get(f"{self.base}/works", self._params(search=query, **{"per-page": str(limit)}))
        if not _ok(r):
            return []
        return [self.parse(w, r.key, query) for w in r.json().get("results", [])]

    def resolve(self, identifier: str) -> Paper | None:
        scheme, value = identifier.split(":", 1)
        path = {"doi": f"doi:{value}", "openalex": value}.get(scheme)
        if path is None:
            return None
        r = self.cache.get(f"{self.base}/works/{path}", self._params())
        return self.parse(r.json(), r.key) if _ok(r) else None


def _strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


class Crossref:
    name = "crossref"
    base = "https://api.crossref.org"

    def __init__(self, cache: ResponseCache, mailto: str = "") -> None:
        self.cache, self.mailto = cache, mailto

    def parse(self, m: dict[str, Any], key: str, query: str = "") -> Paper:
        parts = ((m.get("issued") or {}).get("date-parts") or [[None]])[0]
        authors = [" ".join(x for x in (a.get("given"), a.get("family")) if x) or a.get("name", "")
                   for a in m.get("author", [])]  # fmt: skip
        return Paper(
            id=f"doi:{m['DOI'].lower()}",
            title=(m.get("title") or [""])[0],
            authors=[a for a in authors if a],
            year=parts[0] if parts and isinstance(parts[0], int) else None,
            venue=(m.get("container-title") or [""])[0],
            abstract=_strip_tags(m.get("abstract", "")),
            url=m.get("URL", ""),
            source=self.name,
            cache_key=key,
            query=query,
        )

    def _params(self, **kw: str) -> dict[str, str]:
        return {**kw, **({"mailto": self.mailto} if self.mailto else {})}

    def search(self, query: str, limit: int) -> list[Paper]:
        r = self.cache.get(f"{self.base}/works", self._params(query=query, rows=str(limit)))
        if not _ok(r):
            return []
        items = r.json().get("message", {}).get("items", [])
        return [self.parse(m, r.key, query) for m in items if m.get("DOI")]

    def resolve(self, identifier: str) -> Paper | None:
        scheme, value = identifier.split(":", 1)
        if scheme != "doi":
            return None
        r = self.cache.get(f"{self.base}/works/{value}", self._params())
        return self.parse(r.json()["message"], r.key) if _ok(r) else None


ATOM = "{http://www.w3.org/2005/Atom}"


class Arxiv:
    name = "arxiv"
    base = "http://export.arxiv.org/api/query"

    def __init__(self, cache: ResponseCache) -> None:
        self.cache = cache

    def _parse_feed(self, body: str, key: str, query: str = "") -> list[Paper]:
        out = []
        for e in ET.fromstring(body).findall(f"{ATOM}entry"):
            raw_id = (e.findtext(f"{ATOM}id") or "").rsplit("/abs/", 1)[-1]
            ident = normalise_identifier(f"arxiv:{raw_id}")
            if not ident:
                continue
            published = e.findtext(f"{ATOM}published") or ""
            out.append(Paper(
                id=ident,
                title=" ".join((e.findtext(f"{ATOM}title") or "").split()),
                authors=[a.findtext(f"{ATOM}name") or "" for a in e.findall(f"{ATOM}author")],
                year=int(published[:4]) if published[:4].isdigit() else None,
                venue="arXiv",
                abstract=" ".join((e.findtext(f"{ATOM}summary") or "").split()),
                url=e.findtext(f"{ATOM}id") or "",
                source=self.name,
                cache_key=key,
                query=query,
            ))  # fmt: skip
        return out

    def search(self, query: str, limit: int) -> list[Paper]:
        r = self.cache.get(self.base, {"search_query": f"all:{query}", "max_results": str(limit)})
        return self._parse_feed(r.body, r.key, query) if _ok(r) else []

    def resolve(self, identifier: str) -> Paper | None:
        scheme, value = identifier.split(":", 1)
        if scheme != "arxiv":
            return None
        r = self.cache.get(self.base, {"id_list": value})
        papers = self._parse_feed(r.body, r.key) if _ok(r) else []
        return papers[0] if papers else None


class SemanticScholar:
    name = "semanticscholar"
    base = "https://api.semanticscholar.org/graph/v1"
    fields = "title,authors,year,abstract,venue,externalIds,url"

    def __init__(self, cache: ResponseCache) -> None:
        self.cache = cache

    def parse(self, p: dict[str, Any], key: str, query: str = "") -> Paper:
        ext = p.get("externalIds") or {}
        if ext.get("DOI"):
            ident = f"doi:{ext['DOI'].lower()}"
        elif ext.get("ArXiv"):
            ident = f"arxiv:{ext['ArXiv']}"
        else:
            ident = f"s2:{p['paperId']}"
        return Paper(
            id=ident,
            title=p.get("title") or "",
            authors=[a.get("name", "") for a in p.get("authors") or []],
            year=p.get("year"),
            venue=p.get("venue") or "",
            abstract=p.get("abstract") or "",
            url=p.get("url") or "",
            source=self.name,
            cache_key=key,
            query=query,
        )

    def search(self, query: str, limit: int) -> list[Paper]:
        r = self.cache.get(f"{self.base}/paper/search", {"query": query, "limit": str(limit), "fields": self.fields})
        if not _ok(r):
            return []
        return [self.parse(p, r.key, query) for p in r.json().get("data", [])]

    def resolve(self, identifier: str) -> Paper | None:
        scheme, value = identifier.split(":", 1)
        path = {"doi": f"DOI:{value}", "arxiv": f"ARXIV:{value}"}.get(scheme)
        if path is None:
            return None
        r = self.cache.get(f"{self.base}/paper/{path}", {"fields": self.fields})
        return self.parse(r.json(), r.key) if _ok(r) else None
