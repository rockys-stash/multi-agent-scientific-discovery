"""Unit tests: text handling, identifiers, source parsing, cache, verifier and process log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from discoverylab.literature.cache import CacheMissError, ResponseCache
from discoverylab.literature.sources import Arxiv, Crossref, OpenAlex, SemanticScholar, normalise_identifier
from discoverylab.log import ProcessLog, verify
from discoverylab.models import Citation, Evidence
from discoverylab.text import contains_quote, surname, title_similarity, tokens
from discoverylab.verify import Verifier


def test_quote_matching_tolerates_typography_not_content() -> None:
    src = "Isotonic regression – unlike Platt’s method – is non-\nparametric."
    assert contains_quote(src, "isotonic regression - unlike Platt's method - is nonparametric")
    assert not contains_quote(src, "isotonic regression is parametric")
    assert not contains_quote(src, "   ")


def test_names_titles_and_tokens() -> None:
    assert surname("Niculescu-Mizil, A.") == surname("Alexandru Niculescu-Mizil") == "niculescu-mizil"
    assert surname("José Hernández") == "hernandez"
    assert title_similarity("Predicting good probabilities with supervised learning",
                            "Predicting Good Probabilities With Supervised Learning.") > 0.99  # fmt: skip
    assert title_similarity("Predicting good probabilities", "Deep residual learning") < 0.5
    assert "the" not in tokens("The Brier score")


@pytest.mark.parametrize(("raw", "expected"), [
    ("10.1145/1102351.1102430", "doi:10.1145/1102351.1102430"),
    ("https://doi.org/10.1145/1102351.1102430", "doi:10.1145/1102351.1102430"),
    ("DOI:10.1145/ABC", "doi:10.1145/abc"),
    ("arXiv:1706.04599v2", "arxiv:1706.04599"),
    ("https://arxiv.org/abs/1706.04599", "arxiv:1706.04599"),
    ("https://openalex.org/W2963446712", "openalex:W2963446712"),
    ("Guo et al. 2017", None),
    ("10.1145", None),
])  # fmt: skip
def test_identifier_normalisation(raw: str, expected: str | None) -> None:
    assert normalise_identifier(raw) == expected


# Response bodies below follow each API's documented shape; the field values are invented.
OPENALEX = {"results": [{"id": "https://openalex.org/W1", "doi": "https://doi.org/10.5555/X.1",
                         "display_name": "A title", "publication_year": 2005,
                         "authorships": [{"author": {"display_name": "Ada Example"}}],
                         "primary_location": {"source": {"display_name": "A venue"}},
                         "abstract_inverted_index": {"world": [1], "hello": [0]}}]}  # fmt: skip
CROSSREF = {"message": {"items": [{"DOI": "10.5555/Y.2", "title": ["Y title"], "author": [{"given": "Ben", "family": "Sample"}],
                                   "issued": {"date-parts": [[2019, 3]]}, "container-title": ["Journal"],
                                   "abstract": "<jats:p>Some <jats:italic>text</jats:italic>.</jats:p>"}]}}  # fmt: skip
ARXIV = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2101.00001v3</id>
<title>An  arXiv
 title</title><summary> Abstract text. </summary><published>2021-01-01T00:00:00Z</published>
<author><name>Eli Mock</name></author></entry></feed>"""
S2 = {"data": [{"paperId": "abc", "title": "S2 title", "year": 2020, "abstract": None,
                "authors": [{"name": "Chen Placeholder"}], "externalIds": {"ArXiv": "2001.00002"}}]}  # fmt: skip


def _cache(tmp_path: Path, bodies: dict[str, str], mode: str = "record") -> ResponseCache:
    def fetch(url: str, params: dict[str, str]) -> tuple[int, str]:
        for frag, body in bodies.items():
            if frag in url:
                return 200, body
        return 404, ""

    return ResponseCache(tmp_path / "cache", mode=mode, fetch=fetch)  # type: ignore[arg-type]


def test_source_parsers(tmp_path: Path) -> None:
    c = _cache(tmp_path, {"openalex": json.dumps(OPENALEX), "crossref": json.dumps(CROSSREF),
                          "arxiv": ARXIV, "semanticscholar": json.dumps(S2)})  # fmt: skip
    [oa] = OpenAlex(c).search("x", 5)
    assert (oa.id, oa.abstract, oa.authors, oa.year, oa.venue) == (
        "doi:10.5555/x.1",
        "hello world",
        ["Ada Example"],
        2005,
        "A venue",
    )
    [cr] = Crossref(c).search("x", 5)
    assert (cr.id, cr.authors, cr.year, cr.abstract) == ("doi:10.5555/y.2", ["Ben Sample"], 2019, "Some text .")
    [ax] = Arxiv(c).search("x", 5)
    assert (ax.id, ax.title, ax.abstract, ax.year) == ("arxiv:2101.00001", "An arXiv title", "Abstract text.", 2021)
    [s2] = SemanticScholar(c).search("x", 5)
    assert (s2.id, s2.abstract) == ("arxiv:2001.00002", "")
    assert (
        oa.cache_key and oa.cache_key == c.get("https://api.openalex.org/works", {"search": "x", "per-page": "5"}).key
    )


def test_cache_replay_is_offline_and_exact(tmp_path: Path) -> None:
    rec = _cache(tmp_path, {"example": "body-1"})
    first = rec.get("https://example.org/a", {"q": "1"})
    replay = ResponseCache(tmp_path / "cache", mode="replay")
    again = replay.get("https://example.org/a", {"q": "1"})
    assert (again.body, again.key, again.retrieved_at) == (first.body, first.key, first.retrieved_at)
    with pytest.raises(CacheMissError):
        replay.get("https://example.org/a", {"q": "2"})


def _ev(identifier: str, title: str, author: str, year: int | None, quote: str) -> Evidence:
    return Evidence(
        id="E1",
        claim="c",
        quote=quote,
        citation=Citation(identifier=identifier, title=title, first_author=author, year=year),
    )


def test_verifier_statuses(registry) -> None:  # type: ignore[no-untyped-def]
    v = Verifier(registry)
    title = "Calibration of random forest probabilities with isotonic regression"
    quote = "isotonic regression reduces the Brier score of random forest classifiers"
    ok = v.check_evidence(_ev("https://doi.org/10.5555/TEST.1", title, "A. Example", 2015, quote))
    assert ok.correct and ok.citation.status == "verified" and ok.quote == "found"
    assert (
        v.check_citation(Citation(identifier="10.5555/test.1", title=title, first_author="Example", year=2016)).status
        == "verified"
    )
    for bad in (Citation(identifier="10.5555/test.1", title="Deep residual learning", first_author="Example", year=2015),
                Citation(identifier="10.5555/test.1", title=title, first_author="Someone Else", year=2015),
                Citation(identifier="10.5555/test.1", title=title, first_author="Example", year=2009)):  # fmt: skip
        assert v.check_citation(bad).status == "metadata_mismatch"
    assert v.check_citation(Citation(identifier="10.5555/does.not.exist")).status == "unresolvable"
    assert v.check_citation(Citation(identifier="Example et al. (2015)")).status == "malformed"
    altered = v.check_evidence(
        _ev("10.5555/test.1", title, "Example", 2015, "isotonic regression increases the Brier score")
    )
    assert altered.quote == "not_found" and not altered.correct


def test_process_log_chain_detects_edits(tmp_path: Path) -> None:
    log = ProcessLog(tmp_path / "log.jsonl", clock=lambda: "2026-01-01T00:00:00.000+00:00")
    for i in range(5):
        log.append("director", "message", "literature", {"i": i, "x": 0.1 + 0.2})
    assert verify(log.path).valid and verify(log.path).events == 5
    reopened = ProcessLog(log.path)
    assert reopened.head == log.head
    lines = log.path.read_text().splitlines()
    lines[2] = lines[2].replace('"i":2', '"i":7')
    log.path.write_text("\n".join(lines) + "\n")
    rep = verify(log.path)
    assert not rep.valid and rep.first_bad_seq == 3
    del lines[2]
    log.path.write_text("\n".join(lines) + "\n")
    deleted = verify(log.path)
    assert deleted.first_bad_seq == 3 and "seq" in (deleted.problem or "")


def test_rate_limited_source_is_unverifiable_not_unresolvable(tmp_path: Path) -> None:
    from discoverylab.literature.registry import SourceRegistry

    cache = ResponseCache(tmp_path, fetch=lambda url, params: (429, "Too Many Requests"))
    reg = SourceRegistry([Crossref(cache, "test@example.org")])
    chk = Verifier(reg).check_citation(Citation(identifier="10.5555/real.but.throttled"))
    assert chk.status == "unverifiable"
    assert reg.search("anything") == [] and reg.unavailable["crossref"] == 2


def test_record_mode_refetches_an_earlier_rate_limit(tmp_path: Path) -> None:
    answers = iter([(429, "slow down"), (200, "{}")])
    cache = ResponseCache(tmp_path, fetch=lambda url, params: next(answers))
    assert cache.get("https://x.test/a").status == 429
    assert cache.get("https://x.test/a").status == 200
    assert ResponseCache(tmp_path, mode="replay").get("https://x.test/a").status == 200
