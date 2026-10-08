"""Text normalisation, sentence splitting and lexical similarity (no model downloads needed)."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "while",
        "with",
        "without",
        "whether",
        "our",
        "not",
        "no",
        "than",
        "also",
        "more",
        "most",
        "may",
        "using",
        "use",
        "used",
    ]
)
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")
_QUOTES = dict.fromkeys(map(ord, "‘’‚′"), "'") | dict.fromkeys(map(ord, "“”„″"), '"')


def normalise(text: str) -> str:
    """Unicode NFKC, unified quotes and dashes, de-hyphenated line breaks, collapsed spaces, casefold."""
    t = unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_QUOTES)
    t = re.sub(r"(\w)-\s+(\w)", r"\1\2", t)
    return re.sub(r"\s+", " ", t).strip().casefold()


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalise(text)) if w not in STOP and len(w) > 1]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", re.sub(r"\s+", " ", text).strip())
    return [p.strip() for p in parts if len(p.split()) >= 5]


def contains_quote(source: str, quote: str) -> bool:
    q = normalise(quote).strip(" .\"'")
    return bool(q) and q in normalise(source)


def title_similarity(a: str, b: str) -> float:
    na, nb = " ".join(tokens(a)), " ".join(tokens(b))
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def surname(name: str) -> str:
    """Last token of a personal name, normalised ("Niculescu-Mizil, A." and "A. Niculescu-Mizil" agree)."""
    n = normalise(name)
    n = n.split(",", 1)[0] if "," in n else n.split()[-1] if n.split() else ""
    return re.sub(r"[^a-z\-]", "", unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode())


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs, self.k1, self.b = docs, k1, b
        self.avg = sum(map(len, docs)) / max(len(docs), 1)
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(d) for d in docs]

    def score(self, query: list[str], i: int) -> float:
        tf, dl, s = self.tf[i], len(self.docs[i]), 0.0
        for w in query:
            f = tf.get(w, 0)
            if f:
                s += (
                    self.idf.get(w, 0.0)
                    * f
                    * (self.k1 + 1)
                    / (f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1)))
                )
        return s


def tfidf_cosine(a: str, corpus: list[str]) -> list[float]:
    """Cosine similarity of ``a`` to each corpus document, with IDF from the corpus plus ``a``."""
    docs = [tokens(x) for x in [a, *corpus]]
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)

    def vec(d: list[str]) -> dict[str, float]:
        tf = Counter(d)
        return {w: c * math.log((1 + n) / (1 + df[w])) + c for w, c in tf.items()}

    va = vec(docs[0])
    na = math.sqrt(sum(v * v for v in va.values())) or 1.0
    out = []
    for d in docs[1:]:
        vb = vec(d)
        nb = math.sqrt(sum(v * v for v in vb.values())) or 1.0
        out.append(sum(va[w] * vb.get(w, 0.0) for w in va) / (na * nb))
    return out
