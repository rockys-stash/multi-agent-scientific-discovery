"""The run's written report, and a readability score for it.

The report is generated from the run's artefacts only, so its wording does not depend on how
well the process went. That is the point of E4: a report reads the same whether or not its
citations and conclusions survive checking.
"""

from __future__ import annotations

import re

from discoverylab.models import RunState

_VOWELS = re.compile(r"[aeiouy]+")


def render_report(state: RunState) -> str:
    q = state.question
    lines = [f"# {q.text}", "", "## Prior work", ""]
    for e in state.evidence:
        c = e.citation
        lines.append(f"- {e.claim} “{e.quote}” ({c.first_author or 'unknown'}, {c.year or 'n.d.'}; {c.identifier})")
    if state.gaps:
        lines += ["", "## Gaps", ""] + [f"- {g.description}" for g in state.gaps]
    for h in state.hypotheses:
        lines += ["", f"## Hypothesis {h.id}", "", h.statement]
        for d in (d for d in state.designs if d.hypothesis_id == h.id):
            lines.append(
                f"We compared {', '.join(d.treatments)} against {d.control} on {', '.join(d.datasets)}, "
                f"using {d.folds}-fold cross-validation over {len(d.seeds)} seeds and a {d.test.replace('_', ' ')} test "
                f"on {d.metrics[0] if d.metrics else 'the primary metric'}."
            )
        for k in (k for k in state.conclusions if k.hypothesis_id == h.id):
            lines += ["", k.statement]
            if k.limitations:
                lines += ["", "Limitations: " + " ".join(k.limitations)]
    return "\n".join(lines).strip() + "\n"


def _syllables(word: str) -> int:
    w = word.lower().strip(".,;:!?()\"'“”")
    if not w:
        return 0
    n = len(_VOWELS.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def flesch_reading_ease(text: str) -> float | None:
    """Flesch reading ease (higher reads more easily). A formula, not a judgement of quality."""
    prose = " ".join(line.lstrip("#- ") for line in text.splitlines() if line.strip())
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", prose) if s.strip()]
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", prose)]
    if not sentences or not words:
        return None
    syl = sum(_syllables(w) for w in words)
    return round(206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syl / len(words)), 2)
