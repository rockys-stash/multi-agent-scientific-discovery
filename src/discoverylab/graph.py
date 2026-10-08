"""The research network: a run's artefacts as typed nodes and relations.

Edges that rest on a citation carry the verifier's result, so the console can show which parts
of a conclusion stand on verified sources.
"""

from __future__ import annotations

from typing import Any

from discoverylab.literature.sources import normalise_identifier
from discoverylab.models import RunState


def build_graph(state: RunState, verification: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    checks = {c["evidence_id"]: c for c in (verification or {}).get("evidence", [])}
    nodes: list[dict[str, Any]] = [{"id": "Q", "kind": "question", "label": state.question.text, "stage": "question"}]
    edges: list[dict[str, Any]] = []
    cited = {normalise_identifier(e.citation.identifier) or e.citation.identifier for e in state.evidence}

    for p in state.papers:
        nodes.append({"id": p.id, "kind": "paper", "label": p.title, "stage": "literature",
                      "meta": {"year": p.year, "authors": p.authors[:3], "source": p.source, "cited": p.id in cited}})  # fmt: skip
        edges.append({"source": p.id, "target": "Q", "rel": "retrieved for"})
    paper_ids = {p.id for p in state.papers}
    for e in state.evidence:
        chk = checks.get(e.id)
        status = None if chk is None else ("verified" if chk["correct"] else
                                            chk["citation"]["status"] if chk["citation"]["status"] != "verified" else f"quote_{chk['quote']}")  # fmt: skip
        nodes.append({"id": e.id, "kind": "evidence", "label": e.claim, "stage": "evidence",
                      "meta": {"quote": e.quote, "stance": e.stance, "verification": status}})  # fmt: skip
        target = normalise_identifier(e.citation.identifier) or e.citation.identifier
        if target not in paper_ids:  # cited something never retrieved: show it, flagged
            nodes.append({"id": target, "kind": "paper", "label": e.citation.title or target, "stage": "literature",
                          "meta": {"retrieved": False}})  # fmt: skip
            paper_ids.add(target)
        edges.append({"source": e.id, "target": target, "rel": "quotes", "verification": status})
    for g in state.gaps:
        nodes.append({"id": g.id, "kind": "gap", "label": g.description, "stage": "gaps", "meta": {"kind": g.kind}})
        edges += [{"source": g.id, "target": i, "rel": "rests on"} for i in g.evidence_ids]
    for h in state.hypotheses:
        nodes.append({"id": h.id, "kind": "hypothesis", "label": h.statement, "stage": "hypotheses",
                      "meta": {"iv": h.independent_variable, "dv": h.dependent_variable, "direction": h.expected_direction}})  # fmt: skip
        edges += [{"source": h.id, "target": i, "rel": "addresses"} for i in h.gap_ids]
        edges += [{"source": h.id, "target": i, "rel": "cites"} for i in h.evidence_ids]
    for d in state.designs:
        nodes.append({"id": d.id, "kind": "design", "label": f"{d.control} vs {', '.join(d.treatments)}", "stage": "design",
                      "meta": {"datasets": d.datasets, "metrics": d.metrics, "test": d.test}})  # fmt: skip
        edges.append({"source": d.id, "target": d.hypothesis_id, "rel": "tests"})
    for r in state.results:
        nodes.append({"id": r.id, "kind": "result", "label": f"{len(r.comparisons)} comparisons", "stage": "analysis"})
        edges.append({"source": r.id, "target": r.design_id, "rel": "produced by"})
    for k in state.conclusions:
        nodes.append(
            {
                "id": k.id,
                "kind": "conclusion",
                "label": k.statement,
                "stage": "conclusion",
                "meta": {"verdict": k.verdict},
            }
        )
        edges += [
            {"source": k.id, "target": k.result_id, "rel": "draws on"},
            {"source": k.id, "target": k.hypothesis_id, "rel": "judges"},
        ]
    known = {n["id"] for n in nodes}
    for c in state.critiques:
        if c.target_id not in known:
            continue
        nodes.append({"id": c.id, "kind": "critique", "label": c.message, "stage": "critique",
                      "meta": {"issue": c.issue, "severity": c.severity, "reviewer": c.reviewer, "resolution": c.resolution,
                               "round": c.round}})  # fmt: skip
        edges.append({"source": c.id, "target": c.target_id, "rel": "critiques"})
    known = {n["id"] for n in nodes}
    return {"nodes": nodes, "edges": [e for e in edges if e["source"] in known and e["target"] in known]}
