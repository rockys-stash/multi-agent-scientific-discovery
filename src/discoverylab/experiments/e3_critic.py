"""E3: does the critic catch process faults planted on purpose?

Faults are injected one at a time into copies of completed runs' states, at the stage where they
would arise, and the rule critic reviews that stage. A fault counts as caught when a critique of an
expected type targets the faulted artefact. Semantic faults are included deliberately: the rule
critic is not designed to see them, and the experiment reports that rather than hiding it.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from discoverylab.critic import rule_critique
from discoverylab.experiments.common import Summary, e1_run_dirs, load_run, result_dir, table
from discoverylab.models import RunState, Stage
from discoverylab.reasoners.base import Reasoner
from discoverylab.run import make_reasoner, unavailable

DESCRIPTION = "Fault injection: which planted process faults the critic catches, by stage."

Injector = Callable[[RunState, str, random.Random], bool]  # mutates the state; False if not applicable


@dataclass(frozen=True)
class Fault:
    name: str
    stage: Stage
    targets: Callable[[RunState], list[str]]
    inject: Injector
    expected: frozenset[str]
    kind: str  # "structural" (rule critic's scope) or "semantic" (needs a model critic or a person)


def _ev(s: RunState, i: str) -> Any:
    return next(e for e in s.evidence if e.id == i)


def _swap_word(text: str, rng: random.Random) -> str:
    words = text.split()
    long = [k for k, w in enumerate(words) if len(w.strip(".,;:()")) > 4]
    if not long:
        return text + " notably"
    k = rng.choice(long)
    words[k] = "unrelated" if words[k].lower() != "unrelated" else "different"
    return " ".join(words)


def _negate(claim: str) -> str:
    return f"It is not the case that {claim[0].lower()}{claim[1:]}" if claim else "Nothing."


def _set(obj: Any, **kw: Any) -> bool:
    for k, v in kw.items():
        setattr(obj, k, v)
    return True


def _design(s: RunState, i: str) -> Any:
    return next(d for d in s.designs if d.id == i)


def _hyp(s: RunState, i: str) -> Any:
    return next(h for h in s.hypotheses if h.id == i)


def _flip_verdict(s: RunState, i: str, _: random.Random) -> bool:
    k = next(c for c in s.conclusions if c.id == i)
    k.verdict = "supported" if k.verdict != "supported" else "not_supported"
    return True


def _shift_mean(s: RunState, i: str, _: random.Random) -> bool:
    r = next(r for r in s.results if r.id == i)
    if not r.conditions:
        return False
    c = r.conditions[0]
    c.mean += max(0.05, abs(c.mean) * 0.1)
    return True


def _break_interval(s: RunState, i: str, _: random.Random) -> bool:
    r = next(r for r in s.results if r.id == i)
    if not r.conditions:
        return False
    c = r.conditions[0]
    c.lo, c.hi = c.mean + 0.01, c.mean + 0.02
    return True


def _rotate_metrics(s: RunState, i: str, _: random.Random) -> bool:
    d = _design(s, i)
    if len(d.metrics) < 2:
        return False
    d.metrics = [*d.metrics[1:], d.metrics[0]]
    return True


def _control_as_treatment(s: RunState, i: str, _: random.Random) -> bool:
    d = _design(s, i)
    if not d.treatments:
        return False
    d.control = d.treatments[0]
    return True


def _flip_direction(s: RunState, i: str, _: random.Random) -> bool:
    h = _hyp(s, i)
    h.expected_direction = {"increase": "decrease", "decrease": "increase", "no_difference": "increase"}[
        h.expected_direction
    ]
    return True


FAULTS: list[Fault] = [
    Fault("citation_to_unretrieved_source", Stage.EVIDENCE, lambda s: [e.id for e in s.evidence],
          lambda s, i, r: _set(_ev(s, i).citation, identifier=f"doi:10.5555/injected.{r.randrange(10**6)}"),
          frozenset({"citation_not_retrieved"}), "structural"),
    Fault("altered_quote", Stage.EVIDENCE, lambda s: [e.id for e in s.evidence],
          lambda s, i, r: _set(_ev(s, i), quote=_swap_word(_ev(s, i).quote, r)), frozenset({"quote_not_in_source"}), "structural"),
    Fault("missing_quote", Stage.EVIDENCE, lambda s: [e.id for e in s.evidence],
          lambda s, i, r: _set(_ev(s, i), quote=""), frozenset({"missing_quote"}), "structural"),
    Fault("ungrounded_hypothesis", Stage.HYPOTHESES, lambda s: [h.id for h in s.hypotheses],
          lambda s, i, r: _set(_hyp(s, i), gap_ids=[]), frozenset({"ungrounded_hypothesis"}), "structural"),
    Fault("dangling_reference", Stage.HYPOTHESES, lambda s: [h.id for h in s.hypotheses],
          lambda s, i, r: _set(_hyp(s, i), gap_ids=[*_hyp(s, i).gap_ids, "G999"]), frozenset({"unknown_reference"}), "structural"),
    Fault("control_also_treatment", Stage.DESIGN, lambda s: [d.id for d in s.designs], _control_as_treatment,
          frozenset({"no_control"}), "structural"),
    Fault("single_measurement", Stage.DESIGN, lambda s: [d.id for d in s.designs],
          lambda s, i, r: _set(_design(s, i), seeds=[0], split="stratified_holdout"), frozenset({"insufficient_repeats"}), "structural"),
    Fault("primary_metric_mismatch", Stage.DESIGN, lambda s: [d.id for d in s.designs], _rotate_metrics,
          frozenset({"metric_mismatch"}), "structural"),
    Fault("unknown_dataset", Stage.DESIGN, lambda s: [d.id for d in s.designs],
          lambda s, i, r: _set(_design(s, i), datasets=["no_such_dataset"]), frozenset({"unrunnable_design"}), "structural"),
    Fault("misreported_mean", Stage.ANALYSIS, lambda s: [x.id for x in s.results], _shift_mean,
          frozenset({"arithmetic_error"}), "structural"),
    Fault("interval_excludes_mean", Stage.ANALYSIS, lambda s: [x.id for x in s.results], _break_interval,
          frozenset({"arithmetic_error"}), "structural"),
    Fault("verdict_not_supported_by_statistics", Stage.CONCLUSION, lambda s: [k.id for k in s.conclusions], _flip_verdict,
          frozenset({"overreach", "verdict_mismatch"}), "structural"),
    # Semantic: the artefacts stay structurally consistent, only their meaning is wrong.
    Fault("claim_contradicts_quote", Stage.EVIDENCE, lambda s: [e.id for e in s.evidence],
          lambda s, i, r: _set(_ev(s, i), claim=_negate(_ev(s, i).claim)), frozenset({"claim_not_supported"}), "semantic"),
    Fault("stance_flipped", Stage.EVIDENCE, lambda s: [e.id for e in s.evidence if e.stance != "context"],
          lambda s, i, r: _set(_ev(s, i), stance="contradicts" if _ev(s, i).stance == "supports" else "supports"),
          frozenset({"claim_not_supported"}), "semantic"),
    Fault("hypothesis_direction_flipped", Stage.HYPOTHESES, lambda s: [h.id for h in s.hypotheses], _flip_direction,
          frozenset({"hypothesis_contradicts_evidence"}), "semantic"),
]  # fmt: skip


MODEL_STAGES = (Stage.EVIDENCE, Stage.GAPS, Stage.HYPOTHESES, Stage.DESIGN, Stage.CONCLUSION)


def inject_and_review(
    state: RunState, fault: Fault, target: str, seed: int, model_critic: Reasoner | None = None
) -> dict[str, Any]:
    s = state.model_copy(deep=True)
    if not fault.inject(s, target, random.Random(seed)):
        return {"applicable": False}
    found = rule_critique(s, fault.stage, 1, 1)
    on_target = [c for c in found if c.target_id == target]
    out: dict[str, Any] = {
        "applicable": True,
        "caught": any(c.issue in fault.expected for c in on_target),
        "flagged_target": bool(on_target),
        "issues": sorted({c.issue for c in found}),
    }
    if model_critic is not None and fault.stage in MODEL_STAGES:
        # The model names issues in its own words, so only "flagged the faulted artefact" is scored.
        mc = model_critic.critique(s, fault.stage, 1, 1)
        out["model_flagged_target"] = any(c.target_id == target for c in mc)
        out["model_critiques"] = [{"target_id": c.target_id, "severity": c.severity, "issue": c.issue} for c in mc]
    return out


def clean_false_alarms(state: RunState) -> dict[str, list[str]]:
    """Critiques the rule critic raises on the unmodified final state, per stage."""
    out: dict[str, list[str]] = {}
    for st in (Stage.EVIDENCE, Stage.GAPS, Stage.HYPOTHESES, Stage.DESIGN, Stage.ANALYSIS, Stage.CONCLUSION):
        out[st.value] = [f"{c.target_id}:{c.issue}" for c in rule_critique(state, st, 1, 1)]
    return out


def model_clean_critiques(state: RunState, critic: Reasoner) -> dict[str, list[str]]:
    """Critiques the model critic raises on the unmodified final state, per stage it reviews."""
    return {st.value: [f"{c.target_id}:{c.severity}" for c in critic.critique(state, st, 1, 1)] for st in MODEL_STAGES}


def evaluate(
    states: list[RunState],
    seed: int,
    max_targets: int,
    model_critic: Reasoner | None = None,
    model_trials_per_fault: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    trials: list[dict[str, Any]] = []
    clean: list[dict[str, Any]] = []
    for st in states:
        clean.append({"run_id": st.run_id, "false_alarms": clean_false_alarms(st)})
        for f in FAULTS:
            targets = f.targets(st)
            for t in rng.sample(targets, min(max_targets, len(targets))):
                sd = rng.randrange(2**31)
                res = inject_and_review(st, f, t, sd)
                if res["applicable"]:
                    trials.append({"run_id": st.run_id, "reasoner": st.reasoner, "fault": f.name, "kind": f.kind,
                                   "stage": f.stage.value, "target": t, "seed": sd, **res})  # fmt: skip
    if model_critic is not None:
        # A separate stream, so adding the model critic leaves the rule critic's trials unchanged.
        mrng = random.Random(seed + 1)
        by_state = {st.run_id: st for st in states}
        for f in FAULTS:
            mine = [t for t in trials if t["fault"] == f.name]
            for tr in mrng.sample(mine, min(model_trials_per_fault, len(mine))):
                # same injection seed as the rule trial, so both critics review the identical faulted state
                res = inject_and_review(by_state[tr["run_id"]], f, tr["target"], tr["seed"], model_critic)
                if "model_flagged_target" in res:
                    tr["model_flagged_target"] = res["model_flagged_target"]
                    tr["model_critiques"] = res["model_critiques"]
        for c in clean:
            c["model_critiques"] = model_clean_critiques(by_state[c["run_id"]], model_critic)
    return trials, clean


def summarise(s: Summary, trials: list[dict[str, Any]], clean: list[dict[str, Any]], n_runs: int) -> None:
    s.title = "E3 Critic"
    s.question = "Which injected process faults does the rule critic catch, and which can it not see?"
    rows = []
    for f in FAULTS:
        mine = [t for t in trials if t["fault"] == f.name]
        if not mine:
            rows.append(
                {
                    "fault": f.name.replace("_", " "),
                    "kind": f.kind,
                    "stage": f.stage.value,
                    "trials": 0,
                    "caught": None,
                    "flagged": None,
                }
            )
            continue
        rows.append({"fault": f.name.replace("_", " "), "kind": f.kind, "stage": f.stage.value, "trials": len(mine),
                     "caught": sum(t["caught"] for t in mine) / len(mine),
                     "flagged": sum(t["flagged_target"] for t in mine) / len(mine)})  # fmt: skip
    s.tables.append(table("faults", "Detection by fault (share of trials)", [("fault", "Fault", "l"), ("kind", "Kind", "l"),
        ("stage", "Stage", "l"), ("trials", "Trials", "r"), ("caught", "Caught as the right issue", "r"),
        ("flagged", "Target flagged at all", "r")], rows))  # fmt: skip
    fa = sum(len(v) for c in clean for v in c["false_alarms"].values())
    s.findings.append(f"{len(trials)} injections into {n_runs} completed runs.")
    for kind in ("structural", "semantic"):
        mine = [t for t in trials if t["kind"] == kind]
        if mine:
            s.findings.append(f"{kind.capitalize()} faults caught: {sum(t['caught'] for t in mine)}/{len(mine)}.")
    s.findings.append(f"Critiques on the unmodified final states: {fa}.")
    judged = [t for t in trials if "model_flagged_target" in t]
    if judged:
        mrows = []
        for f in FAULTS:
            mine = [t for t in judged if t["fault"] == f.name]
            if mine:
                mrows.append({"fault": f.name.replace("_", " "), "kind": f.kind, "trials": len(mine),
                              "rule": sum(t["caught"] for t in mine) / len(mine),
                              "model": sum(t["model_flagged_target"] for t in mine) / len(mine)})  # fmt: skip
        s.tables.append(table("model_critic", "Model critic on a sample of the same trials (faulted artefact flagged)",
            [("fault", "Fault", "l"), ("kind", "Kind", "l"), ("trials", "Trials", "r"),
             ("rule", "Rule critic caught", "r"), ("model", "Model critic flagged", "r")], mrows))  # fmt: skip
        for kind in ("structural", "semantic"):
            mine = [t for t in judged if t["kind"] == kind]
            if mine:
                s.findings.append(f"Model critic flagged the faulted artefact in {sum(t['model_flagged_target'] for t in mine)}/{len(mine)} "
                                  f"sampled {kind} trials (rule critic on the same trials: {sum(t['caught'] for t in mine)}/{len(mine)}).")  # fmt: skip
        mc = [v for c in clean if "model_critiques" in c for v in c["model_critiques"].values()]
        n_clean = sum(1 for c in clean if "model_critiques" in c)
        s.findings.append(f"Model critic critiques on the {n_clean} unmodified final states: {sum(len(v) for v in mc)} "
                          f"({sum(1 for v in mc for x in v if x.endswith(':blocking'))} blocking).")  # fmt: skip
        s.notes.append("The model critic names issues in its own words, so it is scored on whether it flagged the faulted artefact, not on the issue label; "
                       "a critique of an unmodified state is not necessarily wrong (the model may find a real problem).")  # fmt: skip
    else:
        s.notes.append(
            "Semantic faults keep every artefact structurally consistent; catching them needs a model critic or a person (not run here)."
        )


def run(cfg: dict[str, Any], runs: Path, results: Path | None = None, model_critic: Reasoner | None = None) -> Path:
    states = [load_run(d)[0] for d in e1_run_dirs(runs, results)]
    name = cfg.get("model_critic")
    why = unavailable(name) if name else None
    if model_critic is None and name and why is None:
        model_critic = make_reasoner(name)
    with result_dir("e3_critic", cfg, DESCRIPTION, results) as (rd, s):
        trials, clean = evaluate(states, int(cfg.get("seed", 0)), int(cfg.get("max_targets_per_fault", 5)),
                                 model_critic, int(cfg.get("model_trials_per_fault", 6)))  # fmt: skip
        if why:
            s.notes.append(f"Model critic not run: {why}.")
        elif model_critic is not None:
            s.notes.append(f"Model critic: {model_critic.name} ({getattr(model_critic, 'model', 'unknown')}).")
        rd.write_json("trials.json", trials)
        rd.write_json("clean.json", clean)
        summarise(s, trials, clean, len(states))
    return rd.path
