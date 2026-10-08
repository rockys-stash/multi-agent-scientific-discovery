"""The experiment toolbox: what the Experiment agent may compose, and how designs are executed.

Agents choose datasets, conditions, metrics and a test by name; this module runs them. No agent
ever writes a number into a result: every value comes from code below.

A condition is ``model`` plus optional ``+modifier`` parts, e.g. ``random_forest+isotonic`` or
``knn+standardize``. Modifiers are applied inside each training fold only, so there is no
leakage from the evaluation data.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from discoverylab.models import AnalysisResult, ConditionResult, ExperimentDesign

Array = np.ndarray[Any, Any]


# ---------- datasets (bundled with scikit-learn; no download) ----------
@dataclass(frozen=True)
class DatasetInfo:
    name: str
    description: str
    loader: Callable[[], tuple[Array, Array]]


def _bc() -> tuple[Array, Array]:
    d = load_breast_cancer()
    return d.data, d.target


def _wine_binary() -> tuple[Array, Array]:
    d = load_wine()
    return d.data, (d.target == 0).astype(int)


def _digits_binary() -> tuple[Array, Array]:
    d = load_digits()
    return d.data, (d.target >= 5).astype(int)


def _imbalanced(loader: Callable[[], tuple[Array, Array]], ratio: float) -> Callable[[], tuple[Array, Array]]:
    """Keep every majority example and a fixed, seeded subsample of the minority class."""

    def load() -> tuple[Array, Array]:
        X, y = loader()
        minority = int(np.argmin(np.bincount(y)))
        idx_min = np.flatnonzero(y == minority)
        idx_maj = np.flatnonzero(y != minority)
        k = max(10, round(ratio * len(idx_maj)))
        keep = np.random.default_rng(12345).choice(idx_min, size=min(k, len(idx_min)), replace=False)
        idx = np.sort(np.concatenate([idx_maj, keep]))
        return X[idx], y[idx]

    return load


DATASETS: dict[str, DatasetInfo] = {
    "breast_cancer": DatasetInfo("breast_cancer", "Wisconsin diagnostic breast cancer (569 x 30, binary)", _bc),
    "wine_binary": DatasetInfo("wine_binary", "Wine recognition, cultivar 0 vs rest (178 x 13)", _wine_binary),
    "digits_binary": DatasetInfo("digits_binary", "Optical digits, 0-4 vs 5-9 (1797 x 64)", _digits_binary),
    "breast_cancer_imb10": DatasetInfo(
        "breast_cancer_imb10",
        "Breast cancer with the minority class subsampled to 10% of the majority",
        _imbalanced(_bc, 0.10),
    ),
    "digits_binary_imb10": DatasetInfo(
        "digits_binary_imb10",
        "Digits 0-4 vs 5-9 with the minority class subsampled to 10% of the majority",
        _imbalanced(_digits_binary, 0.10),
    ),
}


# ---------- conditions ----------
MODELS: dict[str, Callable[[int], BaseEstimator]] = {
    "random_forest": lambda s: RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=1),
    "gradient_boosting": lambda s: GradientBoostingClassifier(random_state=s),
    "logistic_regression": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "knn": lambda s: KNeighborsClassifier(n_neighbors=7),
    "naive_bayes": lambda s: GaussianNB(),
}
MODIFIERS = ("platt", "isotonic", "standardize", "class_weight", "oversample")


class _Oversampled(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Random oversampling of the minority class inside ``fit`` only."""

    def __init__(self, estimator: BaseEstimator, seed: int = 0) -> None:
        self.estimator, self.seed = estimator, seed

    def fit(self, X: Array, y: Array) -> _Oversampled:
        rng = np.random.default_rng(self.seed)
        counts = np.bincount(y)
        top = int(np.max(counts))
        idx: list[Array] = [np.arange(len(y))]
        for c, n in enumerate(counts):
            if 0 < n < top:
                idx.append(np.asarray(rng.choice(np.flatnonzero(y == c), size=top - int(n), replace=True)))
        i = np.concatenate(idx)
        self.estimator_ = clone(self.estimator).fit(X[i], y[i])
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: Array) -> Array:
        return self.estimator_.predict(X)

    def predict_proba(self, X: Array) -> Array:
        return self.estimator_.predict_proba(X)


def parse_condition(name: str) -> tuple[str, list[str]]:
    model, *mods = name.split("+")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
    for m in mods:
        if m not in MODIFIERS:
            raise ValueError(f"unknown modifier {m!r}; choose from {list(MODIFIERS)}")
    return model, mods


def build(name: str, seed: int) -> BaseEstimator:
    model, mods = parse_condition(name)
    est = MODELS[model](seed)
    if "class_weight" in mods:
        if not hasattr(est, "class_weight") and not hasattr(est, "steps"):
            raise ValueError(f"{model} does not support class_weight")
        if hasattr(est, "steps"):
            est.steps[-1][1].set_params(class_weight="balanced")
        else:
            est.set_params(class_weight="balanced")
    if "standardize" in mods:
        est = make_pipeline(StandardScaler(), est)
    if "oversample" in mods:
        est = _Oversampled(est, seed)
    for m in mods:
        if m in ("platt", "isotonic"):
            est = CalibratedClassifierCV(est, method="sigmoid" if m == "platt" else "isotonic", cv=3)
    return est


# ---------- metrics ----------
def expected_calibration_error(y: Array, p: Array, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def _minority_recall(y: Array, pred: Array, p: Array) -> float:
    minority = int(np.argmin(np.bincount(y)))
    return float(recall_score(y, pred, pos_label=minority))


METRICS: dict[str, tuple[Callable[[Array, Array, Array], float], bool]] = {
    # name -> (fn(y, pred, p_positive), higher_is_better)
    "accuracy": (lambda y, pred, p: float(accuracy_score(y, pred)), True),
    "balanced_accuracy": (lambda y, pred, p: float(balanced_accuracy_score(y, pred)), True),
    "f1": (lambda y, pred, p: float(f1_score(y, pred)), True),
    "roc_auc": (lambda y, pred, p: float(roc_auc_score(y, p)), True),
    "brier": (lambda y, pred, p: float(brier_score_loss(y, p)), False),
    "log_loss": (lambda y, pred, p: float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])), False),
    "ece": (lambda y, pred, p: expected_calibration_error(y, p), False),
    "minority_recall": (_minority_recall, True),
}
TESTS = ("paired_wilcoxon", "paired_t")


# ---------- execution ----------
def _bootstrap_ci(x: Array, seed: int = 0, n: int = 2000) -> tuple[float, float]:
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _splits(design: ExperimentDesign, y: Array, seed: int) -> list[tuple[Array, Array]]:
    if design.split == "stratified_kfold":
        return list(StratifiedKFold(design.folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    return [(tr, te)]


def validate(design: ExperimentDesign) -> list[str]:
    """Problems that make a design unrunnable (not judgements of its validity)."""
    problems = []
    for d in design.datasets:
        if d not in DATASETS:
            problems.append(f"unknown dataset {d!r}")
    for c in [design.control, *design.treatments]:
        try:
            parse_condition(c)
        except ValueError as e:
            problems.append(str(e))
    for m in design.metrics:
        if m not in METRICS:
            problems.append(f"unknown metric {m!r}")
    if design.test not in TESTS:
        problems.append(f"unknown test {design.test!r}")
    if not design.metrics:
        problems.append("no metric")
    return problems


def run_design(design: ExperimentDesign, result_id: str) -> AnalysisResult:
    problems = validate(design)
    if problems:
        raise ValueError("; ".join(problems))
    t0 = time.perf_counter()
    conditions = [design.control, *design.treatments]
    scores: dict[tuple[str, str, str], list[float]] = {}
    for ds in design.datasets:
        X, y = DATASETS[ds].loader()
        for seed in design.seeds:
            for tr, te in _splits(design, y, seed):
                for cond in conditions:
                    est = build(cond, seed).fit(X[tr], y[tr])
                    p = est.predict_proba(X[te])[:, 1]
                    pred = est.predict(X[te])
                    for m in design.metrics:
                        scores.setdefault((ds, cond, m), []).append(METRICS[m][0](y[te], pred, p))
    results = []
    for (ds, cond, m), vals in scores.items():
        arr = np.asarray(vals)
        lo, hi = _bootstrap_ci(arr)
        results.append(ConditionResult(condition=cond, dataset=ds, metric=m, values=[float(v) for v in arr],
                                       mean=float(arr.mean()), lo=lo, hi=hi))  # fmt: skip
    comparisons: list[dict[str, float | str | int | bool | None]] = []
    for ds in design.datasets:
        for m in design.metrics:
            base = np.asarray(scores[(ds, design.control, m)])
            for t in design.treatments:
                arr = np.asarray(scores[(ds, t, m)])
                diff = arr - base
                if np.allclose(diff, 0):
                    pval = 1.0
                elif design.test == "paired_wilcoxon":
                    pval = float(stats.wilcoxon(arr, base).pvalue)
                else:
                    pval = float(stats.ttest_rel(arr, base).pvalue)
                sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
                lo, hi = _bootstrap_ci(diff)
                comparisons.append({
                    "dataset": ds, "metric": m, "treatment": t, "control": design.control,
                    "mean_diff": float(diff.mean()), "lo": lo, "hi": hi, "p": pval,
                    "effect_size_dz": float(diff.mean() / sd) if sd > 0 else None,
                    "n": len(diff), "higher_is_better": METRICS[m][1],
                })  # fmt: skip
    _holm(comparisons, [design.metrics[0]])
    return AnalysisResult(
        id=result_id,
        design_id=design.id,
        conditions=results,
        comparisons=comparisons,
        runtime_seconds=round(time.perf_counter() - t0, 3),
    )


def _holm(comps: list[dict[str, float | str | int | bool | None]], primary: list[str]) -> None:
    """Holm-adjusted p-values across the primary-metric comparisons (the family that is tested)."""
    fam = [c for c in comps if c["metric"] in primary]
    order = sorted(range(len(fam)), key=lambda i: float(fam[i]["p"] or 0.0))
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (len(fam) - rank) * float(fam[i]["p"]))  # type: ignore[arg-type]
        running = max(running, adj)
        fam[i]["p_holm"] = running
    for c in comps:
        c.setdefault("p_holm", None)
