"""Leakage-safe pipelines and bounded, reproducible hyperparameter search.

Guarantees (each one is tested):
  * Train-only   : `run_search` receives only the training portion. The locked
                   hold-out rows are never passed in, so they cannot be scored.
  * In-fold fit  : scaling lives inside a Pipeline, so it is fit on the training
                   part of each fold only.
  * Time-aware   : folds come from `splits.cv_splits` (expanding window + gap).
  * Reproducible : candidate configs come from a seeded `ParameterSampler`;
                   models get the same seed; single process (n_jobs=1).
  * Bounded      : hard caps on trials, wall-clock time, and rows per fit.
                   Exceeding a cap is an error (trials) or a clean early stop
                   with partial results (time), never a silent overrun.

Usage:  python -m evalproto.search
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_dataset
from .metrics import PRIMARY_METRIC, compute_metrics
from .splits import SEED, cv_splits, holdout_split

RESULTS = Path(__file__).resolve().parent.parent / "results" / "search.json"

# ---- resource limits (defaults; every one can be lowered per call) ---------
MAX_TRIALS_HARD_CAP = 50  # no call may request more candidates than this
DEFAULT_N_ITER = 12  # candidates per model family
DEFAULT_TIME_BUDGET_S = 240.0  # wall-clock budget per model family
GAP = 24  # hours between train and validation (same as Day 1)
N_SPLITS = 4  # folds per candidate; fewer than Day 1 to save time


class SearchBudgetError(ValueError):
    """Raised when a search request exceeds a hard resource limit."""


# ---- pipelines and search spaces -------------------------------------------
def make_pipeline(family: str, seed: int = SEED) -> Pipeline:
    """Every pipeline = preprocessing + model, so fitting a fold fits both."""
    if family == "logreg":
        model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
        return Pipeline([("scale", StandardScaler()), ("clf", model)])
    if family == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=150, early_stopping=False, class_weight="balanced", random_state=seed
        )
        return Pipeline([("scale", StandardScaler()), ("clf", model)])
    raise ValueError(f"unknown model family: {family!r}")


SPACES = {
    "logreg": {"clf__C": loguniform(1e-3, 1e2)},
    "hgb": {
        "clf__learning_rate": loguniform(0.02, 0.3),
        "clf__max_depth": randint(2, 7),
        "clf__max_leaf_nodes": randint(8, 40),
        "clf__min_samples_leaf": randint(20, 200),
        "clf__l2_regularization": loguniform(1e-3, 10),
    },
}


@dataclass
class SearchResult:
    family: str
    seed: int
    n_requested: int
    n_completed: int
    stopped_early: bool
    elapsed_s: float
    best_params: dict | None
    best_cv_pr_auc: float | None
    trials: list = field(default_factory=list)


def _jsonable(params: dict) -> dict:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


# ---- the search -------------------------------------------------------------
def run_search(
    family: str,
    X_train,
    y_train,
    time_train,
    *,
    n_iter: int = DEFAULT_N_ITER,
    seed: int = SEED,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    max_rows: int | None = None,
    n_splits: int = N_SPLITS,
    gap: int = GAP,
    clock=time.monotonic,
) -> SearchResult:
    """Random search scored by mean CV PR-AUC on TRAINING folds only.

    `X_train/y_train/time_train` must be the training portion; the hold-out is
    never an argument, so it cannot influence the search.
    """
    if n_iter < 1:
        raise SearchBudgetError("n_iter must be >= 1")
    if n_iter > MAX_TRIALS_HARD_CAP:
        raise SearchBudgetError(
            f"n_iter={n_iter} exceeds hard cap of {MAX_TRIALS_HARD_CAP} trials"
        )
    if time_budget_s <= 0:
        raise SearchBudgetError("time_budget_s must be > 0")

    X_train, y_train, time_train = map(np.asarray, (X_train, y_train, time_train))
    folds = list(cv_splits(y_train, time=time_train, n_splits=n_splits, gap=gap))
    if max_rows is not None:  # cap rows used to FIT, keeping the most recent ones
        folds = [(tr[-max_rows:], va) for tr, va in folds]

    candidates = list(ParameterSampler(SPACES[family], n_iter=n_iter, random_state=seed))
    base = make_pipeline(family, seed)

    start = clock()
    trials, stopped_early = [], False
    for i, params in enumerate(candidates):
        if clock() - start >= time_budget_s:
            stopped_early = True
            break
        scores = []
        for tr, va in folds:
            model = clone(base).set_params(**params).fit(X_train[tr], y_train[tr])
            m = compute_metrics(y_train[va], model.predict_proba(X_train[va])[:, 1])
            scores.append(m[PRIMARY_METRIC])
        trials.append(
            {
                "trial": i,
                "params": _jsonable(params),
                "cv_pr_auc_mean": float(np.mean(scores)),
                "cv_pr_auc_std": float(np.std(scores, ddof=1)),
            }
        )

    best = max(trials, key=lambda t: t["cv_pr_auc_mean"], default=None)
    return SearchResult(
        family=family,
        seed=seed,
        n_requested=n_iter,
        n_completed=len(trials),
        stopped_early=stopped_early,
        elapsed_s=float(clock() - start),
        best_params=None if best is None else best["params"],
        best_cv_pr_auc=None if best is None else best["cv_pr_auc_mean"],
        trials=trials,
    )


# ---- end to end -------------------------------------------------------------
def run(n_iter: int = DEFAULT_N_ITER, seed: int = SEED, time_budget_s: float = DEFAULT_TIME_BUDGET_S) -> dict:
    ds = load_dataset()
    X, y = ds.X.to_numpy(), ds.y.to_numpy()
    train_idx, test_idx = holdout_split(y, ds.groups, ds.time, gap=GAP)

    out = {
        "seed": seed,
        "primary_metric": PRIMARY_METRIC,
        "limits": {
            "n_iter": n_iter,
            "hard_cap": MAX_TRIALS_HARD_CAP,
            "time_budget_s_per_family": time_budget_s,
            "n_splits": N_SPLITS,
            "gap_hours": GAP,
            "n_jobs": 1,
        },
        "families": {},
    }
    for family in SPACES:
        # 1) search sees ONLY the training portion
        res = run_search(
            family, X[train_idx], y[train_idx], ds.time[train_idx],
            n_iter=n_iter, seed=seed, time_budget_s=time_budget_s,
        )
        # 2) refit best config on the whole training portion, 3) score hold-out ONCE
        final = make_pipeline(family, seed).set_params(**res.best_params)
        final.fit(X[train_idx], y[train_idx])
        holdout = compute_metrics(y[test_idx], final.predict_proba(X[test_idx])[:, 1])
        out["families"][family] = {**res.__dict__, "holdout": holdout}

    # model choice uses CV score only, never the hold-out
    winner = max(out["families"], key=lambda f: out["families"][f]["best_cv_pr_auc"])
    out["selected_by_cv"] = winner
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    return out


def _print(out: dict) -> None:
    print(f"seed={out['seed']} limits={out['limits']}\n")
    for fam, r in out["families"].items():
        print(f"[{fam}] trials {r['n_completed']}/{r['n_requested']} in {r['elapsed_s']:.1f}s"
              f"{' (stopped early)' if r['stopped_early'] else ''}")
        print(f"  best CV PR-AUC {r['best_cv_pr_auc']:.3f}  params {r['best_params']}")
        h = r["holdout"]
        print(f"  hold-out: PR-AUC {h['pr_auc']:.3f} recall {h['recall']:.3f} "
              f"precision {h['precision']:.3f} accuracy {h['accuracy']:.3f}\n")
    print(f"selected by CV score: {out['selected_by_cv']}")


if __name__ == "__main__":
    _print(run())
