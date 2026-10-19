"""Run baselines under the split policy and write results/baseline.json.

Usage:  python -m evalproto.baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_dataset
from .metrics import PRIMARY_METRIC, compute_metrics
from .splits import SEED, choose_strategy, cv_splits, holdout_split

RESULTS = Path(__file__).resolve().parent.parent / "results" / "baseline.json"


def models():
    return {
        # Floors: any real model must beat these on the primary metric.
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "dummy_prior": DummyClassifier(strategy="prior"),
        # First real baseline: simple, scaled, regularised, interpretable.
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
        ),
    }


def _score(model, X, y):
    return model.predict_proba(X)[:, 1]


def summarise(fold_metrics: list[dict]) -> dict:
    keys = fold_metrics[0].keys()
    return {
        k: {
            "mean": float(np.mean([m[k] for m in fold_metrics])),
            "std": float(np.std([m[k] for m in fold_metrics], ddof=1)),
        }
        for k in keys
    }


def run() -> dict:
    ds = load_dataset()
    X, y = ds.X.to_numpy(), ds.y.to_numpy()
    strategy = choose_strategy(y, ds.groups, ds.time)

    train_idx, test_idx = holdout_split(y, ds.groups, ds.time)
    X_tr, y_tr = X[train_idx], y[train_idx]

    out = {
        "dataset": ds.name,
        "rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "strategy": strategy,
        "primary_metric": PRIMARY_METRIC,
        "holdout": {
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_positive_rate": float(y[train_idx].mean()),
            "test_positive_rate": float(y[test_idx].mean()),
        },
        "models": {},
    }

    for name, model in models().items():
        # 1) CV on the training portion only; test set stays untouched.
        folds = []
        for tr, va in cv_splits(y_tr, None if ds.groups is None else ds.groups[train_idx]):
            m = clone(model).fit(X_tr[tr], y_tr[tr])
            folds.append(compute_metrics(y_tr[va], _score(m, X_tr[va], y_tr[va])))
        # 2) One final hold-out evaluation, fit on the full training portion.
        final = clone(model).fit(X_tr, y_tr)
        holdout = compute_metrics(y[test_idx], _score(final, X[test_idx], y[test_idx]))
        out["models"][name] = {"cv": summarise(folds), "holdout": holdout}

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    return out


def _print(out: dict) -> None:
    print(f"{out['dataset']}: {out['rows']} rows, positive rate {out['positive_rate']:.3f}")
    print(f"strategy={out['strategy']}  primary={out['primary_metric']}")
    h = out["holdout"]
    print(f"holdout: train={h['train_rows']} test={h['test_rows']} "
          f"pos-rate train={h['train_positive_rate']:.3f} test={h['test_positive_rate']:.3f}\n")
    cols = ["pr_auc", "roc_auc", "recall", "precision", "balanced_accuracy", "accuracy"]
    print(f"{'model':<16}{'':<9}" + "".join(f"{c:>19}" for c in cols))
    for name, r in out["models"].items():
        cv = "".join(f"{r['cv'][c]['mean']:>12.3f}±{r['cv'][c]['std']:.3f}" for c in cols)
        ho = "".join(f"{r['holdout'][c]:>19.3f}" for c in cols)
        print(f"{name:<16}{'CV':<9}{cv}")
        print(f"{'':<16}{'holdout':<9}{ho}")


if __name__ == "__main__":
    _print(run())
