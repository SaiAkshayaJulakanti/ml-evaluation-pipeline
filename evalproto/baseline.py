"""Run baselines under the split policy and write results/baseline.json.

Usage:  python -m evalproto.baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, load_dataset
from .metrics import PRIMARY_METRIC, compute_metrics
from .splits import SEED, choose_strategy, cv_splits, holdout_split

RESULTS = Path(__file__).resolve().parent.parent / "results" / "baseline.json"
GAP = 24  # hours dropped between train and test: covers the 24h lag and event persistence
N_SPLITS = 5


class DeviationRule(BaseEstimator, ClassifierMixin):
    """Untrained rule: flag an hour whose consumption deviates more than 50%
    from the same hour last week. score = s / (s + log 1.5), so 0.5 is exactly
    the rule's threshold and the score stays monotone in the deviation."""

    def __init__(self, column: int = 0, threshold: float = float(np.log(1.5))):
        self.column = column
        self.threshold = threshold

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = np.abs(np.asarray(X)[:, self.column])
        p = s / (s + self.threshold)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def models():
    return {
        # Floors: any real model must beat these on the primary metric.
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "dummy_prior": DummyClassifier(strategy="prior"),
        # Domain rule, no training: the bar a learned model has to clear.
        "rule_dev_vs_last_week": DeviationRule(column=FEATURES.index("abs_log_ratio_168")),
        # First learned baseline: scaled, class-balanced logistic regression.
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        ),
    }


def _score(model, X):
    return model.predict_proba(X)[:, 1]


def summarise(fold_metrics: list[dict]) -> dict:
    return {
        k: {
            "mean": float(np.mean([m[k] for m in fold_metrics])),
            "std": float(np.std([m[k] for m in fold_metrics], ddof=1)),
        }
        for k in fold_metrics[0]
    }


def run() -> dict:
    ds = load_dataset()
    X, y = ds.X.to_numpy(), ds.y.to_numpy()
    strategy = choose_strategy(y, ds.groups, ds.time)

    train_idx, test_idx = holdout_split(y, ds.groups, ds.time, gap=GAP)
    X_tr, y_tr, t_tr = X[train_idx], y[train_idx], ds.time[train_idx]

    out = {
        "dataset": ds.name,
        "rows": int(len(y)),
        "buildings": int(len(np.unique(ds.groups))),
        "positive_rate": float(y.mean()),
        "strategy": strategy,
        "gap_hours": GAP,
        "primary_metric": PRIMARY_METRIC,
        "holdout": {
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_hours": [int(t_tr.min()), int(t_tr.max())],
            "test_hours": [int(ds.time[test_idx].min()), int(ds.time[test_idx].max())],
            "train_positive_rate": float(y[train_idx].mean()),
            "test_positive_rate": float(y[test_idx].mean()),
            "test_positives": int(y[test_idx].sum()),
        },
        "models": {},
    }

    for name, model in models().items():
        # 1) expanding-window CV on the training portion only; test stays untouched
        folds = []
        for tr, va in cv_splits(y_tr, time=t_tr, n_splits=N_SPLITS, gap=GAP):
            m = clone(model).fit(X_tr[tr], y_tr[tr])
            folds.append(compute_metrics(y_tr[va], _score(m, X_tr[va])))
        # 2) one final hold-out evaluation, fit on the whole training portion
        final = clone(model).fit(X_tr, y_tr)
        holdout = compute_metrics(y[test_idx], _score(final, X[test_idx]))
        out["models"][name] = {"cv": summarise(folds), "holdout": holdout}

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    return out


def _print(out: dict) -> None:
    print(f"{out['dataset']}: {out['rows']} rows, {out['buildings']} buildings, "
          f"positive rate {out['positive_rate']:.3f}")
    print(f"strategy={out['strategy']} gap={out['gap_hours']}h primary={out['primary_metric']}")
    h = out["holdout"]
    print(f"holdout: train={h['train_rows']} (hours {h['train_hours'][0]}-{h['train_hours'][1]}) "
          f"test={h['test_rows']} (hours {h['test_hours'][0]}-{h['test_hours'][1]}, "
          f"{h['test_positives']} positives)")
    print(f"pos-rate train={h['train_positive_rate']:.3f} test={h['test_positive_rate']:.3f}\n")
    cols = ["pr_auc", "roc_auc", "recall", "precision", "balanced_accuracy", "accuracy"]
    print(f"{'model':<24}{'':<9}" + "".join(f"{c:>19}" for c in cols))
    for name, r in out["models"].items():
        cv = "".join(f"{r['cv'][c]['mean']:>12.3f}±{r['cv'][c]['std']:.3f}" for c in cols)
        ho = "".join(f"{r['holdout'][c]:>19.3f}" for c in cols)
        print(f"{name:<24}{'CV':<9}{cv}")
        print(f"{'':<24}{'holdout':<9}{ho}")


if __name__ == "__main__":
    _print(run())
