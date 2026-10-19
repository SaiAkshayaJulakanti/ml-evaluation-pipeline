"""Metric selection.

Primary metric : PR-AUC (average precision). Positive class (malignant) is the
                 minority (~37%) and the one that matters; PR-AUC is threshold
                 free and sensitive to positive-class ranking quality.
Secondary      : recall (missing a malignant case is the costly error),
                 precision, F1, balanced accuracy, ROC-AUC.
Reported only to expose the trap: plain accuracy (a majority-class predictor
already scores ~63%).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PRIMARY_METRIC = "pr_auc"


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if np.isnan(y_score).any():
        raise ValueError("y_score contains NaN")
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true has a single class; AUC metrics are undefined")
    y_pred = (y_score >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
