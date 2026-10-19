"""Split policy: choose a validation strategy from the data structure and
verify that a split does not leak.

Decision order (first match wins):
  1. time column present   -> time-aware  (train strictly before test)
  2. group column present  -> group-aware (no entity in both train and test)
  3. otherwise             -> stratified   (class ratios preserved)
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    TimeSeriesSplit,
    train_test_split,
)

MIN_CLASS_COUNT = 10  # per class, otherwise stratified CV is not meaningful
SEED = 42


class SplitPolicyError(ValueError):
    """Raised when the data cannot support a valid evaluation split."""


class LeakageError(AssertionError):
    """Raised when a train/test split violates the policy."""


def choose_strategy(y, groups=None, time=None) -> str:
    if time is not None:
        return "time"
    if groups is not None:
        return "group"
    return "stratified"


def _check_class_support(y, n_splits: int) -> None:
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise SplitPolicyError("target has a single class; nothing to evaluate")
    smallest = counts.min()
    if smallest < MIN_CLASS_COUNT or smallest < n_splits:
        raise SplitPolicyError(
            f"smallest class has {smallest} rows; need >= "
            f"max({MIN_CLASS_COUNT}, n_splits={n_splits})"
        )


def holdout_split(y, groups=None, time=None, test_size=0.2, seed=SEED):
    """Single locked hold-out split. Returns (train_idx, test_idx)."""
    strategy = choose_strategy(y, groups, time)
    n = len(y)
    idx = np.arange(n)
    if strategy == "time":
        order = np.argsort(np.asarray(time), kind="stable")
        cut = int(round(n * (1 - test_size)))
        train_idx, test_idx = order[:cut], order[cut:]
    elif strategy == "group":
        sgkf = StratifiedGroupKFold(
            n_splits=round(1 / test_size), shuffle=True, random_state=seed
        )
        train_idx, test_idx = next(sgkf.split(idx, y, groups))
    else:
        _check_class_support(y, n_splits=2)
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, stratify=np.asarray(y), random_state=seed
        )
    assert_no_leakage(train_idx, test_idx, groups=groups, time=time)
    return np.sort(train_idx), np.sort(test_idx)


def cv_splits(y, groups=None, time=None, n_splits=5, n_repeats=3, seed=SEED):
    """Yield (train_idx, val_idx) folds for the data structure, leakage-checked."""
    strategy = choose_strategy(y, groups, time)
    idx = np.arange(len(y))
    if strategy == "time":
        order = np.argsort(np.asarray(time), kind="stable")
        for tr, va in TimeSeriesSplit(n_splits=n_splits).split(order):
            tr, va = order[tr], order[va]
            assert_no_leakage(tr, va, groups=groups, time=time)
            yield tr, va
    elif strategy == "group":
        for tr, va in GroupKFold(n_splits=n_splits).split(idx, y, groups):
            assert_no_leakage(tr, va, groups=groups, time=time)
            yield tr, va
    else:
        _check_class_support(y, n_splits)
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        for tr, va in cv.split(idx, np.asarray(y)):
            assert_no_leakage(tr, va)
            yield tr, va


def assert_no_leakage(train_idx, test_idx, groups=None, time=None) -> None:
    train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise LeakageError("empty train or test set")
    overlap = np.intersect1d(train_idx, test_idx)
    if overlap.size:
        raise LeakageError(f"{overlap.size} rows appear in both train and test")
    if groups is not None:
        g = np.asarray(groups)
        shared = np.intersect1d(g[train_idx], g[test_idx])
        if shared.size:
            raise LeakageError(f"{shared.size} groups appear in both train and test")
    if time is not None:
        t = np.asarray(time)
        if t[train_idx].max() > t[test_idx].min():
            raise LeakageError("training data extends past the start of test data")
