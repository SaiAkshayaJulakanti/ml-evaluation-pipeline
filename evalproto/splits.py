"""Split policy: choose a validation strategy from the data structure and
verify that a split does not leak.

Decision order (first match wins):
  1. time column present   -> time-aware  (train strictly before test, with a gap)
  2. group column present  -> group-aware (no entity in both train and test)
  3. otherwise             -> stratified   (class ratios preserved)

Time is an integer step index (e.g. hours). Time splits cut on UNIQUE time
values, so rows that share a timestamp (several buildings at the same hour)
always land on the same side. When both time and groups exist, time wins and
groups are allowed on both sides: the question being answered is "how well do
we predict the future of sites we already know".
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    train_test_split,
)

MIN_CLASS_COUNT = 10  # per class, otherwise evaluation is not meaningful
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


def _check_eval_fold(y_eval) -> None:
    """An evaluation fold needs enough of both classes for PR/ROC metrics."""
    classes, counts = np.unique(np.asarray(y_eval), return_counts=True)
    if len(classes) < 2 or counts.min() < MIN_CLASS_COUNT:
        raise SplitPolicyError(
            f"evaluation fold has class counts {dict(zip(classes.tolist(), counts.tolist()))}; "
            f"need >= {MIN_CLASS_COUNT} of each class"
        )


def _rows_at(time, times) -> np.ndarray:
    return np.flatnonzero(np.isin(time, times))


def _time_holdout(y, time, test_size, gap):
    time = np.asarray(time)
    ut = np.unique(time)
    cut = int(round(len(ut) * (1 - test_size)))
    train = _rows_at(time, ut[: max(cut - gap, 0)])
    test = _rows_at(time, ut[cut:])
    if len(train) == 0 or len(test) == 0:
        raise SplitPolicyError("time range too short for this test_size and gap")
    _check_eval_fold(np.asarray(y)[test])
    return train, test


def _time_folds(y, time, n_splits, gap):
    """Expanding-window folds: fold k trains on chunks 0..k, validates on k+1."""
    time = np.asarray(time)
    y = np.asarray(y)
    chunks = np.array_split(np.unique(time), n_splits + 1)
    for k in range(n_splits):
        train_times = np.concatenate(chunks[: k + 1])
        train_times = train_times[: max(len(train_times) - gap, 0)]
        tr, va = _rows_at(time, train_times), _rows_at(time, chunks[k + 1])
        if len(tr) == 0:
            raise SplitPolicyError("gap leaves no training data in the first fold")
        _check_eval_fold(y[va])
        yield tr, va


def holdout_split(y, groups=None, time=None, test_size=0.2, seed=SEED, gap=0):
    """Single locked hold-out split. Returns (train_idx, test_idx)."""
    strategy = choose_strategy(y, groups, time)
    n = len(y)
    idx = np.arange(n)
    if strategy == "time":
        train_idx, test_idx = _time_holdout(y, time, test_size, gap)
        assert_no_leakage(train_idx, test_idx, time=time, gap=gap)
    elif strategy == "group":
        sgkf = StratifiedGroupKFold(
            n_splits=round(1 / test_size), shuffle=True, random_state=seed
        )
        train_idx, test_idx = next(sgkf.split(idx, y, groups))
        assert_no_leakage(train_idx, test_idx, groups=groups)
    else:
        _check_class_support(y, n_splits=2)
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, stratify=np.asarray(y), random_state=seed
        )
        assert_no_leakage(train_idx, test_idx)
    return np.sort(train_idx), np.sort(test_idx)


def cv_splits(y, groups=None, time=None, n_splits=5, n_repeats=3, seed=SEED, gap=0):
    """Yield (train_idx, val_idx) folds for the data structure, leakage-checked."""
    strategy = choose_strategy(y, groups, time)
    idx = np.arange(len(y))
    if strategy == "time":
        for tr, va in _time_folds(y, time, n_splits, gap):
            assert_no_leakage(tr, va, time=time, gap=gap)
            yield tr, va
    elif strategy == "group":
        for tr, va in GroupKFold(n_splits=n_splits).split(idx, y, groups):
            assert_no_leakage(tr, va, groups=groups)
            yield tr, va
    else:
        _check_class_support(y, n_splits)
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        for tr, va in cv.split(idx, np.asarray(y)):
            assert_no_leakage(tr, va)
            yield tr, va


def assert_no_leakage(train_idx, test_idx, groups=None, time=None, gap=0) -> None:
    """Raise LeakageError if the split violates the policy.

    `groups`: no group may appear on both sides (omit for time splits, where
    known sites on both sides are intended).
    `time`/`gap`: every test time must exceed every train time by more than `gap` steps.
    """
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
        if t[test_idx].min() - t[train_idx].max() <= gap:
            raise LeakageError(
                f"training data comes within {gap} steps of (or after) the start of test data"
            )
