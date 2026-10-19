"""Tests for the evaluation protocol.

Written with stdlib unittest so they run without extra installs:
    python -m unittest discover -s tests -v
They are also collected unchanged by pytest.
"""
import unittest

import numpy as np

from evalproto.baseline import run
from evalproto.data import load_dataset
from evalproto.metrics import compute_metrics
from evalproto.splits import (
    LeakageError,
    SplitPolicyError,
    assert_no_leakage,
    choose_strategy,
    cv_splits,
    holdout_split,
)


class HappyPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset()
        cls.y = cls.ds.y.to_numpy()

    def test_strategy_selection(self):
        y = np.array([0, 1] * 20)
        self.assertEqual(choose_strategy(y), "stratified")
        self.assertEqual(choose_strategy(y, groups=np.arange(40)), "group")
        self.assertEqual(
            choose_strategy(y, groups=np.arange(40), time=np.arange(40)), "time"
        )

    def test_stratified_holdout_preserves_ratio(self):
        tr, te = holdout_split(self.y)
        self.assertLess(abs(self.y[tr].mean() - self.y.mean()), 0.01)
        self.assertLess(abs(self.y[te].mean() - self.y.mean()), 0.02)
        self.assertEqual(len(np.intersect1d(tr, te)), 0)
        self.assertEqual(len(tr) + len(te), len(self.y))

    def test_cv_folds_disjoint_and_stratified(self):
        n = 0
        for tr, va in cv_splits(self.y, n_splits=5, n_repeats=2):
            n += 1
            self.assertEqual(len(np.intersect1d(tr, va)), 0)
            self.assertLess(abs(self.y[va].mean() - self.y.mean()), 0.05)
        self.assertEqual(n, 10)

    def test_split_is_deterministic(self):
        a, b = holdout_split(self.y), holdout_split(self.y)
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    def test_group_holdout_keeps_groups_apart(self):
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(50), 4)
        y = rng.integers(0, 2, size=200)
        tr, te = holdout_split(y, groups=groups)
        self.assertFalse(set(groups[tr]) & set(groups[te]))

    def test_time_holdout_is_chronological(self):
        t = np.random.default_rng(1).permutation(100)
        y = np.array([0, 1] * 50)
        tr, te = holdout_split(y, time=t)
        self.assertLess(t[tr].max(), t[te].min())

    def test_group_cv_keeps_groups_apart(self):
        rng = np.random.default_rng(2)
        groups = np.repeat(np.arange(60), 5)
        y = rng.integers(0, 2, size=300)
        for tr, va in cv_splits(y, groups=groups, n_splits=5):
            self.assertFalse(set(groups[tr]) & set(groups[va]))

    def test_baseline_beats_dummy_and_exposes_accuracy_trap(self):
        out = run()
        lr, dm = out["models"]["logreg"], out["models"]["dummy_majority"]
        self.assertGreater(lr["cv"]["pr_auc"]["mean"], 0.95)
        # majority predictor looks acceptable on accuracy but never finds a case
        self.assertGreater(dm["holdout"]["accuracy"], 0.55)
        self.assertEqual(dm["holdout"]["recall"], 0.0)
        self.assertGreater(lr["holdout"]["pr_auc"], dm["holdout"]["pr_auc"] + 0.4)


class NegativeAndBoundary(unittest.TestCase):
    def test_leakage_overlapping_rows_detected(self):
        with self.assertRaisesRegex(LeakageError, "both train and test"):
            assert_no_leakage([0, 1, 2], [2, 3])

    def test_leakage_shared_group_detected(self):
        groups = np.array([0, 0, 1, 1])
        with self.assertRaisesRegex(LeakageError, "groups"):
            assert_no_leakage([0, 2], [1, 3], groups=groups)

    def test_leakage_future_in_train_detected(self):
        t = np.array([1, 2, 3, 4])
        with self.assertRaisesRegex(LeakageError, "past the start"):
            assert_no_leakage([0, 3], [1, 2], time=t)

    def test_empty_test_set_rejected(self):
        with self.assertRaisesRegex(LeakageError, "empty"):
            assert_no_leakage([0, 1], [])

    def test_single_class_target_rejected(self):
        with self.assertRaisesRegex(SplitPolicyError, "single class"):
            holdout_split(np.zeros(100, dtype=int))

    def test_rare_class_below_limit_rejected(self):
        y = np.array([0] * 95 + [1] * 5)  # 5 positives < MIN_CLASS_COUNT
        with self.assertRaisesRegex(SplitPolicyError, "smallest class"):
            holdout_split(y)

    def test_rare_class_exactly_at_limit_accepted(self):
        y = np.array([0] * 90 + [1] * 10)
        tr, te = holdout_split(y)
        self.assertGreaterEqual(y[te].sum(), 1)

    def test_metrics_reject_single_class_and_nan(self):
        with self.assertRaisesRegex(ValueError, "single class"):
            compute_metrics([1, 1, 1], [0.2, 0.8, 0.9])
        with self.assertRaisesRegex(ValueError, "NaN"):
            compute_metrics([0, 1], [0.1, float("nan")])


if __name__ == "__main__":
    unittest.main()
