"""Tests for the evaluation protocol.

Written with stdlib unittest so they run without extra installs:
    python -m unittest discover -s tests -v
They are also collected unchanged by pytest.
"""
import unittest

import numpy as np

from evalproto.baseline import GAP, run
from evalproto.data import FEATURES, load_dataset, make_features, simulate_raw
from evalproto.metrics import compute_metrics
from evalproto.splits import (
    LeakageError,
    SplitPolicyError,
    assert_no_leakage,
    choose_strategy,
    cv_splits,
    holdout_split,
)


def imbalanced_y(n=1000, rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, int(n * rate), replace=False)] = 1
    return y


class TimeAwareOnEnergyData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset()
        cls.y, cls.t, cls.g = cls.ds.y.to_numpy(), cls.ds.time, cls.ds.groups

    def test_dataset_structure(self):
        self.assertEqual(choose_strategy(self.y, self.g, self.t), "time")
        self.assertEqual(len(np.unique(self.g)), 20)
        self.assertEqual(list(self.ds.X.columns), FEATURES)
        self.assertFalse(self.ds.X.isna().any().any())
        self.assertTrue(0.01 < self.y.mean() < 0.06)  # rare target

    def test_holdout_is_chronological_with_gap(self):
        tr, te = holdout_split(self.y, self.g, self.t, gap=GAP)
        self.assertGreater(self.t[te].min() - self.t[tr].max(), GAP)
        self.assertEqual(len(np.intersect1d(tr, te)), 0)
        # known buildings on both sides is intended for forecasting
        self.assertEqual(set(self.g[tr]), set(self.g[te]))

    def test_rows_sharing_a_timestamp_stay_together(self):
        tr, te = holdout_split(self.y, self.g, self.t, gap=GAP)
        self.assertFalse(set(self.t[tr]) & set(self.t[te]))

    def test_cv_folds_are_expanding_windows(self):
        folds = list(cv_splits(self.y, time=self.t, n_splits=5, gap=GAP))
        self.assertEqual(len(folds), 5)
        prev_train = 0
        for tr, va in folds:
            self.assertGreater(self.t[va].min() - self.t[tr].max(), GAP)
            self.assertGreater(len(tr), prev_train)
            prev_train = len(tr)

    def test_features_are_causal(self):
        """Changing a future reading must not change any earlier feature row."""
        raw = simulate_raw()
        base = make_features(raw)
        cut = 2000
        perturbed = raw.copy()
        perturbed.loc[perturbed["hour"] >= cut, "consumption"] *= 7.0
        alt = make_features(perturbed)
        before_a = base[base["hour"] < cut].reset_index(drop=True)
        before_b = alt[alt["hour"] < cut].reset_index(drop=True)
        self.assertGreater(len(before_a), 0)
        np.testing.assert_allclose(before_a[FEATURES].to_numpy(), before_b[FEATURES].to_numpy())

    def test_baselines_beat_floor_and_expose_accuracy_trap(self):
        out = run()
        m = out["models"]
        floor = m["dummy_majority"]
        # accuracy trap: 97% accuracy while finding no anomaly at all
        self.assertGreater(floor["holdout"]["accuracy"], 0.95)
        self.assertEqual(floor["holdout"]["recall"], 0.0)
        for name in ("rule_dev_vs_last_week", "logreg"):
            self.assertGreater(m[name]["cv"]["pr_auc"]["mean"], floor["cv"]["pr_auc"]["mean"] + 0.3)
            self.assertGreater(m[name]["holdout"]["pr_auc"], floor["holdout"]["pr_auc"] + 0.3)
        self.assertGreater(m["logreg"]["cv"]["pr_auc"]["mean"], m["rule_dev_vs_last_week"]["cv"]["pr_auc"]["mean"])


class OtherBranches(unittest.TestCase):
    def test_strategy_selection(self):
        y = np.array([0, 1] * 20)
        self.assertEqual(choose_strategy(y), "stratified")
        self.assertEqual(choose_strategy(y, groups=np.arange(40)), "group")
        self.assertEqual(choose_strategy(y, groups=np.arange(40), time=np.arange(40)), "time")

    def test_stratified_holdout_preserves_ratio(self):
        y = imbalanced_y()
        tr, te = holdout_split(y)
        self.assertLess(abs(y[tr].mean() - y.mean()), 0.01)
        self.assertLess(abs(y[te].mean() - y.mean()), 0.02)
        self.assertEqual(len(tr) + len(te), len(y))

    def test_stratified_cv_disjoint_and_stratified(self):
        y = imbalanced_y()
        folds = list(cv_splits(y, n_splits=5, n_repeats=2))
        self.assertEqual(len(folds), 10)
        for tr, va in folds:
            self.assertEqual(len(np.intersect1d(tr, va)), 0)
            self.assertLess(abs(y[va].mean() - y.mean()), 0.02)

    def test_split_is_deterministic(self):
        y = imbalanced_y()
        a, b = holdout_split(y), holdout_split(y)
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    def test_group_holdout_and_cv_keep_groups_apart(self):
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(50), 4)
        y = rng.integers(0, 2, size=200)
        tr, te = holdout_split(y, groups=groups)
        self.assertFalse(set(groups[tr]) & set(groups[te]))
        for tr, va in cv_splits(y, groups=groups, n_splits=5):
            self.assertFalse(set(groups[tr]) & set(groups[va]))


class NegativeAndBoundary(unittest.TestCase):
    def test_leakage_overlapping_rows_detected(self):
        with self.assertRaisesRegex(LeakageError, "both train and test"):
            assert_no_leakage([0, 1, 2], [2, 3])

    def test_leakage_shared_group_detected(self):
        with self.assertRaisesRegex(LeakageError, "groups"):
            assert_no_leakage([0, 2], [1, 3], groups=np.array([0, 0, 1, 1]))

    def test_leakage_future_in_train_detected(self):
        with self.assertRaisesRegex(LeakageError, "within 0 steps"):
            assert_no_leakage([0, 3], [1, 2], time=np.array([1, 2, 3, 4]))

    def test_gap_violation_detected_and_exact_gap_accepted(self):
        t = np.array([0, 10, 12, 20])
        with self.assertRaisesRegex(LeakageError, "within 5 steps"):
            assert_no_leakage([0, 1], [2, 3], time=t, gap=5)  # 12 - 10 = 2 <= 5
        assert_no_leakage([0, 1], [3], time=t, gap=5)  # 20 - 10 = 10 > 5

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

    def test_time_holdout_without_enough_positives_rejected(self):
        t = np.arange(1000)
        y = np.zeros(1000, dtype=int)
        y[:200:10] = 1  # 20 positives, all in the early (training) period
        with self.assertRaisesRegex(SplitPolicyError, "evaluation fold"):
            holdout_split(y, time=t)

    def test_time_range_too_short_for_gap_rejected(self):
        t = np.arange(30)
        y = np.array([0, 1] * 15)
        with self.assertRaisesRegex(SplitPolicyError, "too short"):
            holdout_split(y, time=t, gap=100)

    def test_metrics_reject_single_class_and_nan(self):
        with self.assertRaisesRegex(ValueError, "single class"):
            compute_metrics([1, 1, 1], [0.2, 0.8, 0.9])
        with self.assertRaisesRegex(ValueError, "NaN"):
            compute_metrics([0, 1], [0.1, float("nan")])


if __name__ == "__main__":
    unittest.main()
