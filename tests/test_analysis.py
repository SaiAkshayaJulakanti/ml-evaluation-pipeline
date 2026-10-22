"""Day 3 tests: error analysis and model selection.

Run: python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from evalproto import analysis
from evalproto.analysis import (
    AnalysisError,
    choose_threshold,
    false_positive_origin,
    load_best_params,
    operating_point,
    paired_bootstrap,
    recall_by_event,
    select_model,
    slice_report,
)
from evalproto.data import load_dataset


def card(cv, std=0.005, gap=0.0, worst=0.9):
    return {"cv_pr_auc_mean": cv, "cv_pr_auc_std": std, "overfit_gap": gap, "worst_slice_recall": worst}


class Thresholds(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        self.s = np.array([0.9, 0.8, 0.7, 0.4, 0.6, 0.3, 0.2, 0.1, 0.1, 0.05])

    def test_operating_point_hand_computed(self):
        op = operating_point(self.y, self.s, 0.5)  # alerts: 0.9 0.8 0.7 0.6 -> tp 3, fp 1, fn 1
        self.assertEqual((op["tp"], op["fp"], op["fn"]), (3, 1, 1))
        self.assertAlmostEqual(op["precision"], 0.75)
        self.assertAlmostEqual(op["recall"], 0.75)
        self.assertAlmostEqual(op["alerts_per_1000_hours"], 400.0)

    def test_choose_threshold_is_highest_that_meets_recall(self):
        r = choose_threshold(self.y, self.s, min_recall=0.75)
        self.assertAlmostEqual(r["threshold"], 0.7)  # catches 3 of 4 at the highest possible cut
        self.assertGreaterEqual(r["recall"], 0.75)

    def test_full_recall_boundary_accepted(self):
        r = choose_threshold(self.y, self.s, min_recall=1.0)
        self.assertEqual(r["recall"], 1.0)
        self.assertAlmostEqual(r["threshold"], 0.4)

    def test_invalid_recall_target_rejected(self):
        for bad in (0, -0.1, 1.5):
            with self.assertRaisesRegex(AnalysisError, "min_recall"):
                choose_threshold(self.y, self.s, min_recall=bad)

    def test_no_positives_rejected(self):
        with self.assertRaisesRegex(AnalysisError, "no positives"):
            choose_threshold(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


class Slices(unittest.TestCase):
    def test_small_slice_flagged_not_scored(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 400 + [1] * 60 + [0] * 100 + [1] * 5)
        s = rng.random(len(y))
        labels = np.array(["big"] * 460 + ["tiny"] * 105)
        rows = {r["slice"]: r for r in slice_report(y, s, 0.5, labels)}
        self.assertEqual(rows["tiny"]["status"], "insufficient_positives")
        self.assertNotIn("recall", rows["tiny"])
        self.assertEqual(rows["big"]["status"], "ok")
        self.assertIn("pr_auc", rows["big"])

    def test_exactly_min_positives_is_scored(self):
        y = np.array([0] * 50 + [1] * analysis.MIN_SLICE_POSITIVES)
        s = np.linspace(0, 1, len(y))
        rows = slice_report(y, s, 0.5, np.array(["a"] * len(y)))
        self.assertEqual(rows[0]["status"], "ok")

    def test_one_below_min_positives_is_flagged(self):
        y = np.array([0] * 50 + [1] * (analysis.MIN_SLICE_POSITIVES - 1))
        rows = slice_report(y, np.linspace(0, 1, len(y)), 0.5, np.array(["a"] * len(y)))
        self.assertEqual(rows[0]["status"], "insufficient_positives")

    def test_all_positive_slice_flagged(self):
        y = np.ones(30, dtype=int)
        rows = slice_report(y, np.linspace(0, 1, 30), 0.5, np.array(["a"] * 30))
        self.assertEqual(rows[0]["status"], "no_negatives")

    def test_recall_by_event_type_and_position(self):
        y = np.array([1, 1, 1, 1, 0])
        s = np.array([0.9, 0.2, 0.9, 0.9, 0.1])
        etype = np.array([1, 1, 2, 2, 0])
        epos = np.array([0, 1, 0, 1, -1])
        r = recall_by_event(y, s, 0.5, etype, epos)
        self.assertAlmostEqual(r["by_type"]["spike"]["recall"], 0.5)
        self.assertAlmostEqual(r["by_type"]["drop"]["recall"], 1.0)
        self.assertAlmostEqual(r["by_position"]["first hour"]["recall"], 1.0)
        self.assertAlmostEqual(r["by_position"]["hour 2 of event"]["recall"], 0.5)


class ErrorAnalysis(unittest.TestCase):
    def test_false_positive_after_real_event_is_found(self):
        # building 0: real anomaly at hour 10; alerts at hour 34 (10+24, contaminated lag),
        # at hour 60 (not lag related) and at a normal hour of building 1.
        building = np.array([0, 0, 0, 0, 1])
        hour = np.array([10, 34, 60, 70, 34])
        y = np.array([1, 0, 0, 0, 0])
        s = np.array([0.9, 0.9, 0.9, 0.1, 0.9])
        r = false_positive_origin(y, s, 0.5, building, hour)
        self.assertEqual(r["false_positives"], 3)
        self.assertAlmostEqual(r["share_after_real_event"], 1 / 3)
        self.assertAlmostEqual(r["chance_share_among_normal_hours"], 1 / 4)

    def test_lag_168_also_counted(self):
        building, hour = np.array([0, 0]), np.array([0, 168])
        r = false_positive_origin(np.array([1, 0]), np.array([0.9, 0.9]), 0.5, building, hour)
        self.assertAlmostEqual(r["share_after_real_event"], 1.0)

    def test_no_false_positives_does_not_crash(self):
        r = false_positive_origin(np.array([1, 0]), np.array([0.9, 0.1]), 0.5, np.array([0, 0]), np.array([0, 5]))
        self.assertEqual(r["false_positives"], 0)
        self.assertIsNone(r["share_after_real_event"])


class Bootstrap(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.day = np.repeat(np.arange(60), 20)
        self.y = (rng.random(1200) < 0.1).astype(int)
        self.good = self.y * 0.8 + rng.random(1200) * 0.2
        self.noise = rng.random(1200)

    def test_clearly_better_model_detected(self):
        r = paired_bootstrap(self.y, self.good, self.noise, self.day, n_boot=100)
        self.assertTrue(r["a_better_than_b"])
        self.assertGreater(r["ci95"][0], 0)

    def test_identical_models_interval_covers_zero(self):
        r = paired_bootstrap(self.y, self.good, self.good, self.day, n_boot=100)
        self.assertFalse(r["a_better_than_b"])
        self.assertEqual(r["diff_mean"], 0.0)

    def test_deterministic_for_a_seed(self):
        a = paired_bootstrap(self.y, self.good, self.noise, self.day, n_boot=50, seed=1)
        b = paired_bootstrap(self.y, self.good, self.noise, self.day, n_boot=50, seed=1)
        self.assertEqual(a, b)

    def test_too_few_days_rejected(self):
        with self.assertRaisesRegex(AnalysisError, "10 distinct days"):
            paired_bootstrap(self.y[:100], self.good[:100], self.noise[:100], self.day[:100] % 5)

    def test_all_resamples_single_class_rejected(self):
        y = np.zeros(400, dtype=int)
        with self.assertRaisesRegex(AnalysisError, "degenerate"):
            paired_bootstrap(y, self.noise[:400], self.good[:400], np.repeat(np.arange(20), 20), n_boot=20)


class Selection(unittest.TestCase):
    def test_clear_winner_chosen_by_cv(self):
        d = select_model({"rule": card(0.52), "logreg": card(0.59), "hgb": card(0.99)})
        self.assertEqual(d["selected"], "hgb")
        self.assertEqual(d["decided_by"], "highest CV PR-AUC")
        self.assertEqual(d["runner_up"], "logreg")

    def test_tie_goes_to_fewest_assumptions(self):
        d = select_model({"hgb": card(0.991, std=0.005), "hgb_unweighted": card(0.990, std=0.005)})
        self.assertEqual(d["best_by_cv"], "hgb")
        self.assertEqual(d["selected"], "hgb_unweighted")
        self.assertEqual(d["decided_by"], "tie rule (fewest assumptions)")

    def test_lead_just_beyond_margin_is_not_a_tie(self):
        d = select_model({"hgb": card(0.991, std=0.005), "hgb_unweighted": card(0.985, std=0.005)})
        self.assertEqual(d["selected"], "hgb")

    def test_lead_exactly_at_margin_is_a_tie(self):
        d = select_model({"hgb": card(0.990, std=0.005), "hgb_unweighted": card(0.985, std=0.005)})
        self.assertEqual(d["selected"], "hgb_unweighted")

    def test_overfit_model_excluded_even_if_best(self):
        d = select_model({"hgb": card(0.99, gap=0.20), "logreg": card(0.59, gap=0.0)})
        self.assertEqual(d["selected"], "logreg")
        self.assertFalse(d["per_model"]["hgb"]["eligible"])
        self.assertIn("overfit", d["per_model"]["hgb"]["failed_gates"][0])

    def test_overfit_gap_exactly_at_limit_passes(self):
        d = select_model({"hgb": card(0.99, gap=analysis.OVERFIT_GAP_LIMIT)})
        self.assertTrue(d["per_model"]["hgb"]["eligible"])

    def test_weak_slice_excludes_model(self):
        d = select_model({"hgb": card(0.99, worst=0.5), "logreg": card(0.59, worst=0.9)})
        self.assertEqual(d["selected"], "logreg")
        self.assertIn("worst slice", d["per_model"]["hgb"]["failed_gates"][0])

    def test_no_eligible_model_raises(self):
        with self.assertRaisesRegex(AnalysisError, "no model passes"):
            select_model({"hgb": card(0.99, gap=0.5), "logreg": card(0.59, worst=0.1)})

    def test_unknown_model_in_a_tie_raises(self):
        with self.assertRaisesRegex(AnalysisError, "PREFERENCE"):
            select_model({"hgb": card(0.99), "mystery": card(0.99)})


class Inputs(unittest.TestCase):
    def test_missing_search_results_gives_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "evalproto.search"):
            load_best_params(Path("/nonexistent/search.json"))

    def test_slice_metadata_is_aligned_and_consistent(self):
        ds = load_dataset()
        self.assertTrue(np.array_equal(ds.meta["hour"].to_numpy(), ds.time))
        self.assertTrue(np.array_equal(ds.meta["building"].to_numpy(), ds.groups))
        self.assertTrue(((ds.meta["event_type"] > 0).to_numpy() == ds.y.to_numpy().astype(bool)).all())
        self.assertTrue(((ds.meta["event_pos"] >= 0).to_numpy() == ds.y.to_numpy().astype(bool)).all())
        self.assertEqual(len(ds.meta), len(ds.X))


class EndToEnd(unittest.TestCase):
    def test_full_run_is_train_side_only_and_consistent(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(analysis, "RESULTS", Path(d) / "m.json"):
            out = analysis.run(n_boot=30)
        sel = out["decision"]["selected"]
        self.assertIn(sel, out["models"])
        for name, m in out["models"].items():
            # threshold comes from out-of-fold scores and reaches the recall target there
            self.assertGreaterEqual(m["threshold_from_oof"]["recall"], analysis.TARGET_RECALL)
            self.assertLess(m["cv_pr_auc_min_fold"], m["cv_pr_auc_mean"] + 1e-9)
            self.assertEqual(len(m["holdout_threshold_sweep"]), 9)
            self.assertTrue(all(s["status"] in {"ok", "insufficient_positives", "no_negatives"}
                                for s in m["slices_oof"]["building"]))
        # the weak baselines are clearly beaten, and the bootstrap says so
        self.assertTrue(out["paired_bootstrap_holdout"][f"{sel}_minus_rule"]["a_better_than_b"])
        self.assertTrue(out["paired_bootstrap_holdout"][f"{sel}_minus_logreg"]["a_better_than_b"])


if __name__ == "__main__":
    unittest.main()
