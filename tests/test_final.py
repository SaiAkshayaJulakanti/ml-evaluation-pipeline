"""Day 5 tests: release gates, one-time hold-out evaluation, packaging. Everything writes to temp folders."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evalproto import final as F
from evalproto import tracking as T
from evalproto.inference import load_package
from evalproto.monitoring import monitor

GOOD = {"holdout_pr_auc": 0.99, "holdout_precision": 0.95, "holdout_recall": 0.95}


class TestGates(unittest.TestCase):
    def check(self, cv=0.99, worst=0.9, **over):
        return F.check_gates({**GOOD, **over}, cv, worst)

    def test_good_model_released(self):
        self.assertTrue(self.check()["release"])

    def test_each_gate_can_fail_alone(self):
        cases = {"holdout_pr_auc": dict(holdout_pr_auc=0.89, cv=0.89), "holdout_precision": dict(holdout_precision=0.89),
                 "holdout_recall": dict(holdout_recall=0.89), "cv_to_holdout_drop": dict(cv=0.99 + 0.03 - 0.0, ),
                 "worst_building_recall": dict(worst=0.79)}
        for gate, kw in cases.items():
            r = self.check(**kw)
            self.assertFalse(r["release"], gate)
            self.assertFalse(r["gates"][gate]["ok"], gate)
            self.assertEqual(sum(not g["ok"] for g in r["gates"].values()), 1, gate)

    def test_boundaries(self):
        self.assertTrue(self.check(holdout_precision=0.90)["release"])  # exactly at the limit passes
        self.assertFalse(self.check(holdout_precision=0.8999)["release"])
        self.assertTrue(self.check(cv=0.9677, holdout_pr_auc=0.9477)["gates"]["cv_to_holdout_drop"]["ok"])  # drop 0.02
        self.assertFalse(self.check(cv=0.9679, holdout_pr_auc=0.9477)["gates"]["cv_to_holdout_drop"]["ok"])
        self.assertTrue(self.check(worst=0.80)["release"])

    def test_nan_and_missing_fail(self):
        self.assertFalse(self.check(holdout_recall=float("nan"))["release"])
        self.assertFalse(self.check(worst=None)["release"])  # no building could be judged -> no release
        self.assertFalse(self.check(worst=float("nan"))["release"])

    def test_better_on_holdout_than_cv_is_fine(self):
        self.assertTrue(self.check(cv=0.95, holdout_pr_auc=0.99)["gates"]["cv_to_holdout_drop"]["ok"])

    def test_useless_detector_is_rejected(self):
        """Negative control: random scores must not pass the gates."""
        from sklearn.metrics import average_precision_score
        from evalproto.analysis import operating_point
        rng = np.random.default_rng(0)
        y = (rng.random(13720) < 0.027).astype(int)
        s = rng.random(len(y))
        op = operating_point(y, s, 0.95)
        m = {"holdout_pr_auc": average_precision_score(y, s), "holdout_precision": op["precision"], "holdout_recall": op["recall"]}
        r = F.check_gates(m, 0.99, 0.5)
        self.assertFalse(r["release"])
        self.assertFalse(r["gates"]["holdout_pr_auc"]["ok"])

    def test_declared_gate_values(self):
        """The gates were fixed before the evaluation; changing them should be a visible, deliberate edit."""
        self.assertEqual(F.GATES, {"min_holdout_pr_auc": 0.90, "min_holdout_precision": 0.90, "min_holdout_recall": 0.90,
                                   "max_cv_to_holdout_drop": 0.02, "min_worst_building_recall": 0.80, "min_slice_positives": 10})


class TestEvaluateOnce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.res = F.evaluate_final(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs_exist_and_are_consistent(self):
        r = self.root
        for rel in ("results/final_evaluation.json", "results/holdout_ledger.json", "results/experiments.jsonl",
                    "models/final/package.json", "models/final/model.joblib"):
            self.assertTrue((r / rel).exists(), rel)
        log = T.read_log(r / "results" / "experiments.jsonl")
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["stage"], "final_evaluation")
        self.assertEqual(log[0]["run_id"], self.res["run_id"])
        ledger = F.read_ledger(r / "results" / "holdout_ledger.json")
        self.assertEqual([e["config_id"] for e in ledger["entries"]], [self.res["config_id"]])

    def test_selected_candidate_released_and_matches_day3(self):
        self.assertEqual(self.res["candidate"], T.selected_model_name())
        self.assertTrue(self.res["release"])
        sel = json.loads(T.SELECTION_JSON.read_text())["models"][self.res["candidate"]]
        self.assertLessEqual(abs(self.res["metrics"]["cv_pr_auc_mean"] - sel["cv_pr_auc_mean"]), T.TOLERANCES["cv_pr_auc_mean"])

    def test_declared_gates_are_stored_with_result(self):
        self.assertEqual(self.res["gates_declared"], F.GATES)

    def test_small_building_slices_are_not_judged(self):
        statuses = {s["status"] for s in self.res["slices"]["building"]}
        self.assertTrue(statuses <= {"ok", "insufficient_positives", "no_negatives"})
        for s in self.res["slices"]["building"]:
            if s["positives"] < F.GATES["min_slice_positives"]:
                self.assertEqual(s["status"], "insufficient_positives")
                self.assertNotIn("recall", s)

    def test_second_evaluation_of_same_configuration_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            first = F.evaluate_final(root)
            log_before = (root / "results" / "experiments.jsonl").read_text()
            with self.assertRaises(F.HoldoutAlreadyUsedError):
                F.evaluate_final(root)
            # nothing was scored or written by the refused call
            self.assertEqual((root / "results" / "experiments.jsonl").read_text(), log_before)
            self.assertEqual(len(F.read_ledger(root / "results" / "holdout_ledger.json")["entries"]), 1)
            self.assertEqual(first["config_id"], self.res["config_id"])

    def test_changed_configuration_is_a_new_candidate_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = F.evaluate_final(root)
            b = F.evaluate_final(root, param_overrides={"clf__max_depth": 3})
            self.assertNotEqual(a["config_id"], b["config_id"])
            ledger = F.read_ledger(root / "results" / "holdout_ledger.json")
            self.assertEqual(len(ledger["entries"]), 2)  # every look at the hold-out is visible

    def test_weak_candidate_fails_the_gates(self):
        """Negative case with the real pipeline: a single decision stump must not be released."""
        with tempfile.TemporaryDirectory() as d:
            res = F.evaluate_final(Path(d), param_overrides={"clf__max_depth": 1, "clf__max_iter": 1})
            self.assertFalse(res["release"])
            meta = json.loads((Path(d) / "models" / "final" / "package.json").read_text())
            self.assertFalse(meta["release"])  # the package says so

    def test_never_writes_outside_given_root(self):
        real = F.ROOT / "results" / "holdout_ledger.json"
        before = real.read_bytes() if real.exists() else None
        with tempfile.TemporaryDirectory() as d:
            F.evaluate_final(Path(d))
        self.assertEqual(real.read_bytes() if real.exists() else None, before)


class TestPackageAndMonitoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.res = F.evaluate_final(cls.root)
        cls.pipe = load_package(F.package_dir(cls.root))
        a, b = F.write_samples(cls.root / "data")
        cls.healthy, cls.drifted = pd.read_csv(a), pd.read_csv(b)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _signals(self, raw):
        frame = self.pipe.score_frame(raw)
        labels = raw.sort_values(["building", "hour"])["anomaly"].to_numpy()
        return frame, monitor(frame, self.pipe.meta["monitoring_reference"], labels)

    def test_package_scores_hold_out_like_the_evaluation(self):
        frame, sig = self._signals(self.healthy)
        lp = sig["label_performance"]
        self.assertEqual(lp["status"], "ok")
        self.assertGreater(lp["recall"], 0.95)
        self.assertGreater(lp["precision"], 0.95)

    def test_healthy_holdout_window_has_no_output_side_alert(self):
        _, sig = self._signals(self.healthy)
        self.assertIn(sig["overall"], ("ok", "warn"))  # temperature drifts by design (warn at most)
        self.assertEqual(sig["score_drift"]["status"], "ok")

    def test_drifted_window_raises_alert(self):
        _, sig = self._signals(self.drifted)
        self.assertEqual(sig["overall"], "alert")
        self.assertEqual(sig["input_quality"]["status"], "alert")

    def test_package_metadata_complete(self):
        meta = self.pipe.meta
        for k in ("candidate", "config_id", "run_id", "trained_on", "threshold_rule", "code", "environment",
                  "monitoring_reference", "holdout_metrics", "gates", "release"):
            self.assertIn(k, meta)
        self.assertEqual(self.pipe.threshold, self.res["threshold"])

    def test_rebuild_does_not_touch_ledger_or_log(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            F.rebuild_package(root)
            self.assertTrue((root / "models" / "final" / "model.joblib").exists())
            self.assertFalse((root / "results" / "holdout_ledger.json").exists())
            self.assertFalse((root / "results" / "experiments.jsonl").exists())

    def test_rebuilt_package_predicts_like_evaluated_one(self):
        with tempfile.TemporaryDirectory() as d:
            F.rebuild_package(Path(d))
            rebuilt = load_package(F.package_dir(Path(d)))
        pd.testing.assert_frame_equal(self.pipe.predict(self.healthy), rebuilt.predict(self.healthy))

    def test_samples_have_required_columns(self):
        for df in (self.healthy, self.drifted):
            self.assertTrue({"building", "hour", "temperature", "building_size", "consumption", "anomaly"} <= set(df.columns))
        self.assertGreater((self.drifted["temperature"] - self.healthy["temperature"]).min(), 11.9)


if __name__ == "__main__":
    unittest.main()
