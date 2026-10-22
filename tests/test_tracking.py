"""Day 4 tests: experiment tracking and reproducibility (stdlib unittest, pytest also collects them)."""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evalproto import tracking as T
from evalproto.data import load_dataset


def fake_run(**metric_overrides):
    """A minimal record with the same shape as run_experiment()'s output."""
    metrics = {"cv_pr_auc_mean": 0.99, "cv_pr_auc_std": 0.004, "cv_pr_auc_folds": [0.99] * 4,
               "holdout_pr_auc": 0.997, "threshold": 0.30, "holdout_precision": 0.979,
               "holdout_recall": 0.981, "holdout_false_positives": 8, "holdout_false_negatives": 7}
    metrics.update(metric_overrides)
    return {
        "run_id": None,
        "config": {"model": "m", "model_class": "Pipeline", "params": {"a": 1}, "seed": 42,
                   "split": {"gap_hours": 24}, "preprocessing": {"scaling": "x"},
                   "dataset": {"fingerprint": "abc", "rows": 10}},
        "config_id": "0123456789abcdef",
        "code": {"git_commit": "c0ffee", "git_dirty": False, "source_hash": "h1"},
        "environment": {"python": "3.13"},
        "metrics": metrics,
        "holdout_scores_sha256": "s1",
    }


class TestFingerprint(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.1, 0.2, 0.3, 0.4]})
        self.y = pd.Series([0, 1, 0, 0])
        self.t = np.arange(4)
        self.g = np.array([0, 0, 1, 1])

    def fp(self, X=None, y=None, t=None, g=None):
        return T.fingerprint_dataset(self.X if X is None else X, self.y if y is None else y,
                                     self.t if t is None else t, self.g if g is None else g)

    def test_same_data_same_hash(self):
        self.assertEqual(self.fp(), self.fp(X=self.X.copy()))

    def test_one_changed_cell_changes_hash(self):
        X = self.X.copy()
        X.iloc[2, 1] += 1e-9
        self.assertNotEqual(self.fp(), self.fp(X=X))

    def test_one_flipped_label_changes_hash(self):
        self.assertNotEqual(self.fp(), self.fp(y=pd.Series([0, 1, 1, 0])))

    def test_row_order_changes_hash(self):
        order = [1, 0, 2, 3]
        self.assertNotEqual(self.fp(), self.fp(X=self.X.iloc[order], y=self.y.iloc[order],
                                               t=self.t[order], g=self.g[order]))

    def test_column_rename_changes_hash(self):
        self.assertNotEqual(self.fp(), self.fp(X=self.X.rename(columns={"a": "z"})))

    def test_time_and_groups_matter(self):
        self.assertNotEqual(self.fp(), self.fp(t=self.t + 1))
        self.assertNotEqual(self.fp(), self.fp(g=np.array([0, 1, 1, 1])))

    def test_rounded_hash_ignores_float_noise_but_catches_real_change(self):
        noisy = self.X + 3e-13  # machine-level float noise
        self.assertNotEqual(self.fp(), self.fp(X=noisy))  # exact hash does see it
        r = lambda X: T.fingerprint_dataset(X, self.y, self.t, self.g, decimals=T.IDENTITY_DECIMALS)
        self.assertEqual(r(self.X), r(noisy))  # identity hash does not
        changed = self.X.copy()
        changed.iloc[1, 0] += 0.01  # a real change
        self.assertNotEqual(r(self.X), r(changed))
        changed.iloc[1, 0] = self.X.iloc[1, 0] + 0.5
        self.assertNotEqual(r(self.X), r(changed))

    def test_negative_zero_equals_zero_in_rounded_hash(self):
        Xz = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 2.0]})
        Xn = pd.DataFrame({"a": [-0.0, 1.0], "b": [-1e-13, 2.0]})
        f = lambda X: T.fingerprint_dataset(X, pd.Series([0, 1]), decimals=T.IDENTITY_DECIMALS)
        self.assertEqual(f(Xz), f(Xn))

    def test_real_dataset_hash_is_stable(self):
        a, b = load_dataset(), load_dataset()
        self.assertEqual(T.fingerprint_dataset(a.X, a.y, a.time, a.groups),
                         T.fingerprint_dataset(b.X, b.y, b.time, b.groups))


class TestCodeRevision(unittest.TestCase):
    def test_outside_git_falls_back_to_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            rev = T.code_revision(Path(d))
        self.assertEqual(rev, {"git_commit": "unknown", "git_dirty": None})

    def test_clean_and_dirty_repo(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
            run("init", "-q")
            (repo / "evalproto").mkdir()
            (repo / "evalproto" / "m.py").write_text("x = 1\n")
            run("add", "-A")
            run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
            clean = T.code_revision(repo)
            self.assertEqual(len(clean["git_commit"]), 40)
            self.assertFalse(clean["git_dirty"])
            (repo / "evalproto" / "m.py").write_text("x = 2\n")  # modify tracked code
            self.assertTrue(T.code_revision(repo)["git_dirty"])

    def test_code_hash_detects_edit_without_git(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "a.py").write_text("x = 1\n")
            h1 = T.code_hash(p)
            (p / "a.py").write_text("x = 2\n")
            self.assertNotEqual(h1, T.code_hash(p))
            (p / "a.py").write_text("x = 1\n")
            self.assertEqual(h1, T.code_hash(p))


class TestLog(unittest.TestCase):
    def test_append_only_and_run_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log.jsonl"
            r1 = T.log_run(fake_run(), path)
            first_line = path.read_text().splitlines()[0]
            r2 = T.log_run(fake_run(), path)
            lines = path.read_text().splitlines()
            self.assertEqual(lines[0], first_line)  # earlier line untouched
            self.assertEqual(len(lines), 2)
            self.assertNotEqual(r1["run_id"], r2["run_id"])
            self.assertEqual([r["run_id"] for r in T.read_log(path)], [r1["run_id"], r2["run_id"]])

    def test_missing_log_is_empty(self):
        self.assertEqual(T.read_log(Path("/nonexistent/none.jsonl")), [])

    def test_corrupt_line_raises_not_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log.jsonl"
            T.log_run(fake_run(), path)
            with path.open("a") as f:
                f.write('{"run_id": "trunc\n')
            with self.assertRaises(T.TrackingError):
                T.read_log(path)

    def test_verify_on_empty_log_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(T.TrackingError):
                T.verify(Path(d) / "log.jsonl")

    def test_verify_unknown_run_id_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log.jsonl"
            T.log_run(fake_run(), path)
            with self.assertRaises(T.TrackingError):
                T.verify(path, reference_id="run-999")


class TestCompare(unittest.TestCase):
    def test_identical_runs_are_consistent(self):
        res = T.compare(fake_run(), fake_run())
        self.assertTrue(res["consistent"])
        self.assertEqual(res["max_abs_diff_pr_auc"], 0.0)
        self.assertFalse(res["code_changed"])

    def test_drift_inside_tolerance_passes(self):
        res = T.compare(fake_run(), fake_run(holdout_pr_auc=0.997 - 0.0015))
        self.assertTrue(res["consistent"])

    def test_boundary_exactly_at_tolerance_passes_just_beyond_fails(self):
        tol = {"holdout_pr_auc": 0.002}
        ref = fake_run(holdout_pr_auc=0.990)
        self.assertTrue(T.compare(ref, fake_run(holdout_pr_auc=0.992), tol)["consistent"])
        beyond = T.compare(ref, fake_run(holdout_pr_auc=0.9921), tol)
        self.assertFalse(beyond["consistent"])
        self.assertFalse(beyond["metrics"]["holdout_pr_auc"]["ok"])

    def test_count_boundary(self):
        ref = fake_run()
        self.assertTrue(T.compare(ref, fake_run(holdout_false_positives=10))["consistent"])  # +2 allowed
        self.assertFalse(T.compare(ref, fake_run(holdout_false_positives=11))["consistent"])  # +3 not

    def test_large_drift_fails(self):
        res = T.compare(fake_run(), fake_run(holdout_recall=0.90))
        self.assertFalse(res["consistent"])
        self.assertFalse(res["metrics"]["holdout_recall"]["ok"])
        self.assertTrue(res["metrics"]["holdout_pr_auc"]["ok"])  # only the drifted metric fails

    def test_nan_metric_fails_never_passes(self):
        res = T.compare(fake_run(), fake_run(holdout_pr_auc=float("nan")))
        self.assertFalse(res["consistent"])

    def test_missing_metric_raises(self):
        broken = fake_run()
        del broken["metrics"]["threshold"]
        with self.assertRaises(T.TrackingError):
            T.compare(fake_run(), broken)

    def test_different_dataset_is_not_comparable(self):
        other = fake_run()
        other["config"]["dataset"]["fingerprint"] = "different"
        with self.assertRaises(T.ComparabilityError) as cm:
            T.compare(fake_run(), other)
        self.assertIn("dataset fingerprint", str(cm.exception))

    def test_different_params_or_seed_is_not_comparable(self):
        for key, value in (("params", {"a": 2}), ("seed", 7), ("model", "other"), ("split", {"gap_hours": 0})):
            other = fake_run()
            other["config"][key] = value
            with self.assertRaises(T.ComparabilityError, msg=key):
                T.compare(fake_run(), other)

    def test_code_change_is_reported_not_fatal(self):
        other = fake_run()
        other["code"]["source_hash"] = "h2"
        res = T.compare(fake_run(), other)
        self.assertTrue(res["consistent"])
        self.assertTrue(res["code_changed"])


class TestEndToEnd(unittest.TestCase):
    """Runs the real selected experiment (about 5 s each)."""

    @classmethod
    def setUpClass(cls):
        cls.first = T.run_experiment()
        cls.second = T.run_experiment()

    def test_record_has_all_required_parts(self):
        r = self.first
        for key in ("config", "config_id", "code", "environment", "metrics", "artifacts"):
            self.assertIn(key, r)
        self.assertEqual(len(r["config"]["dataset"]["fingerprint"]), 64)
        self.assertIn("git_commit", r["code"])
        self.assertEqual(len(r["code"]["source_hash"]), 64)
        self.assertIn("scikit-learn", r["environment"])
        self.assertIn("features", r["config"]["preprocessing"])
        self.assertIn("clf__learning_rate", r["config"]["params"])
        self.assertEqual(set(r["artifacts"]), {"results/search.json", "results/model_selection.json"})
        self.assertTrue(all(len(v) == 64 for v in r["artifacts"].values()))

    def test_rerun_is_consistent_and_identical(self):
        res = T.compare(self.first, self.second)
        self.assertTrue(res["consistent"])
        self.assertEqual(res["max_abs_diff_pr_auc"], 0.0)
        self.assertTrue(res["holdout_scores_identical"])
        self.assertEqual(self.first["config_id"], self.second["config_id"])

    def test_matches_day3_decision(self):
        m = self.first["metrics"]
        sel = json.loads(T.SELECTION_JSON.read_text())["models"][self.first["config"]["model"]]
        # model_selection.json may come from another machine / library version, so use the
        # stated tolerance rather than exact equality (a 0.0004 CV difference was seen on Python 3.14)
        self.assertLessEqual(abs(m["cv_pr_auc_mean"] - sel["cv_pr_auc_mean"]), T.TOLERANCES["cv_pr_auc_mean"])
        self.assertLessEqual(abs(m["threshold"] - sel["threshold_from_oof"]["threshold"]), T.TOLERANCES["threshold"])

    def test_changed_hyperparameter_is_detected(self):
        changed = T.run_experiment(param_overrides={"clf__max_depth": 3})
        with self.assertRaises(T.ComparabilityError):
            T.compare(self.first, changed)
        self.assertNotEqual(self.first["config_id"], changed["config_id"])

    def test_changed_data_is_detected_and_not_compared(self):
        ds = load_dataset()
        X = ds.X.copy()
        X.iloc[0, 0] += 0.5  # one cell
        altered = type(ds)(X=X, y=ds.y, groups=ds.groups, time=ds.time, name=ds.name, meta=ds.meta)
        rec = T.run_experiment(dataset=altered)
        self.assertNotEqual(rec["config"]["dataset"]["fingerprint"], self.first["config"]["dataset"]["fingerprint"])
        with self.assertRaises(T.ComparabilityError):
            T.compare(self.first, rec)

    def test_tampered_log_fails_verification(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log.jsonl"
            rec = copy.deepcopy(self.first)
            rec["metrics"]["holdout_pr_auc"] -= 0.05  # a logged number that the code cannot reproduce
            T.log_run(rec, path)
            res = T.verify(path)
            self.assertFalse(res["consistent"])
            self.assertFalse(res["metrics"]["holdout_pr_auc"]["ok"])

    def test_untampered_log_verifies(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log.jsonl"
            T.log_run(self.first, path)
            self.assertTrue(T.verify(path)["consistent"])

    def test_unknown_model_rejected(self):
        with self.assertRaises(T.TrackingError):
            T.run_experiment("no_such_model")


if __name__ == "__main__":
    unittest.main()
