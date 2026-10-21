"""Day 2 tests: leakage-safe pipelines and bounded, reproducible search.

Run: python -m unittest discover -s tests -v
"""
import unittest
from unittest import mock

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evalproto import search
from evalproto.baseline import GAP
from evalproto.data import load_dataset
from evalproto.search import (
    MAX_TRIALS_HARD_CAP,
    SearchBudgetError,
    make_pipeline,
    run_search,
)
from evalproto.splits import holdout_split


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ds = load_dataset()
        cls.X, cls.y, cls.t = ds.X.to_numpy(), ds.y.to_numpy(), ds.time
        cls.train_idx, cls.test_idx = holdout_split(cls.y, ds.groups, ds.time, gap=GAP)
        # small slice keeps tests fast; still time-ordered, still has positives
        cls.Xtr = cls.X[cls.train_idx][:12000]
        cls.ytr = cls.y[cls.train_idx][:12000]
        cls.ttr = cls.t[cls.train_idx][:12000]

    def small_search(self, family="logreg", **kw):
        kw.setdefault("n_iter", 3)
        kw.setdefault("n_splits", 3)
        return run_search(family, self.Xtr, self.ytr, self.ttr, **kw)


class Reproducibility(Fixture):
    def test_same_seed_gives_identical_trials(self):
        a, b = self.small_search(seed=7), self.small_search(seed=7)
        self.assertEqual(a.trials, b.trials)
        self.assertEqual(a.best_params, b.best_params)

    def test_hgb_same_seed_identical(self):
        a = self.small_search("hgb", n_iter=2, seed=3)
        b = self.small_search("hgb", n_iter=2, seed=3)
        self.assertEqual(a.trials, b.trials)

    def test_different_seed_samples_different_configs(self):
        a, b = self.small_search(seed=1), self.small_search(seed=2)
        self.assertNotEqual([t["params"] for t in a.trials], [t["params"] for t in b.trials])


class TrainOnlyAndInFold(Fixture):
    def test_search_never_sees_holdout_rows(self):
        """Every row passed to fit/predict must come from the training portion."""
        seen = []
        real_fit = Pipeline.fit

        def spy_fit(self_, X, y=None, **kw):
            seen.append(np.asarray(X)[:, 0].copy())  # column 0 = temperature
            return real_fit(self_, X, y, **kw)

        # holdout temperatures, to prove none of them appear in any fit call
        test_temps = set(np.round(self.X[self.test_idx][:, 0], 9))
        with mock.patch.object(Pipeline, "fit", spy_fit):
            self.small_search()
        self.assertTrue(seen)
        for col in seen:
            self.assertFalse(test_temps & set(np.round(col, 9)))

    def test_every_fit_ends_before_validation_with_gap(self):
        """Time order inside the search: training rows precede validation by > GAP."""
        from evalproto.splits import cv_splits

        for tr, va in cv_splits(self.ytr, time=self.ttr, n_splits=3, gap=GAP):
            self.assertGreater(self.ttr[va].min() - self.ttr[tr].max(), GAP)

    def test_scaler_is_fit_on_training_fold_only(self):
        """Preprocessing is inside the pipeline: it is fit on each fold's train rows, not all rows."""
        from evalproto.splits import cv_splits

        fitted_on = []

        class SpyScaler(StandardScaler):
            def fit(self, X, y=None, **kw):
                fitted_on.append(len(X))
                return super().fit(X, y, **kw)

        folds = list(cv_splits(self.ytr, time=self.ttr, n_splits=3, gap=GAP))
        for tr, _ in folds:
            pipe = make_pipeline("logreg")
            pipe.steps[0] = ("scale", SpyScaler())
            pipe.fit(self.Xtr[tr], self.ytr[tr])
        self.assertEqual(fitted_on, [len(tr) for tr, _ in folds])
        self.assertTrue(all(n < len(self.Xtr) for n in fitted_on))

    def test_pipeline_contains_preprocessing(self):
        for fam in ("logreg", "hgb"):
            names = [n for n, _ in make_pipeline(fam).steps]
            self.assertEqual(names, ["scale", "clf"])


class Limits(Fixture):
    def test_trial_cap_enforced(self):
        with self.assertRaisesRegex(SearchBudgetError, "hard cap"):
            self.small_search(n_iter=MAX_TRIALS_HARD_CAP + 1)

    def test_trial_cap_boundary_accepted(self):
        """n_iter == hard cap is allowed (cheap run on a tiny slice)."""
        r = run_search(
            "logreg", self.Xtr[:4000], self.ytr[:4000], self.ttr[:4000],
            n_iter=MAX_TRIALS_HARD_CAP, n_splits=2, time_budget_s=1e6,
        )
        self.assertEqual(r.n_completed, MAX_TRIALS_HARD_CAP)

    def test_zero_trials_rejected(self):
        with self.assertRaisesRegex(SearchBudgetError, "n_iter"):
            self.small_search(n_iter=0)

    def test_nonpositive_time_budget_rejected(self):
        with self.assertRaisesRegex(SearchBudgetError, "time_budget_s"):
            self.small_search(time_budget_s=0)

    def test_time_budget_stops_early_with_partial_results(self):
        ticks = iter(range(0, 1000, 10))  # fake clock: +10 s per call
        r = self.small_search(n_iter=6, time_budget_s=25, clock=lambda: next(ticks))
        self.assertTrue(r.stopped_early)
        self.assertLess(r.n_completed, 6)
        self.assertGreaterEqual(r.n_completed, 1)
        self.assertIsNotNone(r.best_params)

    def test_zero_budget_exhausted_before_first_trial(self):
        ticks = iter([0, 100, 200, 300])  # already over budget at first check
        r = self.small_search(n_iter=3, time_budget_s=1, clock=lambda: next(ticks))
        self.assertTrue(r.stopped_early)
        self.assertEqual(r.n_completed, 0)
        self.assertIsNone(r.best_params)

    def test_max_rows_caps_training_rows_per_fit(self):
        sizes = []
        real_fit = Pipeline.fit

        def spy(self_, X, y=None, **kw):
            sizes.append(len(X))
            return real_fit(self_, X, y, **kw)

        with mock.patch.object(Pipeline, "fit", spy):
            self.small_search(n_iter=1, max_rows=500)
        self.assertTrue(sizes and max(sizes) <= 500)

    def test_unknown_family_rejected(self):
        with self.assertRaises(ValueError):
            make_pipeline("svm")
        with self.assertRaises(KeyError):
            self.small_search("svm")


class NegativeControl(Fixture):
    def test_shuffled_labels_collapse_to_chance(self):
        """If the search leaked, shuffled labels would still score high. They must not."""
        rng = np.random.default_rng(0)
        y_shuffled = rng.permutation(self.ytr)
        r = run_search("hgb", self.Xtr, y_shuffled, self.ttr, n_iter=2, n_splits=3, seed=1)
        chance = float(y_shuffled.mean())
        self.assertLess(r.best_cv_pr_auc, chance + 0.05)
        real = run_search("hgb", self.Xtr, self.ytr, self.ttr, n_iter=2, n_splits=3, seed=1)
        self.assertGreater(real.best_cv_pr_auc, 5 * chance)


class EndToEnd(Fixture):
    def test_results_file_and_cv_based_selection(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d, mock.patch.object(search, "RESULTS", Path(d) / "s.json"):
            out = search.run(n_iter=2, time_budget_s=120)
        fams = out["families"]
        best_cv = max(fams, key=lambda f: fams[f]["best_cv_pr_auc"])
        self.assertEqual(out["selected_by_cv"], best_cv)
        for r in fams.values():
            self.assertLessEqual(r["n_completed"], r["n_requested"])
            self.assertGreater(r["holdout"]["pr_auc"], 0.3)


if __name__ == "__main__":
    unittest.main()
