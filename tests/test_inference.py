"""Day 5 tests: inference pipeline and model package."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from evalproto import inference as I
from evalproto.data import FEATURES, make_features, simulate_raw

RAW = simulate_raw(n_buildings=3, n_days=20)  # 3 buildings x 480 hours, small and fast
COLS = ["building", "hour", "temperature", "building_size", "consumption"]


def raw_frame(n_hours=480, buildings=(0, 1)):
    return RAW[RAW["building"].isin(buildings) & (RAW["hour"] < n_hours)][COLS].reset_index(drop=True)


def small_pipeline(threshold=0.5, ranges=None):
    f = make_features(RAW)
    model = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(f[FEATURES].to_numpy(), f["anomaly"])
    return I.InferencePipeline(model=model, threshold=threshold, ranges=ranges or {}), f


class TestFeatureParity(unittest.TestCase):
    def test_features_equal_training_features(self):
        """Serving must compute the same numbers as training (training/serving skew check)."""
        served = I.compute_features(I.validate_raw(RAW[COLS]))
        served = served[served["status"] == "scored"]
        trained = make_features(RAW)
        m = served.merge(trained, on=["building", "hour"], suffixes=("_s", "_t"))
        self.assertEqual(len(m), len(trained))
        for c in FEATURES:
            np.testing.assert_allclose(m[f"{c}_s"], m[f"{c}_t"], rtol=0, atol=1e-12, err_msg=c)

    def test_scores_equal_model_on_training_features(self):
        pipe, f = small_pipeline()
        out = pipe.score_frame(RAW[COLS])
        out = out[out["status"] == "scored"].merge(f[["building", "hour"] + FEATURES], on=["building", "hour"], suffixes=("", "_t"))
        expected = pipe.model.predict_proba(out[[c + "_t" if c + "_t" in out else c for c in FEATURES]].to_numpy())[:, 1]
        np.testing.assert_allclose(out["score"].to_numpy(), expected, atol=1e-12)

    def test_features_are_causal(self):
        pipe, _ = small_pipeline()
        raw = raw_frame(buildings=(0,))
        a = pipe.score_frame(raw)
        changed = raw.copy()
        changed.loc[changed["hour"] >= 400, "consumption"] *= 3  # change the FUTURE only
        b = pipe.score_frame(changed)
        early = a["hour"] < 400
        pd.testing.assert_series_equal(a.loc[early, "score"], b.loc[early, "score"])


class TestHistoryBoundary(unittest.TestCase):
    def test_168_rows_cannot_be_scored_169_can(self):
        pipe, _ = small_pipeline()
        r168 = pipe.score_frame(raw_frame(168, buildings=(0,)))
        self.assertEqual(len(r168), 168)
        self.assertTrue((r168["status"] == "insufficient_history").all())
        self.assertTrue(r168["score"].isna().all())
        self.assertFalse(r168["alert"].any())
        r169 = pipe.score_frame(raw_frame(169, buildings=(0,)))
        self.assertEqual((r169["status"] == "scored").sum(), 1)
        self.assertTrue(np.isfinite(r169.loc[r169["status"] == "scored", "score"]).all())

    def test_history_is_per_building(self):
        pipe, _ = small_pipeline()
        out = pipe.score_frame(raw_frame(200, buildings=(0, 1)))
        self.assertEqual(out.groupby("building")["status"].apply(lambda s: (s == "scored").sum()).tolist(), [32, 32])


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.pipe, _ = small_pipeline()
        self.raw = raw_frame(300, buildings=(0,))

    def bad(self, df, msg=None):
        with self.assertRaises(I.InputError) as cm:
            self.pipe.predict(df)
        if msg:
            self.assertIn(msg, str(cm.exception))

    def test_missing_column(self):
        self.bad(self.raw.drop(columns=["temperature"]), "missing columns")

    def test_nan_inf_and_text(self):
        for bad_value in (np.nan, np.inf, "abc"):
            df = self.raw.copy()
            df["consumption"] = df["consumption"].astype(object)
            df.loc[5, "consumption"] = bad_value
            self.bad(df)

    def test_non_positive_consumption(self):
        for v in (0.0, -1.0):
            df = self.raw.copy()
            df.loc[3, "consumption"] = v
            self.bad(df, "consumption must be > 0")

    def test_non_positive_building_size(self):
        df = self.raw.copy()
        df.loc[0, "building_size"] = 0
        self.bad(df, "building_size")

    def test_duplicates(self):
        self.bad(pd.concat([self.raw, self.raw.iloc[[10]]]), "duplicate")

    def test_hour_gap_is_rejected(self):
        self.bad(self.raw.drop(index=[100]), "consecutive")

    def test_fractional_or_negative_hour(self):
        df = self.raw.copy()
        df["hour"] = df["hour"].astype(float)
        df.loc[4, "hour"] = 4.5
        self.bad(df, "integer")
        df2 = self.raw.copy()
        df2["hour"] = df2["hour"] - 1  # first hour becomes -1
        self.bad(df2, "integer")

    def test_empty_and_wrong_type(self):
        self.bad(self.raw.iloc[0:0])
        with self.assertRaises(I.InputError):
            self.pipe.predict([1, 2, 3])

    def test_unsorted_input_gives_sorted_output_and_same_scores(self):
        shuffled = self.raw.sample(frac=1, random_state=1)
        a, b = self.pipe.predict(self.raw), self.pipe.predict(shuffled)
        pd.testing.assert_frame_equal(a, b)

    def test_extra_columns_ignored(self):
        a = self.pipe.predict(self.raw)
        b = self.pipe.predict(self.raw.assign(anomaly=0, note="x"))
        pd.testing.assert_frame_equal(a, b)


class TestRangeWarning(unittest.TestCase):
    def test_boundary_inclusive(self):
        raw = raw_frame(300, buildings=(0,))
        f = I.compute_features(I.validate_raw(raw))
        top = float(f.loc[f["status"] == "scored", "temperature"].max())  # warm-up rows are never warned
        pipe, _ = small_pipeline(ranges={"temperature": [-100.0, top]})
        out = pipe.score_frame(raw)
        self.assertTrue((out["warning"] == "").all())  # exactly at the maximum: not flagged
        pipe2, _ = small_pipeline(ranges={"temperature": [-100.0, top - 1e-6]})
        out2 = pipe2.score_frame(raw)
        flagged = out2[out2["warning"] != ""]
        self.assertGreaterEqual(len(flagged), 1)
        self.assertTrue(flagged["warning"].str.contains("out_of_range:temperature").all())
        self.assertTrue(flagged["score"].notna().all())  # still scored, only warned


class TestRangeSlack(unittest.TestCase):
    def test_float_noise_is_not_a_warning_but_a_real_excess_is(self):
        raw = raw_frame(300, buildings=(0,))
        f = I.compute_features(I.validate_raw(raw))
        top = float(f.loc[f["status"] == "scored", "building_size"].max())
        for excess, expect_warning in ((1e-12, False), (1e-3, True)):
            pipe, _ = small_pipeline(ranges={"building_size": [0.0, top - excess]})
            out = pipe.score_frame(raw)
            self.assertEqual(bool((out["warning"] != "").any()), expect_warning, excess)

    def test_csv_round_trip_does_not_create_warnings(self):
        raw = raw_frame(300, buildings=(0,))
        f = I.compute_features(I.validate_raw(raw))
        sc = f[f["status"] == "scored"]
        ranges = {c: [float(sc[c].min()), float(sc[c].max())] for c in ("temperature", "building_size", "log_consumption")}
        pipe, _ = small_pipeline(ranges=ranges)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.csv"
            raw.to_csv(path, index=False)
            out = pipe.score_frame(pd.read_csv(path))
        self.assertTrue((out["warning"] == "").all())


class TestPackage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "pkg"
        self.pipe, _ = small_pipeline(threshold=0.3, ranges={"temperature": [-5.0, 40.0]})
        I.save_package(self.dir, self.pipe, {"candidate": "test"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_same_predictions(self):
        loaded = I.load_package(self.dir)
        raw = raw_frame(300, buildings=(0, 1))
        pd.testing.assert_frame_equal(self.pipe.predict(raw), loaded.predict(raw))
        self.assertEqual(loaded.threshold, 0.3)
        self.assertEqual(loaded.meta["candidate"], "test")

    def test_package_records_versions_and_hash(self):
        meta = json.loads((self.dir / "package.json").read_text())
        for k in ("model_sha256", "scikit_learn_version", "python_version", "features", "threshold"):
            self.assertIn(k, meta)

    def test_altered_model_file_is_rejected(self):
        with (self.dir / "model.joblib").open("ab") as f:
            f.write(b"x")
        with self.assertRaisesRegex(I.PackageError, "hash"):
            I.load_package(self.dir)

    def test_missing_files_rejected(self):
        (self.dir / "package.json").unlink()
        with self.assertRaises(I.PackageError):
            I.load_package(self.dir)
        with self.assertRaises(I.PackageError):
            I.load_package(Path(self.tmp.name) / "nothing")

    def test_version_mismatch_rejected_unless_allowed(self):
        p = self.dir / "package.json"
        meta = json.loads(p.read_text())
        meta["scikit_learn_version"] = "0.0.1"
        p.write_text(json.dumps(meta))
        with self.assertRaisesRegex(I.PackageError, "scikit-learn"):
            I.load_package(self.dir)
        self.assertIsNotNone(I.load_package(self.dir, allow_version_mismatch=True))

    def test_feature_list_mismatch_rejected(self):
        p = self.dir / "package.json"
        meta = json.loads(p.read_text())
        meta["features"] = meta["features"][:-1]
        p.write_text(json.dumps(meta))
        with self.assertRaisesRegex(I.PackageError, "feature"):
            I.load_package(self.dir)


if __name__ == "__main__":
    unittest.main()
