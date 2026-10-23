"""Day 5 tests: monitoring signals."""
import unittest

import numpy as np
import pandas as pd

from evalproto import monitoring as M
from evalproto.data import FEATURES

RNG = np.random.default_rng(0)
N_REF = 5000


def make_train():
    return pd.DataFrame({c: RNG.normal(size=N_REF) for c in FEATURES})


TRAIN = make_train()
OOF = np.clip(RNG.beta(0.3, 6, N_REF), 0, 1)
REF = M.build_reference(TRAIN, OOF, oof_alert_rate=0.03)


def window(n=2000, shift=None, alert_rate=0.03, warning_share=0.0, history_share=0.0, seed=1):
    r = np.random.default_rng(seed)
    X = pd.DataFrame({c: r.normal(size=n) for c in FEATURES})
    for c, d in (shift or {}).items():
        X[c] += d
    df = X.copy()
    df["score"] = np.clip(r.beta(0.3, 6, n), 0, 1)
    df["alert"] = r.random(n) < alert_rate
    df["status"] = "scored"
    df["warning"] = ""
    df.loc[: int(n * warning_share) - 1, "warning"] = "out_of_range:temperature;" if warning_share else ""
    df.loc[n - int(n * history_share):, "status"] = "insufficient_history"
    return df


class TestPsi(unittest.TestCase):
    def test_same_distribution_is_small(self):
        self.assertLess(M.psi(M.make_reference(TRAIN["temperature"]), window()["temperature"]), 0.05)

    def test_shift_is_large(self):
        ref = M.make_reference(TRAIN["temperature"])
        self.assertGreater(M.psi(ref, window(shift={"temperature": 2.0})["temperature"]), M.PSI_ALERT)

    def test_identical_data_is_zero(self):
        ref = M.make_reference(TRAIN["temperature"])
        self.assertAlmostEqual(M.psi(ref, TRAIN["temperature"]), 0.0, places=9)

    def test_values_outside_reference_range_are_counted_in_outer_bins(self):
        ref = M.make_reference(TRAIN["temperature"])
        self.assertGreater(M.psi(ref, np.full(500, 1000.0)), M.PSI_ALERT)

    def test_errors(self):
        with self.assertRaises(M.MonitoringError):
            M.make_reference(np.arange(10))  # too few reference values
        with self.assertRaises(M.MonitoringError):
            M.psi(M.make_reference(TRAIN["temperature"]), [np.nan, np.inf])  # nothing finite

    def test_binary_feature_works(self):
        ref = M.make_reference(RNG.integers(0, 2, 1000))
        self.assertLess(M.psi(ref, RNG.integers(0, 2, 1000)), 0.05)

    def test_status_boundaries(self):
        self.assertEqual(M._status_high(0.10, M.PSI_WARN, M.PSI_ALERT), "ok")  # exactly at warn: ok
        self.assertEqual(M._status_high(0.1001, M.PSI_WARN, M.PSI_ALERT), "warn")
        self.assertEqual(M._status_high(0.25, M.PSI_WARN, M.PSI_ALERT), "warn")
        self.assertEqual(M._status_high(0.2501, M.PSI_WARN, M.PSI_ALERT), "alert")


class TestMonitor(unittest.TestCase):
    def test_healthy_window_is_ok(self):
        sig = M.monitor(window(), REF)
        self.assertEqual(sig["overall"], "ok", M.format_report(sig))

    def test_feature_drift_alone_is_capped_at_warn(self):
        sig = M.monitor(window(shift={"temperature": 3.0}), REF)
        self.assertEqual(sig["feature_drift"]["temperature"]["status"], "alert")
        self.assertEqual(sig["overall"], "warn")  # leading indicator, not an output failure

    def test_out_of_range_inputs_raise_alert(self):
        self.assertEqual(M.monitor(window(warning_share=0.06), REF)["overall"], "alert")
        self.assertEqual(M.monitor(window(warning_share=0.03), REF)["input_quality"]["status"], "warn")
        self.assertEqual(M.monitor(window(warning_share=0.0), REF)["input_quality"]["status"], "ok")

    def test_start_up_window_without_history_warns_only(self):
        sig = M.monitor(window(n=3000, history_share=0.4), REF)
        self.assertEqual(sig["input_quality"]["status"], "warn")

    def test_alert_rate_ratio_thresholds(self):
        self.assertEqual(M._ratio_status(0.03, 0.03)[1], "ok")
        self.assertEqual(M._ratio_status(0.045, 0.03)[1], "ok")  # exactly 1.5x: not above the warn limit
        self.assertEqual(M._ratio_status(0.0451, 0.03)[1], "warn")
        self.assertEqual(M._ratio_status(0.075, 0.03)[1], "warn")  # exactly 2.5x
        self.assertEqual(M._ratio_status(0.0751, 0.03)[1], "alert")
        self.assertEqual(M._ratio_status(0.01, 0.03)[1], "alert")  # three times FEWER alerts also alerts
        self.assertEqual(M._ratio_status(0.0, 0.03)[1], "alert")  # silent model

    def test_alert_storm_and_silence_in_a_window(self):
        self.assertEqual(M.monitor(window(alert_rate=0.15), REF)["alert_rate"]["status"], "alert")
        self.assertEqual(M.monitor(window(alert_rate=0.0), REF)["alert_rate"]["status"], "alert")

    def test_score_drift(self):
        w = window()
        w["score"] = np.clip(w["score"] + 0.4, 0, 1)
        self.assertEqual(M.monitor(w, REF)["score_drift"]["status"], "alert")

    def test_labels_not_supplied_or_too_few(self):
        self.assertEqual(M.monitor(window(), REF)["label_performance"]["status"], "not_evaluable")
        w = window()
        few = np.zeros(len(w), dtype=int)
        few[:19] = 1  # 19 positives: below the minimum of 20
        self.assertEqual(M.monitor(w, REF, few)["label_performance"]["status"], "not_evaluable")

    def test_label_performance_ok_and_alert(self):
        w = window(n=3000)
        y = w["alert"].to_numpy().astype(int)  # perfect detector
        self.assertEqual(M.monitor(w, REF, y)["label_performance"]["status"], "ok")
        y_bad = y.copy()
        y_bad[np.flatnonzero(y == 1)[: int(0.2 * y.sum())]] = 0  # precision falls to 0.80
        sig = M.monitor(w, REF, y_bad)
        self.assertEqual(sig["label_performance"]["status"], "alert")
        self.assertEqual(sig["overall"], "alert")

    def test_label_length_mismatch(self):
        with self.assertRaises(M.MonitoringError):
            M.monitor(window(), REF, np.zeros(5))

    def test_empty_and_tiny_windows(self):
        with self.assertRaises(M.MonitoringError):
            M.monitor(window().iloc[0:0], REF)
        sig = M.monitor(window(n=100), REF)  # too few rows for drift statistics
        self.assertEqual(sig["score_drift"]["status"], "not_evaluable")
        self.assertEqual(sig["overall"], "ok")

    def test_alert_rate_needs_enough_expected_alerts(self):
        # reference rate 0.03: 667 rows -> 20.01 expected alerts (judged), 600 rows -> 18 (not judged)
        self.assertNotEqual(M.monitor(window(n=667, alert_rate=0.15), REF)["alert_rate"]["status"], "not_evaluable")
        self.assertEqual(M.monitor(window(n=600, alert_rate=0.15), REF)["alert_rate"]["status"], "not_evaluable")

    def test_no_scored_rows(self):
        w = window(n=300)
        w["status"] = "insufficient_history"
        sig = M.monitor(w, REF)
        self.assertEqual(sig["alert_rate"]["status"], "not_evaluable")

    def test_report_is_printable(self):
        self.assertIn("overall:", M.format_report(M.monitor(window(), REF)))


if __name__ == "__main__":
    unittest.main()
