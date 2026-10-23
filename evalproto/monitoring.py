"""Monitoring signals for the deployed model (Day 5).

The model card lists what to watch; this module computes it. Input is the full frame from
`InferencePipeline.score_frame` (features, score, alert, status, warning) and the reference
stored in the package. Every signal returns a value, its thresholds and a status
(`ok` / `warn` / `alert` / `not_evaluable`); the overall status is the worst one.

Signals (no labels needed unless stated):
  input_quality      share of rows with insufficient history or out-of-range warnings
  feature_drift      population stability index (PSI) per feature against the training data
  score_drift        PSI of the model score against out-of-fold training scores (fixed bins)
  alert_rate         observed alert share vs the expected one (ratio; z-score shown for information)
  label_performance  precision and recall once confirmed labels arrive (needs labels)

PSI rule of thumb: < 0.10 stable, 0.10 to 0.25 some shift, > 0.25 large shift. These cut-offs
are conventional, not derived from this data.

Overall status: feature drift is a LEADING indicator, not proof that outputs got worse (on the
simulated hold-out, temperature drifts strongly while recall stays at 0.98). So feature drift on
its own can raise the overall status to `warn` at most; `alert` needs an output-side signal
(score drift, alert rate, input quality or labels). Alert-rate cut-offs are ratios, not z-scores,
because anomalies come in 1 to 4 hour events and the base rate itself varies between periods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES

PSI_WARN, PSI_ALERT = 0.10, 0.25
RATIO_WARN, RATIO_ALERT = 1.5, 2.5  # observed/expected alert rate, or its inverse
SCORE_EDGES = (0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95)  # fixed score bins (deciles of a near-constant bulk are unstable)
QUALITY_WARN, QUALITY_ALERT = 0.01, 0.05  # share of rows with a warning
HISTORY_WARN = 0.30  # share of rows without 168 h history (high right after start-up)
MIN_ROWS_FOR_DRIFT = 200
MIN_EXPECTED_ALERTS = 20  # alert rate is judged only when the window should contain this many alerts
MIN_LABEL_POSITIVES = 20
LABEL_FLOOR = 0.90  # recall and precision below this raise an alert (same as the release gates)
SEVERITY = {"ok": 0, "not_evaluable": 0, "warn": 1, "alert": 2}


class MonitoringError(ValueError):
    pass


# ---- PSI --------------------------------------------------------------------
def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, values, side="right")
    return np.bincount(idx, minlength=len(edges) + 1) / len(values)


def make_reference(values, n_bins: int = 10, edges=None) -> dict:
    """Bin edges (quantiles of the reference data unless given) and the share of data per bin."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < n_bins * 5:
        raise MonitoringError(f"need at least {n_bins * 5} reference values, got {len(v)}")
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])) if edges is None \
        else np.asarray(sorted(set(edges)), dtype=float)
    return {"edges": edges.tolist(), "proportions": _proportions(v, edges).tolist(), "n": int(len(v))}


def psi(reference: dict, values, eps: float = 1e-4) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        raise MonitoringError("no finite values to compare")
    p = np.clip(np.asarray(reference["proportions"], dtype=float), eps, None)
    q = np.clip(_proportions(v, np.asarray(reference["edges"], dtype=float)), eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def build_reference(X_train: pd.DataFrame, oof_scores, oof_alert_rate: float) -> dict:
    """Reference stored in the package: feature bins, score bins, expected alert rate."""
    return {
        "features": {c: make_reference(X_train[c].to_numpy()) for c in FEATURES},
        "score": make_reference(oof_scores, edges=SCORE_EDGES),
        "expected_alert_rate": float(oof_alert_rate),
    }


# ---- signals ----------------------------------------------------------------
def _status_high(value: float, warn: float, alert: float) -> str:
    return "alert" if value > alert else "warn" if value > warn else "ok"


def _ratio_status(obs: float, exp: float) -> tuple[float, str]:
    """Ratio of observed to expected alert rate (>= 1 means more alerts than expected)."""
    if exp <= 0:
        return float("inf") if obs > 0 else 1.0, "alert" if obs > 0 else "ok"
    ratio = obs / exp
    worst = ratio if ratio >= 1 else (1 / ratio if ratio > 0 else float("inf"))
    return ratio, _status_high(worst, RATIO_WARN, RATIO_ALERT)


def _psi_signal(ref: dict, values, rows: int) -> dict:
    if rows < MIN_ROWS_FOR_DRIFT:
        return {"status": "not_evaluable", "reason": f"fewer than {MIN_ROWS_FOR_DRIFT} scored rows"}
    value = psi(ref, values)
    return {"psi": value, "status": _status_high(value, PSI_WARN, PSI_ALERT)}


def monitor(frame: pd.DataFrame, reference: dict, labels=None) -> dict:
    """Compute all signals for one monitoring window (a scored frame)."""
    if frame is None or len(frame) == 0:
        raise MonitoringError("empty monitoring window")
    scored = frame[frame["status"] == "scored"]
    n_all, n = len(frame), len(scored)
    sig: dict = {}

    hist_share = float((frame["status"] == "insufficient_history").mean())
    warn_share = float((scored["warning"] != "").mean()) if n else 0.0
    sig["input_quality"] = {
        "insufficient_history_share": hist_share, "out_of_range_share": warn_share,
        "status": max((_status_high(warn_share, QUALITY_WARN, QUALITY_ALERT),
                       "warn" if hist_share > HISTORY_WARN else "ok"), key=SEVERITY.get),
    }

    sig["feature_drift"] = {c: _psi_signal(reference["features"][c], scored[c], n) for c in FEATURES}
    sig["score_drift"] = _psi_signal(reference["score"], scored["score"], n)

    if n == 0:
        sig["alert_rate"] = {"status": "not_evaluable", "reason": "no scored rows"}
    elif reference["expected_alert_rate"] * n < MIN_EXPECTED_ALERTS:
        sig["alert_rate"] = {"status": "not_evaluable",
                             "reason": f"fewer than {MIN_EXPECTED_ALERTS} alerts expected in this window"}
    else:
        exp, obs = reference["expected_alert_rate"], float(scored["alert"].mean())
        sd = np.sqrt(exp * (1 - exp) / n)
        z = (obs - exp) / sd if sd > 0 else 0.0
        ratio, status = _ratio_status(obs, exp)
        sig["alert_rate"] = {"expected": exp, "observed": obs, "alerts_per_1000_h": 1000 * obs,
                             "ratio": float(ratio), "z": float(z), "status": status}

    if labels is None:
        sig["label_performance"] = {"status": "not_evaluable", "reason": "no labels supplied"}
    else:
        y = np.asarray(labels)
        if len(y) != n_all:
            raise MonitoringError("labels must have one value per row of the frame")
        m = (frame["status"] == "scored").to_numpy()
        yt, pred = y[m].astype(int), frame["alert"].to_numpy()[m]
        pos = int(yt.sum())
        if pos < MIN_LABEL_POSITIVES:
            sig["label_performance"] = {"positives": pos, "status": "not_evaluable",
                                        "reason": f"fewer than {MIN_LABEL_POSITIVES} confirmed anomalies"}
        else:
            tp = int((pred & (yt == 1)).sum())
            recall, precision = tp / pos, tp / max(int(pred.sum()), 1)
            sig["label_performance"] = {"positives": pos, "recall": recall, "precision": precision,
                                        "status": "alert" if min(recall, precision) < LABEL_FLOOR else "ok"}

    output_side = [sig["input_quality"]["status"], sig["score_drift"]["status"], sig["alert_rate"]["status"],
                   sig["label_performance"]["status"]]
    feature_side = ["warn" if v["status"] in ("warn", "alert") else v["status"] for v in sig["feature_drift"].values()]
    sig["overall"] = max(output_side + feature_side, key=SEVERITY.get)  # feature drift caps at warn
    sig["rows"] = {"total": n_all, "scored": n}
    return sig


def format_report(sig: dict) -> str:
    lines = [f"overall: {sig['overall'].upper()}  (rows {sig['rows']['total']}, scored {sig['rows']['scored']})"]
    q = sig["input_quality"]
    lines.append(f"input_quality   {q['status']:14} no-history {q['insufficient_history_share']:.1%}  "
                 f"out-of-range {q['out_of_range_share']:.1%}")
    s = sig["score_drift"]
    lines.append(f"score_drift     {s['status']:14} " + (f"PSI {s['psi']:.3f}" if "psi" in s else s.get("reason", "")))
    a = sig["alert_rate"]
    lines.append(f"alert_rate      {a['status']:14} " + (f"{a['alerts_per_1000_h']:.1f} per 1000 h "
                 f"(expected {1000 * a['expected']:.1f}, ratio {a['ratio']:.2f})" if "z" in a else a.get("reason", "")))
    lp = sig["label_performance"]
    lines.append(f"label_perf      {lp['status']:14} " + (f"recall {lp['recall']:.3f} precision {lp['precision']:.3f}"
                 if "recall" in lp else lp.get("reason", "")))
    for c, v in sig["feature_drift"].items():
        lines.append(f"  drift {c:20} {v['status']:14} " + (f"PSI {v['psi']:.3f}" if "psi" in v else v.get("reason", "")))
    return "\n".join(lines)
