"""Error analysis and model selection (Day 3).

Question answered: which model should we pick, and do we trust that choice?
The answer is built from several measured pieces of evidence, not one score:

  1. Out-of-fold (OOF) scores from time-aware CV on the TRAINING portion only
     -> fold stability, thresholds, slices and error analysis all come from here.
  2. Overfit check: train score vs CV score vs hold-out score.
  3. Class imbalance: lift over the no-skill floor; effect of class weighting.
  4. Threshold trade-offs: precision/recall/alerts at several operating points.
  5. Slices: building, time of day, weekday/weekend, temperature band, anomaly
     type and position inside an event. Tiny slices are flagged, not scored.
  6. Error analysis: where false negatives and false positives come from.
  7. Paired bootstrap on the hold-out: is the winner's lead bigger than noise?

The hold-out is scored once per model, AFTER the threshold and the choice are
fixed from training-side evidence. It confirms the choice; it does not make it.

Usage:  python -m evalproto.analysis        (needs results/search.json from Day 2)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_recall_curve

from .baseline import DeviationRule
from .data import FEATURES, load_dataset
from .metrics import PRIMARY_METRIC, compute_metrics
from .search import GAP, N_SPLITS, SEED, make_pipeline
from .splits import cv_splits, holdout_split

ROOT = Path(__file__).resolve().parent.parent
SEARCH_JSON = ROOT / "results" / "search.json"
RESULTS = ROOT / "results" / "model_selection.json"

# ---- policy constants (documented in docs/MODEL_SELECTION.md) ---------------
TARGET_RECALL = 0.95  # operating point: catch at least 95% of anomalies...
MIN_SLICE_POSITIVES = 20  # ...slices with fewer anomalies are flagged, not scored
OVERFIT_GAP_LIMIT = 0.05  # train PR-AUC minus CV PR-AUC above this = overfit
WORST_SLICE_RECALL_FLOOR = 0.80  # no big slice may fall below this recall
N_BOOT = 300  # paired bootstrap resamples (days are resampled, not rows)
# Tie-break order, fewest modelling assumptions first. Used only when models are
# statistically tied on CV PR-AUC (within one fold standard deviation of the best).
PREFERENCE = ("rule", "logreg", "hgb_unweighted", "hgb")


class AnalysisError(ValueError):
    """Raised when the evidence needed for a decision cannot be produced."""


# ---- models -----------------------------------------------------------------
def load_best_params(path: Path = SEARCH_JSON) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m evalproto.search` first.")
    fam = json.loads(Path(path).read_text())["families"]
    return {k: v["best_params"] for k, v in fam.items()}


def build_models(best: dict) -> dict:
    """Candidates compared on Day 3. All are scale->clf pipelines except the rule."""
    return {
        "rule": DeviationRule(column=FEATURES.index("abs_log_ratio_168")),
        "logreg": make_pipeline("logreg").set_params(**best["logreg"]),
        "hgb": make_pipeline("hgb").set_params(**best["hgb"]),
        # same tuned boosting, class weighting switched off: isolates the imbalance effect
        "hgb_unweighted": make_pipeline("hgb").set_params(**best["hgb"], clf__class_weight=None),
    }


def _score(model, X):
    return model.predict_proba(X)[:, 1]


# ---- evidence 1: out-of-fold scores ----------------------------------------
def oof_scores(model, X, y, t, n_splits: int = N_SPLITS, gap: int = GAP):
    """Time-aware out-of-fold scores on the training portion.

    Returns (row_idx, scores, fold_id). Rows in the earliest chunk are never
    validated (expanding window), so they have no OOF score.
    """
    idx, sc, fid = [], [], []
    for k, (tr, va) in enumerate(cv_splits(y, time=t, n_splits=n_splits, gap=gap)):
        m = clone(model).fit(X[tr], y[tr])
        idx.append(va)
        sc.append(_score(m, X[va]))
        fid.append(np.full(len(va), k))
    return np.concatenate(idx), np.concatenate(sc), np.concatenate(fid)


def fold_pr_auc(y, scores, fold_id) -> list[float]:
    return [float(average_precision_score(y[fold_id == k], scores[fold_id == k]))
            for k in np.unique(fold_id)]


# ---- evidence 4: thresholds -------------------------------------------------
def choose_threshold(y, scores, min_recall: float = TARGET_RECALL) -> dict:
    """Highest threshold (fewest alerts) that still reaches `min_recall`.

    Raises AnalysisError if the target recall cannot be reached at all.
    """
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if not 0 < min_recall <= 1:
        raise AnalysisError("min_recall must be in (0, 1]")
    if y.sum() == 0:
        raise AnalysisError("no positives: cannot choose a threshold")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    ok = np.flatnonzero(recall[:-1] >= min_recall)
    if ok.size == 0:
        raise AnalysisError(f"recall {min_recall} is not reachable with these scores")
    i = ok[-1]  # recall falls as the threshold rises, so the last ok index = highest threshold
    return {"threshold": float(thresholds[i]), "precision": float(precision[i]), "recall": float(recall[i])}


def operating_point(y, scores, thr: float) -> dict:
    y, scores = np.asarray(y), np.asarray(scores)
    pred = scores >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return {
        "threshold": float(thr),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "alerts_per_1000_hours": 1000.0 * (tp + fp) / len(y),
        "false_alarms_per_1000_hours": 1000.0 * fp / len(y),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ---- evidence 5: slices -----------------------------------------------------
def slice_report(y, scores, thr: float, labels, min_positives: int = MIN_SLICE_POSITIVES) -> list[dict]:
    """Per-slice metrics at one threshold. Slices with too few anomalies are flagged
    `insufficient_positives` and carry no metrics, so nobody reads noise as a result."""
    y, scores, labels = np.asarray(y), np.asarray(scores), np.asarray(labels)
    out = []
    for lab in pd.unique(labels):
        m = labels == lab
        n, pos = int(m.sum()), int(y[m].sum())
        row = {"slice": str(lab), "rows": n, "positives": pos}
        if pos < min_positives or pos == n:
            row["status"] = "insufficient_positives" if pos < min_positives else "no_negatives"
        else:
            op = operating_point(y[m], scores[m], thr)
            row.update(status="ok", pr_auc=float(average_precision_score(y[m], scores[m])),
                       recall=op["recall"], precision=op["precision"])
        out.append(row)
    return sorted(out, key=lambda r: r["slice"])


def temperature_band(temp, edges=None) -> np.ndarray:
    temp = np.asarray(temp)
    edges = np.quantile(temp, [1 / 3, 2 / 3]) if edges is None else edges
    return np.where(temp < edges[0], "cold", np.where(temp < edges[1], "mild", "warm"))


def recall_by_event(y, scores, thr, event_type, event_pos) -> dict:
    """Recall among TRUE anomalies, split by anomaly type and position inside the event."""
    y, scores = np.asarray(y), np.asarray(scores)
    pos = y == 1
    caught = scores >= thr
    out = {"by_type": {}, "by_position": {}}
    for code, name in ((1, "spike"), (2, "drop")):
        m = pos & (np.asarray(event_type) == code)
        out["by_type"][name] = {"anomalies": int(m.sum()), "recall": float(caught[m].mean()) if m.any() else None}
    for p in sorted(np.unique(np.asarray(event_pos)[pos])):
        m = pos & (np.asarray(event_pos) == p)
        label = "first hour" if p == 0 else f"hour {int(p) + 1} of event"
        out["by_position"][label] = {"anomalies": int(m.sum()), "recall": float(caught[m].mean())}
    return out


# ---- evidence 6: error analysis --------------------------------------------
def false_positive_origin(y, scores, thr, building, hour, lags=(24, 168)) -> dict:
    """How many false alarms fire exactly `lag` hours after a real anomaly in the
    same building? Those are the lag features being contaminated by an earlier event.
    Compared with the same share among ALL normal hours (the chance level)."""
    y, scores = np.asarray(y), np.asarray(scores)
    building, hour = np.asarray(building), np.asarray(hour)
    anomalous = set(zip(building[y == 1].tolist(), hour[y == 1].tolist()))

    def after_event(b, h):
        return any((b, h - lag) in anomalous for lag in lags)

    flag = np.fromiter((after_event(b, h) for b, h in zip(building.tolist(), hour.tolist())), bool, len(y))
    fp = (scores >= thr) & (y == 0)
    neg = y == 0
    return {
        "false_positives": int(fp.sum()),
        "share_after_real_event": float(flag[fp].mean()) if fp.any() else None,
        "chance_share_among_normal_hours": float(flag[neg].mean()),
        "enrichment": float(flag[fp].mean() / flag[neg].mean()) if fp.any() and flag[neg].mean() > 0 else None,
    }


# ---- evidence 7: paired bootstrap ------------------------------------------
def paired_bootstrap(y, s_a, s_b, day, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """95% interval for PR-AUC(a) - PR-AUC(b), resampling whole days (rows within a
    day are correlated, so resampling rows would be too optimistic)."""
    y, s_a, s_b, day = map(np.asarray, (y, s_a, s_b, day))
    days = np.unique(day)
    if len(days) < 10:
        raise AnalysisError("need at least 10 distinct days to bootstrap")
    by_day = {d: np.flatnonzero(day == d) for d in days}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(days, size=len(days), replace=True)
        rows = np.concatenate([by_day[d] for d in pick])
        if y[rows].min() == y[rows].max():
            continue  # a resample with one class has no PR-AUC; skip it
        diffs.append(average_precision_score(y[rows], s_a[rows]) - average_precision_score(y[rows], s_b[rows]))
    if len(diffs) < n_boot * 0.9:
        raise AnalysisError("too many degenerate bootstrap resamples")
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff_mean": float(np.mean(diffs)), "ci95": [float(lo), float(hi)], "n_boot": len(diffs),
            "a_better_than_b": bool(lo > 0)}


# ---- selection --------------------------------------------------------------
def select_model(card: dict) -> dict:
    """Decision rule applied to the scorecard (documented in docs/MODEL_SELECTION.md).

    1. Gates. A model failing any gate is not eligible:
         - overfit gap (train PR-AUC - CV PR-AUC) <= OVERFIT_GAP_LIMIT
         - worst-building recall at the operating threshold >= WORST_SLICE_RECALL_FLOOR
    2. Rank eligible models by mean CV PR-AUC.
    3. Tie rule. Every eligible model within one CV-fold standard deviation of the
       best is "tied" with it: a lead that small is noise. Among tied models pick
       the one with the fewest modelling assumptions (PREFERENCE order).
    Models are never ranked by hold-out score.
    """
    rows = {}
    for name, c in card.items():
        reasons = []
        if c["overfit_gap"] > OVERFIT_GAP_LIMIT:
            reasons.append(f"overfit gap {c['overfit_gap']:.3f} > {OVERFIT_GAP_LIMIT}")
        if c["worst_slice_recall"] is not None and c["worst_slice_recall"] < WORST_SLICE_RECALL_FLOOR:
            reasons.append(f"worst slice recall {c['worst_slice_recall']:.3f} < {WORST_SLICE_RECALL_FLOOR}")
        rows[name] = {"eligible": not reasons, "failed_gates": reasons, "cv_pr_auc": c["cv_pr_auc_mean"]}
    eligible = [n for n, r in rows.items() if r["eligible"]]
    if not eligible:
        raise AnalysisError("no model passes the gates; do not ship any of them")
    best = max(eligible, key=lambda n: rows[n]["cv_pr_auc"])
    margin = card[best]["cv_pr_auc_std"]
    tied = [n for n in eligible if rows[n]["cv_pr_auc"] >= rows[best]["cv_pr_auc"] - margin - 1e-12]
    unknown = [n for n in tied if n not in PREFERENCE]
    if unknown:
        raise AnalysisError(f"models missing from PREFERENCE: {unknown}")
    chosen = min(tied, key=PREFERENCE.index)
    others = sorted((n for n in eligible if n != chosen), key=lambda n: rows[n]["cv_pr_auc"], reverse=True)
    return {
        "selected": chosen,
        "runner_up": others[0] if others else None,
        "best_by_cv": best,
        "tied_with_best": sorted(tied, key=PREFERENCE.index),
        "tie_margin": margin,
        "decided_by": "tie rule (fewest assumptions)" if chosen != best else "highest CV PR-AUC",
        "per_model": rows,
    }


# ---- end to end -------------------------------------------------------------
def run(n_boot: int = N_BOOT) -> dict:
    ds = load_dataset()
    X, y, t, meta = ds.X.to_numpy(), ds.y.to_numpy(), ds.time, ds.meta
    train_idx, test_idx = holdout_split(y, ds.groups, t, gap=GAP)
    Xtr, ytr, ttr = X[train_idx], y[train_idx], t[train_idx]
    mtr, mte = meta.iloc[train_idx].reset_index(drop=True), meta.iloc[test_idx].reset_index(drop=True)
    models = build_models(load_best_params())

    out = {"seed": SEED, "primary_metric": PRIMARY_METRIC,
           "policy": {"target_recall": TARGET_RECALL, "min_slice_positives": MIN_SLICE_POSITIVES,
                      "overfit_gap_limit": OVERFIT_GAP_LIMIT, "worst_slice_recall_floor": WORST_SLICE_RECALL_FLOOR},
           "base_rate": {"train": float(ytr.mean()), "holdout": float(y[test_idx].mean())},
           "models": {}}
    card, holdout_scores = {}, {}

    for name, model in models.items():
        idx, oof, fid = oof_scores(model, Xtr, ytr, ttr)
        y_oof, m_oof = ytr[idx], mtr.iloc[idx].reset_index(drop=True)
        folds = fold_pr_auc(y_oof, oof, fid)
        final = clone(model).fit(Xtr, ytr)
        train_pr = float(average_precision_score(ytr, _score(final, Xtr)))
        s_test = _score(final, X[test_idx])
        holdout_scores[name] = s_test

        thr = choose_threshold(y_oof, oof)  # fixed from TRAINING-side evidence
        cv_mean = float(np.mean(folds))

        # slices on OOF scores at the chosen threshold
        slices = {
            "building": slice_report(y_oof, oof, thr["threshold"], m_oof["building"].to_numpy()),
            "time_of_day": slice_report(y_oof, oof, thr["threshold"],
                                        np.where((m_oof["hod"] >= 8) & (m_oof["hod"] < 18), "work hours", "night/evening")),
            "day_type": slice_report(y_oof, oof, thr["threshold"], np.where(m_oof["is_weekend"] == 1, "weekend", "weekday")),
            "temperature": slice_report(y_oof, oof, thr["threshold"], temperature_band(m_oof["temperature"])),
        }
        big_ok = [s for s in slices["building"] if s["status"] == "ok"]
        worst = min(big_ok, key=lambda s: s["recall"]) if big_ok else None

        card[name] = {
            "cv_pr_auc_mean": cv_mean, "cv_pr_auc_std": float(np.std(folds, ddof=1)),
            "cv_pr_auc_min_fold": float(min(folds)), "train_pr_auc": train_pr,
            "overfit_gap": train_pr - cv_mean,
            "worst_slice_recall": None if worst is None else worst["recall"],
        }
        out["models"][name] = {
            **card[name],
            "fold_pr_auc": folds,
            "lift_over_no_skill_cv": cv_mean / float(y_oof.mean()),
            "threshold_from_oof": thr,
            "oof_at_default_0.5": operating_point(y_oof, oof, 0.5),
            "holdout_pr_auc": float(average_precision_score(y[test_idx], s_test)),
            "holdout_at_default_0.5": operating_point(y[test_idx], s_test, 0.5),
            "holdout_at_chosen_threshold": operating_point(y[test_idx], s_test, thr["threshold"]),
            "holdout_threshold_sweep": [operating_point(y[test_idx], s_test, th) for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)],
            "worst_building": None if worst is None else {"building": worst["slice"], "recall": worst["recall"], "positives": worst["positives"]},
            "slices_oof": slices,
            "recall_by_event_oof": recall_by_event(y_oof, oof, thr["threshold"], m_oof["event_type"], m_oof["event_pos"]),
            "false_positive_origin_oof": false_positive_origin(y_oof, oof, thr["threshold"], m_oof["building"], m_oof["hour"]),
        }

    decision = select_model(card)
    out["decision"] = decision
    sel = decision["selected"]
    day = (mte["hour"] // 24).to_numpy()
    out["paired_bootstrap_holdout"] = {
        f"{sel}_minus_{other}": paired_bootstrap(y[test_idx], holdout_scores[sel], holdout_scores[other], day, n_boot=n_boot)
        for other in models if other != sel
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    return out


def _print(out: dict) -> None:
    print(f"base rate train {out['base_rate']['train']:.3f}, hold-out {out['base_rate']['holdout']:.3f}\n")
    print(f"{'model':<16}{'CV PR-AUC':>12}{'min fold':>10}{'train':>8}{'gap':>8}{'worst bldg rec':>16}{'thr':>7}{'HO prec':>9}{'HO rec':>8}")
    for n, m in out["models"].items():
        h = m["holdout_at_chosen_threshold"]
        wr = m["worst_slice_recall"]
        print(f"{n:<16}{m['cv_pr_auc_mean']:>12.3f}{m['cv_pr_auc_min_fold']:>10.3f}{m['train_pr_auc']:>8.3f}"
              f"{m['overfit_gap']:>8.3f}{(wr if wr is not None else float('nan')):>16.3f}"
              f"{m['threshold_from_oof']['threshold']:>7.3f}{h['precision']:>9.3f}{h['recall']:>8.3f}")
    d = out["decision"]
    print(f"\nbest by CV: {d['best_by_cv']}  tied within {d['tie_margin']:.4f}: {d['tied_with_best']}")
    print(f"selected: {d['selected']} ({d['decided_by']})  runner-up: {d['runner_up']}")
    for n, r in d["per_model"].items():
        print(f"  {n}: eligible={r['eligible']} {r['failed_gates']}")
    print("\npaired bootstrap on hold-out (PR-AUC difference, 95% interval):")
    for k, v in out["paired_bootstrap_holdout"].items():
        print(f"  {k}: {v['diff_mean']:+.3f} {v['ci95']}")


if __name__ == "__main__":
    _print(run())
