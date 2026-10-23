"""Final validation (Day 5): one formal hold-out evaluation, release gates, packaged model.

What this module does, in order:

  1. Freeze the candidate: the model chosen on Day 3, its Day 2 hyper-parameters, and the alert
     threshold fixed from out-of-fold TRAINING scores (never from the hold-out).
  2. Ledger check: a configuration that already went through the formal evaluation cannot be
     evaluated again (`HoldoutAlreadyUsedError`). Repeated looks at a hold-out turn it into
     training data; the ledger makes every look visible and blocks the repeat.
  3. Score the hold-out ONCE (through `tracking.run_experiment`, so the experiment record is
     written from the same scoring pass) and check the release gates that are declared below,
     before any result exists.
  4. Per-building and time-of-day results on the hold-out, comparison table from Day 3.
  5. Package the evaluated model (the one trained on the training portion only) with its
     monitoring reference, and write results/final_evaluation.json.

Honest note: the hold-out was also scored for confirmation on Days 2 to 4. No decision used
those numbers (selection used out-of-fold scores only), but the hold-out is therefore not
untouched; the ledger starts with the Day 5 evaluation and cannot see earlier looks.

Usage:
  python -m evalproto.final evaluate           # the one formal evaluation + package + report
  python -m evalproto.final build              # rebuild the package only (no hold-out scoring)
  python -m evalproto.final sample             # write data/sample_raw.csv and data/sample_drifted.csv
  python -m evalproto.final predict IN.csv [OUT.csv]
  python -m evalproto.final monitor IN.csv     # uses an `anomaly` column as labels if present
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import tracking
from .analysis import RESULTS as SELECTION_JSON
from .analysis import operating_point, slice_report
from .data import FEATURES, simulate_raw
from .inference import InferencePipeline, load_package, save_package
from .monitoring import build_reference, format_report, monitor

ROOT = Path(__file__).resolve().parent.parent

# Release gates. Declared BEFORE the hold-out is scored; changing them afterwards would be
# tuning to the test set. Values: a gate that a useful detector must clear, not a tuned optimum.
GATES = {
    "min_holdout_pr_auc": 0.90,
    "min_holdout_precision": 0.90,
    "min_holdout_recall": 0.90,
    "max_cv_to_holdout_drop": 0.02,  # CV PR-AUC minus hold-out PR-AUC (optimism check)
    "min_worst_building_recall": 0.80,  # over buildings with enough anomalies
    "min_slice_positives": 10,  # smaller slices are reported but not judged
}


class FinalEvaluationError(ValueError):
    pass


class HoldoutAlreadyUsedError(FinalEvaluationError):
    """This configuration was already formally evaluated on the hold-out."""


# ---- gates ------------------------------------------------------------------
def check_gates(metrics: dict, cv_pr_auc_mean: float, worst_building_recall: float | None,
                gates: dict = GATES) -> dict:
    """Compare hold-out metrics with the declared gates. A missing or NaN value fails."""

    def ok(value, limit, kind):
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
        eps = 1e-12  # float noise: 0.9677 - 0.9477 is 0.0200000000000000018, not "above 0.02"
        return value >= limit - eps if kind == "min" else value <= limit + eps

    drop = cv_pr_auc_mean - metrics["holdout_pr_auc"]
    rows = {
        "holdout_pr_auc": (metrics["holdout_pr_auc"], gates["min_holdout_pr_auc"], "min"),
        "holdout_precision": (metrics["holdout_precision"], gates["min_holdout_precision"], "min"),
        "holdout_recall": (metrics["holdout_recall"], gates["min_holdout_recall"], "min"),
        "cv_to_holdout_drop": (drop, gates["max_cv_to_holdout_drop"], "max"),
        "worst_building_recall": (worst_building_recall, gates["min_worst_building_recall"], "min"),
    }
    out = {k: {"value": v, "limit": lim, "rule": ">=" if kind == "min" else "<=", "ok": bool(ok(v, lim, kind))}
           for k, (v, lim, kind) in rows.items()}
    return {"gates": out, "release": all(g["ok"] for g in out.values())}


# ---- ledger -----------------------------------------------------------------
LEDGER_NOTE = ("Each entry is one formal hold-out evaluation. A configuration (config_id) can appear once. "
               "Earlier confirmation scores on Days 2 to 4 are not recorded here.")


def read_ledger(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"note": LEDGER_NOTE, "entries": []}
    return json.loads(path.read_text())


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


# ---- helpers ----------------------------------------------------------------
def _comparison_table() -> dict:
    """Day 3 evidence for all candidates (read from results/model_selection.json)."""
    if not Path(SELECTION_JSON).exists():
        raise FileNotFoundError(f"{SELECTION_JSON} not found. Run `python -m evalproto.analysis` first.")
    sel = json.loads(Path(SELECTION_JSON).read_text())
    table = {}
    for name, m in sel["models"].items():
        table[name] = {
            "cv_pr_auc_mean": m["cv_pr_auc_mean"], "cv_pr_auc_std": m["cv_pr_auc_std"],
            "overfit_gap": m["overfit_gap"], "threshold": m["threshold_from_oof"]["threshold"],
            "oof_precision_at_95_recall": m["threshold_from_oof"]["precision"],
            "day3_holdout_confirmation": m["holdout_at_chosen_threshold"],
        }
    return {"decision": sel["decision"], "models": table}


def _make_pipeline(cap: dict) -> tuple[InferencePipeline, dict]:
    """Package object and monitoring reference from a captured (fitted) experiment."""
    plan = cap["plan"]
    ds, train_idx = plan["ds"], plan["train_idx"]
    Xtr = ds.X.iloc[train_idx]
    thr = cap["threshold"]["threshold"]
    reference = build_reference(Xtr, cap["oof"], float((cap["oof"] >= thr).mean()))
    ranges = {c: [float(Xtr[c].min()), float(Xtr[c].max())] for c in ("temperature", "building_size", "log_consumption")}
    return InferencePipeline(model=cap["model"], threshold=thr, ranges=ranges), reference


def _package_meta(plan: dict, reference: dict, extra: dict) -> dict:
    ds, tr = plan["ds"], plan["train_idx"]
    return {
        "candidate": plan["model_name"], "config_id": plan["config_id"],
        "trained_on": {"rows": int(len(tr)), "first_hour": int(ds.time[tr].min()), "last_hour": int(ds.time[tr].max()),
                       "dataset_name": ds.name, "dataset_identity_hash": plan["config"]["dataset"]["fingerprint"]},
        "threshold_rule": "highest threshold with >= 95% recall on out-of-fold training scores",
        "code": {**tracking.code_revision(), "source_hash": tracking.code_hash()},
        "environment": tracking.environment(),
        "monitoring_reference": reference,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }


def package_dir(root: Path) -> Path:
    return Path(root) / "models" / "final"


# ---- the formal evaluation --------------------------------------------------
def evaluate_final(root: Path = ROOT, model_name: str | None = None, dataset=None,
                   param_overrides: dict | None = None, gates: dict = GATES) -> dict:
    """The one formal hold-out evaluation. Writes results, ledger, experiment record and package."""
    root = Path(root)
    ledger_path = root / "results" / "holdout_ledger.json"
    plan = tracking.plan_experiment(model_name, param_overrides, dataset)  # no hold-out scoring yet
    ledger = read_ledger(ledger_path)
    if any(e["config_id"] == plan["config_id"] for e in ledger["entries"]):
        raise HoldoutAlreadyUsedError(
            f"configuration {plan['config_id']} was already evaluated on the hold-out "
            f"(see {ledger_path}). Re-scoring it would make the hold-out part of the tuning loop. "
            f"Use `python -m evalproto.tracking verify` to check reproducibility instead.")
    comparison = _comparison_table()

    cap: dict = {}
    record = tracking.run_experiment(model_name, param_overrides, dataset=plan["ds"], capture=cap, stage="final_evaluation")
    record = tracking.log_run(record, root / "results" / "experiments.jsonl")

    ds, test_idx = plan["ds"], plan["test_idx"]
    y_te, s_te = ds.y.to_numpy()[test_idx], cap["s_test"]
    thr = cap["threshold"]["threshold"]
    meta_te = ds.meta.iloc[test_idx].reset_index(drop=True)
    hod = meta_te["hod"].to_numpy()
    min_pos = gates["min_slice_positives"]
    slices = {
        "building": slice_report(y_te, s_te, thr, meta_te["building"].to_numpy(), min_positives=min_pos),
        "time_of_day": slice_report(y_te, s_te, thr, np.where((hod >= 8) & (hod < 18), "work hours", "night/evening"),
                                    min_positives=min_pos),
        "day_type": slice_report(y_te, s_te, thr, np.where(meta_te["is_weekend"] == 1, "weekend", "weekday"),
                                 min_positives=min_pos),
    }
    judged = [r["recall"] for r in slices["building"] if r["status"] == "ok"]
    worst = min(judged) if judged else None
    verdict = check_gates(record["metrics"], record["metrics"]["cv_pr_auc_mean"], worst, gates)
    op = operating_point(y_te, s_te, thr)

    pipeline, reference = _make_pipeline(cap)
    pkg = _package_meta(plan, reference, {
        "run_id": record["run_id"], "release": verdict["release"],
        "holdout_metrics": {k: record["metrics"][k] for k in (
            "holdout_pr_auc", "holdout_precision", "holdout_recall", "holdout_false_positives", "holdout_false_negatives")},
        "gates": verdict["gates"]})
    save_package(package_dir(root), pipeline, pkg)

    result = {
        "candidate": plan["model_name"], "config_id": plan["config_id"], "run_id": record["run_id"],
        "threshold": thr, "metrics": record["metrics"],
        "holdout": {"rows": int(len(test_idx)), "anomalies": int(y_te.sum()), "base_rate": float(y_te.mean()),
                    "no_skill_pr_auc": float(y_te.mean()), "operating_point": op},
        "gates_declared": gates, **verdict, "slices": slices,
        "comparison_day3": comparison, "package": str(package_dir(root).relative_to(root)),
        "code": record["code"], "environment": record["environment"],
    }
    _write_json(root / "results" / "final_evaluation.json", result)
    ledger["entries"].append({"config_id": plan["config_id"], "candidate": plan["model_name"],
                              "evaluated_utc": record["created_utc"], "run_id": record["run_id"],
                              "release": verdict["release"], "dataset_identity_hash": plan["config"]["dataset"]["fingerprint"]})
    _write_json(ledger_path, ledger)
    return result


def rebuild_package(root: Path = ROOT, model_name: str | None = None, dataset=None) -> Path:
    """Rebuild the package from training data only. Does not score or read the hold-out labels."""
    from sklearn.base import clone

    from .analysis import choose_threshold, oof_scores

    root = Path(root)
    plan = tracking.plan_experiment(model_name, None, dataset)
    ds, tr = plan["ds"], plan["train_idx"]
    X, y, t = ds.X.to_numpy()[tr], ds.y.to_numpy()[tr], ds.time[tr]
    idx, oof, _ = oof_scores(plan["model"], X, y, t)
    thr = choose_threshold(y[idx], oof)
    cap = {"plan": plan, "model": clone(plan["model"]).fit(X, y), "oof": oof, "threshold": thr}
    pipeline, reference = _make_pipeline(cap)
    prev = package_dir(root) / "package.json"
    extra = {}
    if prev.exists():  # keep the recorded evaluation of the same configuration
        old = json.loads(prev.read_text())
        if old.get("config_id") == plan["config_id"]:
            extra = {k: old[k] for k in ("run_id", "release", "holdout_metrics", "gates") if k in old}
    return save_package(package_dir(root), pipeline, _package_meta(plan, reference, extra))


# ---- sample data and commands ----------------------------------------------
def write_samples(out_dir: Path) -> tuple[Path, Path]:
    """Raw readings of all buildings for hold-out period hours 2920-3559, plus a drifted copy."""
    raw = simulate_raw()
    s = raw[(raw["hour"] >= 2920) & (raw["hour"] < 3560)]
    s = s[["building", "hour", "temperature", "building_size", "consumption", "anomaly"]].reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b = out_dir / "sample_raw.csv", out_dir / "sample_drifted.csv"
    s.to_csv(a, index=False)
    d = s.copy()  # a unit/sensor change: temperature +12 and building sizes x3
    d["temperature"] += 12
    d["building_size"] *= 3
    d.to_csv(b, index=False)
    return a, b


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    try:
        if cmd == "evaluate":
            res = evaluate_final()
            m = res["metrics"]
            print(f"candidate {res['candidate']}  config {res['config_id']}  threshold {res['threshold']:.4f}")
            print(f"hold-out PR-AUC {m['holdout_pr_auc']:.4f}  precision {m['holdout_precision']:.3f}  "
                  f"recall {m['holdout_recall']:.3f}  FP {m['holdout_false_positives']}  FN {m['holdout_false_negatives']}")
            for k, g in res["gates"].items():
                print(f"  gate {k:24} {g['value']:.4f} {g['rule']} {g['limit']}  {'PASS' if g['ok'] else 'FAIL'}")
            print("RELEASE:", "yes" if res["release"] else "NO", "| package:", res["package"])
            return 0 if res["release"] else 1
        if cmd == "build":
            print("package written to", rebuild_package())
            return 0
        if cmd == "sample":
            for p in write_samples(ROOT / "data"):
                print("wrote", p)
            return 0
        if cmd in ("predict", "monitor") and len(argv) >= 2:
            pipe = load_package(package_dir(ROOT))
            raw = pd.read_csv(argv[1])
            frame = pipe.score_frame(raw)
            if cmd == "predict":
                out = frame[["building", "hour", "score", "alert", "status", "warning"]]
                target = Path(argv[2]) if len(argv) > 2 else ROOT / "data" / "predictions.csv"
                target.parent.mkdir(exist_ok=True)
                out.to_csv(target, index=False)
                print(f"{len(out)} rows, {int((frame['status'] == 'scored').sum())} scored, "
                      f"{int(frame['alert'].sum())} alerts -> {target}")
                return 0
            labels = raw.sort_values(["building", "hour"])["anomaly"].to_numpy() if "anomaly" in raw.columns else None
            sig = monitor(frame, pipe.meta["monitoring_reference"], labels)
            print(format_report(sig))
            return 0 if sig["overall"] != "alert" else 1
        print(__doc__)
        return 2
    except (FinalEvaluationError, tracking.TrackingError, FileNotFoundError, ValueError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
