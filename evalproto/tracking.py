"""Experiment tracking and reproducibility check (Day 4).

Question answered: can we say exactly what produced a result, and does running
it again give the same result?

Every run of the SELECTED experiment (the model chosen on Day 3) is written as
one JSON line to `results/experiments.jsonl` (append-only) with:

  dataset      row count, feature list and a SHA-256 fingerprint of X, y, time, groups
  preprocessing  the transformation steps and the lag/window constants
  params       the model's hyper-parameters, the seed and the split policy
  code         git commit, whether tracked code was modified, and a hash of evalproto/*.py
  environment  python / numpy / pandas / scikit-learn versions
  metrics      CV PR-AUC (mean, std, per fold), hold-out PR-AUC, threshold, precision, recall
  artifacts    SHA-256 of the input and output files (search.json, model_selection.json)

`compare` then checks a re-run against a logged run:
  * the CONFIGURATION must be identical (dataset fingerprint, params, seed, split).
    If it is not, the runs are not comparable and `ComparabilityError` is raised,
    instead of reporting a misleading pass or fail.
  * every metric must agree within a stated absolute tolerance (TOLERANCES).

Usage:
  python -m evalproto.tracking run      # run the selected experiment and log it
  python -m evalproto.tracking verify   # re-run the latest logged run and compare
  python -m evalproto.tracking show     # list logged runs
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from .analysis import (
    SEARCH_JSON,
    RESULTS as SELECTION_JSON,
    build_models,
    choose_threshold,
    fold_pr_auc,
    load_best_params,
    oof_scores,
    operating_point,
    _score,
)
from .data import FEATURES, LAG_LONG, LAG_SHORT, load_dataset
from .search import GAP, N_SPLITS, SEED
from .splits import holdout_split

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "experiments.jsonl"
HOLDOUT_FRACTION = 0.2
# Dataset identity: features rounded to this many decimals before hashing. Different
# machines/library versions give float noise around 1e-13 (measured: a column sum of
# hod_sin was +0.0 on one machine and -0.0 on another), which an exact hash would call
# "a different dataset". 4 decimals ignores that noise but still catches any real change.
IDENTITY_DECIMALS = 4

# Stated tolerance for "the re-run is consistent": absolute difference per metric.
# On the same machine and library versions the observed difference is 0; the
# tolerance exists so a different CPU / BLAS / minor library version does not
# produce a false alarm. Counts are integers (alerts that flip near the threshold).
TOLERANCES = {
    "cv_pr_auc_mean": 0.002,
    "cv_pr_auc_std": 0.002,
    "holdout_pr_auc": 0.002,
    # widened from 0.02 to 0.05 after measuring a 0.024 shift between Python 3.13 and 3.14 on
    # IDENTICAL data: the threshold is the 95%-recall cut point, so it moves in steps
    "threshold": 0.05,
    "holdout_precision": 0.01,
    "holdout_recall": 0.01,
    "holdout_false_positives": 2,
    "holdout_false_negatives": 2,
}


class TrackingError(ValueError):
    """Bad input to the tracking code (unknown model, empty log, missing metric)."""


class ComparabilityError(TrackingError):
    """Two runs have different configurations, so comparing their metrics is meaningless."""


# ---- fingerprints -----------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    path = Path(path)
    return sha256_bytes(path.read_bytes()) if path.exists() else None


def fingerprint_dataset(X, y, time=None, groups=None, decimals: int | None = None) -> str:
    """SHA-256 over values AND order of X, y, time and groups.

    Any changed cell, label, row order, shape or column set changes the hash.
    Each array is prefixed with its name and shape so that moving data between
    arrays cannot give the same hash.
    """
    h = hashlib.sha256()
    cols = list(X.columns) if hasattr(X, "columns") else []
    h.update(json.dumps(cols).encode())
    xv = np.asarray(X, dtype=np.float64)
    if decimals is not None:  # tolerate float noise below this many decimals
        xv = np.round(xv, decimals) + 0.0  # + 0.0 turns -0.0 into 0.0
    parts = {"X": xv, "y": np.asarray(y, dtype=np.int64)}
    if time is not None:
        parts["time"] = np.asarray(time, dtype=np.int64)
    if groups is not None:
        parts["groups"] = np.asarray(groups, dtype=np.int64)
    for name, arr in parts.items():
        h.update(f"|{name}:{arr.shape}|".encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def code_hash(package_dir: Path | None = None) -> str:
    """Hash of all evalproto/*.py (name + content). Works without git."""
    package_dir = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(package_dir.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def code_revision(repo: Path | None = None) -> dict:
    """Git commit of the code plus a dirty flag. Falls back to 'unknown' outside git."""
    repo = Path(repo) if repo else ROOT

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=20)

    try:
        head = git("rev-parse", "HEAD")
        if head.returncode != 0:
            raise OSError(head.stderr.strip())
        # untracked files inside evalproto/ also count as "code not fully committed"
        dirty = git("status", "--porcelain", "--", "evalproto", "tests", "requirements.txt")
        return {"git_commit": head.stdout.strip(), "git_dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "unknown", "git_dirty": None}


def environment() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "platform": platform.system()}


# ---- the experiment ---------------------------------------------------------
def _simple_params(model) -> dict:
    """JSON-safe hyper-parameters of a model/pipeline (scalars and strings only)."""
    if not hasattr(model, "get_params"):
        return {}
    out = {}
    for k, v in sorted(model.get_params(deep=True).items()):
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
    return out


def selected_model_name(path: Path = SELECTION_JSON) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m evalproto.analysis` first.")
    return json.loads(Path(path).read_text())["decision"]["selected"]


def preprocessing_description() -> dict:
    return {
        "features": list(FEATURES),
        "feature_engineering": f"causal lag ratios at {LAG_SHORT} h and {LAG_LONG} h, hour-of-day sin/cos, "
                               f"first {LAG_LONG} h of each building dropped",
        "scaling": "StandardScaler inside the pipeline, fit on each training fold only (rule baseline: none)",
        "missing_values": "none after the warm-up rows are dropped",
    }


def plan_experiment(model_name: str | None = None, param_overrides: dict | None = None, dataset=None) -> dict:
    """Everything that defines an experiment, WITHOUT scoring the hold-out.

    Returns the model, data, split indices, configuration and `config_id`. The Day 5
    hold-out ledger uses this to decide whether a configuration was already evaluated
    before any hold-out score is computed.
    """
    model_name = model_name or selected_model_name()
    models = build_models(load_best_params())
    if model_name not in models:
        raise TrackingError(f"unknown model {model_name!r}; choose from {sorted(models)}")
    model = models[model_name]
    if param_overrides:
        model = model.set_params(**param_overrides)
    ds = dataset or load_dataset()
    y, t = ds.y.to_numpy(), ds.time
    train_idx, test_idx = holdout_split(y, ds.groups, t, test_size=HOLDOUT_FRACTION, seed=SEED, gap=GAP)
    config = {
        "model": model_name,
        "model_class": type(model).__name__,
        "params": _simple_params(model),
        "seed": SEED,
        "split": {"holdout_fraction": HOLDOUT_FRACTION, "gap_hours": GAP, "cv": "expanding window",
                  "n_splits": N_SPLITS, "train_rows": int(len(train_idx)), "holdout_rows": int(len(test_idx))},
        "dataset": {"name": ds.name, "rows": int(len(ds.X)), "positives": int(y.sum()),
                    "fingerprint": fingerprint_dataset(ds.X, ds.y, ds.time, ds.groups, IDENTITY_DECIMALS),
                    "fingerprint_decimals": IDENTITY_DECIMALS},
        "preprocessing": preprocessing_description(),
    }
    return {"model_name": model_name, "model": model, "ds": ds, "train_idx": train_idx, "test_idx": test_idx,
            "config": config, "config_id": sha256_bytes(json.dumps(config, sort_keys=True).encode())[:16]}


def run_experiment(model_name: str | None = None, param_overrides: dict | None = None,
                   dataset=None, capture: dict | None = None, stage: str | None = None) -> dict:
    """Run one experiment end to end and return its full tracking record.

    `param_overrides` and `dataset` exist so tests can show that a changed
    configuration is detected; normal use leaves them as None. `capture`, if a dict,
    receives the fitted model and the scores (used by Day 5 so the hold-out is scored
    only once); `stage` labels the record (e.g. "final_evaluation").
    """
    plan = plan_experiment(model_name, param_overrides, dataset)
    model, ds, config = plan["model"], plan["ds"], plan["config"]
    train_idx, test_idx = plan["train_idx"], plan["test_idx"]
    X, y, t = ds.X.to_numpy(), ds.y.to_numpy(), ds.time
    Xtr, ytr, ttr = X[train_idx], y[train_idx], t[train_idx]

    idx, oof, fid = oof_scores(model, Xtr, ytr, ttr)
    folds = fold_pr_auc(ytr[idx], oof, fid)
    thr = choose_threshold(ytr[idx], oof)  # from training-side evidence only
    final = clone(model).fit(Xtr, ytr)
    s_test = _score(final, X[test_idx])
    op = operating_point(y[test_idx], s_test, thr["threshold"])

    metrics = {
        "cv_pr_auc_mean": float(np.mean(folds)),
        "cv_pr_auc_std": float(np.std(folds)),
        "cv_pr_auc_folds": [float(f) for f in folds],
        "holdout_pr_auc": float(average_precision_score(y[test_idx], s_test)),
        "threshold": thr["threshold"],
        "holdout_precision": op["precision"],
        "holdout_recall": op["recall"],
        "holdout_false_positives": op["fp"],
        "holdout_false_negatives": op["fn"],
    }
    if capture is not None:
        capture.update(plan=plan, model=final, oof_idx=idx, oof=oof, s_test=s_test, threshold=thr)
    record = {
        "run_id": None,  # assigned by log_run
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "config_id": plan["config_id"],
        "code": {**code_revision(), "source_hash": code_hash()},
        "environment": environment(),
        "metrics": metrics,
        # exact (bit-level) hash: informational only, may differ between machines
        "dataset_exact_sha256": fingerprint_dataset(ds.X, ds.y, ds.time, ds.groups),
        "holdout_scores_sha256": sha256_bytes(np.round(s_test, 6).astype(np.float64).tobytes()),
        "artifacts": {
            "results/search.json": sha256_file(SEARCH_JSON),
            "results/model_selection.json": sha256_file(SELECTION_JSON),
        },
    }
    if stage:
        record["stage"] = stage
    return record


# ---- the log ----------------------------------------------------------------
def read_log(path: Path = LOG) -> list[dict]:
    """All logged runs, oldest first. A corrupt line raises, it is never skipped silently."""
    path = Path(path)
    if not path.exists():
        return []
    runs = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise TrackingError(f"{path} line {n} is not valid JSON ({e}); the log was edited or truncated") from e
    return runs


def log_run(record: dict, path: Path = LOG) -> dict:
    """Append a record. Existing lines are never rewritten (append-only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(read_log(path)) + 1
    record = {**record, "run_id": f"run-{n:03d}-{record['config_id'][:6]}"}
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


# ---- comparison -------------------------------------------------------------
def config_differences(a: dict, b: dict) -> list[str]:
    """Names of the configuration parts that differ between two records."""
    diffs = []
    ca, cb = a["config"], b["config"]
    if ca["dataset"]["fingerprint"] != cb["dataset"]["fingerprint"]:
        diffs.append("dataset fingerprint")
    for key in ("model", "model_class", "params", "seed", "split", "preprocessing"):
        if ca[key] != cb[key]:
            diffs.append(key)
    return diffs


def compare(reference: dict, rerun: dict, tolerances: dict | None = None) -> dict:
    """Compare a re-run with a reference run.

    Raises ComparabilityError if the configurations differ. Otherwise returns
    per-metric differences and `consistent` = every difference <= its tolerance.
    A missing or non-finite metric counts as a failure, never as a pass.
    """
    tol = TOLERANCES if tolerances is None else tolerances
    diffs = config_differences(reference, rerun)
    if diffs:
        raise ComparabilityError("configurations differ (" + ", ".join(diffs) + "); "
                                 "metrics of different experiments cannot be compared")
    rows = {}
    for name, allowed in tol.items():
        if name not in reference["metrics"] or name not in rerun["metrics"]:
            raise TrackingError(f"metric {name!r} missing from a run")
        a, b = reference["metrics"][name], rerun["metrics"][name]
        finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b))
        diff = abs(a - b) if finite else float("nan")
        rows[name] = {"reference": a, "rerun": b, "abs_diff": diff, "tolerance": allowed,
                      "ok": bool(finite and diff <= allowed + 1e-12)}
    return {
        "consistent": all(r["ok"] for r in rows.values()),
        "metrics": rows,
        "max_abs_diff_pr_auc": max((rows[k]["abs_diff"] for k in ("cv_pr_auc_mean", "holdout_pr_auc") if k in rows),
                                   default=float("nan")),
        "code_changed": reference["code"]["source_hash"] != rerun["code"]["source_hash"],
        "environment_changed": reference["environment"] != rerun["environment"],
        "reference_env": {k: reference["environment"].get(k) for k in ("python", "numpy", "scikit-learn")},
        "rerun_env": {k: rerun["environment"].get(k) for k in ("python", "numpy", "scikit-learn")},
        "data_bit_identical": reference.get("dataset_exact_sha256") == rerun.get("dataset_exact_sha256"),
        "holdout_scores_identical": reference.get("holdout_scores_sha256") == rerun.get("holdout_scores_sha256"),
    }


def verify(path: Path = LOG, reference_id: str | None = None) -> dict:
    """Re-run the experiment of a logged run and compare. Latest run by default."""
    runs = read_log(path)
    if not runs:
        raise TrackingError(f"no logged runs in {path}; run `python -m evalproto.tracking run` first")
    if reference_id is None:
        ref = runs[-1]
    else:
        match = [r for r in runs if r["run_id"] == reference_id]
        if not match:
            raise TrackingError(f"run {reference_id!r} not found in {path}")
        ref = match[0]
    rerun = run_experiment(ref["config"]["model"])
    return {"reference_run": ref["run_id"], "rerun": rerun, **compare(ref, rerun)}


# ---- command line -----------------------------------------------------------
def _print_run(r: dict) -> None:
    m, c = r["metrics"], r["code"]
    print(f"{r['run_id']}  model={r['config']['model']}  data={r['config']['dataset']['fingerprint'][:10]}  "
          f"commit={c['git_commit'][:8]}{'*' if c['git_dirty'] else ''}  "
          f"CV PR-AUC={m['cv_pr_auc_mean']:.4f}  hold-out PR-AUC={m['holdout_pr_auc']:.4f}  "
          f"P={m['holdout_precision']:.3f} R={m['holdout_recall']:.3f}")


def _print_comparison(res: dict) -> None:
    print(f"\nre-run vs {res['reference_run']}")
    print(f"{'metric':26}{'reference':>12}{'re-run':>12}{'|diff|':>10}{'tolerance':>11}  ok")
    for k, r in res["metrics"].items():
        print(f"{k:26}{r['reference']:>12.5f}{r['rerun']:>12.5f}{r['abs_diff']:>10.6f}{r['tolerance']:>11}  "
              f"{'yes' if r['ok'] else 'NO'}")
    print(f"environment: reference {res['reference_env']} | re-run {res['rerun_env']}")
    print(f"data bit-identical: {res['data_bit_identical']}; "
          f"hold-out scores bit-identical (6 dp): {res['holdout_scores_identical']}; "
          f"code changed: {res['code_changed']}; environment changed: {res['environment_changed']}")
    print("RESULT:", "consistent within tolerance" if res["consistent"] else "NOT consistent")


def describe_data() -> None:
    """Print the dataset fingerprint at several precisions plus column sums (to compare machines)."""
    ds = load_dataset()
    print("env:", environment())
    print("rows", len(ds.X), "positives", int(ds.y.sum()))
    print("labels/time/groups hash:", sha256_bytes(np.asarray(ds.y, dtype=np.int64).tobytes() +
          np.asarray(ds.time, dtype=np.int64).tobytes() + np.asarray(ds.groups, dtype=np.int64).tobytes())[:16])
    for d in (None, 12, 10, 8, 6, 4):
        print(f"fingerprint decimals={d}:", fingerprint_dataset(ds.X, ds.y, ds.time, ds.groups, d)[:16])
    for c in ds.X.columns:
        print(f"sum {c:20s} {ds.X[c].sum():.12f}")
    print("first row:", [f"{v:.17g}" for v in ds.X.iloc[0]])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "run"
    try:
        if cmd == "run":
            rec = log_run(run_experiment())
            _print_run(rec)
            print(f"logged to {LOG.relative_to(ROOT)}")
            return 0
        if cmd == "verify":
            res = verify()
            _print_comparison(res)
            return 0 if res["consistent"] else 1
        if cmd == "data":
            describe_data()
            return 0
        if cmd == "show":
            for r in read_log():
                _print_run(r)
            return 0
        print(__doc__)
        return 2
    except (TrackingError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
