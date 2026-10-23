"""Inference pipeline and model package (Day 5).

Turns RAW hourly meter readings into anomaly alerts using the frozen final model:

    raw readings -> validation -> causal features -> model score -> alert (score >= threshold)

Raw input columns (one row per building and hour):
    building        any id; each building is scored on its own history
    hour            integer hour index; hour 0 = Monday 00:00 (hour of day = hour % 24,
                    weekend = (hour // 24) % 7 >= 5), same convention as the training data
    temperature     degrees C
    building_size   positive number
    consumption     > 0 (the features use its logarithm)

Rules (each one raises `InputError` with a message, nothing is silently repaired):
  * required columns present, no missing / infinite values, consumption > 0
  * no duplicate (building, hour); hours of a building must be consecutive (the 24 h and
    168 h lags are taken by row position, so a gap would silently give wrong features)
  * the first 168 hours of a building cannot be scored: status `insufficient_history`
  * values outside the range seen in training are still scored but carry a warning

A package on disk is `model.joblib` + `package.json`. Loading checks the model file hash and
the scikit-learn version, because a pickle that was altered or built by another version can
fail silently or run foreign code. Only load packages you built yourself.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .data import FEATURES, LAG_LONG, LAG_SHORT

REQUIRED_COLUMNS = ["building", "hour", "temperature", "building_size", "consumption"]
RANGE_FEATURES = ["temperature", "building_size", "log_consumption"]
RANGE_SLACK = 1e-9  # relative tolerance of the training-range check


class InputError(ValueError):
    """The raw readings cannot be scored (bad schema, values or hour gaps)."""


class PackageError(ValueError):
    """The model package is missing, altered, or built with another library version."""


# ---- validation and features ------------------------------------------------
def validate_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy sorted by (building, hour) or raise InputError."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise InputError("input must be a non-empty DataFrame")
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise InputError(f"missing columns: {missing}")
    df = raw.copy()
    for c in ("hour", "temperature", "building_size", "consumption"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[REQUIRED_COLUMNS].isna().any(axis=1) | ~np.isfinite(
        df[["hour", "temperature", "building_size", "consumption"]].to_numpy(dtype=float)).all(axis=1)
    if bad.any():
        raise InputError(f"{int(bad.sum())} row(s) have missing, non-numeric or infinite values")
    if (df["consumption"] <= 0).any():
        raise InputError("consumption must be > 0 (features use its logarithm)")
    if (df["building_size"] <= 0).any():
        raise InputError("building_size must be > 0")
    if not (df["hour"] == np.floor(df["hour"])).all() or (df["hour"] < 0).any():
        raise InputError("hour must be a non-negative integer hour index")
    df["hour"] = df["hour"].astype(np.int64)
    if df.duplicated(["building", "hour"]).any():
        raise InputError("duplicate (building, hour) rows")
    df = df.sort_values(["building", "hour"]).reset_index(drop=True)
    gaps = df.groupby("building")["hour"].diff().dropna()
    if (gaps != 1).any():
        raise InputError("hours of a building must be consecutive (found a gap); lag features would be wrong")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal features exactly as in training (data.make_features), for validated input.

    Rows without a full 168 h history keep NaN lag features and are marked
    `insufficient_history` instead of being dropped.
    """
    out = df.copy()
    hod = out["hour"] % 24
    out["is_weekend"] = (((out["hour"] // 24) % 7) >= 5).astype(int)
    g = out.groupby("building")["consumption"]
    out["log_consumption"] = np.log(out["consumption"])
    for lag, name in ((LAG_SHORT, "24"), (LAG_LONG, "168")):
        out[f"log_ratio_{name}"] = np.log(out["consumption"] / g.shift(lag))
        out[f"abs_log_ratio_{name}"] = out[f"log_ratio_{name}"].abs()
    out["hod_sin"] = np.sin(2 * np.pi * hod / 24)
    out["hod_cos"] = np.cos(2 * np.pi * hod / 24)
    out["status"] = np.where(out[FEATURES].isna().any(axis=1), "insufficient_history", "scored")
    return out


# ---- the pipeline -----------------------------------------------------------
@dataclass
class InferencePipeline:
    model: object
    threshold: float
    ranges: dict = field(default_factory=dict)  # feature -> [min, max] seen in training
    meta: dict = field(default_factory=dict)

    def score_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Features, score, alert, status and warning for every validated input row."""
        df = compute_features(validate_raw(raw))
        ok = df["status"] == "scored"
        df["score"] = np.nan
        if ok.any():
            df.loc[ok, "score"] = self.model.predict_proba(df.loc[ok, FEATURES].to_numpy())[:, 1]
        df["alert"] = ok & (df["score"] >= self.threshold)
        warn = pd.Series("", index=df.index)
        for feat, (lo, hi) in self.ranges.items():
            if feat in df.columns:
                slack = RANGE_SLACK * max(1.0, abs(lo), abs(hi))  # CSV/JSON round trips change floats at ~1e-14
                out_of = ok & ((df[feat] < lo - slack) | (df[feat] > hi + slack))
                warn = warn.where(~out_of, warn + f"out_of_range:{feat};")
        df["warning"] = warn
        return df

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Compact result: building, hour, score, alert, status, warning."""
        return self.score_frame(raw)[["building", "hour", "score", "alert", "status", "warning"]]


# ---- package on disk --------------------------------------------------------
def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_package(out_dir: Path, pipeline: InferencePipeline, meta: dict) -> Path:
    """Write model.joblib and package.json (metadata + model hash + library versions)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(pipeline.model, model_path)
    info = {
        **meta,
        "threshold": float(pipeline.threshold),
        "features": list(FEATURES),
        "ranges": pipeline.ranges,
        "required_columns": REQUIRED_COLUMNS,
        "model_sha256": sha256_file(model_path),
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    (out_dir / "package.json").write_text(json.dumps(info, indent=2, sort_keys=True))
    return out_dir


def load_package(out_dir: Path, allow_version_mismatch: bool = False) -> InferencePipeline:
    out_dir = Path(out_dir)
    meta_path, model_path = out_dir / "package.json", out_dir / "model.joblib"
    if not meta_path.exists() or not model_path.exists():
        raise PackageError(f"{out_dir} is not a complete package (needs package.json and model.joblib)")
    meta = json.loads(meta_path.read_text())
    if sha256_file(model_path) != meta.get("model_sha256"):
        raise PackageError("model.joblib does not match the hash in package.json (file was changed)")
    if meta.get("scikit_learn_version") != sklearn.__version__ and not allow_version_mismatch:
        raise PackageError(f"package was built with scikit-learn {meta.get('scikit_learn_version')}, "
                           f"this environment has {sklearn.__version__}; rebuild it (python -m evalproto.final build)")
    if meta.get("features") != list(FEATURES):
        raise PackageError("package feature list differs from the code's FEATURES; rebuild the package")
    model = joblib.load(model_path)
    return InferencePipeline(model=model, threshold=float(meta["threshold"]),
                             ranges=meta.get("ranges", {}), meta=meta)
