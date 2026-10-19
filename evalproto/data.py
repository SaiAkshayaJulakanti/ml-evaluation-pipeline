"""Dataset selection.

Dataset: SIMULATED hourly energy-meter data (20 buildings, 150 days).
Target : `anomaly` = 1 when the hour is inside an injected consumption event
         (spike 1.5-2.5x or drop to 0.2-0.5x of expected, lasting 1-4 hours).

It is simulated (deterministic, seed 42) because this sandbox cannot download
real data. The structure mirrors real efficiency-monitoring data, and it is
the structure that drives the split policy (see docs/SPLIT_POLICY.md):

  * hourly timestamps         -> time-aware validation, with a gap
  * many buildings (groups)   -> same buildings appear in train and test by
                                 design (forecasting known sites)
  * rare target (~3%)         -> PR-AUC / recall, never plain accuracy
  * temperature drifts down over the period -> later data differs from earlier
    data, so a random split would be optimistic

All features are causal: they use only the current and past hours of the same
building (tested in tests/test_protocol.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 42
N_BUILDINGS = 20
N_DAYS = 150
LAG_SHORT, LAG_LONG = 24, 168  # same hour yesterday / same hour last week
WARMUP = LAG_LONG  # rows without a full lag window are dropped


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    groups: np.ndarray | None  # building id; None => rows are independent
    time: np.ndarray | None  # integer hour index; None => no temporal ordering
    name: str


def simulate_raw(seed: int = SEED, n_buildings: int = N_BUILDINGS, n_days: int = N_DAYS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_hours = n_days * 24
    hours = np.arange(n_hours)
    hod, day = hours % 24, hours // 24
    dow = day % 7
    # shared weather: seasonal cooling (25C -> 7C) plus a daily swing
    base_temp = 16 + 9 * np.cos(2 * np.pi * day / 300) + 4 * np.sin(2 * np.pi * (hod - 9) / 24)
    workhour = ((hod >= 8) & (hod < 18) & (dow < 5)).astype(float)

    frames = []
    for b in range(n_buildings):
        size = rng.lognormal(np.log(100), 0.5)
        temp = base_temp + rng.normal(0, 1.5, n_hours) + rng.normal(0, 1)
        expected = size * (0.3 + 0.7 * workhour) * (1 + 0.03 * np.abs(temp - 18))
        cons = expected * (1 + rng.normal(0, 0.05, n_hours))
        label = np.zeros(n_hours, dtype=int)
        t = 0
        while t < n_hours:
            if rng.random() < 0.012:
                length = int(rng.integers(1, 5))
                factor = rng.uniform(1.5, 2.5) if rng.random() < 0.7 else rng.uniform(0.2, 0.5)
                cons[t : t + length] *= factor
                label[t : t + length] = 1
                t += length
            else:
                t += 1
        frames.append(
            pd.DataFrame(
                {
                    "building": b,
                    "hour": hours,
                    "hod": hod,
                    "is_weekend": (dow >= 5).astype(int),
                    "temperature": temp,
                    "building_size": size,
                    "consumption": cons,
                    "anomaly": label,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Causal feature engineering: row (building, hour) uses hours <= `hour` only."""
    df = raw.sort_values(["building", "hour"]).reset_index(drop=True)
    g = df.groupby("building")["consumption"]
    df["log_consumption"] = np.log(df["consumption"])
    for lag, name in ((LAG_SHORT, "24"), (LAG_LONG, "168")):
        df[f"log_ratio_{name}"] = np.log(df["consumption"] / g.shift(lag))
        df[f"abs_log_ratio_{name}"] = df[f"log_ratio_{name}"].abs()
    df["hod_sin"] = np.sin(2 * np.pi * df["hod"] / 24)
    df["hod_cos"] = np.cos(2 * np.pi * df["hod"] / 24)
    df = df.dropna().sort_values(["hour", "building"]).reset_index(drop=True)
    return df


FEATURES = [
    "temperature",
    "building_size",
    "hod_sin",
    "hod_cos",
    "is_weekend",
    "log_consumption",
    "log_ratio_24",
    "log_ratio_168",
    "abs_log_ratio_24",
    "abs_log_ratio_168",
]


def load_dataset() -> Dataset:
    df = make_features(simulate_raw())
    return Dataset(
        X=df[FEATURES].copy(),
        y=df["anomaly"].rename("anomaly"),
        groups=df["building"].to_numpy(),
        time=df["hour"].to_numpy(dtype=np.int64),
        name="simulated_energy_meter",
    )


if __name__ == "__main__":  # write a CSV for inspection (data/ is git-ignored)
    from pathlib import Path

    ds = load_dataset()
    out = Path(__file__).resolve().parent.parent / "data" / "energy_meter.csv"
    out.parent.mkdir(exist_ok=True)
    full = ds.X.assign(building=ds.groups, hour=ds.time, anomaly=ds.y.to_numpy())
    full.to_csv(out, index=False)
    print(f"wrote {out} ({len(full)} rows)")
