"""Dataset selection.

Dataset: Wisconsin Diagnostic Breast Cancer (bundled with scikit-learn, so it
works offline and is reproducible). Target: malignant (1) vs benign (0).

Data structure facts that drive the split policy (see docs/SPLIT_POLICY.md):
  * one row per patient sample  -> no repeated entities, no group column
  * no timestamp column         -> no temporal ordering
  * imbalanced binary target    -> stratification is required
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    groups: np.ndarray | None  # None => rows are independent
    time: np.ndarray | None  # None => no temporal ordering
    name: str


def load_dataset() -> Dataset:
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()
    # sklearn encodes malignant=0; flip so the positive (rare, costly) class is
    # malignant=1 and recall/precision refer to cancer detection.
    y = (1 - bunch.target).rename("malignant")
    return Dataset(X=X, y=y, groups=None, time=None, name="wdbc_breast_cancer")
