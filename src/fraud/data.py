"""Dataset loading, validation and leakage-safe splitting."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
# SHA-256 of the public ULB/Kaggle release (284,807 rows, 492 frauds).
KNOWN_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
FEATURES = [c for c in EXPECTED_COLUMNS if c not in ("Time", "Class")]


@dataclass(frozen=True)
class Split:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    description: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load the CSV, validate its schema and drop exact duplicate rows.

    Duplicates are dropped *before* splitting so the same transaction cannot
    appear in both train and test.
    """
    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected columns: {list(df.columns)}")
    if df.isna().any().any():
        raise ValueError("dataset contains missing values")
    if not set(df["Class"].unique()) <= {0, 1}:
        raise ValueError("Class must be binary")
    n_raw, frauds_raw = len(df), int(df["Class"].sum())
    dup_mask = df.duplicated()
    df = df.loc[~dup_mask].reset_index(drop=True)
    info = {
        "rows_raw": n_raw,
        "frauds_raw": frauds_raw,
        "exact_duplicates_dropped": int(dup_mask.sum()),
        "rows": len(df),
        "frauds": int(df["Class"].sum()),
        "fraud_rate": float(df["Class"].mean()),
        "time_span_hours": float(df["Time"].max() / 3600),
    }
    return df, info


def temporal_split(df: pd.DataFrame, train: float = 0.6, val: float = 0.2) -> Split:
    """Chronological split: fit on the past, select thresholds on the next window, test on the last.

    This is the primary protocol because a deployed model only ever scores
    transactions that occur after the data it was trained on.
    """
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(df)
    a, b = int(n * train), int(n * (train + val))
    parts = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    return Split(
        *(x for p in parts for x in (p[FEATURES], p["Class"].to_numpy())),
        description=f"temporal {train:.0%}/{val:.0%}/{1 - train - val:.0%} by Time",
    )


def stratified_split(df: pd.DataFrame, seed: int, train: float = 0.6, val: float = 0.2) -> Split:
    """Random stratified split, used only as a secondary robustness check."""
    from sklearn.model_selection import train_test_split

    rest = 1 - train
    tr, tmp = train_test_split(df, test_size=rest, stratify=df["Class"], random_state=seed)
    va, te = train_test_split(tmp, test_size=(rest - val) / rest, stratify=tmp["Class"], random_state=seed)
    parts = tr, va, te
    return Split(
        *(x for p in parts for x in (p[FEATURES], p["Class"].to_numpy())),
        description=f"stratified random {train:.0%}/{val:.0%}/{rest - val:.0%}, seed {seed}",
    )
