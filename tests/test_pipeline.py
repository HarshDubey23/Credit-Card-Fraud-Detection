"""Tests run on synthetic data shaped like the real dataset, so CI needs no download."""
import numpy as np
import pandas as pd
import pytest

from fraud import data as D
from fraud import evaluate as E
from fraud import models as M


def synthetic(n=6000, fraud_rate=0.02, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    X = rng.normal(size=(n, 28)) + y[:, None] * rng.normal(1.5, 0.2, size=28)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    df["Amount"] = rng.gamma(2, 40, n)
    df["Class"] = y
    return df


@pytest.fixture
def csv(tmp_path):
    df = synthetic()
    df = pd.concat([df, df.iloc[:25]])  # inject exact duplicates
    p = tmp_path / "creditcard.csv"
    df.to_csv(p, index=False)
    return p


def test_load_drops_duplicates_and_validates(csv):
    df, info = D.load(csv)
    assert info["exact_duplicates_dropped"] == 25
    assert not df.duplicated().any()
    assert info["rows"] == len(df)


def test_load_rejects_wrong_schema(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "Class": [0]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        D.load(p)


def test_temporal_split_is_chronological_and_disjoint(csv):
    df, _ = D.load(csv)
    s = D.temporal_split(df)
    assert len(s.X_train) + len(s.X_val) + len(s.X_test) == len(df)
    # Time is not a feature, so recover it by index order: every test row is later than every train row
    ordered = df.sort_values("Time", kind="stable").reset_index(drop=True)
    a, b = len(s.X_train), len(s.X_train) + len(s.X_val)
    assert ordered["Time"].iloc[:a].max() <= ordered["Time"].iloc[b:].min()
    assert "Time" not in s.X_train.columns and "Class" not in s.X_train.columns


def test_stratified_split_preserves_rate(csv):
    df, _ = D.load(csv)
    s = D.stratified_split(df, seed=1)
    rates = [s.y_train.mean(), s.y_val.mean(), s.y_test.mean()]
    assert max(rates) - min(rates) < 0.01


def test_scaler_is_fit_on_train_only(csv):
    df, _ = D.load(csv)
    s = D.temporal_split(df)
    model = M.build("logreg_balanced", 0, 1.0).fit(s.X_train, s.y_train)
    np.testing.assert_allclose(model.named_steps["scale"].mean_, s.X_train.mean().to_numpy(), rtol=1e-9)


@pytest.mark.parametrize("name", M.MODELS)
def test_every_model_fits_and_scores(csv, name):
    df, _ = D.load(csv)
    s = D.temporal_split(df)
    pw = float((s.y_train == 0).sum() / (s.y_train == 1).sum())
    p = M.build(name, 0, pw).fit(s.X_train, s.y_train).predict_proba(s.X_test)[:, 1]
    assert p.shape == (len(s.X_test),) and np.isfinite(p).all()
    if name not in ("prior", "isolation_forest"):
        assert E.summary(s.y_test, p)["roc_auc"] > 0.9  # signal is strong in the synthetic data


def test_threshold_max_f1_on_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = E.threshold_max_f1(y, p)
    assert E.at_threshold(y, p, t)["f1"] == 1.0


def test_min_cost_threshold_tracks_review_cost():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    p = np.clip(0.5 * y + rng.random(5000) * 0.6, 0, 1)
    amount = rng.gamma(2, 50, 5000)
    cheap = E.threshold_min_cost(y, p, amount, review_cost=0.5)
    pricey = E.threshold_min_cost(y, p, amount, review_cost=200)
    assert cheap <= pricey  # expensive reviews => fewer alerts


def test_at_threshold_counts_and_cost():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    amt = np.array([100.0, 50.0, 10.0, 10.0])
    r = E.at_threshold(y, p, 0.5, amt, review_cost=5)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 1)
    assert r["missed_fraud_amount"] == 50.0 and r["cost"] == 50.0 + 5 * 2


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.05).astype(int)
    p = np.clip(0.4 * y + rng.random(4000) * 0.7, 0, 1)
    b = E.bootstrap(y, {"m": p}, {"m": 0.5}, n=200)
    lo, hi = b["m"]["pr_auc"]
    assert lo <= E.summary(y, p)["pr_auc"] <= hi


def test_ece_is_zero_for_perfectly_calibrated_bins():
    p = np.repeat([0.1, 0.5, 0.9], 1000)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(900)], np.r_[np.ones(500), np.zeros(500)],
                        np.r_[np.ones(900), np.zeros(100)]])
    assert E.ece(y, p, bins=3) < 1e-9
