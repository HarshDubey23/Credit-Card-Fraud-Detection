"""Metrics, threshold selection, calibration and bootstrap uncertainty."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score)


def threshold_max_f1(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def threshold_min_cost(y: np.ndarray, p: np.ndarray, amount: np.ndarray, review_cost: float) -> float:
    """Threshold minimising (missed fraud amount + review_cost * alerts) on the given split.

    A missed fraud costs its transaction amount; every alert (TP or FP) costs one
    manual review. This is a stylised cost model, documented as such in the README.
    """
    order = np.argsort(-p)
    p_s, y_s, a_s = p[order], y[order], amount[order]
    # flagging the top-k transactions: cost = fraud amount not in top-k + review_cost * k
    fraud_amt = a_s * y_s
    missed = fraud_amt.sum() - np.cumsum(fraud_amt)
    cost = missed + review_cost * np.arange(1, len(p_s) + 1)
    k = int(np.argmin(cost))
    return float(p_s[k])


def at_threshold(y: np.ndarray, p: np.ndarray, t: float, amount: np.ndarray | None = None,
                 review_cost: float | None = None) -> dict:
    pred = p >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out = {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "precision": precision, "recall": recall,
           "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
           "alerts_per_1k": 1000 * (tp + fp) / len(y)}
    if amount is not None and review_cost is not None:
        missed = float(np.sum(amount[~pred & (y == 1)]))
        out["missed_fraud_amount"] = missed
        out["cost"] = missed + review_cost * (tp + fp)
        out["fraud_amount_caught_share"] = 1 - missed / max(float(np.sum(amount[y == 1])), 1e-12)
    return out


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins (equal-width bins are empty here)."""
    order = np.argsort(p)
    total = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            total += len(idx) * abs(p[idx].mean() - y[idx].mean())
    return total / len(y)


def summary(y: np.ndarray, p: np.ndarray) -> dict:
    return {"pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
            "ece": float(ece(y, p))}


def bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], thresholds: dict[str, float],
              n: int = 1000, seed: int = 0) -> dict:
    """Percentile 95% CIs for PR-AUC and recall@threshold; paired across models."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    samples = {m: {"pr_auc": [], "recall": []} for m in scores}
    for _ in range(n):
        # stratified resample keeps the (tiny) number of positives stable
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yb = y[idx]
        for m, p in scores.items():
            pb = p[idx]
            samples[m]["pr_auc"].append(average_precision_score(yb, pb))
            samples[m]["recall"].append(float(np.mean(pb[yb == 1] >= thresholds[m])))
    out = {}
    for m, s in samples.items():
        out[m] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in s.items()}
        out[m]["_pr_auc_samples"] = np.asarray(s["pr_auc"])
    return out


def paired_diff(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_a_not_better": float(np.mean(d <= 0))}
