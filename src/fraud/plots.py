"""Figures generated from the actual run. Palette: validated categorical slots 1-3 + neutral gray."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
})

LABEL = {"prior": "Prior (no skill)", "isolation_forest": "Isolation Forest (original notebook)",
         "logreg": "Logistic regression", "logreg_balanced": "Logistic regression, class-weighted",
         "logreg_smote": "Logistic regression + SMOTE", "hist_gb": "Hist. gradient boosting",
         "xgboost": "XGBoost", "xgboost_weighted": "XGBoost, class-weighted", "xgboost_smote": "XGBoost + SMOTE"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pr_curves(out, split, scores, path):
    best = out["selected_model"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, color in ((best, BLUE), ("logreg_balanced", ORANGE), ("isolation_forest", AQUA)):
        if name not in scores:
            continue
        prec, rec, _ = precision_recall_curve(split.y_test, scores[name])
        ap = out["models"][name]["runs"][0]["test"]["pr_auc"]
        ax.plot(rec, prec, color=color, label=f"{LABEL[name]}  (AP {ap:.3f})")
    base = split.y_test.mean()
    ax.axhline(base, color=GRAY, linestyle="--", linewidth=1.2, label=f"Prior ({base:.4f})")
    ax.set(xlabel="Recall (share of frauds caught)", ylabel="Precision (share of alerts that are fraud)",
           xlim=(0, 1), ylim=(0, 1.02), title="Precision-recall on the held-out final 20% of time")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.18), fontsize=8.5)
    _save(fig, path)


def model_comparison(out, path):
    names = [m for m in out["models"] if m != "prior"]
    names.sort(key=lambda m: out["models"][m]["runs"][0]["test"]["pr_auc"])
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(names) + 1.2))
    y = np.arange(len(names))
    for i, m in enumerate(names):
        ap = out["models"][m]["runs"][0]["test"]["pr_auc"]
        lo, hi = out["bootstrap_ci95_seed0"][m]["pr_auc"]
        color = BLUE if m == out["selected_model"] else GRAY
        ax.barh(i, ap, height=0.55, color=color)
        ax.plot([lo, hi], [i, i], color=INK, linewidth=1.2)
        ax.text(hi + 0.01, i, f"{ap:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [LABEL[m] for m in names], fontsize=8.5)
    ax.set(xlim=(0, 1), xlabel="Test PR-AUC (bars) with 95% bootstrap CI (lines)",
           title="Model comparison, temporal split (seed 0)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def calibration(out, split, p_raw, p_cal, path, bins=8):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    top = 4 * int(split.y_test.sum())  # nearly all frauds sit in the top-scored few hundred rows
    for p, color, label in ((p_raw, ORANGE, "Uncalibrated"), (p_cal, BLUE, "Isotonic (fit on validation)")):
        order = np.argsort(p)
        # equal-mass bins over the top-scored rows; the remaining ~99.5% all score near 0
        idx = np.array_split(order[-top:], bins)
        xs = [p[i].mean() for i in idx]
        ys = [split.y_test[i].mean() for i in idx]
        ax.plot(xs, ys, color=color, marker="o", markersize=5, label=label)
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.2, label="Perfect calibration")
    c = out["calibration"]
    ax.set(xlabel="Mean predicted fraud probability", ylabel="Observed fraud rate",
           title=f"Reliability, top {top} test scores ({LABEL[c['model']]})", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _save(fig, path)


def cost_sensitivity(out, path):
    s = out["review_cost_sensitivity"]
    c = [r["review_cost"] for r in s]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(c, [r["recall"] for r in s], color=BLUE, marker="o", label="Recall")
    ax.plot(c, [r["precision"] for r in s], color=ORANGE, marker="o", label="Precision")
    ax.plot(c, [r["fraud_amount_caught_share"] for r in s], color=AQUA, marker="o",
            label="Share of fraud amount caught")
    ax.set_xscale("log")
    ax.set_xticks(c, [f"€{x:g}" for x in c])
    ax.set(ylim=(0, 1.02), xlabel="Assumed cost of one manual review (threshold chosen on validation)",
           ylabel="Test-set value", title="How the operating point moves with review cost")
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.42))
    _save(fig, path)


def confusion(out, path):
    best = out["selected_model"]
    r = out["models"][best]["runs"][0]["test_at_val_f1_threshold"]
    m = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(np.log10(m + 1), cmap="Blues", vmin=0, vmax=np.log10(m.max() + 1) * 1.15)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12,
                color="white" if v > 1000 else INK)
    ax.set_xticks([0, 1], ["Pred. legit", "Pred. fraud"])
    ax.set_yticks([0, 1], ["Legit", "Fraud"])
    ax.grid(False)
    ax.set_title(f"{LABEL[best]}, val-F1 threshold", fontsize=10)
    _save(fig, path)


def make_all(out, split, scores, p_cal, fig_dir: Path):
    pr_curves(out, split, scores, fig_dir / "pr_curves.png")
    model_comparison(out, fig_dir / "model_comparison.png")
    calibration(out, split, scores[out["selected_model"]], p_cal, fig_dir / "calibration.png")
    cost_sensitivity(out, fig_dir / "cost_sensitivity.png")
    confusion(out, fig_dir / "confusion_matrix.png")
