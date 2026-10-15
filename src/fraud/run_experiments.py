"""Run the full evaluation protocol and write results/ and figures/.

Usage:
    python -m fraud.run_experiments --data path/to/creditcard.csv [--seeds 5] [--bootstrap 1000]
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import xgboost
from sklearn.isotonic import IsotonicRegression

from . import data as D
from . import evaluate as E
from . import models as M
from . import plots

ROOT = Path(__file__).resolve().parents[2]
STOCHASTIC = {"isolation_forest", "logreg_smote", "hist_gb", "xgboost", "xgboost_weighted", "xgboost_smote"}


def fit_score(name: str, split: D.Split, seed: int):
    pos_weight = float((split.y_train == 0).sum() / max((split.y_train == 1).sum(), 1))
    t0 = time.perf_counter()
    model = M.build(name, seed, pos_weight).fit(split.X_train, split.y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_test = model.predict_proba(split.X_test)[:, 1]
    infer_us = 1e6 * (time.perf_counter() - t0) / len(split.X_test)
    p_val = model.predict_proba(split.X_val)[:, 1]
    return model, p_val, p_test, fit_s, infer_us


def evaluate_model(name, split, p_val, p_test, review_cost):
    a_val, a_test = split.X_val["Amount"].to_numpy(), split.X_test["Amount"].to_numpy()
    t_f1 = E.threshold_max_f1(split.y_val, p_val)
    t_cost = E.threshold_min_cost(split.y_val, p_val, a_val, review_cost)
    return {
        "val": E.summary(split.y_val, p_val),
        "test": E.summary(split.y_test, p_test),
        "test_at_val_f1_threshold": E.at_threshold(split.y_test, p_test, t_f1, a_test, review_cost),
        "test_at_val_cost_threshold": E.at_threshold(split.y_test, p_test, t_cost, a_test, review_cost),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--review-cost", type=float, default=5.0,
                    help="assumed cost of one manual review, in the dataset's currency (EUR)")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--models", nargs="*", default=M.MODELS)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    t_start = time.time()
    df, info = D.load(args.data)
    info["sha256"] = D.sha256(args.data)
    info["sha256_matches_public_release"] = info["sha256"] == D.KNOWN_SHA256
    split = D.temporal_split(df)
    info["split"] = {"protocol": split.description,
                     **{k: {"rows": len(y), "frauds": int(y.sum())}
                        for k, y in (("train", split.y_train), ("val", split.y_val), ("test", split.y_test))}}
    print(json.dumps(info, indent=2))

    # --- 1. main comparison on the temporal split, several seeds ---------------------------
    results, seed0_scores, seed0_val, fitted = {}, {}, {}, {}
    for name in args.models:
        seeds = range(args.seeds) if name in STOCHASTIC else [0]
        runs = []
        for seed in seeds:
            model, p_val, p_test, fit_s, infer_us = fit_score(name, split, seed)
            r = evaluate_model(name, split, p_val, p_test, args.review_cost)
            r.update(seed=seed, fit_seconds=fit_s, inference_us_per_tx=infer_us)
            runs.append(r)
            if seed == 0:
                seed0_scores[name], seed0_val[name], fitted[name] = p_test, p_val, model
        pr = [r["test"]["pr_auc"] for r in runs]
        results[name] = {"runs": runs, "test_pr_auc_mean": float(np.mean(pr)),
                         "test_pr_auc_sd": float(np.std(pr, ddof=1)) if len(pr) > 1 else 0.0,
                         "n_seeds": len(runs)}
        print(f"{name:18s} test PR-AUC {np.mean(pr):.3f} ± {results[name]['test_pr_auc_sd']:.3f}  "
              f"(val {np.mean([r['val']['pr_auc'] for r in runs]):.3f})")

    # model selection uses VALIDATION PR-AUC only
    selectable = [m for m in results if m != "prior"]
    best = max(selectable, key=lambda m: np.mean([r["val"]["pr_auc"] for r in results[m]["runs"]]))
    print("selected on validation:", best)

    # --- 2. bootstrap CIs on seed-0 test scores + paired comparisons -------------------------
    thr = {m: results[m]["runs"][0]["test_at_val_f1_threshold"]["threshold"] for m in seed0_scores}
    boot = E.bootstrap(split.y_test, seed0_scores, thr, n=args.bootstrap)
    pairs = {}
    for a, b in [(best, "logreg_balanced"), ("xgboost", "xgboost_weighted"), ("xgboost", "xgboost_smote"),
                 ("logreg", "logreg_smote"), (best, "isolation_forest")]:
        if a in boot and b in boot and a != b:
            pairs[f"{a} - {b}"] = E.paired_diff(boot[a]["_pr_auc_samples"], boot[b]["_pr_auc_samples"])
    for m in boot:
        boot[m].pop("_pr_auc_samples")

    # --- 3. calibration of the selected model (isotonic fit on validation only) -------------
    iso = IsotonicRegression(out_of_bounds="clip").fit(seed0_val[best], split.y_val)
    p_cal_test = iso.predict(seed0_scores[best])
    p_cal_val = iso.predict(seed0_val[best])
    calibration = {
        "model": best,
        "uncalibrated_test": E.summary(split.y_test, seed0_scores[best]),
        "isotonic_test": E.summary(split.y_test, p_cal_test),
        "isotonic_test_at_val_f1_threshold": E.at_threshold(
            split.y_test, p_cal_test, E.threshold_max_f1(split.y_val, p_cal_val)),
    }

    # --- 4. review-cost sensitivity for the selected model ----------------------------------
    a_val, a_test = split.X_val["Amount"].to_numpy(), split.X_test["Amount"].to_numpy()
    sensitivity = []
    for c in (1, 2, 5, 10, 25, 50):
        t = E.threshold_min_cost(split.y_val, seed0_val[best], a_val, c)
        sensitivity.append({"review_cost": c, **E.at_threshold(split.y_test, seed0_scores[best], t, a_test, c)})

    # --- 5. robustness: how much does a random split flatter the models? --------------------
    random_split = {}
    for name in ("logreg_balanced", best):
        vals = []
        for seed in range(args.seeds):
            s = D.stratified_split(df, seed)
            _, _, p_test, _, _ = fit_score(name, s, seed)
            vals.append(E.summary(s.y_test, p_test)["pr_auc"])
        random_split[name] = {"pr_auc_mean": float(np.mean(vals)), "pr_auc_sd": float(np.std(vals, ddof=1)),
                              "temporal_pr_auc_mean": results[name]["test_pr_auc_mean"]}

    # --- 6. error analysis on the selected model ---------------------------------------------
    t_best = thr[best]
    y, p = split.y_test, seed0_scores[best]
    tp_amt, fn_amt = a_test[(y == 1) & (p >= t_best)], a_test[(y == 1) & (p < t_best)]
    fp_amt = a_test[(y == 0) & (p >= t_best)]
    buckets = [0, 1, 10, 100, 500, np.inf]
    by_amount = []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        m = (a_test >= lo) & (a_test < hi) & (y == 1)
        caught = int(np.sum(m & (p >= t_best)))
        by_amount.append({"amount_range": f"[{lo:g}, {hi:g})", "frauds": int(m.sum()), "caught": caught,
                          "recall": caught / m.sum() if m.sum() else None})
    errors = {"model": best, "threshold": t_best,
              "fraud_amount_median_caught": float(np.median(tp_amt)) if len(tp_amt) else None,
              "fraud_amount_median_missed": float(np.median(fn_amt)) if len(fn_amt) else None,
              "false_positive_amount_median": float(np.median(fp_amt)) if len(fp_amt) else None,
              "recall_by_amount": by_amount}
    importance = None
    if hasattr(fitted[best], "feature_importances_"):
        imp = fitted[best].feature_importances_
        importance = sorted(zip(D.FEATURES, map(float, imp)), key=lambda x: -x[1])[:10]
        errors["top_features_gain"] = importance

    out = {"dataset": info, "selected_model": best, "review_cost_assumption_eur": args.review_cost,
           "models": results, "bootstrap_ci95_seed0": boot, "paired_bootstrap_pr_auc": pairs,
           "calibration": calibration, "review_cost_sensitivity": sensitivity,
           "random_vs_temporal_split": random_split, "error_analysis": errors,
           "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                           "xgboost": xgboost.__version__, "platform": platform.platform(),
                           "runtime_seconds": round(time.time() - t_start, 1),
                           "n_bootstrap": args.bootstrap, "n_seeds": args.seeds}}
    (args.out / "metrics.json").write_text(json.dumps(out, indent=2))
    plots.make_all(out, split, seed0_scores, p_cal_test, fig_dir)
    print(f"done in {time.time() - t_start:.0f}s -> {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
