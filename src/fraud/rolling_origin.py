"""Rolling-origin evaluation inside the development window (first 80% of time).

The final 20% test window used for the headline results is never touched here.
The development window is cut into 5 chronological blocks; fold k trains on
blocks 1..k and tests on block k+1 (k = 2, 3, 4), so every test block lies in
the future of its training data. This gives several out-of-time estimates instead
of one, which shows how stable the model ranking is.

Usage:
    python -m fraud.rolling_origin --data path/to/creditcard.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import data as D
from . import evaluate as E
from . import models as M

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["isolation_forest", "logreg_balanced", "hist_gb", "xgboost"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=5)
    args = ap.parse_args(argv)

    df, _ = D.load(args.data)
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    dev = df.iloc[: int(len(df) * 0.8)]  # identical boundary to temporal_split's test cut
    blocks = np.array_split(np.arange(len(dev)), args.blocks)

    folds = []
    for k in range(2, args.blocks):
        tr = dev.iloc[np.concatenate(blocks[:k])]
        te = dev.iloc[blocks[k]]
        Xtr, ytr = tr[D.FEATURES], tr["Class"].to_numpy()
        Xte, yte = te[D.FEATURES], te["Class"].to_numpy()
        pw = float((ytr == 0).sum() / (ytr == 1).sum())
        row = {"fold": k - 1, "train_rows": len(tr), "train_frauds": int(ytr.sum()),
               "test_rows": len(te), "test_frauds": int(yte.sum()), "pr_auc": {}}
        for name in MODELS:
            p = M.build(name, 0, pw).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            row["pr_auc"][name] = E.summary(yte, p)["pr_auc"]
        folds.append(row)
        print(row)

    summary = {m: {"mean": float(np.mean([f["pr_auc"][m] for f in folds])),
                   "sd": float(np.std([f["pr_auc"][m] for f in folds], ddof=1))} for m in MODELS}
    diffs = [f["pr_auc"]["xgboost"] - f["pr_auc"]["logreg_balanced"] for f in folds]
    out = {"protocol": f"expanding window over the first 80% of time, {args.blocks} blocks, "
                       f"{len(folds)} out-of-time folds; final test window untouched",
           "folds": folds, "summary": summary,
           "xgboost_minus_logreg_balanced": {"per_fold": diffs, "folds_xgboost_better": int(sum(d > 0 for d in diffs))}}
    (ROOT / "results/rolling_origin.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["summary"], indent=2), out["xgboost_minus_logreg_balanced"])


if __name__ == "__main__":
    main()
