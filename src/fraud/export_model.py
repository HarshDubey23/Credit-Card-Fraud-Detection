"""Refit the selected model exactly as evaluated and export it for serving.

The exported model is checked against results/metrics.json: its test PR-AUC must
equal the published seed-0 value, which proves the served model is the evaluated one.

Usage:
    python -m fraud.export_model --data path/to/creditcard.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import data as D
from . import evaluate as E
from . import models as M

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args(argv)

    metrics = json.loads((ROOT / "results/metrics.json").read_text())
    name = metrics["selected_model"]
    run0 = metrics["models"][name]["runs"][0]

    df, _ = D.load(args.data)
    split = D.temporal_split(df)
    pos_weight = float((split.y_train == 0).sum() / (split.y_train == 1).sum())
    model = M.build(name, 0, pos_weight).fit(split.X_train, split.y_train)
    p_test = model.predict_proba(split.X_test)[:, 1]
    pr_auc = E.summary(split.y_test, p_test)["pr_auc"]
    if abs(pr_auc - run0["test"]["pr_auc"]) > 1e-9:
        raise SystemExit(f"exported model PR-AUC {pr_auc} != published {run0['test']['pr_auc']}")

    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / "model.json"
    model.save_model(path)
    card = {
        "model": name,
        "features": D.FEATURES,
        "threshold": run0["test_at_val_f1_threshold"]["threshold"],
        "threshold_rule": "maximises F1 on the validation window",
        "test_pr_auc": pr_auc,
        "test_precision_at_threshold": run0["test_at_val_f1_threshold"]["precision"],
        "test_recall_at_threshold": run0["test_at_val_f1_threshold"]["recall"],
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dataset_sha256": metrics["dataset"]["sha256"],
    }
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
