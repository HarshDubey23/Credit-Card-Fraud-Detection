"""Model zoo. Every model is an sklearn-compatible estimator exposing a fraud score.

Scalers and resamplers live *inside* the pipeline, so they are fit on the
training split only.
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class IsolationForestScorer(BaseEstimator, ClassifierMixin):
    """Unsupervised baseline equivalent to the original notebook.

    Trained without labels; contamination is set to the training fraud rate,
    as the notebook did. Higher score = more anomalous.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 0):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.model_ = Pipeline([
            ("scale", StandardScaler()),
            ("iforest", IsolationForest(n_estimators=self.n_estimators,
                                        contamination=float(np.mean(y)),
                                        random_state=self.random_state, n_jobs=-1)),
        ]).fit(X)
        return self

    def predict_proba(self, X):
        s = -self.model_.score_samples(X)  # monotone anomaly score, not a probability
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return np.column_stack([1 - s, s])


def build(name: str, seed: int, pos_weight: float):
    if name == "prior":
        return DummyClassifier(strategy="prior")
    if name == "isolation_forest":
        return IsolationForestScorer(random_state=seed)
    if name == "logreg":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000))])
    if name == "logreg_balanced":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    if name == "logreg_smote":
        return ImbPipeline([("scale", StandardScaler()),
                            ("smote", SMOTE(random_state=seed)),
                            ("clf", LogisticRegression(max_iter=2000))])
    if name == "hist_gb":
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400,
                                              l2_regularization=1.0, random_state=seed)
    if name == "xgboost":
        return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
                             random_state=seed, n_jobs=-1, tree_method="hist")
    if name == "xgboost_weighted":
        return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
                             scale_pos_weight=pos_weight,
                             random_state=seed, n_jobs=-1, tree_method="hist")
    if name == "xgboost_smote":
        return ImbPipeline([("smote", SMOTE(random_state=seed)),
                            ("clf", XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  eval_metric="aucpr", random_state=seed,
                                                  n_jobs=-1, tree_method="hist"))])
    raise KeyError(name)


MODELS = ["prior", "isolation_forest", "logreg", "logreg_balanced", "logreg_smote",
          "hist_gb", "xgboost", "xgboost_weighted", "xgboost_smote"]
