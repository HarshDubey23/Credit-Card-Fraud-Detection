"""Minimal scoring API for the exported model.

    uvicorn fraud.serve:app --app-dir src

POST /score   one or more transactions -> fraud score and decision at the validated threshold
GET  /model   model card (threshold, test metrics, artifact hashes)
GET  /health  liveness
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from xgboost import XGBClassifier

MODEL_DIR = Path(os.environ.get("FRAUD_MODEL_DIR", Path(__file__).resolve().parents[2] / "model"))
MAX_BATCH = 1000


class Transaction(BaseModel):
    """The ULB schema: anonymised PCA components V1-V28 plus Amount (EUR). Time is not used."""

    V: list[float] = Field(..., min_length=28, max_length=28, description="V1..V28 in order")
    Amount: float = Field(..., ge=0, le=1_000_000)

    @field_validator("V")
    @classmethod
    def finite(cls, v: list[float]) -> list[float]:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("V values must be finite")
        return v


class ScoreRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=MAX_BATCH)


class Score(BaseModel):
    fraud_score: float
    flagged: bool


class ScoreResponse(BaseModel):
    threshold: float
    model_sha256: str
    results: list[Score]


@lru_cache(maxsize=1)
def load():
    card_path, model_path = MODEL_DIR / "model_card.json", MODEL_DIR / "model.json"
    if not (card_path.exists() and model_path.exists()):
        raise RuntimeError(f"model artifacts missing in {MODEL_DIR}; run python -m fraud.export_model")
    card = json.loads(card_path.read_text())
    model = XGBClassifier()
    model.load_model(model_path)
    return model, card


app = FastAPI(title="Fraud scoring API",
              description="Benchmark model trained on the public ULB credit-card dataset. "
                          "For demonstration only; not for real payment decisions.")


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Echoing the offending input back would fail on NaN/inf, so return locations and messages only.
    errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", "")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors[:20]})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_card():
    return load()[1]


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    try:
        model, card = load()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    X = np.array([[*t.V, t.Amount] for t in req.transactions], dtype=float)
    p = model.predict_proba(X)[:, 1]
    t = float(card["threshold"])
    return ScoreResponse(threshold=t, model_sha256=card["model_sha256"],
                         results=[Score(fraud_score=float(x), flagged=bool(x >= t)) for x in p])
