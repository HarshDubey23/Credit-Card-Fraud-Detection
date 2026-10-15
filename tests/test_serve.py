"""API tests against a small model trained on synthetic data (no dataset download needed)."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier


@pytest.fixture
def client(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    X = rng.normal(size=(2000, 29)) + y[:, None] * 2.0
    m = XGBClassifier(n_estimators=20, max_depth=2).fit(X, y)
    m.save_model(tmp_path / "model.json")
    (tmp_path / "model_card.json").write_text(json.dumps({"threshold": 0.5, "model_sha256": "test"}))
    import fraud.serve as serve
    monkeypatch.setattr(serve, "MODEL_DIR", tmp_path)
    serve.load.cache_clear()
    yield TestClient(serve.app)
    serve.load.cache_clear()


def tx(shift=0.0, amount=10.0):
    return {"V": [shift] * 28, "Amount": amount}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_scores_and_flags(client):
    r = client.post("/score", json={"transactions": [tx(0.0), tx(3.0, amount=3.0)]})
    assert r.status_code == 200
    body = r.json()
    assert body["threshold"] == 0.5 and len(body["results"]) == 2
    low, high = (x["fraud_score"] for x in body["results"])
    assert 0 <= low < high <= 1
    assert body["results"][1]["flagged"] is True


@pytest.mark.parametrize("bad", [
    {"transactions": []},
    {"transactions": [{"V": [0.0] * 27, "Amount": 1.0}]},
    {"transactions": [{"V": [0.0] * 28, "Amount": -1.0}]},
    {"transactions": [tx()] * 1001},
])
def test_rejects_invalid_input(client, bad):
    assert client.post("/score", json=bad).status_code == 422


def test_rejects_non_finite(client):
    body = '{"transactions": [{"V": [' + ",".join(["NaN"] * 28) + '], "Amount": 1}]}'
    r = client.post("/score", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_missing_model_returns_503(tmp_path, monkeypatch):
    import fraud.serve as serve
    monkeypatch.setattr(serve, "MODEL_DIR", tmp_path)
    serve.load.cache_clear()
    r = TestClient(serve.app).post("/score", json={"transactions": [tx()]})
    serve.load.cache_clear()
    assert r.status_code == 503
