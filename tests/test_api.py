"""Tests for the FastAPI live web app (api/main.py). Uses FastAPI's
TestClient (no real server/socket needed) and the same in-process demo
pipeline the app builds on first request — the first test in this module
pays that one-time cost (~30-60s, trains a small VQC), subsequent tests
reuse the cached singleton."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_frontend_is_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "QGFDA" in resp.text or "Quantum" in resp.text


def test_overview_has_expected_fields():
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("total_transactions", "fraud_cases", "pr_auc", "recall", "decision_summary", "qubits"):
        assert key in body
    assert body["is_synthetic"] is True


def test_transactions_list_respects_limit():
    resp = client.get("/api/transactions", params={"limit": 5})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) <= 5
    if rows:
        assert {"transaction_id", "classical_risk", "final_risk", "decision"}.issubset(rows[0])


def test_transactions_list_decision_filter():
    resp = client.get("/api/transactions", params={"limit": 50, "decision": "BLOCK"})
    assert resp.status_code == 200
    rows = resp.json()
    assert all(r["decision"] == "BLOCK" for r in rows)


def test_score_transaction_end_to_end():
    resp = client.post("/api/transactions/score", json={"amount_multiplier": 1.5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"].startswith("TXN-LIVE-")
    assert 0.0 <= body["classical_risk"] <= 1.0
    assert body["decision"] in ("APPROVE", "REVIEW", "BLOCK")


def test_score_transaction_rejects_invalid_multiplier():
    resp = client.post("/api/transactions/score", json={"amount_multiplier": -1})
    assert resp.status_code == 422


def test_quantum_summary():
    resp = client.get("/api/quantum/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["qubits"] > 0
    assert body["backend"]


def test_build_circuit_returns_image():
    resp = client.post("/api/quantum/circuit", json={
        "feature_map": "zz", "ansatz": "real_amplitudes", "qubits": 3, "reps": 1,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["logical_depth"] > 0
    assert len(body["image_base64_png"]) > 1000  # a real PNG, not an empty placeholder


def test_build_circuit_rejects_unknown_feature_map():
    resp = client.post("/api/quantum/circuit", json={
        "feature_map": "not-a-map", "ansatz": "real_amplitudes", "qubits": 3, "reps": 1,
    })
    assert resp.status_code == 422  # pydantic Literal validation


def test_feature_selection_returns_candidates():
    resp = client.post("/api/feature-selection", json={"candidate_counts": [2, 4], "seed": 42})
    assert resp.status_code == 200
    rows = resp.json()
    assert {r["n_features"] for r in rows} == {2, 4}


def test_drift_report():
    resp = client.get("/api/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("NORMAL", "WARNING", "CRITICAL")
    assert len(body["reference_histogram"]) > 0


def test_robustness_perturbation():
    resp = client.post("/api/drift/robustness", json={"amount_pct": 0.2})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["flip_rate"] <= 1.0


@pytest.mark.slow
def test_run_classical_experiment_via_api():
    resp = client.post("/api/experiments/run", json={
        "dataset": "ulb", "experiment_type": "classical", "model": "logistic_regression",
        "use_synthetic": True, "seed": 42,
    })
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert rows[0]["status"] == "completed"


def test_experiments_list_and_export():
    resp = client.get("/api/experiments")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    resp_csv = client.get("/api/experiments/export.csv")
    assert resp_csv.status_code == 200
    assert "text/csv" in resp_csv.headers["content-type"]
