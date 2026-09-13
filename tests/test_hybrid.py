"""Tests for the hybrid architecture: uncertainty gate, risk fusion,
cost-sensitive decision engine, and the end-to-end QGFDA pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.classical.xgboost_model import XGBoostModel
from src.hybrid.cost_sensitive_decision import (
    CostModel,
    Decision,
    compute_expected_loss_three_way,
    decide,
    optimize_thresholds,
    summarize_decisions,
)
from src.hybrid.qgfda import QGFDA, QGFDAConfig
from src.hybrid.risk_fusion import RiskFusion, fuse_only_where_routed
from src.hybrid.uncertainty_gate import UncertaintyGate
from src.quantum.vqc_model import VQCModel


# ---------------------------------------------------------------- gate ----

def test_gate_routes_ambiguous_scores_to_quantum():
    gate = UncertaintyGate(gate_threshold=0.8, risk_band=(0.15, 0.85), max_quantum_fraction=None)
    scores = np.array([0.02, 0.5, 0.98, 0.6])
    decision = gate.route(scores)
    assert decision.route_to_quantum.tolist() == [False, True, False, True]


def test_gate_never_uses_ground_truth_label():
    """The gate's route() signature only accepts scores, not labels —
    this test documents/enforces that routing cannot depend on y_true."""
    import inspect

    sig = inspect.signature(UncertaintyGate.route)
    assert "y_true" not in sig.parameters and "y" not in sig.parameters


def test_gate_enforces_resource_cap():
    gate = UncertaintyGate(gate_threshold=1.0, risk_band=(0.0, 1.0), max_quantum_fraction=0.25)
    scores = np.random.default_rng(0).uniform(0, 1, size=100)
    decision = gate.route(scores)
    assert decision.route_to_quantum.sum() <= 25


def test_gate_zero_threshold_and_full_band_routes_everything():
    gate = UncertaintyGate(gate_threshold=0.0, risk_band=(0.0, 1.0), max_quantum_fraction=None)
    scores = np.random.default_rng(0).uniform(0, 1, size=50)
    decision = gate.route(scores)
    assert decision.route_to_quantum.all()


# -------------------------------------------------------------- fusion ----

def test_weighted_fusion_matches_manual_computation():
    fusion = RiskFusion(method="weighted", weights={"classical": 0.7, "quantum": 0.3})
    classical = np.array([0.8, 0.2])
    quantum = np.array([0.4, 0.6])
    result = fusion.fuse(classical, quantum)
    expected = 0.7 * classical + 0.3 * quantum
    np.testing.assert_allclose(result.final_score, expected)


def test_confidence_weighted_fusion_favors_more_confident_score():
    fusion = RiskFusion(method="confidence_weighted")
    classical = np.array([0.95])  # very confident legit... wait high score = confident fraud
    quantum = np.array([0.5])     # totally unsure
    result = fusion.fuse(classical, quantum)
    # classical confidence (0.95) >> quantum confidence (0.0), so final should sit near classical
    assert abs(result.final_score[0] - classical[0]) < abs(result.final_score[0] - quantum[0])


def test_logistic_stacking_requires_fit_before_fuse():
    fusion = RiskFusion(method="logistic_stacking")
    with pytest.raises(RuntimeError):
        fusion.fuse(np.array([0.5]), np.array([0.5]))


def test_logistic_stacking_fits_and_fuses():
    rng = np.random.default_rng(0)
    classical = rng.uniform(0, 1, 100)
    quantum = rng.uniform(0, 1, 100)
    y = ((classical + quantum) / 2 > 0.5).astype(int)
    fusion = RiskFusion(method="logistic_stacking").fit(classical, quantum, y)
    result = fusion.fuse(classical, quantum)
    assert np.all((result.final_score >= 0) & (result.final_score <= 1))


def test_fuse_only_where_routed_leaves_others_untouched():
    fusion = RiskFusion(method="weighted", weights={"classical": 0.5, "quantum": 0.5})
    classical = np.array([0.1, 0.9, 0.5])
    routed = np.array([False, False, True])
    quantum_routed_only = np.array([0.9])  # only for index 2
    final = fuse_only_where_routed(classical, quantum_routed_only, routed, fusion)
    assert final[0] == classical[0]
    assert final[1] == classical[1]
    assert final[2] == pytest.approx(0.5 * 0.5 + 0.5 * 0.9)


# ------------------------------------------------------- decision engine --

def test_decide_three_way_partition():
    scores = np.array([0.05, 0.5, 0.95])
    decisions = decide(scores, review_threshold=0.3, block_threshold=0.7)
    assert decisions.tolist() == [Decision.APPROVE.value, Decision.REVIEW.value, Decision.BLOCK.value]


def test_decide_rejects_invalid_threshold_order():
    with pytest.raises(ValueError):
        decide(np.array([0.5]), review_threshold=0.8, block_threshold=0.2)


def test_expected_loss_counts_missed_fraud_and_false_blocks():
    y_true = np.array([1, 0, 1, 0])
    decisions = np.array([Decision.APPROVE.value, Decision.BLOCK.value, Decision.BLOCK.value, Decision.APPROVE.value])
    cost_model = CostModel(fraud_loss=100, false_positive_cost=10, review_cost=1)
    loss = compute_expected_loss_three_way(y_true, decisions, cost_model)
    # fn=1 (fraud approved at idx0) -> 100; fp=1 (legit blocked at idx1) -> 10
    assert loss == pytest.approx(110)


def test_optimize_thresholds_beats_naive_always_approve():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([rng.uniform(0, 0.4, 90), rng.uniform(0.6, 1.0, 10)])
    cost_model = CostModel(fraud_loss=500, false_positive_cost=25, review_cost=5)
    best = optimize_thresholds(y_true, scores, cost_model, search_grid_points=20)

    always_approve_loss = compute_expected_loss_three_way(
        y_true, np.full(len(y_true), Decision.APPROVE.value), cost_model
    )
    assert best.expected_loss <= always_approve_loss


def test_summarize_decisions_rates_sum_to_one():
    y_true = np.array([0, 1, 0, 1])
    decisions = np.array([Decision.APPROVE.value, Decision.REVIEW.value, Decision.BLOCK.value, Decision.APPROVE.value])
    summary = summarize_decisions(y_true, decisions, CostModel())
    assert summary.approve_rate + summary.review_rate + summary.block_rate == pytest.approx(1.0)


# ------------------------------------------------------------- QGFDA -----

def test_qgfda_end_to_end_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 5))
    y = ((X[:, 0] + X[:, 1] + rng.normal(scale=0.5, size=n)) > 0).astype(int)
    # force a small amount of class imbalance so the gate/decision path is exercised meaningfully
    split = int(n * 0.7)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    config = QGFDAConfig(
        quantum_feature_indices=[0, 1, 2],
        gate_threshold=0.8,
        risk_band=(0.15, 0.85),
        max_quantum_fraction=0.5,
        fusion_method="weighted",
        fusion_weights={"classical": 0.6, "quantum": 0.4},
        cost_model=CostModel(fraud_loss=500, false_positive_cost=25, review_cost=5),
        threshold_search_grid_points=15,
        quantum_train_sample_size=40,
        random_state=42,
    )
    classical_model = XGBoostModel(calibrate=False)
    quantum_model = VQCModel(num_qubits=3, backend="simulator", shots=64, seed=42, optimizer="cobyla", optimizer_maxiter=10)

    model = QGFDA(classical_model, quantum_model, config)
    model.fit(X_train, y_train, X_val, y_val)

    predictions = model.predict(X_val)
    assert predictions.final_score.shape == (len(y_val),)
    assert set(predictions.decision.tolist()).issubset({d.value for d in Decision})
    assert 0.0 <= predictions.quantum_fraction <= 0.5 + 1e-9

    frame = predictions.to_frame()
    assert len(frame) == len(y_val)
    assert {"transaction_id", "classical_risk", "quantum_risk", "final_risk", "decision"}.issubset(frame.columns)


def test_qgfda_raises_if_predict_called_before_fit():
    config = QGFDAConfig(quantum_feature_indices=[0, 1])
    classical_model = XGBoostModel(calibrate=False)
    quantum_model = VQCModel(num_qubits=2, backend="simulator", shots=64, seed=42, optimizer_maxiter=5)
    model = QGFDA(classical_model, quantum_model, config)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 5)))
