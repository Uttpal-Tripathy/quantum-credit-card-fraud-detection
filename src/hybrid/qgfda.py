"""QGFDA: Quantum-Gated Fraud Detection Architecture — the project's primary
research contribution (spec section 12).

Pipeline:
    Transaction
      -> Classical fast-path model (scores every transaction)
      -> Uncertainty Gate (routes only ambiguous/low-confidence transactions
         onward, using classical-model confidence available at inference
         time — never the ground-truth label)
      -> Quantum ML (scores only the routed subset, on QAFS-selected features)
      -> Risk Fusion (combines classical + quantum scores for routed rows;
         everyone else keeps their classical score)
      -> Cost-Sensitive Decision Engine (APPROVE / REVIEW / BLOCK)

This module orchestrates already-implemented pieces
(src/classical/*, src/quantum/*, src/hybrid/uncertainty_gate.py,
src/hybrid/risk_fusion.py, src/hybrid/cost_sensitive_decision.py) — it holds
no modeling logic of its own beyond the wiring, so each stage stays testable
in isolation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.hybrid.cost_sensitive_decision import (
    CostModel,
    ThresholdConfig,
    decide,
    optimize_thresholds,
    summarize_decisions,
)
from src.hybrid.risk_fusion import RiskFusion, fuse_only_where_routed
from src.hybrid.uncertainty_gate import GateDecision, UncertaintyGate
from src.utils.logging import get_logger

logger = get_logger("hybrid.qgfda")


@dataclass
class QGFDAPredictions:
    classical_score: np.ndarray
    quantum_score: np.ndarray          # NaN where not routed to quantum
    final_score: np.ndarray
    routed_to_quantum: np.ndarray
    decision: np.ndarray
    classical_latency_s: np.ndarray
    quantum_latency_s: np.ndarray      # 0.0 where not routed
    quantum_fraction: float

    def to_frame(self, transaction_ids: list[str] | None = None, amounts: np.ndarray | None = None) -> pd.DataFrame:
        n = len(self.final_score)
        return pd.DataFrame({
            "transaction_id": transaction_ids or [f"TXN-{i:08d}" for i in range(n)],
            "amount": amounts if amounts is not None else np.full(n, np.nan),
            "classical_risk": self.classical_score,
            "quantum_risk": self.quantum_score,
            "final_risk": self.final_score,
            "routed_to_quantum": self.routed_to_quantum,
            "decision": self.decision,
        })


@dataclass
class QGFDAConfig:
    quantum_feature_indices: list[int]
    gate_threshold: float = 0.80
    risk_band: tuple[float, float] = (0.15, 0.85)
    max_quantum_fraction: float | None = 0.25
    fusion_method: str = "weighted"
    fusion_weights: dict[str, float] = field(default_factory=lambda: {"classical": 0.5, "quantum": 0.5})
    cost_model: CostModel = field(default_factory=CostModel)
    threshold_search_grid_points: int = 50
    quantum_train_sample_size: int | None = 300
    random_state: int = 42


class QGFDA:
    """End-to-end hybrid architecture. `classical_model` and `quantum_model`
    must each expose `.fit(X, y)` and `.predict_proba(X)` (all wrappers in
    src/classical/* and src/quantum/* already do)."""

    def __init__(self, classical_model, quantum_model, config: QGFDAConfig):
        self.classical_model = classical_model
        self.quantum_model = quantum_model
        self.config = config
        self.gate = UncertaintyGate(
            gate_threshold=config.gate_threshold,
            risk_band=config.risk_band,
            max_quantum_fraction=config.max_quantum_fraction,
        )
        self.fusion = RiskFusion(
            method=config.fusion_method,
            weights=config.fusion_weights,
            random_state=config.random_state,
        )
        self.thresholds: ThresholdConfig | None = None
        self._is_fit = False

    def _quantum_columns(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.config.quantum_feature_indices]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "QGFDA":
        logger.info(f"Fitting classical fast path on {len(y_train)} training rows.")
        self.classical_model.fit(X_train, y_train)

        rng = np.random.default_rng(self.config.random_state)
        sample_size = self.config.quantum_train_sample_size
        if sample_size is not None and sample_size < len(y_train):
            idx = rng.choice(len(y_train), size=sample_size, replace=False)
        else:
            idx = np.arange(len(y_train))

        logger.info(f"Fitting quantum path on {len(idx)} sampled training rows "
                     f"({len(self.config.quantum_feature_indices)} qubits).")
        self.quantum_model.fit(self._quantum_columns(X_train)[idx], y_train[idx])

        classical_val_scores = self.classical_model.predict_proba(X_val)
        gate_decision = self.gate.route(classical_val_scores)
        routed_idx = np.where(gate_decision.route_to_quantum)[0]

        if len(routed_idx) == 0:
            logger.warning("Uncertainty gate routed zero validation rows to quantum; "
                            "fusion/threshold tuning will rely on classical scores only.")
            quantum_val_scores_routed = np.array([])
        else:
            quantum_val_scores_routed = self.quantum_model.predict_proba(
                self._quantum_columns(X_val)[routed_idx]
            )
            if self.fusion.requires_fit():
                self.fusion.fit(
                    classical_val_scores[routed_idx], quantum_val_scores_routed, y_val[routed_idx]
                )

        final_val_scores = fuse_only_where_routed(
            classical_val_scores, quantum_val_scores_routed, gate_decision.route_to_quantum, self.fusion
        )

        self.thresholds = optimize_thresholds(
            y_val, final_val_scores, self.config.cost_model, self.config.threshold_search_grid_points
        )
        logger.info(f"Optimized decision thresholds: {self.thresholds}")
        self._is_fit = True
        return self

    def predict(self, X: np.ndarray) -> QGFDAPredictions:
        if not self._is_fit or self.thresholds is None:
            raise RuntimeError("QGFDA must be .fit() before .predict().")

        start = time.perf_counter()
        classical_scores = self.classical_model.predict_proba(X)
        classical_latency = (time.perf_counter() - start) / max(len(X), 1)
        classical_latency_s = np.full(len(X), classical_latency)

        gate_decision: GateDecision = self.gate.route(classical_scores)
        routed_idx = np.where(gate_decision.route_to_quantum)[0]

        quantum_score_full = np.full(len(X), np.nan)
        quantum_latency_s = np.zeros(len(X))
        if len(routed_idx) > 0:
            start_q = time.perf_counter()
            quantum_scores_routed = self.quantum_model.predict_proba(self._quantum_columns(X)[routed_idx])
            elapsed_q = (time.perf_counter() - start_q) / len(routed_idx)
            quantum_score_full[routed_idx] = quantum_scores_routed
            quantum_latency_s[routed_idx] = elapsed_q

        final_scores = fuse_only_where_routed(
            classical_scores,
            quantum_score_full[routed_idx] if len(routed_idx) > 0 else np.array([]),
            gate_decision.route_to_quantum,
            self.fusion,
        )
        decisions = decide(final_scores, self.thresholds.review_threshold, self.thresholds.block_threshold)

        return QGFDAPredictions(
            classical_score=classical_scores,
            quantum_score=quantum_score_full,
            final_score=final_scores,
            routed_to_quantum=gate_decision.route_to_quantum,
            decision=decisions,
            classical_latency_s=classical_latency_s,
            quantum_latency_s=quantum_latency_s,
            quantum_fraction=gate_decision.quantum_fraction,
        )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        from src.evaluation.metrics import compute_metrics

        predictions = self.predict(X)
        binary_metrics = compute_metrics(
            y, predictions.final_score, threshold=self.thresholds.review_threshold,
            fraud_loss=self.config.cost_model.fraud_loss,
            false_positive_cost=self.config.cost_model.false_positive_cost,
        )
        decision_summary = summarize_decisions(y, predictions.decision, self.config.cost_model)

        return {
            "predictions": predictions,
            "binary_metrics": binary_metrics,
            "decision_summary": decision_summary,
            "quantum_fraction": predictions.quantum_fraction,
            "avg_classical_latency_s": float(np.mean(predictions.classical_latency_s)),
            "avg_quantum_latency_s": (
                float(np.mean(predictions.quantum_latency_s[predictions.routed_to_quantum]))
                if predictions.routed_to_quantum.any() else 0.0
            ),
            "thresholds": self.thresholds,
        }
