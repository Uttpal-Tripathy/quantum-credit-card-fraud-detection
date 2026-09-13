"""Uncertainty gate: decides, using only information available at inference
time, whether a transaction can be resolved by the fast classical model or
needs the (expensive, resource-constrained) quantum path.

Critically — per spec section 12 — routing is based on the classical model's
own predicted-probability confidence, NEVER on the ground-truth fraud label.
A transaction is "uncertain" when its classical score falls in the
configured risk band around the decision boundary (default [0.15, 0.85]) —
i.e. the classical model itself is unsure — or, symmetrically, when the
score is high enough to be plausibly fraud but the calibrated confidence is
still below `gate_threshold`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GateDecision:
    route_to_quantum: np.ndarray  # boolean mask
    confidence: np.ndarray
    quantum_fraction: float


class UncertaintyGate:
    """Configurable confidence-band gate.

    A transaction is routed to the quantum path if:
      - its classical score falls inside `risk_band` (genuinely ambiguous), OR
      - classical confidence (= max(p, 1-p)) is below `gate_threshold`.
    Both conditions express the same idea from two angles; either can be
    disabled independently for ablation (spec experiment E: "hybrid without
    gate" sets `gate_threshold=0.0` and `risk_band=(0,1)` so nothing is
    classically resolved).
    """

    def __init__(
        self,
        gate_threshold: float = 0.80,
        risk_band: tuple[float, float] = (0.15, 0.85),
        max_quantum_fraction: float | None = 0.25,
    ):
        if not (0.0 <= gate_threshold <= 1.0):
            raise ValueError("gate_threshold must be in [0, 1]")
        lo, hi = risk_band
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError("risk_band must satisfy 0 <= lo <= hi <= 1")
        self.gate_threshold = gate_threshold
        self.risk_band = risk_band
        self.max_quantum_fraction = max_quantum_fraction

    def route(self, classical_scores: np.ndarray) -> GateDecision:
        scores = np.asarray(classical_scores, dtype=float)
        confidence = np.maximum(scores, 1 - scores)

        lo, hi = self.risk_band
        in_risk_band = (scores >= lo) & (scores <= hi)
        low_confidence = confidence < self.gate_threshold
        route_to_quantum = in_risk_band | low_confidence

        if self.max_quantum_fraction is not None:
            route_to_quantum = self._enforce_resource_cap(route_to_quantum, confidence)

        quantum_fraction = float(route_to_quantum.mean()) if len(route_to_quantum) else 0.0
        return GateDecision(route_to_quantum=route_to_quantum, confidence=confidence, quantum_fraction=quantum_fraction)

    def _enforce_resource_cap(self, route_to_quantum: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        """If more transactions are flagged uncertain than the resource
        budget (`max_quantum_fraction` of the batch) allows, keep only the
        LEAST confident ones — a documented, deterministic prioritization
        rule rather than an arbitrary truncation."""
        n = len(route_to_quantum)
        budget = int(np.floor(n * self.max_quantum_fraction))
        n_flagged = int(route_to_quantum.sum())
        if n_flagged <= budget:
            return route_to_quantum

        candidate_idx = np.where(route_to_quantum)[0]
        # lowest confidence first = most uncertain first
        ranked = candidate_idx[np.argsort(confidence[candidate_idx])]
        keep = set(ranked[:budget].tolist())

        capped = np.zeros_like(route_to_quantum)
        capped[list(keep)] = True
        return capped
