"""Risk fusion: combine classical and quantum risk scores into one final
score. Multiple strategies are implemented and evaluated head-to-head (spec
section 13) — the project does not assume any one fusion method wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

FusionMethod = Literal[
    "weighted", "confidence_weighted", "logistic_stacking", "gbm_stacking", "learned_gate"
]


@dataclass
class FusionResult:
    final_score: np.ndarray
    method: str
    weights_used: dict[str, float] | None = None


class RiskFusion:
    """Fits (where applicable) and applies one fusion strategy. Only
    'logistic_stacking', 'gbm_stacking', and 'learned_gate' require `.fit`;
    'weighted' and 'confidence_weighted' are closed-form and configurable via
    `weights` without any training data."""

    def __init__(
        self,
        method: FusionMethod = "weighted",
        weights: dict[str, float] | None = None,
        random_state: int = 42,
    ):
        self.method = method
        self.weights = weights or {"classical": 0.5, "quantum": 0.5}
        self.random_state = random_state
        self._model = None

    def requires_fit(self) -> bool:
        return self.method in {"logistic_stacking", "gbm_stacking", "learned_gate"}

    def fit(
        self,
        classical_scores: np.ndarray,
        quantum_scores: np.ndarray,
        y_true: np.ndarray,
        was_routed_to_quantum: np.ndarray | None = None,
    ) -> "RiskFusion":
        if not self.requires_fit():
            return self

        X = self._stack_features(classical_scores, quantum_scores, was_routed_to_quantum)

        if self.method == "logistic_stacking":
            self._model = LogisticRegression(max_iter=1000, class_weight="balanced")
        elif self.method in ("gbm_stacking", "learned_gate"):
            self._model = GradientBoostingClassifier(random_state=self.random_state)
        self._model.fit(X, y_true)
        return self

    def _stack_features(
        self,
        classical_scores: np.ndarray,
        quantum_scores: np.ndarray,
        was_routed_to_quantum: np.ndarray | None,
    ) -> np.ndarray:
        classical_scores = np.asarray(classical_scores, dtype=float)
        quantum_scores = np.asarray(quantum_scores, dtype=float)
        features = [classical_scores, quantum_scores]
        if self.method == "learned_gate" and was_routed_to_quantum is not None:
            features.append(np.asarray(was_routed_to_quantum, dtype=float))
        return np.column_stack(features)

    def fuse(
        self,
        classical_scores: np.ndarray,
        quantum_scores: np.ndarray,
        was_routed_to_quantum: np.ndarray | None = None,
    ) -> FusionResult:
        classical_scores = np.asarray(classical_scores, dtype=float)
        quantum_scores = np.asarray(quantum_scores, dtype=float)

        if self.method == "weighted":
            wc, wq = self.weights["classical"], self.weights["quantum"]
            final = wc * classical_scores + wq * quantum_scores
            return FusionResult(final_score=final, method=self.method, weights_used=self.weights)

        if self.method == "confidence_weighted":
            classical_confidence = np.maximum(classical_scores, 1 - classical_scores)
            quantum_confidence = np.maximum(quantum_scores, 1 - quantum_scores)
            total = classical_confidence + quantum_confidence
            total = np.where(total == 0, 1.0, total)
            wc = classical_confidence / total
            wq = quantum_confidence / total
            final = wc * classical_scores + wq * quantum_scores
            return FusionResult(final_score=final, method=self.method, weights_used=None)

        if self.method in {"logistic_stacking", "gbm_stacking", "learned_gate"}:
            if self._model is None:
                raise RuntimeError(f"RiskFusion(method='{self.method}') must be .fit() before .fuse().")
            X = self._stack_features(classical_scores, quantum_scores, was_routed_to_quantum)
            final = self._model.predict_proba(X)[:, 1]
            return FusionResult(final_score=final, method=self.method, weights_used=None)

        raise ValueError(f"Unknown fusion method: {self.method}")


def fuse_only_where_routed(
    classical_scores: np.ndarray,
    quantum_scores: np.ndarray | None,
    routed_to_quantum: np.ndarray,
    fusion: RiskFusion,
) -> np.ndarray:
    """Apply fusion ONLY to transactions routed to the quantum path; every
    other transaction keeps its classical score untouched. This is how the
    full QGFDA architecture (as opposed to the 'fuse everything' ablation
    arm) actually combines scores."""
    final = classical_scores.copy().astype(float)
    if quantum_scores is None or not np.any(routed_to_quantum):
        return final

    idx = np.where(routed_to_quantum)[0]
    fused = fusion.fuse(
        classical_scores[idx],
        quantum_scores,
        was_routed_to_quantum=np.ones(len(idx)),
    )
    final[idx] = fused.final_score
    return final
