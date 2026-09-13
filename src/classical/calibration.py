"""Probability calibration and cost-aware threshold tuning shared by every
classical model.

Fraud models are consumed downstream by the cost-sensitive decision engine
(src/hybrid/cost_sensitive_decision.py), which needs well-calibrated
probabilities, not just a good ranking — hence calibration is a first-class
step here rather than an afterthought.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, roc_curve


def calibrate_model(estimator, method: str = "isotonic", cv: int = 3):
    """Wrap a fitted-or-unfitted sklearn-compatible estimator in
    CalibratedClassifierCV. `method`: 'isotonic' (flexible, needs more data)
    or 'sigmoid' (Platt scaling, safer on small/very imbalanced folds)."""
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError(f"Unknown calibration method: {method}")
    return CalibratedClassifierCV(estimator, method=method, cv=cv)


@dataclass
class ThresholdResult:
    threshold: float
    method: str
    score_at_threshold: float


def tune_threshold_expected_loss(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fraud_loss: float,
    false_positive_cost: float,
    n_points: int = 200,
) -> ThresholdResult:
    """Sweep thresholds in [0, 1] and pick the one minimizing
    ExpectedLoss = FN * fraud_loss + FP * false_positive_cost (review cost is
    handled separately by the 3-way decision engine; this binary sweep only
    trades off block-vs-approve for a raw classifier)."""
    thresholds = np.linspace(0.0, 1.0, n_points)
    best_threshold, best_loss = 0.5, np.inf

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        loss = fn * fraud_loss + fp * false_positive_cost
        if loss < best_loss:
            best_loss = loss
            best_threshold = float(t)

    return ThresholdResult(threshold=best_threshold, method="expected_loss", score_at_threshold=best_loss)


def tune_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray, n_points: int = 200) -> ThresholdResult:
    thresholds = np.linspace(0.0, 1.0, n_points)
    best_threshold, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(t)
    return ThresholdResult(threshold=best_threshold, method="f1", score_at_threshold=best_f1)


def tune_threshold_youden_j(y_true: np.ndarray, y_proba: np.ndarray) -> ThresholdResult:
    """Youden's J statistic: maximizes TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return ThresholdResult(threshold=float(thresholds[best_idx]), method="youden_j", score_at_threshold=float(j[best_idx]))


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    method: str = "expected_loss",
    fraud_loss: float = 500.0,
    false_positive_cost: float = 25.0,
    n_points: int = 200,
) -> ThresholdResult:
    if method == "expected_loss":
        return tune_threshold_expected_loss(y_true, y_proba, fraud_loss, false_positive_cost, n_points)
    if method == "f1":
        return tune_threshold_f1(y_true, y_proba, n_points)
    if method == "youden_j":
        return tune_threshold_youden_j(y_true, y_proba)
    raise ValueError(f"Unknown threshold tuning method: {method}")
