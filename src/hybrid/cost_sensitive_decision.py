"""Cost-sensitive 3-way decision engine: APPROVE / REVIEW / BLOCK.

Fraud detection is not binary classification — a missed fraud, a false
block, and a manual review all cost different amounts (spec section 14).
Given a final risk score in [0, 1] and two thresholds (review_threshold <
block_threshold), transactions are partitioned into three actions. Both
thresholds are optimized by minimizing total ExpectedLoss on a validation
set (via a threshold grid search) rather than chosen arbitrarily, and every
choice is recorded for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Decision(str, Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass
class CostModel:
    fraud_loss: float = 500.0
    false_positive_cost: float = 25.0
    review_cost: float = 5.0

    @classmethod
    def from_config(cls, cost_config: dict) -> "CostModel":
        """Build from configs/experiments.yaml's `cost_model` block, which
        also carries optional `block_threshold`/`review_threshold` overrides
        consumed elsewhere (by optimize_thresholds) — silently ignore any
        keys this dataclass doesn't define."""
        return cls(
            fraud_loss=cost_config.get("fraud_loss", 500.0),
            false_positive_cost=cost_config.get("false_positive_cost", 25.0),
            review_cost=cost_config.get("review_cost", 5.0),
        )


@dataclass
class ThresholdConfig:
    review_threshold: float
    block_threshold: float
    expected_loss: float
    search_grid_points: int


def decide(scores: np.ndarray, review_threshold: float, block_threshold: float) -> np.ndarray:
    """Vectorized 3-way decision: score < review_threshold -> APPROVE;
    review_threshold <= score < block_threshold -> REVIEW; score >=
    block_threshold -> BLOCK."""
    if not (0.0 <= review_threshold <= block_threshold <= 1.0):
        raise ValueError("Require 0 <= review_threshold <= block_threshold <= 1.")
    scores = np.asarray(scores, dtype=float)
    decisions = np.full(scores.shape, Decision.APPROVE.value, dtype=object)
    decisions[scores >= review_threshold] = Decision.REVIEW.value
    decisions[scores >= block_threshold] = Decision.BLOCK.value
    return decisions


def compute_expected_loss_three_way(
    y_true: np.ndarray,
    decisions: np.ndarray,
    cost_model: CostModel,
) -> float:
    """ExpectedLoss = FN(missed fraud, i.e. fraud approved) * fraud_loss
                     + FP(legit blocked) * false_positive_cost
                     + n_reviewed * review_cost
    A REVIEW is assumed to eventually resolve correctly at `review_cost` and
    is not double-counted as FN/FP, reflecting analyst adjudication."""
    y_true = np.asarray(y_true).astype(int)
    is_fraud = y_true == 1

    approved = decisions == Decision.APPROVE.value
    blocked = decisions == Decision.BLOCK.value
    reviewed = decisions == Decision.REVIEW.value

    fn = int(np.sum(is_fraud & approved))          # fraud that slipped through
    fp = int(np.sum(~is_fraud & blocked))           # legitimate transaction blocked
    n_reviewed = int(np.sum(reviewed))

    return fn * cost_model.fraud_loss + fp * cost_model.false_positive_cost + n_reviewed * cost_model.review_cost


def optimize_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_model: CostModel,
    search_grid_points: int = 50,
) -> ThresholdConfig:
    """Grid search over (review_threshold, block_threshold) pairs with
    review_threshold <= block_threshold, minimizing total ExpectedLoss.
    O(grid^2) decisions over the full score array — fine at
    search_grid_points<=50 (1275 combinations) for datasets up to ~1M rows;
    reduce search_grid_points for very large validation sets."""
    grid = np.linspace(0.0, 1.0, search_grid_points)
    best = ThresholdConfig(review_threshold=0.5, block_threshold=0.5, expected_loss=np.inf, search_grid_points=search_grid_points)

    for review_t in grid:
        for block_t in grid:
            if block_t < review_t:
                continue
            decisions = decide(scores, review_t, block_t)
            loss = compute_expected_loss_three_way(y_true, decisions, cost_model)
            if loss < best.expected_loss:
                best = ThresholdConfig(
                    review_threshold=float(review_t),
                    block_threshold=float(block_t),
                    expected_loss=loss,
                    search_grid_points=search_grid_points,
                )
    return best


@dataclass
class DecisionSummary:
    n_approve: int
    n_review: int
    n_block: int
    approve_rate: float
    review_rate: float
    block_rate: float
    expected_loss: float


def summarize_decisions(y_true: np.ndarray, decisions: np.ndarray, cost_model: CostModel) -> DecisionSummary:
    n = len(decisions)
    n_approve = int(np.sum(decisions == Decision.APPROVE.value))
    n_review = int(np.sum(decisions == Decision.REVIEW.value))
    n_block = int(np.sum(decisions == Decision.BLOCK.value))
    return DecisionSummary(
        n_approve=n_approve,
        n_review=n_review,
        n_block=n_block,
        approve_rate=n_approve / n if n else 0.0,
        review_rate=n_review / n if n else 0.0,
        block_rate=n_block / n if n else 0.0,
        expected_loss=compute_expected_loss_three_way(y_true, decisions, cost_model),
    )
