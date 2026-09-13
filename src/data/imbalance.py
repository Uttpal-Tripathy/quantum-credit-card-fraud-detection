"""Class-imbalance handling.

Fraud datasets are extremely imbalanced (ULB: 0.17% positive). We prefer
class-weighting (no synthetic rows, no risk of leaking synthetic fraud
patterns into validation) as the default, with SMOTE/undersampling available
for explicit ablation experiments only — and only ever fit on the TRAIN
split.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def compute_class_weights(y: pd.Series | np.ndarray) -> dict[int, float]:
    """Balanced class weights: weight_c = n_samples / (n_classes * n_c)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    return {int(c): n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts)}


def compute_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """XGBoost-style scale_pos_weight = n_negative / n_positive."""
    y = np.asarray(y)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0:
        raise ValueError("No positive-class samples found; cannot compute scale_pos_weight.")
    return float(n_neg / n_pos)


def resample_train_only(
    X: np.ndarray,
    y: np.ndarray,
    strategy: Literal["none", "smote", "random_undersample"] = "none",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample — TRAIN SPLIT ONLY. Never call this on validation/test data:
    doing so would let synthetic or duplicated minority rows leak evaluation
    signal. Default 'none' relies on class-weighting instead (see
    compute_class_weights / compute_scale_pos_weight)."""
    if strategy == "none":
        return X, y

    if strategy == "smote":
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(random_state=random_state)
    elif strategy == "random_undersample":
        from imblearn.under_sampling import RandomUnderSampler

        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown resampling strategy: {strategy}")

    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res
