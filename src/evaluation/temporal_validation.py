"""Temporal validation: chronological splitting, rolling-window / forward-
chaining evaluation, and drift-over-time measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation.metrics import ClassificationMetrics, compute_metrics


@dataclass
class RollingWindowResult:
    window_index: int
    train_range: tuple[float, float]
    test_range: tuple[float, float]
    n_train: int
    n_test: int
    metrics: ClassificationMetrics


def rolling_window_splits(
    frame: pd.DataFrame,
    time_column: str,
    n_folds: int = 5,
    train_window_fraction: float = 0.6,
    step_fraction: float = 0.1,
) -> list[dict[str, pd.DataFrame]]:
    """Forward-chaining / rolling-window splits: each fold trains on a window
    of `train_window_fraction` of the (time-sorted) data and tests on the
    next `step_fraction` slice, sliding forward `n_folds` times. This is the
    "Month 1-3 train / Month 4 test, then roll forward" pattern from the spec,
    generalized to fractional windows so it works across datasets with very
    different time ranges."""
    ordered = frame.sort_values(time_column).reset_index(drop=True)
    n = len(ordered)
    train_size = int(n * train_window_fraction)
    step_size = max(1, int(n * step_fraction))

    folds = []
    start = 0
    for _ in range(n_folds):
        train_end = start + train_size
        test_end = train_end + step_size
        if test_end > n:
            break
        folds.append({
            "train": ordered.iloc[start:train_end].reset_index(drop=True),
            "test": ordered.iloc[train_end:test_end].reset_index(drop=True),
        })
        start += step_size
    return folds


def evaluate_rolling_windows(
    frame: pd.DataFrame,
    time_column: str,
    target: str,
    fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame], tuple[np.ndarray, np.ndarray]],
    n_folds: int = 5,
    train_window_fraction: float = 0.6,
    step_fraction: float = 0.1,
    threshold: float = 0.5,
) -> list[RollingWindowResult]:
    """Run `fit_predict_fn(train_frame, test_frame) -> (y_test, y_proba)` over
    each rolling window and collect per-fold metrics, so callers can plot
    performance degradation over time (see docs on temporal drift)."""
    folds = rolling_window_splits(frame, time_column, n_folds, train_window_fraction, step_fraction)
    results = []
    for i, fold in enumerate(folds):
        y_test, y_proba = fit_predict_fn(fold["train"], fold["test"])
        metrics = compute_metrics(y_test, y_proba, threshold=threshold)
        results.append(RollingWindowResult(
            window_index=i,
            train_range=(float(fold["train"][time_column].min()), float(fold["train"][time_column].max())),
            test_range=(float(fold["test"][time_column].min()), float(fold["test"][time_column].max())),
            n_train=len(fold["train"]),
            n_test=len(fold["test"]),
            metrics=metrics,
        ))
    return results


def degradation_summary(results: list[RollingWindowResult]) -> dict[str, float]:
    """Summarize how much PR-AUC degrades from the first to the last rolling
    window — the headline temporal-drift number for the research report."""
    if len(results) < 2:
        return {"pr_auc_first": float("nan"), "pr_auc_last": float("nan"), "pr_auc_delta": float("nan")}
    first, last = results[0].metrics.pr_auc, results[-1].metrics.pr_auc
    return {
        "pr_auc_first": first,
        "pr_auc_last": last,
        "pr_auc_delta": last - first,
    }
