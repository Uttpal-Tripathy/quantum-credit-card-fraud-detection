"""Tests for evaluation metrics, statistical tests, and temporal validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_metrics, latency_stats
from src.evaluation.statistical_tests import (
    bootstrap_metric_ci,
    drift_status,
    ks_drift_test,
    mcnemar_test,
    population_stability_index,
)
from src.evaluation.temporal_validation import degradation_summary, rolling_window_splits


def test_compute_metrics_perfect_classifier():
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.0, 0.1, 0.05, 0.9, 0.95])
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.fpr == 0.0
    assert metrics.pr_auc == pytest.approx(1.0)


def test_compute_metrics_handles_no_positive_class():
    y_true = np.zeros(10)
    y_proba = np.random.default_rng(0).uniform(0, 1, 10)
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    assert np.isnan(metrics.pr_auc)
    assert metrics.n_positive == 0


def test_compute_metrics_expected_loss_scales_with_costs():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.1, 0.1, 0.9, 0.9])  # both classes fully misclassified at threshold 0.5
    cheap = compute_metrics(y_true, y_proba, threshold=0.5, fraud_loss=100, false_positive_cost=10)
    expensive = compute_metrics(y_true, y_proba, threshold=0.5, fraud_loss=1000, false_positive_cost=10)
    assert expensive.expected_loss > cheap.expected_loss


def test_latency_stats_percentiles_are_ordered():
    latencies = np.array([0.001, 0.002, 0.003, 0.004, 0.1])
    stats = latency_stats(latencies)
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]


def test_latency_stats_empty_input():
    stats = latency_stats([])
    assert np.isnan(stats["mean_ms"])


def test_bootstrap_metric_ci_contains_point_estimate():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.concatenate([rng.uniform(0, 0.6, 50), rng.uniform(0.4, 1.0, 50)])
    ci = bootstrap_metric_ci(y_true, y_proba, roc_auc_score, n_bootstrap=200, random_state=1)
    assert ci.ci_low <= ci.point_estimate <= ci.ci_high


def test_mcnemar_test_no_disagreement_returns_p_value_one():
    y_true = np.array([0, 1, 0, 1])
    result = mcnemar_test(y_true, y_true, y_true)
    assert result.p_value == 1.0
    assert result.significant_at_05 is False


def test_mcnemar_test_detects_clear_difference():
    y_true = np.array([1] * 30 + [0] * 30)
    pred_good = y_true.copy()  # always correct
    pred_bad = 1 - y_true      # always wrong
    result = mcnemar_test(y_true, pred_good, pred_bad)
    assert result.significant_at_05 is True


def test_population_stability_index_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=1000)
    psi = population_stability_index(ref, ref.copy())
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_population_stability_index_positive_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(loc=0, size=1000)
    shifted = rng.normal(loc=3, size=1000)
    psi = population_stability_index(ref, shifted)
    assert psi > 0.25


def test_ks_drift_test_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(loc=0, size=500)
    shifted = rng.normal(loc=2, size=500)
    result = ks_drift_test(ref, shifted, alpha=0.01)
    assert result.drifted is True


def test_drift_status_thresholds():
    assert drift_status(0.05, psi_warning=0.1, psi_critical=0.25) == "NORMAL"
    assert drift_status(0.15, psi_warning=0.1, psi_critical=0.25) == "WARNING"
    assert drift_status(0.3, psi_warning=0.1, psi_critical=0.25) == "CRITICAL"


def test_rolling_window_splits_produce_non_overlapping_test_windows():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"t": np.arange(1000), "y": rng.integers(0, 2, 1000)})
    folds = rolling_window_splits(frame, "t", n_folds=3, train_window_fraction=0.5, step_fraction=0.1)
    assert len(folds) <= 3
    for fold in folds:
        assert fold["train"]["t"].max() < fold["test"]["t"].min() or fold["train"]["t"].max() <= fold["test"]["t"].min()


def test_degradation_summary_needs_at_least_two_folds():
    summary = degradation_summary([])
    assert np.isnan(summary["pr_auc_delta"])
