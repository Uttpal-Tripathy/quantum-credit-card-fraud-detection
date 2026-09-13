"""Statistical significance testing (for comparing models honestly) and
distributional drift tests (PSI / KS), used by the drift dashboard page.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class BootstrapCI:
    metric_name: str
    point_estimate: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    confidence: float


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric_fn,
    metric_name: str = "metric",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> BootstrapCI:
    """Non-parametric bootstrap confidence interval for any metric_fn(y_true,
    y_proba) -> float. Used to attach uncertainty to PR-AUC / recall / etc.
    before claiming one model beats another."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    point = metric_fn(y_true, y_proba)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_true[idx], y_proba[idx])

    alpha = 1 - confidence
    lo, hi = np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(
        metric_name=metric_name,
        point_estimate=float(point),
        ci_low=float(lo),
        ci_high=float(hi),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )


@dataclass
class McNemarResult:
    statistic: float
    p_value: float
    significant_at_05: bool
    n_disagreements: int


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> McNemarResult:
    """Paired test for whether two classifiers' error rates differ
    significantly on the SAME test set — the right test for "is Model A
    really better than Model B", not two independent CIs eyeballed together."""
    y_true = np.asarray(y_true)
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    n_disagreements = b + c

    if n_disagreements == 0:
        return McNemarResult(statistic=0.0, p_value=1.0, significant_at_05=False, n_disagreements=0)

    if n_disagreements < 25:
        # exact binomial test for small disagreement counts
        p_value = float(stats.binomtest(min(b, c), n_disagreements, 0.5).pvalue)
        statistic = float(abs(b - c))
    else:
        statistic = float((abs(b - c) - 1) ** 2 / n_disagreements)
        p_value = float(1 - stats.chi2.cdf(statistic, df=1))

    return McNemarResult(
        statistic=statistic, p_value=p_value, significant_at_05=p_value < 0.05, n_disagreements=n_disagreements
    )


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between a reference ('expected') distribution and a new ('actual')
    one. Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 major
    shift — thresholds configurable via configs/experiments.yaml (drift.*)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    quantiles = np.linspace(0, 1, n_bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) < 3:
        return 0.0  # degenerate distribution (e.g. near-constant feature)

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = np.clip(expected_counts / max(len(expected), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(len(actual), 1), 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


@dataclass
class KSDriftResult:
    statistic: float
    p_value: float
    drifted: bool


def ks_drift_test(reference: np.ndarray, current: np.ndarray, alpha: float = 0.01) -> KSDriftResult:
    """Two-sample Kolmogorov-Smirnov test for whether a feature's marginal
    distribution has shifted between a reference window and the current one."""
    statistic, p_value = stats.ks_2samp(reference, current)
    return KSDriftResult(statistic=float(statistic), p_value=float(p_value), drifted=p_value < alpha)


def drift_status(psi: float, psi_warning: float = 0.1, psi_critical: float = 0.25) -> str:
    """Map a PSI value to the dashboard's traffic-light drift status."""
    if psi >= psi_critical:
        return "CRITICAL"
    if psi >= psi_warning:
        return "WARNING"
    return "NORMAL"
