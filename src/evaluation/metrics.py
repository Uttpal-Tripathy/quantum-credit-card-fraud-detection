"""Evaluation metrics for imbalanced fraud detection.

PR-AUC (average precision) is the PRIMARY metric throughout this project —
never accuracy, which is meaningless at a ~0.17% fraud rate (a model that
predicts "legitimate" for everything scores >99.8% accuracy while catching
zero fraud). ROC-AUC, precision/recall/F1, FPR/FNR, MCC, Brier score, and
expected monetary loss are all reported alongside it (see docs/methodology.md).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    pr_auc: float
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    mcc: float
    brier_score: float
    expected_loss: float
    n_samples: int
    n_positive: int
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_expected_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fraud_loss: float = 500.0,
    false_positive_cost: float = 25.0,
) -> float:
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return fn * fraud_loss + fp * false_positive_cost


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    fraud_loss: float = 500.0,
    false_positive_cost: float = 25.0,
) -> ClassificationMetrics:
    """Compute the full metric suite at a fixed decision threshold. Ranking
    metrics (pr_auc/roc_auc/average_precision/brier_score) are threshold-free
    and computed from probabilities; the rest depend on `threshold`."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    n_samples = len(y_true)
    n_positive = int(y_true.sum())

    if n_positive == 0 or n_positive == n_samples:
        roc_auc = float("nan")
        pr_auc = float("nan")
        avg_precision = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_proba))
        precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        pr_auc = float(trapezoid(precisions[::-1], recalls[::-1]))
        avg_precision = float(average_precision_score(y_true, y_proba))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return ClassificationMetrics(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        average_precision=avg_precision,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        fpr=fpr,
        fnr=fnr,
        mcc=float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        brier_score=float(brier_score_loss(y_true, y_proba)),
        expected_loss=compute_expected_loss(y_true, y_pred, fraud_loss, false_positive_cost),
        n_samples=n_samples,
        n_positive=n_positive,
        threshold=threshold,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )


def latency_stats(latencies_s: list[float] | np.ndarray) -> dict[str, float]:
    """Summary latency stats (mean/p50/p95/p99) in milliseconds, for the
    dashboard and resource-analysis plots."""
    arr = np.asarray(latencies_s, dtype=float) * 1000.0
    if arr.size == 0:
        return {"mean_ms": float("nan"), "p50_ms": float("nan"), "p95_ms": float("nan"), "p99_ms": float("nan")}
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }
