"""Classical model explainability: feature importance, SHAP, permutation
importance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


@dataclass
class ExplanationResult:
    method: str
    feature_names: list[str]
    importances: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": self.importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def permutation_importance_explanation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "average_precision",
) -> ExplanationResult:
    """Model-agnostic importance: works for any estimator exposing
    predict/predict_proba, including calibrated classical models. Uses
    average_precision (~PR-AUC) as the scoring metric, consistent with this
    project's primary metric."""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring, n_jobs=-1
    )
    return ExplanationResult(
        method="permutation_importance",
        feature_names=feature_names,
        importances=result.importances_mean,
    )


def shap_explanation(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    max_background_samples: int = 200,
) -> tuple[ExplanationResult, np.ndarray]:
    """KernelExplainer/TreeExplainer-based SHAP values (auto-selected by the
    `shap` library based on model type). Returns both a global importance
    summary (mean |SHAP value| per feature) and the raw per-sample SHAP
    matrix for local explanations. `X_background` is subsampled for tractable
    runtime on kernel-based explainers."""
    import shap

    if len(X_background) > max_background_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_background), size=max_background_samples, replace=False)
        X_background = X_background[idx]

    predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class-1 (fraud) SHAP values
    except Exception:
        explainer = shap.KernelExplainer(lambda x: predict_fn(x)[:, 1] if predict_fn(x).ndim > 1 else predict_fn(x), X_background)
        shap_values = explainer.shap_values(X_explain, nsamples="auto")

    shap_values = np.asarray(shap_values)
    mean_abs = np.abs(shap_values).mean(axis=0)

    return ExplanationResult(method="shap", feature_names=feature_names, importances=mean_abs), shap_values


def native_feature_importance(model, feature_names: list[str]) -> ExplanationResult:
    """Use a model's own `.feature_importance(feature_names)` method if
    available (all wrappers in src/classical/* implement this) — cheapest
    and most faithful explanation for tree/linear models."""
    if not hasattr(model, "feature_importance"):
        raise AttributeError(f"{type(model).__name__} has no native feature_importance method.")
    importance_dict = model.feature_importance(feature_names)
    return ExplanationResult(
        method="native",
        feature_names=list(importance_dict.keys()),
        importances=np.array(list(importance_dict.values())),
    )
