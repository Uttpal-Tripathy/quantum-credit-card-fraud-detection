"""Logistic Regression baseline (linear, fast, highly interpretable — the
sanity-check floor every other model must beat)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.classical.calibration import calibrate_model
from src.config.settings import load_models_config


@dataclass
class FitResult:
    training_time_s: float
    n_train: int


class LogisticRegressionModel:
    """Thin, serializable wrapper around sklearn's LogisticRegression with
    optional probability calibration."""

    name = "logistic_regression"

    def __init__(self, config: dict[str, Any] | None = None, calibrate: bool = True):
        self.config = config or load_models_config()["logistic_regression"]
        self.calibrate = calibrate
        lr_kwargs: dict[str, Any] = dict(
            C=self.config.get("C", 1.0),
            class_weight=self.config.get("class_weight", "balanced"),
            max_iter=self.config.get("max_iter", 2000),
            solver=self.config.get("solver", "lbfgs"),
        )
        # Only pass `penalty` when it differs from sklearn's default ('l2'):
        # newer sklearn deprecates the explicit kwarg in favor of l1_ratio
        # and warns even when the value matches the default.
        penalty = self.config.get("penalty", "l2")
        if penalty != "l2":
            lr_kwargs["penalty"] = penalty
        self._estimator = LogisticRegression(**lr_kwargs)
        self.model = calibrate_model(
            self._estimator,
            method=load_models_config().get("calibration", {}).get("method", "isotonic"),
            cv=load_models_config().get("calibration", {}).get("cv", 3),
        ) if calibrate else self._estimator
        self._fit_result: FitResult | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        start = time.perf_counter()
        self.model.fit(X, y)
        elapsed = time.perf_counter() - start
        self._fit_result = FitResult(training_time_s=elapsed, n_train=len(y))
        return self._fit_result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict_proba_timed(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        proba = self.predict_proba(X)
        elapsed = time.perf_counter() - start
        return proba, elapsed

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Absolute standardized coefficients as a simple importance proxy."""
        estimator = self._estimator
        if not hasattr(estimator, "coef_"):
            raise RuntimeError("Model must be fit before requesting feature importance.")
        coefs = np.abs(estimator.coef_[0])
        return dict(sorted(zip(feature_names, coefs.tolist()), key=lambda kv: -kv[1]))

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionModel":
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance._estimator = getattr(instance.model, "estimator", instance.model)
        instance.config = {}
        instance.calibrate = True
        instance._fit_result = None
        return instance
