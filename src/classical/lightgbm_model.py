"""LightGBM baseline — fast gradient boosting, strong on large tabular data
(useful for PaySim/IEEE-CIS scale)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMClassifier

from src.classical.calibration import calibrate_model
from src.config.settings import load_models_config


@dataclass
class FitResult:
    training_time_s: float
    n_train: int


class LightGBMModel:
    name = "lightgbm"

    def __init__(self, config: dict[str, Any] | None = None, calibrate: bool = True, random_state: int = 42):
        self.config = config or load_models_config()["lightgbm"]
        self.calibrate = calibrate
        self._estimator = LGBMClassifier(
            n_estimators=self.config.get("n_estimators", 400),
            max_depth=self.config.get("max_depth", -1),
            num_leaves=self.config.get("num_leaves", 63),
            learning_rate=self.config.get("learning_rate", 0.05),
            subsample=self.config.get("subsample", 0.8),
            colsample_bytree=self.config.get("colsample_bytree", 0.8),
            objective=self.config.get("objective", "binary"),
            is_unbalance=self.config.get("is_unbalance", True),
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        cal_cfg = load_models_config().get("calibration", {})
        self.model = calibrate_model(
            self._estimator, method=cal_cfg.get("method", "isotonic"), cv=cal_cfg.get("cv", 3)
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
        if not hasattr(self._estimator, "feature_importances_"):
            raise RuntimeError("Model must be fit before requesting feature importance.")
        importances = self._estimator.feature_importances_
        return dict(sorted(zip(feature_names, importances.tolist()), key=lambda kv: -kv[1]))

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance._estimator = getattr(instance.model, "estimator", instance.model)
        instance.config = {}
        instance.calibrate = True
        instance._fit_result = None
        return instance
