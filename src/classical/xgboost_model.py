"""XGBoost baseline — typically the strongest classical baseline for tabular
imbalanced fraud data."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from xgboost import XGBClassifier

from src.classical.calibration import calibrate_model
from src.config.settings import load_models_config
from src.data.imbalance import compute_scale_pos_weight


@dataclass
class FitResult:
    training_time_s: float
    n_train: int
    scale_pos_weight_used: float


class XGBoostModel:
    name = "xgboost"

    def __init__(self, config: dict[str, Any] | None = None, calibrate: bool = True, random_state: int = 42):
        self.config = config or load_models_config()["xgboost"]
        self.calibrate = calibrate
        self.random_state = random_state
        self._estimator: XGBClassifier | None = None
        self.model = None
        self._fit_result: FitResult | None = None

    def _build_estimator(self, scale_pos_weight: float) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=self.config.get("n_estimators", 400),
            max_depth=self.config.get("max_depth", 6),
            learning_rate=self.config.get("learning_rate", 0.05),
            subsample=self.config.get("subsample", 0.8),
            colsample_bytree=self.config.get("colsample_bytree", 0.8),
            eval_metric=self.config.get("eval_metric", "aucpr"),
            tree_method=self.config.get("tree_method", "hist"),
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        spw = self.config.get("scale_pos_weight") or compute_scale_pos_weight(y)
        self._estimator = self._build_estimator(spw)

        cal_cfg = load_models_config().get("calibration", {})
        self.model = calibrate_model(
            self._estimator, method=cal_cfg.get("method", "isotonic"), cv=cal_cfg.get("cv", 3)
        ) if self.calibrate else self._estimator

        start = time.perf_counter()
        self.model.fit(X, y)
        elapsed = time.perf_counter() - start
        self._fit_result = FitResult(training_time_s=elapsed, n_train=len(y), scale_pos_weight_used=spw)
        return self._fit_result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict_proba_timed(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        proba = self.predict_proba(X)
        elapsed = time.perf_counter() - start
        return proba, elapsed

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        if self._estimator is None or not hasattr(self._estimator, "feature_importances_"):
            raise RuntimeError("Model must be fit before requesting feature importance.")
        importances = self._estimator.feature_importances_
        return dict(sorted(zip(feature_names, importances.tolist()), key=lambda kv: -kv[1]))

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance._estimator = getattr(instance.model, "estimator", instance.model)
        instance.config = {}
        instance.calibrate = True
        instance.random_state = 42
        instance._fit_result = None
        return instance
