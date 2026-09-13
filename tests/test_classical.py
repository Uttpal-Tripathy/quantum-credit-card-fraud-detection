"""Tests for classical baseline models, calibration, and threshold tuning."""

from __future__ import annotations

import numpy as np
import pytest

from src.classical.calibration import (
    tune_threshold_expected_loss,
    tune_threshold_f1,
    tune_threshold_youden_j,
)
from src.classical.lightgbm_model import LightGBMModel
from src.classical.logistic_regression import LogisticRegressionModel
from src.classical.random_forest import RandomForestModel
from src.classical.xgboost_model import XGBoostModel


@pytest.fixture
def separable_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.mark.parametrize("ModelClass", [LogisticRegressionModel, RandomForestModel, XGBoostModel, LightGBMModel])
def test_model_fits_and_predicts_valid_probabilities(ModelClass, separable_data):
    X, y = separable_data
    model = ModelClass(calibrate=False)
    fit_result = model.fit(X, y)
    assert fit_result.training_time_s >= 0
    assert fit_result.n_train == len(y)

    proba = model.predict_proba(X)
    assert proba.shape == (len(y),)
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.parametrize("ModelClass", [LogisticRegressionModel, XGBoostModel])
def test_model_learns_better_than_random(ModelClass, separable_data):
    from sklearn.metrics import roc_auc_score

    X, y = separable_data
    model = ModelClass(calibrate=False)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert roc_auc_score(y, proba) > 0.85


def test_calibrated_model_still_predicts_valid_probabilities(separable_data):
    X, y = separable_data
    model = LogisticRegressionModel(calibrate=True)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert np.all((proba >= 0) & (proba <= 1))


def test_native_feature_importance_returns_all_features(separable_data):
    X, y = separable_data
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    model = RandomForestModel(calibrate=False)
    model.fit(X, y)
    importances = model.feature_importance(feature_names)
    assert set(importances.keys()) == set(feature_names)
    assert all(v >= 0 for v in importances.values())


def test_tune_threshold_expected_loss_picks_a_valid_threshold():
    y_true = np.array([0] * 90 + [1] * 10)
    y_proba = np.concatenate([np.random.default_rng(1).uniform(0, 0.4, 90), np.random.default_rng(2).uniform(0.6, 1.0, 10)])
    result = tune_threshold_expected_loss(y_true, y_proba, fraud_loss=500, false_positive_cost=25)
    assert 0.0 <= result.threshold <= 1.0
    assert result.method == "expected_loss"


def test_tune_threshold_f1_beats_extreme_thresholds():
    rng = np.random.default_rng(3)
    y_true = np.array([0] * 80 + [1] * 20)
    y_proba = np.concatenate([rng.uniform(0, 0.5, 80), rng.uniform(0.5, 1.0, 20)])
    result = tune_threshold_f1(y_true, y_proba)
    assert result.score_at_threshold > 0


def test_tune_threshold_youden_j_within_bounds():
    rng = np.random.default_rng(4)
    y_true = np.array([0] * 80 + [1] * 20)
    y_proba = np.concatenate([rng.uniform(0, 0.5, 80), rng.uniform(0.5, 1.0, 20)])
    result = tune_threshold_youden_j(y_true, y_proba)
    assert 0.0 <= result.threshold <= 1.0
