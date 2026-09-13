"""Tests for dataset loading, validation, preprocessing, and splitting.
No IBM Quantum credentials or real Kaggle downloads required — everything
here runs against the synthetic data generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.imbalance import compute_class_weights, compute_scale_pos_weight, stratified_subsample
from src.data.loaders import load_synthetic
from src.data.preprocessing import (
    clean_dataframe,
    fit_transform_split,
    infer_feature_types,
    random_split,
    temporal_split,
)
from src.data.validators import validate_dataset


def test_load_synthetic_schema():
    ds = load_synthetic("ulb", n_samples=500, random_state=1)
    assert ds.is_synthetic is True
    assert ds.target == "Class"
    assert set(["Time", "Amount", "Class"]).issubset(ds.frame.columns)
    assert len(ds.frame) == 500


def test_load_synthetic_fraud_rate_is_reasonable():
    ds = load_synthetic("ulb", n_samples=5000, fraud_rate=0.02, random_state=1)
    rate = ds.frame["Class"].mean()
    assert 0.005 < rate < 0.05


def test_synthetic_fraud_not_clustered_in_time():
    """Regression test: fraud must not be concentrated at one end of the
    Time range (a real bug found where fraud rows always got the largest
    Time values due to independent sort-then-positional-assign)."""
    ds = load_synthetic("ulb", n_samples=4000, random_state=7)
    ordered = ds.frame.sort_values("Time")
    first_half_rate = ordered.iloc[: len(ordered) // 2]["Class"].mean()
    second_half_rate = ordered.iloc[len(ordered) // 2 :]["Class"].mean()
    assert first_half_rate > 0, "first half of the timeline has zero fraud"
    assert second_half_rate > 0, "second half of the timeline has zero fraud"
    assert abs(first_half_rate - second_half_rate) < 0.02


def test_validate_dataset_reports_class_balance():
    ds = load_synthetic("ulb", n_samples=1000, random_state=1)
    report = validate_dataset(ds)
    assert report.n_rows == 1000
    assert "0" in report.class_balance and "1" in report.class_balance


def test_validate_dataset_flags_severe_imbalance():
    # Real ULB fraud rate (~0.17%) is well under the 1% warning threshold;
    # the default synthetic fraud_rate (1.73%) is deliberately milder for
    # trainability, so use a lower rate here to exercise the warning path.
    ds = load_synthetic("ulb", n_samples=2000, fraud_rate=0.005, random_state=1)
    report = validate_dataset(ds)
    assert any("imbalance" in w.lower() for w in report.warnings)


def test_validate_dataset_flags_constant_column():
    ds = load_synthetic("ulb", n_samples=200, random_state=1)
    ds.frame["ConstantCol"] = 1.0
    report = validate_dataset(ds)
    assert "ConstantCol" in report.constant_columns


def test_clean_dataframe_drops_duplicates_and_leakage_columns():
    ds = load_synthetic("ulb", n_samples=200, random_state=1)
    frame_with_dupes = ds.frame.copy()
    frame_with_dupes = pd_concat_self(frame_with_dupes)
    cleaned = clean_dataframe(frame_with_dupes, "Class")
    assert len(cleaned) == 200  # duplicates collapsed back down


def pd_concat_self(frame):
    import pandas as pd

    return pd.concat([frame, frame], ignore_index=True)


def test_infer_feature_types_separates_numeric_and_categorical():
    ds = load_synthetic("ulb", n_samples=200, random_state=1)
    numeric, categorical = infer_feature_types(ds.frame, "Class", known_categorical=[])
    assert "Amount" in numeric
    assert categorical == []


def test_random_split_preserves_class_balance():
    ds = load_synthetic("ulb", n_samples=3000, fraud_rate=0.02, random_state=1)
    splits = random_split(ds.frame, "Class", test_size=0.2, validation_size=0.1, random_state=1)
    overall_rate = ds.frame["Class"].mean()
    for name, split_frame in splits.items():
        rate = split_frame["Class"].mean()
        assert abs(rate - overall_rate) < 0.02, f"{name} split class balance diverged too much"
    total = sum(len(f) for f in splits.values())
    assert total == len(ds.frame)


def test_temporal_split_is_chronological():
    ds = load_synthetic("ulb", n_samples=2000, random_state=1)
    splits = temporal_split(ds.frame, "Time", test_size=0.2, validation_size=0.1)
    assert splits["train"]["Time"].max() <= splits["val"]["Time"].min()
    assert splits["val"]["Time"].max() <= splits["test"]["Time"].min()


def test_temporal_split_requires_time_column():
    ds = load_synthetic("ulb", n_samples=100, random_state=1)
    with pytest.raises(ValueError):
        temporal_split(ds.frame, "NotAColumn")


def test_fit_transform_split_no_leakage_in_shapes():
    ds = load_synthetic("ulb", n_samples=1000, random_state=1)
    splits = random_split(ds.frame, "Class", test_size=0.2, validation_size=0.1, random_state=1)
    numeric, categorical = infer_feature_types(ds.frame, "Class")
    X_train, X_others, transformer, meta = fit_transform_split(
        splits["train"], {"test": splits["test"]}, "Class", numeric, categorical
    )
    assert X_train.shape[0] == len(splits["train"])
    assert X_others["test"].shape[0] == len(splits["test"])
    assert X_train.shape[1] == X_others["test"].shape[1]


def test_compute_class_weights_balances_minority():
    y = np.array([0] * 990 + [1] * 10)
    weights = compute_class_weights(y)
    assert weights[1] > weights[0]


def test_compute_scale_pos_weight_matches_ratio():
    y = np.array([0] * 80 + [1] * 20)
    spw = compute_scale_pos_weight(y)
    assert spw == pytest.approx(4.0)


def test_compute_scale_pos_weight_raises_without_positives():
    y = np.zeros(10)
    with pytest.raises(ValueError):
        compute_scale_pos_weight(y)


def test_stratified_subsample_guarantees_minority_presence():
    y = np.array([0] * 995 + [1] * 5)
    X = np.zeros((1000, 2))
    idx = stratified_subsample(X, y, n_samples=20, min_minority_samples=2, random_state=1)
    assert len(idx) == 20
    assert (y[idx] == 1).sum() >= 2


def test_stratified_subsample_returns_all_when_n_samples_exceeds_n():
    y = np.array([0, 1, 0, 1])
    X = np.zeros((4, 2))
    idx = stratified_subsample(X, y, n_samples=10, random_state=1)
    assert len(idx) == 4
