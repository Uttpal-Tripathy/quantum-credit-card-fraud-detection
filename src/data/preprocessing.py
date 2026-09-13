"""Preprocessing pipeline: missing values, duplicates, encoding, scaling,
reproducible splits (random + temporal), and feature metadata tracking.

Design choices (documented per spec section 6):
  - Scaling/encoding statistics are fit on the TRAIN split only, then applied
    to validation/test, to prevent leakage.
  - Duplicate rows are dropped before splitting (not after), so no duplicate
    of a training transaction can appear in the test split.
  - `PREPROCESSING_VERSION` is recorded in every experiment record so results
    remain traceable to the exact pipeline that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PREPROCESSING_VERSION = "v1"


@dataclass
class FeatureMetadata:
    numeric_features: list[str]
    categorical_features: list[str]
    dropped_features: list[str]
    target: str
    time_column: str | None
    output_feature_names: list[str] = field(default_factory=list)
    preprocessing_version: str = PREPROCESSING_VERSION


def build_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """A leakage-safe ColumnTransformer: median-impute + standard-scale numeric,
    most-frequent-impute + one-hot encode categorical."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def clean_dataframe(
    frame: pd.DataFrame,
    target: str,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Drop exact duplicates and explicitly-flagged leakage/id columns."""
    drop_columns = drop_columns or []
    cleaned = frame.drop(columns=[c for c in drop_columns if c in frame.columns])
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    if target not in cleaned.columns:
        raise ValueError(f"Target column '{target}' missing after cleaning.")
    return cleaned


def infer_feature_types(
    frame: pd.DataFrame,
    target: str,
    known_categorical: list[str] | None = None,
    max_numeric_cardinality_as_categorical: int = 0,
) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric vs categorical. `known_categorical`
    (from configs/datasets.yaml) always wins; everything else is inferred
    from dtype."""
    known_categorical = known_categorical or []
    feature_cols = [c for c in frame.columns if c != target]

    categorical = [c for c in feature_cols if c in known_categorical]
    numeric = []
    for c in feature_cols:
        if c in categorical:
            continue
        if pd.api.types.is_numeric_dtype(frame[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def fit_transform_split(
    train_frame: pd.DataFrame,
    other_frames: dict[str, pd.DataFrame],
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray], ColumnTransformer, FeatureMetadata]:
    """Fit the ColumnTransformer on `train_frame` only, then transform every
    frame in `other_frames` (e.g. {'val': ..., 'test': ...}) with those same
    fitted statistics. Returns (X_train, {name: X_...}, fitted_transformer, metadata)."""
    transformer = build_preprocessing_pipeline(numeric_features, categorical_features)
    X_train = transformer.fit_transform(train_frame[numeric_features + categorical_features])

    X_others = {
        name: transformer.transform(f[numeric_features + categorical_features])
        for name, f in other_frames.items()
    }

    try:
        output_names = list(transformer.get_feature_names_out())
    except Exception:
        output_names = [f"f{i}" for i in range(X_train.shape[1])]

    metadata = FeatureMetadata(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        dropped_features=[],
        target=target,
        time_column=None,
        output_feature_names=output_names,
    )
    return X_train, X_others, transformer, metadata


def random_split(
    frame: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Stratified random train/validation/test split (class-balance preserved
    in every split)."""
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        frame, test_size=test_size, stratify=frame[target], random_state=random_state
    )
    relative_val_size = validation_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val[target],
        random_state=random_state,
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def temporal_split(
    frame: pd.DataFrame,
    time_column: str,
    test_size: float = 0.2,
    validation_size: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Chronological split: earliest transactions train, latest test. No
    shuffling, no stratification — this intentionally preserves any temporal
    drift so downstream evaluation can measure it (see evaluation/temporal_validation.py)."""
    if time_column not in frame.columns:
        raise ValueError(f"time_column '{time_column}' not found in frame.")

    ordered = frame.sort_values(time_column).reset_index(drop=True)
    n = len(ordered)
    n_test = int(n * test_size)
    n_val = int(n * validation_size)
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError("test_size + validation_size too large for temporal_split.")

    return {
        "train": ordered.iloc[:n_train].reset_index(drop=True),
        "val": ordered.iloc[n_train:n_train + n_val].reset_index(drop=True),
        "test": ordered.iloc[n_train + n_val:].reset_index(drop=True),
    }


def split_data(
    frame: pd.DataFrame,
    target: str,
    strategy: Literal["random", "temporal"] = "random",
    time_column: str | None = None,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    if strategy == "temporal":
        if not time_column:
            raise ValueError("time_column is required for strategy='temporal'.")
        return temporal_split(frame, time_column, test_size, validation_size)
    return random_split(frame, target, test_size, validation_size, random_state)
