"""Shared, cached data/model access for every dashboard page.

All transaction-level data shown anywhere in the dashboard is either
synthetic (src/data/loaders.load_synthetic) or drawn from experiment result
files already on disk — never real card data (spec section 22).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.classical.xgboost_model import XGBoostModel
from src.config.settings import load_experiments_config, load_quantum_config
from src.data.loaders import load_synthetic
from src.data.preprocessing import clean_dataframe, fit_transform_split, infer_feature_types, random_split
from src.hybrid.cost_sensitive_decision import CostModel
from src.hybrid.qgfda import QGFDA, QGFDAConfig
from src.quantum.vqc_model import VQCModel
from src.utils.experiment_tracker import REGISTRY_PATH, load_registry


@st.cache_resource(show_spinner="Training demo QGFDA pipeline on synthetic data...")
def get_demo_qgfda(
    n_samples: int = 6000,
    qubits: int = 3,
    gate_threshold: float = 0.80,
    max_quantum_fraction: float = 0.25,
    fusion_method: str = "weighted",
    seed: int = 42,
):
    """Trains one QGFDA instance on a synthetic ULB-schema sample and caches
    it for the lifetime of the Streamlit session (st.cache_resource — the
    trained model object, including the quantum primitives, is not
    JSON-serializable, so cache_data would not work here)."""
    ds = load_synthetic("ulb", n_samples=n_samples, random_state=seed)
    frame = clean_dataframe(ds.frame, ds.target)
    numeric, categorical = infer_feature_types(frame, ds.target, ds.categorical_features)
    splits = random_split(frame, ds.target, test_size=0.2, validation_size=0.15, random_state=seed)
    X_train, X_others, transformer, meta = fit_transform_split(
        splits["train"], {"val": splits["val"], "test": splits["test"]}, ds.target, numeric, categorical
    )
    y_train = splits["train"][ds.target].values
    y_val = splits["val"][ds.target].values
    y_test = splits["test"][ds.target].values

    exp_cfg = load_experiments_config()
    quantum_cfg = load_quantum_config()["quantum"]

    config = QGFDAConfig(
        quantum_feature_indices=[0, 1, 2][:qubits],
        gate_threshold=gate_threshold,
        risk_band=tuple(exp_cfg["hybrid"]["gate_risk_band"]),
        max_quantum_fraction=max_quantum_fraction,
        fusion_method=fusion_method,
        fusion_weights=exp_cfg["hybrid"]["fusion_weights"],
        cost_model=CostModel.from_config(exp_cfg["cost_model"]),
        threshold_search_grid_points=30,
        quantum_train_sample_size=150,
        random_state=seed,
    )
    classical_model = XGBoostModel(calibrate=False, random_state=seed)
    quantum_model = VQCModel(
        num_qubits=qubits, feature_map_name=quantum_cfg["feature_map"], feature_map_reps=1,
        ansatz_name=quantum_cfg["ansatz"], ansatz_reps=2, optimizer="cobyla", optimizer_maxiter=40,
        backend="simulator", shots=256, seed=seed,
    )

    model = QGFDA(classical_model, quantum_model, config)
    model.fit(X_train, y_train, X_others["val"], y_val)
    result = model.evaluate(X_others["test"], y_test)

    amounts = splits["test"]["Amount"].values if "Amount" in splits["test"].columns else None
    frame_out = result["predictions"].to_frame(amounts=amounts)
    frame_out["actual_fraud"] = y_test

    return {
        "model": model,
        "result": result,
        "frame": frame_out,
        "feature_map": model.quantum_model.feature_map,
        "ansatz": getattr(model.quantum_model, "ansatz", None),
    }


@st.cache_data(show_spinner=False)
def get_registry_df() -> pd.DataFrame:
    records = load_registry()
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    numeric_cols = ["pr_auc", "roc_auc", "recall", "precision", "f1", "fpr", "mcc", "expected_loss",
                     "qubits", "circuit_depth", "shots", "training_time_s", "inference_time_s"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def registry_csv_path() -> Path:
    return REGISTRY_PATH
