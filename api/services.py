"""Service layer: builds and caches one QGFDA demo pipeline in-process at
startup (see api/main.py's lifespan), and exposes plain-Python functions the
routers call. Kept separate from the routers so the business logic has no
FastAPI-specific dependencies and stays testable on its own.

This mirrors dashboard/components/data.py's approach (same underlying src/
pipeline) but uses a simple threading.Lock-guarded singleton instead of
Streamlit's cache decorators, since this runs under uvicorn instead.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import matplotlib

# Must be set before any other module imports matplotlib.pyplot (which
# api.routers.quantum does indirectly via quantum_circuit_info below).
# Without this, matplotlib defaults to a GUI backend (e.g. TkAgg) on some
# platforms, which is not thread-safe — since FastAPI/Starlette runs sync
# route handlers (including circuit rendering) in worker threads, that
# caused "main thread is not in main loop" errors under concurrent requests.
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src.classical.xgboost_model import XGBoostModel
from src.config.settings import load_experiments_config, load_quantum_config
from src.data.loaders import load_synthetic
from src.data.preprocessing import clean_dataframe, fit_transform_split, infer_feature_types, random_split
from src.evaluation.statistical_tests import drift_status, ks_drift_test, population_stability_index
from src.hybrid.cost_sensitive_decision import CostModel
from src.hybrid.qgfda import QGFDA, QGFDAConfig
from src.quantum.ansatzes import ANSATZ_NAMES, build_ansatz
from src.quantum.circuit_metrics import compute_circuit_metrics
from src.quantum.feature_maps import FEATURE_MAP_NAMES, build_feature_map
from src.quantum.qubo_feature_selection import qafs_select_features
from src.data.imbalance import stratified_subsample
from src.quantum.vqc_model import VQCModel
from src.utils.experiment_tracker import load_registry
from src.utils.logging import get_logger

logger = get_logger("api.services")

_lock = threading.Lock()


@dataclass
class DemoState:
    model: QGFDA
    result: dict
    frame: pd.DataFrame
    transformer: Any
    numeric_features: list[str]
    categorical_features: list[str]
    target: str
    time_column: str | None
    amount_column: str | None
    built_at: float = field(default_factory=time.time)


_demo_state: DemoState | None = None


def build_demo_state(n_samples: int = 6000, qubits: int = 3, seed: int = 42) -> DemoState:
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
        quantum_feature_indices=list(range(qubits)),
        gate_threshold=exp_cfg["hybrid"]["gate_threshold"],
        risk_band=tuple(exp_cfg["hybrid"]["gate_risk_band"]),
        max_quantum_fraction=exp_cfg["hybrid"]["max_quantum_fraction"],
        fusion_method=exp_cfg["hybrid"]["fusion_method"],
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

    return DemoState(
        model=model, result=result, frame=frame_out, transformer=transformer,
        numeric_features=numeric, categorical_features=categorical, target=ds.target,
        time_column=ds.time_column, amount_column=ds.amount_column,
    )


def get_demo_state(force_rebuild: bool = False) -> DemoState:
    global _demo_state
    with _lock:
        if _demo_state is None or force_rebuild:
            logger.info("Building demo QGFDA pipeline (this trains a small VQC, ~30-60s)...")
            _demo_state = build_demo_state()
            logger.info("Demo pipeline ready.")
        return _demo_state


# ------------------------------------------------------------------ views --

def overview() -> dict:
    state = get_demo_state()
    result = state.result
    frame = state.frame
    metrics = result["binary_metrics"]
    decision_summary = result["decision_summary"]

    n_total = len(frame)
    n_fraud = int(frame["actual_fraud"].sum())

    return {
        "total_transactions": n_total,
        "fraud_cases": n_fraud,
        "fraud_rate": n_fraud / n_total if n_total else 0.0,
        "quantum_inference_pct": result["quantum_fraction"],
        "pr_auc": metrics.pr_auc,
        "roc_auc": metrics.roc_auc,
        "recall": metrics.recall,
        "precision": metrics.precision,
        "f1": metrics.f1,
        "fpr": metrics.fpr,
        "mcc": metrics.mcc,
        "avg_classical_latency_ms": result["avg_classical_latency_s"] * 1000,
        "avg_quantum_latency_ms": result["avg_quantum_latency_s"] * 1000,
        "qubits": state.model.quantum_model.num_qubits,
        "circuit_depth": compute_circuit_metrics(state.model.quantum_model.feature_map).logical_depth,
        "decision_summary": {
            "approve": decision_summary.n_approve,
            "review": decision_summary.n_review,
            "block": decision_summary.n_block,
            "approve_rate": decision_summary.approve_rate,
            "review_rate": decision_summary.review_rate,
            "block_rate": decision_summary.block_rate,
            "expected_loss": decision_summary.expected_loss,
        },
        "is_synthetic": True,
        "backend": state.model.quantum_model.primitives.backend_info.backend_name,
    }


def list_transactions(
    limit: int = 50,
    decisions: list[str] | None = None,
    routed_only: bool = False,
) -> list[dict]:
    state = get_demo_state()
    view = state.frame
    if decisions:
        view = view[view["decision"].isin(decisions)]
    if routed_only:
        view = view[view["routed_to_quantum"]]
    view = view.sort_values("final_risk", ascending=False).head(limit)

    records = []
    for _, row in view.iterrows():
        records.append({
            "transaction_id": row["transaction_id"],
            "amount": None if pd.isna(row["amount"]) else float(row["amount"]),
            "classical_risk": float(row["classical_risk"]),
            "quantum_risk": None if pd.isna(row["quantum_risk"]) else float(row["quantum_risk"]),
            "final_risk": float(row["final_risk"]),
            "routed_to_quantum": bool(row["routed_to_quantum"]),
            "decision": row["decision"],
        })
    return records


def score_new_transaction(amount_multiplier: float = 1.0) -> dict:
    """Draws one fresh synthetic transaction, preprocesses it through the
    SAME fitted transformer the demo model was trained with, and runs it
    through the live QGFDA pipeline — a genuine end-to-end inference call,
    not a lookup."""
    state = get_demo_state()
    seed = int(time.time() * 1000) % 1_000_000
    ds = load_synthetic("ulb", n_samples=1, random_state=seed)
    raw = ds.frame.copy()
    if state.amount_column in raw.columns:
        raw[state.amount_column] = raw[state.amount_column] * amount_multiplier

    cols = state.numeric_features + state.categorical_features
    X = state.transformer.transform(raw[cols])
    predictions = state.model.predict(X)

    return {
        "transaction_id": f"TXN-LIVE-{uuid.uuid4().hex[:10].upper()}",
        "amount": float(raw[state.amount_column].iloc[0]) if state.amount_column in raw.columns else None,
        "classical_risk": float(predictions.classical_score[0]),
        "quantum_risk": None if np.isnan(predictions.quantum_score[0]) else float(predictions.quantum_score[0]),
        "final_risk": float(predictions.final_score[0]),
        "routed_to_quantum": bool(predictions.routed_to_quantum[0]),
        "decision": str(predictions.decision[0]),
        "synthetic_ground_truth_label": int(raw[state.target].iloc[0]),
        "classical_latency_ms": float(predictions.classical_latency_s[0]) * 1000,
        "quantum_latency_ms": float(predictions.quantum_latency_s[0]) * 1000,
    }


def quantum_circuit_info(
    feature_map: str = "zz", ansatz: str = "real_amplitudes", qubits: int = 4, reps: int = 2,
) -> dict:
    if feature_map not in FEATURE_MAP_NAMES:
        raise ValueError(f"Unknown feature_map '{feature_map}'. Choose from {FEATURE_MAP_NAMES}.")
    if ansatz not in ANSATZ_NAMES:
        raise ValueError(f"Unknown ansatz '{ansatz}'. Choose from {ANSATZ_NAMES}.")

    fm = build_feature_map(feature_map, qubits, reps)
    az = build_ansatz(ansatz, qubits, reps)
    circuit = fm.compose(az)
    metrics = compute_circuit_metrics(circuit)

    fig = circuit.decompose().draw("mpl", style="iqp")
    import io
    import base64

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "feature_map": feature_map,
        "ansatz": ansatz,
        "qubits": qubits,
        "reps": reps,
        "logical_depth": metrics.logical_depth,
        "transpiled_depth": metrics.transpiled_depth,
        "two_qubit_gate_count": metrics.two_qubit_gate_count,
        "image_base64_png": image_b64,
    }


def demo_quantum_lab_summary() -> dict:
    state = get_demo_state()
    qm = state.model.quantum_model
    return {
        "qubits": qm.num_qubits,
        "shots": qm.shots,
        "optimizer": qm.optimizer_name,
        "backend": qm.primitives.backend_info.backend_name,
        "noise_enabled": qm.primitives.backend_info.is_noisy,
    }


_qafs_cache: dict[tuple, list[dict]] = {}


def feature_selection(candidate_counts: tuple[int, ...] = (2, 4, 6, 8), seed: int = 42) -> list[dict]:
    key = (candidate_counts, seed)
    if key in _qafs_cache:
        return _qafs_cache[key]

    ds = load_synthetic("ulb", n_samples=4000, random_state=seed)
    frame = clean_dataframe(ds.frame, ds.target)
    numeric, categorical = infer_feature_types(frame, ds.target, ds.categorical_features)
    splits = random_split(frame, ds.target, test_size=0.2, validation_size=0.15, random_state=seed)
    X_train, X_others, transformer, meta = fit_transform_split(
        splits["train"], {"val": splits["val"]}, ds.target, numeric, categorical
    )
    y_train = splits["train"][ds.target].values
    y_val = splits["val"][ds.target].values
    feature_names = meta.output_feature_names

    tr_idx = stratified_subsample(X_train, y_train, 300, random_state=seed)
    val_idx = stratified_subsample(X_others["val"], y_val, 150, random_state=seed)
    weights = {"alpha": 1.0, "beta": 0.5, "gamma": 0.2, "delta": 0.3, "epsilon": 0.2, "zeta": 0.2}

    candidates = qafs_select_features(
        X_train[tr_idx], y_train[tr_idx], X_others["val"][val_idx], y_val[val_idx],
        feature_names, list(candidate_counts), weights, random_state=seed,
    )
    rows = [{
        "n_features": c.n_features, "qubit_cost": c.qubit_cost, "pr_auc": c.pr_auc, "fpr": c.fpr,
        "circuit_depth": c.circuit_depth, "latency_ms": c.latency_s * 1000, "J": c.J,
        "solver": c.solver_method,
        "selected_features": [f.replace("numeric__", "") for f in c.selected_features],
    } for c in candidates]
    _qafs_cache[key] = rows
    return rows


def drift_report() -> dict:
    state = get_demo_state()
    frame = state.frame
    half = len(frame) // 2
    reference = frame["amount"].dropna().values[:half]
    current = frame["amount"].dropna().values[half:]

    psi = population_stability_index(reference, current)
    ks_result = ks_drift_test(reference, current, alpha=0.01)
    status = drift_status(psi, psi_warning=0.1, psi_critical=0.25)

    return {
        "psi": psi,
        "ks_statistic": ks_result.statistic,
        "ks_p_value": ks_result.p_value,
        "drifted": ks_result.drifted,
        "status": status,
        "reference_histogram": np.histogram(np.log1p(reference), bins=30)[0].tolist(),
        "current_histogram": np.histogram(np.log1p(current), bins=30)[0].tolist(),
    }


def robustness_perturbation(amount_pct: float = 0.10, seed: int = 42) -> dict:
    state = get_demo_state()
    n = 300
    ds = load_synthetic("ulb", n_samples=2000, random_state=seed + 1)
    frame_ = clean_dataframe(ds.frame, ds.target)
    numeric, categorical = infer_feature_types(frame_, ds.target, ds.categorical_features)
    splits = random_split(frame_, ds.target, test_size=0.2, validation_size=0.1, random_state=seed + 1)
    X_train, X_others, transformer, meta = fit_transform_split(
        splits["train"], {"test": splits["test"]}, ds.target, numeric, categorical
    )
    X_test = X_others["test"][:n]
    amount_idx = numeric.index("Amount") if "Amount" in numeric else 0

    base_scores = state.model.classical_model.predict_proba(X_test)
    perturbed = X_test.copy()
    perturbed[:, amount_idx] = perturbed[:, amount_idx] * (1 + amount_pct)
    perturbed_scores = state.model.classical_model.predict_proba(perturbed)

    score_delta = perturbed_scores - base_scores
    flip_rate = float(((base_scores >= 0.5) != (perturbed_scores >= 0.5)).mean())

    return {
        "amount_pct": amount_pct,
        "mean_abs_score_change": float(np.abs(score_delta).mean()),
        "flip_rate": flip_rate,
        "score_delta_histogram": np.histogram(score_delta, bins=30)[0].tolist(),
        "score_delta_bin_edges": np.histogram(score_delta, bins=30)[1].tolist(),
    }


def experiment_registry_rows() -> list[dict]:
    return load_registry()
