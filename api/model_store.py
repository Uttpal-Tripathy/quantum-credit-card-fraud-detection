"""Persists the demo QGFDA pipeline's trained artifacts to disk so the API
doesn't retrain a VQC from scratch (30-60s) on every process restart — a
real production concern for cold-start latency.

What's persisted:
  - The classical model (joblib — plain sklearn/XGBoost objects pickle fine).
  - The preprocessing transformer (joblib).
  - The trained quantum estimator's weights, via qiskit-machine-learning's
    own `to_dill`/`from_dill` (plain pickle fails on VQC: it holds a closure
    — `VQC._get_interpret.<locals>.parity` — that only dill can serialize).
  - A JSON metadata file with everything needed to deterministically rebuild
    the surrounding architecture (feature/ansatz config, gate/fusion/cost
    settings, the fitted decision thresholds, and the exact dataset
    parameters needed to regenerate a matching synthetic evaluation split).

Training is skipped entirely on a warm load; only a fresh (fast) inference
pass over the evaluation split is run to repopulate `result`/`frame`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib

from src.config.settings import REPO_ROOT
from src.utils.logging import get_logger

logger = get_logger("api.model_store")

MODEL_DIR = REPO_ROOT / "models" / "demo_pipeline"
CLASSICAL_PATH = MODEL_DIR / "classical_model.joblib"
TRANSFORMER_PATH = MODEL_DIR / "transformer.joblib"
QUANTUM_ESTIMATOR_PATH = MODEL_DIR / "vqc_estimator.dill"
META_PATH = MODEL_DIR / "meta.json"


def is_available() -> bool:
    return META_PATH.exists() and CLASSICAL_PATH.exists() and TRANSFORMER_PATH.exists() and QUANTUM_ESTIMATOR_PATH.exists()


def save(state) -> None:
    """`state` is an api.services.DemoState. Called once right after the
    demo pipeline finishes training."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(state.model.classical_model, CLASSICAL_PATH)
    joblib.dump(state.transformer, TRANSFORMER_PATH)
    state.model.quantum_model._estimator.to_dill(str(QUANTUM_ESTIMATOR_PATH))  # noqa: SLF001

    qm = state.model.quantum_model
    cfg = state.model.config
    meta = {
        "saved_at": time.time(),
        "numeric_features": state.numeric_features,
        "categorical_features": state.categorical_features,
        "target": state.target,
        "time_column": state.time_column,
        "amount_column": state.amount_column,
        "quantum_feature_indices": cfg.quantum_feature_indices,
        "gate_threshold": cfg.gate_threshold,
        "risk_band": list(cfg.risk_band),
        "max_quantum_fraction": cfg.max_quantum_fraction,
        "fusion_method": cfg.fusion_method,
        "fusion_weights": cfg.fusion_weights,
        "cost_model": {
            "fraud_loss": cfg.cost_model.fraud_loss,
            "false_positive_cost": cfg.cost_model.false_positive_cost,
            "review_cost": cfg.cost_model.review_cost,
        },
        "thresholds": {
            "review_threshold": state.model.thresholds.review_threshold,
            "block_threshold": state.model.thresholds.block_threshold,
            "expected_loss": state.model.thresholds.expected_loss,
            "search_grid_points": state.model.thresholds.search_grid_points,
        },
        "quantum_model_config": {
            "num_qubits": qm.num_qubits,
            "feature_map_name": "zz",
            "feature_map_reps": 1,
            "ansatz_name": "real_amplitudes",
            "ansatz_reps": 2,
            "optimizer": qm.optimizer_name,
            "shots": qm.shots,
            "backend": "simulator",
            "seed": cfg.random_state,
        },
        "random_state": cfg.random_state,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"Saved demo pipeline artifacts to {MODEL_DIR}")


def load_meta() -> dict | None:
    if not META_PATH.exists():
        return None
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def load_classical_model():
    return joblib.load(CLASSICAL_PATH)


def load_transformer():
    return joblib.load(TRANSFORMER_PATH)


def load_quantum_estimator():
    from qiskit_machine_learning.algorithms import VQC

    return VQC.from_dill(str(QUANTUM_ESTIMATOR_PATH))


def clear() -> None:
    """Delete persisted artifacts, forcing a full retrain on next load."""
    for path in (CLASSICAL_PATH, TRANSFORMER_PATH, QUANTUM_ESTIMATOR_PATH, META_PATH):
        path.unlink(missing_ok=True)
    logger.info("Cleared persisted demo pipeline artifacts.")
