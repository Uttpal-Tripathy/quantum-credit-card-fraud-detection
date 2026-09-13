"""Pydantic request/response models for the QGFDA API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    amount_multiplier: float = Field(1.0, gt=0, le=100, description="Scales the synthetic transaction's Amount.")


class QuantumLabRequest(BaseModel):
    feature_map: Literal["zz", "z", "pauli"] = "zz"
    ansatz: Literal["real_amplitudes", "efficient_su2", "two_local"] = "real_amplitudes"
    qubits: int = Field(4, ge=2, le=10)
    reps: int = Field(2, ge=1, le=4)


class FeatureSelectionRequest(BaseModel):
    candidate_counts: list[int] = Field(default=[2, 4, 6, 8])
    seed: int = 42


class RobustnessRequest(BaseModel):
    amount_pct: float = Field(0.10, ge=0.01, le=0.50)


class RunExperimentRequest(BaseModel):
    dataset: Literal["ulb", "ieee_cis", "paysim", "banksim", "fraud_detection_handbook"] = "ulb"
    experiment_type: Literal["classical", "quantum", "hybrid_ablation", "noise_sweep", "temporal"] = "classical"
    model: str = "xgboost"
    qubits: int = Field(4, ge=2, le=10)
    ablation_arm: str = "I_full_qgfda"
    backend: Literal["simulator", "noisy_simulator", "ibm_quantum"] = "simulator"
    shots: int = Field(256, ge=64, le=4096)
    use_synthetic: bool = True
    seed: int = 42
