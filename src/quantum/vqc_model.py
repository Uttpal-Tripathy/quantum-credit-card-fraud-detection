"""Feature Map -> Ansatz -> Variational Quantum Classifier (VQC).

Configurable feature map, ansatz, optimizer, qubit count, shots (all read
from configs/quantum.yaml by default, overridable per call).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B, SPSA

from src.quantum.ansatzes import build_ansatz
from src.quantum.circuit_metrics import compute_circuit_metrics
from src.quantum.feature_maps import build_feature_map
from src.quantum.noise_models import QuantumPrimitives, get_primitives

_OPTIMIZERS = {
    "cobyla": lambda maxiter: COBYLA(maxiter=maxiter),
    "spsa": lambda maxiter: SPSA(maxiter=maxiter),
    "l_bfgs_b": lambda maxiter: L_BFGS_B(maxiter=maxiter),
}


@dataclass
class QuantumFitResult:
    training_time_s: float
    n_train: int
    num_qubits: int
    circuit_depth: int
    transpiled_depth: int
    shots: int
    backend_name: str
    optimizer: str
    final_objective_value: float | None


class VQCModel:
    name = "vqc"

    def __init__(
        self,
        num_qubits: int,
        feature_map_name: str = "zz",
        feature_map_reps: int = 2,
        entanglement: str = "linear",
        ansatz_name: str = "real_amplitudes",
        ansatz_reps: int = 3,
        optimizer: str = "cobyla",
        optimizer_maxiter: int = 150,
        backend: str = "simulator",
        shots: int = 1024,
        seed: int = 42,
    ):
        if optimizer not in _OPTIMIZERS:
            raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from {sorted(_OPTIMIZERS)}.")

        self.num_qubits = num_qubits
        self.feature_map = build_feature_map(feature_map_name, num_qubits, feature_map_reps, entanglement)
        self.ansatz = build_ansatz(ansatz_name, num_qubits, ansatz_reps, entanglement)
        self.primitives: QuantumPrimitives = get_primitives(backend=backend, shots=shots, seed=seed)
        self.optimizer_name = optimizer
        self.shots = shots

        self._objective_values: list[float] = []
        self._estimator = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=_OPTIMIZERS[optimizer](optimizer_maxiter),
            sampler=self.primitives.sampler,
            callback=lambda weights, value: self._objective_values.append(float(value)),
        )
        self._fit_result: QuantumFitResult | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> QuantumFitResult:
        if X.shape[1] != self.num_qubits:
            raise ValueError(
                f"VQCModel configured for {self.num_qubits} qubits but received "
                f"{X.shape[1]}-dimensional features. Apply feature selection first."
            )
        metrics = compute_circuit_metrics(self.feature_map.compose(self.ansatz))

        start = time.perf_counter()
        self._estimator.fit(X, y)
        elapsed = time.perf_counter() - start

        self._fit_result = QuantumFitResult(
            training_time_s=elapsed,
            n_train=len(y),
            num_qubits=self.num_qubits,
            circuit_depth=metrics.logical_depth,
            transpiled_depth=metrics.transpiled_depth,
            shots=self.shots,
            backend_name=self.primitives.backend_info.backend_name,
            optimizer=self.optimizer_name,
            final_objective_value=self._objective_values[-1] if self._objective_values else None,
        )
        return self._fit_result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """VQC.predict returns hard class labels; for risk fusion / cost-
        sensitive thresholding we need the soft score, so use VQC's own
        `predict_proba` (shape (n, n_classes)) and take P(class==1)."""
        proba = np.asarray(self._estimator.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.clip(proba[:, 1], 0.0, 1.0)
        return np.clip(proba.ravel(), 0.0, 1.0)

    def predict_proba_timed(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        proba = self.predict_proba(X)
        elapsed = time.perf_counter() - start
        return proba, elapsed
