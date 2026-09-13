"""Quantum Kernel -> QSVC classifier.

QSVC wraps sklearn's SVC with a precomputed quantum kernel matrix. Kernel
evaluation is the dominant cost (O(n^2) circuit evaluations for training),
so this wrapper always records qubits/shots/circuit-depth/backend alongside
the fitted model for the experiment registry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit_machine_learning.algorithms import QSVC

from src.quantum.circuit_metrics import compute_circuit_metrics
from src.quantum.feature_maps import build_feature_map
from src.quantum.noise_models import QuantumPrimitives, get_primitives
from src.quantum.quantum_kernel import build_quantum_kernel


@dataclass
class QuantumFitResult:
    training_time_s: float
    n_train: int
    num_qubits: int
    circuit_depth: int
    transpiled_depth: int
    shots: int
    backend_name: str


class QSVCModel:
    name = "qsvc"

    def __init__(
        self,
        num_qubits: int,
        feature_map_name: str = "zz",
        feature_map_reps: int = 2,
        entanglement: str = "linear",
        backend: str = "simulator",
        shots: int = 1024,
        seed: int = 42,
        C: float = 1.0,
    ):
        self.num_qubits = num_qubits
        self.feature_map = build_feature_map(feature_map_name, num_qubits, feature_map_reps, entanglement)
        self.primitives: QuantumPrimitives = get_primitives(backend=backend, shots=shots, seed=seed)
        self.kernel_bundle = build_quantum_kernel(self.feature_map, self.primitives)
        self._estimator = QSVC(quantum_kernel=self.kernel_bundle.kernel, C=C)
        self._fit_result: QuantumFitResult | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> QuantumFitResult:
        if X.shape[1] != self.num_qubits:
            raise ValueError(
                f"QSVCModel configured for {self.num_qubits} qubits but received "
                f"{X.shape[1]}-dimensional features. Apply feature selection first."
            )
        metrics = compute_circuit_metrics(self.feature_map)

        start = time.perf_counter()
        self._estimator.fit(X, y)
        elapsed = time.perf_counter() - start

        self._fit_result = QuantumFitResult(
            training_time_s=elapsed,
            n_train=len(y),
            num_qubits=self.num_qubits,
            circuit_depth=metrics.logical_depth,
            transpiled_depth=metrics.transpiled_depth,
            shots=self.primitives.sampler.default_shots if hasattr(self.primitives.sampler, "default_shots") else -1,
            backend_name=self.primitives.backend_info.backend_name,
        )
        return self._fit_result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """QSVC (SVC-based) exposes decision_function, not calibrated
        probabilities by default; we min-max squash the decision function
        into [0, 1] as a monotonic risk score, consistent with how the rest
        of the pipeline (risk fusion, decision engine) consumes scores."""
        scores = self._estimator.decision_function(X)
        return _sigmoid_squash(scores)

    def predict_proba_timed(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        proba = self.predict_proba(X)
        elapsed = time.perf_counter() - start
        return proba, elapsed


def _sigmoid_squash(scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-scores))
