"""Qiskit QNN-based classifier: EstimatorQNN or SamplerQNN feeding a
classical output layer (a 1-D logistic calibration head on the QNN's raw
score), matching spec section 9D ("neural-network/classical output layer").

EstimatorQNN (qnn_type='estimator'): circuit -> single Z^{\otimes n}
expectation value in [-1, 1] -> classical logistic head -> P(fraud).
SamplerQNN (qnn_type='sampler'): circuit -> bitstring distribution ->
parity-interpreted 2-class probabilities -> classical logistic head.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B, SPSA
from sklearn.linear_model import LogisticRegression

from src.quantum.ansatzes import build_ansatz
from src.quantum.circuit_metrics import compute_circuit_metrics
from src.quantum.feature_maps import build_feature_map
from src.quantum.noise_models import QuantumPrimitives, get_primitives

_OPTIMIZERS = {
    "cobyla": lambda maxiter: COBYLA(maxiter=maxiter),
    "spsa": lambda maxiter: SPSA(maxiter=maxiter),
    "l_bfgs_b": lambda maxiter: L_BFGS_B(maxiter=maxiter),
}


def _parity(x: int) -> int:
    return bin(x).count("1") % 2


@dataclass
class QuantumFitResult:
    training_time_s: float
    n_train: int
    num_qubits: int
    circuit_depth: int
    transpiled_depth: int
    shots: int
    backend_name: str
    qnn_type: str
    optimizer: str


class QNNModel:
    name = "qnn"

    def __init__(
        self,
        num_qubits: int,
        feature_map_name: str = "zz",
        feature_map_reps: int = 2,
        entanglement: str = "linear",
        ansatz_name: str = "real_amplitudes",
        ansatz_reps: int = 3,
        qnn_type: str = "estimator",
        optimizer: str = "cobyla",
        optimizer_maxiter: int = 150,
        backend: str = "simulator",
        shots: int = 1024,
        seed: int = 42,
    ):
        if qnn_type not in {"estimator", "sampler"}:
            raise ValueError("qnn_type must be 'estimator' or 'sampler'")
        if optimizer not in _OPTIMIZERS:
            raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from {sorted(_OPTIMIZERS)}.")

        self.num_qubits = num_qubits
        self.qnn_type = qnn_type
        self.shots = shots
        self.feature_map = build_feature_map(feature_map_name, num_qubits, feature_map_reps, entanglement)
        self.ansatz = build_ansatz(ansatz_name, num_qubits, ansatz_reps, entanglement)
        self.circuit = self.feature_map.compose(self.ansatz)
        self.primitives: QuantumPrimitives = get_primitives(backend=backend, shots=shots, seed=seed)

        if qnn_type == "estimator":
            self.qnn = EstimatorQNN(
                circuit=self.circuit,
                estimator=self.primitives.estimator,
                input_params=list(self.feature_map.parameters),
                weight_params=list(self.ansatz.parameters),
            )
            loss = "squared_error"
        else:
            self.qnn = SamplerQNN(
                circuit=self.circuit,
                sampler=self.primitives.sampler,
                input_params=list(self.feature_map.parameters),
                weight_params=list(self.ansatz.parameters),
                interpret=_parity,
                output_shape=2,
            )
            loss = "cross_entropy"

        self._classifier = NeuralNetworkClassifier(
            neural_network=self.qnn,
            loss=loss,
            optimizer=_OPTIMIZERS[optimizer](optimizer_maxiter),
            one_hot=(qnn_type == "sampler"),
        )
        self._output_head = LogisticRegression(max_iter=1000)
        self.optimizer_name = optimizer
        self._fit_result: QuantumFitResult | None = None

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the trained QNN only (no output head)."""
        weights = self._classifier.weights
        raw = self.qnn.forward(X, weights)
        raw = np.asarray(raw)
        if raw.ndim == 2:
            # sampler QNN: 2-class distribution -> use P(class==1) as the raw score
            return raw[:, 1] if raw.shape[1] >= 2 else raw.ravel()
        return raw.ravel()

    def fit(self, X: np.ndarray, y: np.ndarray) -> QuantumFitResult:
        if X.shape[1] != self.num_qubits:
            raise ValueError(
                f"QNNModel configured for {self.num_qubits} qubits but received "
                f"{X.shape[1]}-dimensional features. Apply feature selection first."
            )
        metrics = compute_circuit_metrics(self.circuit)

        # EstimatorQNN's squared_error loss expects {-1, +1}-style targets for
        # a symmetric Z-expectation output; SamplerQNN uses one-hot 0/1 labels.
        y_train = np.where(y == 1, 1.0, -1.0) if self.qnn_type == "estimator" else y

        start = time.perf_counter()
        self._classifier.fit(X, y_train)
        raw_train_scores = self._raw_scores(X).reshape(-1, 1)
        self._output_head.fit(raw_train_scores, y)
        elapsed = time.perf_counter() - start

        self._fit_result = QuantumFitResult(
            training_time_s=elapsed,
            n_train=len(y),
            num_qubits=self.num_qubits,
            circuit_depth=metrics.logical_depth,
            transpiled_depth=metrics.transpiled_depth,
            shots=self.shots,
            backend_name=self.primitives.backend_info.backend_name,
            qnn_type=self.qnn_type,
            optimizer=self.optimizer_name,
        )
        return self._fit_result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_scores = self._raw_scores(X).reshape(-1, 1)
        return self._output_head.predict_proba(raw_scores)[:, 1]

    def predict_proba_timed(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        proba = self.predict_proba(X)
        elapsed = time.perf_counter() - start
        return proba, elapsed
