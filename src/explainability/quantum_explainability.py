"""Quantum model "explainability".

IMPORTANT HONESTY NOTE (spec section 18): a quantum circuit's structure is
NOT equivalent to a classical SHAP/permutation-importance explanation. SHAP
attributes a prediction to input features via cooperative game theory;
nothing analogous with the same theoretical guarantees exists for
variational quantum circuits at the time of writing. What this module
provides instead is transparent *reporting* of what the quantum model
actually did — which features (qubits) it used, the circuit structure, the
score it produced, and how confident that score is — so users can audit the
quantum path without a false equivalence claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit

from src.quantum.circuit_metrics import compute_circuit_metrics


@dataclass
class QuantumExplanation:
    selected_features: list[str]
    num_qubits: int
    feature_map_name: str
    ansatz_name: str | None
    circuit_depth: int
    transpiled_depth: int
    two_qubit_gate_count: int
    shots: int
    backend_name: str
    quantum_score: float
    confidence: float
    disclaimer: str = (
        "This is a transparency report of the quantum circuit's structure and "
        "output, not a feature-attribution explanation. It is NOT equivalent "
        "to classical SHAP/permutation importance, and no such equivalence is "
        "claimed by this project."
    )


def explain_quantum_prediction(
    feature_map: QuantumCircuit,
    ansatz: QuantumCircuit | None,
    selected_feature_names: list[str],
    quantum_score: float,
    shots: int,
    backend_name: str,
    feature_map_name: str = "unknown",
    ansatz_name: str | None = None,
) -> QuantumExplanation:
    circuit = feature_map.compose(ansatz) if ansatz is not None else feature_map
    metrics = compute_circuit_metrics(circuit)
    confidence = float(max(quantum_score, 1 - quantum_score))

    return QuantumExplanation(
        selected_features=selected_feature_names,
        num_qubits=metrics.num_qubits,
        feature_map_name=feature_map_name,
        ansatz_name=ansatz_name,
        circuit_depth=metrics.logical_depth,
        transpiled_depth=metrics.transpiled_depth,
        two_qubit_gate_count=metrics.two_qubit_gate_count,
        shots=shots,
        backend_name=backend_name,
        quantum_score=float(quantum_score),
        confidence=confidence,
    )


def feature_sensitivity_scan(
    predict_fn,
    x_row: np.ndarray,
    feature_names: list[str],
    perturbation_fraction: float = 0.1,
) -> dict[str, float]:
    """A cheap, honestly-labeled sensitivity proxy: perturb one feature at a
    time by +/-`perturbation_fraction` of its value and measure the resulting
    change in the quantum model's score. This is a local finite-difference
    sensitivity measure, NOT a Shapley-value attribution — it does not
    satisfy efficiency/symmetry/additivity axioms the way SHAP does, and is
    presented to users with that caveat."""
    base_score = float(predict_fn(x_row.reshape(1, -1))[0])
    sensitivities = {}
    for i, name in enumerate(feature_names):
        perturbed = x_row.copy()
        delta = abs(perturbed[i]) * perturbation_fraction or perturbation_fraction
        perturbed[i] += delta
        new_score = float(predict_fn(perturbed.reshape(1, -1))[0])
        sensitivities[name] = new_score - base_score
    return sensitivities
