"""Tests for the quantum ML stack. Everything here runs on the local Aer
simulator — no IBM Quantum credentials required. Tests use tiny qubit counts
and shot/iteration budgets to stay fast; `@pytest.mark.hardware` marks the
one test that would need real IBM Quantum access (deselected by default via
`-m "not hardware"`, see pyproject.toml).
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from src.quantum.ansatzes import build_ansatz
from src.quantum.circuit_metrics import compute_circuit_metrics
from src.quantum.feature_maps import build_feature_map
from src.quantum.noise_models import get_primitives
from src.quantum.qnn_model import QNNModel
from src.quantum.qsvc_model import QSVCModel
from src.quantum.quantum_pca import classical_pca
from src.quantum.qubo_feature_selection import (
    build_qubo_matrix,
    qubo_to_ising,
    solve_qubo_simulated_annealing,
)
from src.quantum.vqc_model import VQCModel


@pytest.fixture
def toy_binary_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_build_feature_map_dimensions():
    for name in ("zz", "z", "pauli"):
        fm = build_feature_map(name, num_qubits=3, reps=2)
        assert fm.num_qubits == 3


def test_build_feature_map_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_feature_map("not-a-real-map", num_qubits=2)


def test_build_ansatz_dimensions():
    for name in ("real_amplitudes", "efficient_su2", "two_local"):
        ansatz = build_ansatz(name, num_qubits=3, reps=2)
        assert ansatz.num_qubits == 3


def test_circuit_metrics_qubit_count_and_positive_depth():
    fm = build_feature_map("zz", num_qubits=4, reps=2)
    metrics = compute_circuit_metrics(fm)
    assert metrics.num_qubits == 4
    assert metrics.logical_depth > 0
    assert metrics.transpiled_depth > 0
    assert metrics.transpiled_gate_count >= metrics.two_qubit_gate_count


def test_get_primitives_simulator_backend():
    primitives = get_primitives(backend="simulator", shots=128, seed=1)
    assert primitives.backend_info.is_simulator is True
    assert primitives.backend_info.is_noisy is False


def test_get_primitives_noisy_simulator_backend():
    primitives = get_primitives(backend="noisy_simulator", shots=128, seed=1)
    assert primitives.backend_info.is_noisy is True


def test_get_primitives_ibm_quantum_falls_back_without_credentials(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    primitives = get_primitives(backend="ibm_quantum", shots=128, seed=1)
    assert primitives.backend_info.is_simulator is True
    assert primitives.backend_info.fallback_reason is not None


def test_get_primitives_rejects_unknown_backend():
    with pytest.raises(ValueError):
        get_primitives(backend="quantum_cloud_9000", shots=128, seed=1)


def test_classical_pca_reduces_dimensions():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    result = classical_pca(X, n_components=3)
    assert result.transformed.shape == (50, 3)
    assert len(result.explained_variance_ratio) == 3


def test_qubo_to_ising_matches_brute_force_objective():
    """Verifies the closed-form QUBO->Ising conversion in
    src/quantum/qubo_feature_selection.py by brute-force enumeration: the
    Ising Hamiltonian's diagonal entries (in the computational basis) plus
    offset must reproduce x^T Q x for every bitstring x."""
    rng = np.random.default_rng(1)
    d = 4
    Q_raw = rng.normal(size=(d, d))
    Q = (Q_raw + Q_raw.T) / 2  # symmetrize, as qubo_to_ising assumes

    hamiltonian, offset = qubo_to_ising(Q)
    H_matrix = hamiltonian.to_matrix().real

    for bits in product([0, 1], repeat=d):
        x = np.array(bits, dtype=float)
        qubo_value = float(x @ Q @ x)

        basis_index = 0
        for i, bit in enumerate(bits):
            basis_index |= bit << i
        ising_value = float(H_matrix[basis_index, basis_index].real) + offset

        assert qubo_value == pytest.approx(ising_value, abs=1e-6), f"mismatch at bits={bits}"


def test_solve_qubo_simulated_annealing_selects_relevant_features():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    Q = build_qubo_matrix(X, y, n_select=2, random_state=42)
    solution = solve_qubo_simulated_annealing(Q, n_iterations=800, random_state=42)
    selected = set(np.where(solution.selected)[0].tolist())
    assert selected == {0, 1}, f"expected the two truly-relevant features, got {selected}"


def test_qsvc_fits_and_predicts_valid_probabilities(toy_binary_data):
    X, y = toy_binary_data
    model = QSVCModel(num_qubits=3, backend="simulator", shots=64, seed=42)
    fit_result = model.fit(X, y)
    assert fit_result.num_qubits == 3
    proba = model.predict_proba(X[:5])
    assert np.all((proba >= 0) & (proba <= 1))


def test_qsvc_rejects_wrong_feature_dimension(toy_binary_data):
    X, y = toy_binary_data
    model = QSVCModel(num_qubits=5, backend="simulator", shots=64, seed=42)
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_vqc_fits_and_predicts_valid_probabilities(toy_binary_data):
    X, y = toy_binary_data
    model = VQCModel(num_qubits=3, backend="simulator", shots=64, seed=42, optimizer="cobyla", optimizer_maxiter=10)
    fit_result = model.fit(X, y)
    assert fit_result.num_qubits == 3
    proba = model.predict_proba(X[:5])
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.parametrize("qnn_type", ["estimator", "sampler"])
def test_qnn_fits_and_predicts_valid_probabilities(toy_binary_data, qnn_type):
    X, y = toy_binary_data
    model = QNNModel(num_qubits=3, qnn_type=qnn_type, backend="simulator", shots=64, seed=42,
                      optimizer="cobyla", optimizer_maxiter=10)
    fit_result = model.fit(X, y)
    assert fit_result.qnn_type == qnn_type
    proba = model.predict_proba(X[:5])
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.hardware
def test_ibm_quantum_real_hardware_access():
    """Only runs when explicitly requested (deselected by default via
    `pytest -m "not hardware"`) and IBM_QUANTUM_TOKEN is set."""
    import os

    if not os.environ.get("IBM_QUANTUM_TOKEN"):
        pytest.skip("IBM_QUANTUM_TOKEN not set")
    primitives = get_primitives(backend="ibm_quantum", shots=128, seed=1)
    assert primitives.backend_info.is_simulator is False
