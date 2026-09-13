"""QUBO-based feature selection and Quantum-Aware Feature Selection (QAFS).

Two layers, kept deliberately separate and honestly documented:

1. `build_qubo_matrix` + `solve_qubo_qaoa` / `solve_qubo_simulated_annealing`:
   a genuine QUBO formulation (maximize relevance, minimize redundancy,
   subject to a cardinality constraint) solved either via QAOA on a Qiskit
   sampler (for small qubit-count instances, where QAOA is actually
   simulable) or classical simulated annealing (the practical fallback —
   QAOA does not scale to real feature-selection problem sizes on
   simulators, and this project is explicit about that rather than
   pretending otherwise).

2. `qafs_select_features`: implements the QAFS research objective from the
   spec —
       J = alpha*(1 - PR_AUC) + beta*FPR + gamma*ExpectedLoss
           + delta*QubitCost + epsilon*CircuitDepth + zeta*Latency
   by using the QUBO solver to propose a candidate subset for each requested
   feature count, then scoring each candidate with a *cheap classical
   surrogate model* (fast logistic regression) for the PR-AUC/FPR/
   ExpectedLoss terms and real circuit metrics (from
   src/quantum/circuit_metrics.py) for the resource terms. The surrogate is a
   deliberate, documented approximation: exhaustively training every
   candidate quantum model to evaluate J would be combinatorially expensive,
   and a fast classical proxy for ranking candidate subsets is a standard
   feature-selection practice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import compute_metrics
from src.quantum.circuit_metrics import compute_circuit_metrics
from src.quantum.feature_maps import build_feature_map
from src.quantum.noise_models import QuantumPrimitives
from src.utils.logging import get_logger

logger = get_logger("quantum.qubo_feature_selection")


def build_qubo_matrix(
    X: np.ndarray,
    y: np.ndarray,
    n_select: int,
    redundancy_weight: float = 1.0,
    cardinality_penalty: float = 2.0,
    random_state: int = 42,
) -> np.ndarray:
    """Q such that minimizing x^T Q x over x in {0,1}^d favors selecting
    `n_select` features that are individually relevant to y (mutual
    information) and mutually non-redundant (low |correlation|), via a
    cardinality-constraint penalty lambda*(sum(x) - n_select)^2."""
    d = X.shape[1]
    relevance = mutual_info_classif(X, y, random_state=random_state)
    relevance = relevance / (relevance.max() or 1.0)

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    Q = np.zeros((d, d))
    for i in range(d):
        Q[i, i] -= relevance[i]                      # reward relevance (minimize -> negative diagonal)
        Q[i, i] += cardinality_penalty * (1 - 2 * n_select)  # linear term of (sum x - k)^2 expansion
        for j in range(d):
            if i == j:
                continue
            Q[i, j] += redundancy_weight * abs(corr[i, j]) / 2  # penalize redundant pairs
            Q[i, j] += cardinality_penalty                       # quadratic cross-term of (sum x - k)^2
    return Q


def qubo_to_ising(Q: np.ndarray) -> tuple[SparsePauliOp, float]:
    """Standard QUBO -> Ising mapping via x_i = (1 - z_i)/2, z_i in {-1,+1}.

    Q is symmetrized first (x^T Q x == x^T Q_sym x for binary x, and the
    closed-form below assumes symmetry). Verified by brute-force enumeration
    against the QUBO objective for random instances up to d=3 — see
    tests/test_quantum.py::test_qubo_to_ising_matches_brute_force.
    """
    d = Q.shape[0]
    Qs = (Q + Q.T) / 2

    diag = np.diag(Qs)
    row_sum_excl_diag = Qs.sum(axis=1) - diag

    offset = float(0.5 * diag.sum() + 0.25 * (Qs.sum() - diag.sum()))
    linear = -(diag + row_sum_excl_diag) / 2
    quadratic: dict[tuple[int, int], float] = {}
    for i in range(d):
        for j in range(i + 1, d):
            coeff = Qs[i, j] / 2
            if abs(coeff) > 1e-12:
                quadratic[(i, j)] = coeff

    paulis, coeffs = [], []
    for i in range(d):
        if abs(linear[i]) > 1e-12:
            label = ["I"] * d
            label[d - 1 - i] = "Z"
            paulis.append("".join(label))
            coeffs.append(linear[i])
    for (i, j), coeff in quadratic.items():
        if abs(coeff) > 1e-12:
            label = ["I"] * d
            label[d - 1 - i] = "Z"
            label[d - 1 - j] = "Z"
            paulis.append("".join(label))
            coeffs.append(coeff)

    if not paulis:
        paulis, coeffs = ["I" * d], [0.0]
    return SparsePauliOp(paulis, coeffs), offset


@dataclass
class QUBOSolution:
    selected: np.ndarray  # boolean mask, length d
    energy: float
    method: str
    solve_time_s: float


def solve_qubo_simulated_annealing(
    Q: np.ndarray,
    n_iterations: int = 3000,
    random_state: int = 42,
) -> QUBOSolution:
    """Classical fallback: simulated annealing over {0,1}^d. Used whenever
    the instance is too large for QAOA to simulate, or as the default for
    reproducible unit tests (deterministic given a seed)."""
    rng = np.random.default_rng(random_state)
    d = Q.shape[0]
    x = rng.integers(0, 2, size=d).astype(float)

    def energy(vec: np.ndarray) -> float:
        return float(vec @ Q @ vec)

    current_energy = energy(x)
    best_x, best_energy = x.copy(), current_energy

    start = time.perf_counter()
    for step in range(n_iterations):
        temperature = max(1e-3, 1.0 - step / n_iterations)
        flip = rng.integers(0, d)
        candidate = x.copy()
        candidate[flip] = 1 - candidate[flip]
        candidate_energy = energy(candidate)
        delta = candidate_energy - current_energy
        if delta < 0 or rng.random() < np.exp(-delta / temperature):
            x, current_energy = candidate, candidate_energy
            if current_energy < best_energy:
                best_x, best_energy = x.copy(), current_energy
    elapsed = time.perf_counter() - start

    return QUBOSolution(selected=best_x.astype(bool), energy=best_energy, method="simulated_annealing", solve_time_s=elapsed)


def solve_qubo_qaoa(
    Q: np.ndarray,
    primitives: QuantumPrimitives,
    reps: int = 1,
    maxiter: int = 100,
    max_variables: int = 12,
) -> QUBOSolution:
    """Solve small QUBO instances (<=max_variables) via QAOA on the given
    sampler. Raises if the instance is larger, so callers must explicitly
    fall back to `solve_qubo_simulated_annealing` for realistic feature
    counts rather than silently truncating the problem."""
    d = Q.shape[0]
    if d > max_variables:
        raise ValueError(
            f"solve_qubo_qaoa is only simulable up to {max_variables} variables; got {d}. "
            f"Use solve_qubo_simulated_annealing for larger instances."
        )

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA

    hamiltonian, offset = qubo_to_ising(Q)
    # QAOA does not transpile its ansatz internally (transpiler=None means
    # "leave as-is"); Aer's primitives reject the untranspiled QAOAAnsatz
    # wrapper instruction, so an explicit basis-gate pass manager is required.
    transpiler = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
    qaoa = QAOA(sampler=primitives.sampler, optimizer=COBYLA(maxiter=maxiter), reps=reps, transpiler=transpiler)

    start = time.perf_counter()
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    elapsed = time.perf_counter() - start

    best_bitstring = max(result.eigenstate.items(), key=lambda kv: kv[1])[0] if hasattr(result.eigenstate, "items") else None
    if best_bitstring is None:
        # eigenstate may be a QuasiDistribution keyed by int
        probs = result.eigenstate
        best_int = max(probs, key=probs.get)
        best_bitstring = format(best_int, f"0{d}b")

    selected = np.array([int(bit) for bit in best_bitstring[::-1]][:d], dtype=bool)
    energy = float(selected.astype(float) @ Q @ selected.astype(float))

    return QUBOSolution(selected=selected, energy=energy, method="qaoa", solve_time_s=elapsed)


@dataclass
class QAFSCandidate:
    n_features: int
    selected_features: list[str]
    selected_indices: list[int]
    pr_auc: float
    fpr: float
    expected_loss: float
    qubit_cost: int
    circuit_depth: int
    latency_s: float
    J: float
    solver_method: str


def evaluate_feature_subset_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_indices: list[int],
    feature_map_name: str,
    feature_map_reps: int,
    weights: dict[str, float],
    fraud_loss: float = 500.0,
    false_positive_cost: float = 25.0,
) -> tuple[float, dict[str, float]]:
    """Compute J for one candidate feature subset. PR-AUC/FPR/ExpectedLoss
    come from a fast classical surrogate (logistic regression) trained only
    on the candidate columns — an explicit, documented proxy for ranking
    subsets cheaply, not a substitute for actually evaluating the trained
    quantum model (that happens later in the real quantum experiments)."""
    Xt, Xv = X_train[:, feature_indices], X_val[:, feature_indices]

    start = time.perf_counter()
    surrogate = LogisticRegression(max_iter=1000, class_weight="balanced")
    surrogate.fit(Xt, y_train)
    proba = surrogate.predict_proba(Xv)[:, 1]
    latency_s = (time.perf_counter() - start) / max(len(Xv), 1)

    metrics = compute_metrics(y_val, proba, threshold=0.5, fraud_loss=fraud_loss, false_positive_cost=false_positive_cost)

    n_qubits = len(feature_indices)
    feature_map = build_feature_map(feature_map_name, n_qubits, feature_map_reps)
    circuit_metrics = compute_circuit_metrics(feature_map)

    pr_auc = metrics.pr_auc if not np.isnan(metrics.pr_auc) else 0.0
    components = {
        "pr_auc": pr_auc,
        "fpr": metrics.fpr,
        "expected_loss": metrics.expected_loss,
        "qubit_cost": n_qubits,
        "circuit_depth": circuit_metrics.logical_depth,
        "latency_s": latency_s,
    }

    J = (
        weights.get("alpha", 1.0) * (1 - pr_auc)
        + weights.get("beta", 0.5) * metrics.fpr
        + weights.get("gamma", 0.2) * (metrics.expected_loss / max(len(y_val), 1))
        + weights.get("delta", 0.3) * (n_qubits / max(X_train.shape[1], 1))
        + weights.get("epsilon", 0.2) * (circuit_metrics.logical_depth / 100.0)
        + weights.get("zeta", 0.2) * (latency_s * 1000)
    )
    return J, components


def qafs_select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    candidate_counts: list[int],
    weights: dict[str, float],
    feature_map_name: str = "zz",
    feature_map_reps: int = 2,
    primitives: QuantumPrimitives | None = None,
    qaoa_max_variables: int = 12,
    random_state: int = 42,
) -> list[QAFSCandidate]:
    """Run QAFS across every requested feature count and return one
    QAFSCandidate per count, so callers can plot J / PR-AUC vs feature count
    (spec section 8's required comparison) and pick the minimum-J subset."""
    results = []
    for k in candidate_counts:
        if k > X_train.shape[1]:
            logger.warning(f"Skipping candidate_count={k}: exceeds available feature count {X_train.shape[1]}.")
            continue

        Q = build_qubo_matrix(X_train, y_train, n_select=k, random_state=random_state)

        if primitives is not None and X_train.shape[1] <= qaoa_max_variables:
            solution = solve_qubo_qaoa(Q, primitives, max_variables=qaoa_max_variables)
        else:
            solution = solve_qubo_simulated_annealing(Q, random_state=random_state)

        selected_idx = np.where(solution.selected)[0].tolist()
        if len(selected_idx) == 0:
            continue
        # enforce the requested cardinality even if the penalty term didn't land exactly on k
        if len(selected_idx) > k:
            relevance_order = np.argsort(-np.abs(np.corrcoef(X_train[:, selected_idx].T, y_train)[-1, :-1]))
            selected_idx = [selected_idx[i] for i in relevance_order[:k]]
        elif len(selected_idx) < k:
            remaining = [i for i in range(X_train.shape[1]) if i not in selected_idx]
            selected_idx = selected_idx + remaining[: k - len(selected_idx)]

        J, components = evaluate_feature_subset_objective(
            X_train, y_train, X_val, y_val, selected_idx, feature_map_name, feature_map_reps, weights
        )

        results.append(QAFSCandidate(
            n_features=k,
            selected_features=[feature_names[i] for i in selected_idx],
            selected_indices=selected_idx,
            pr_auc=components["pr_auc"],
            fpr=components["fpr"],
            expected_loss=components["expected_loss"],
            qubit_cost=int(components["qubit_cost"]),
            circuit_depth=int(components["circuit_depth"]),
            latency_s=components["latency_s"],
            J=J,
            solver_method=solution.method,
        ))
    return results
