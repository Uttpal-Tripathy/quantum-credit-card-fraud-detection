"""Quantum-inspired / quantum-assisted dimensionality reduction.

Honesty note (spec section 8 + section 18): genuine quantum PCA (via density-
matrix exponentiation / quantum phase estimation) requires quantum RAM and
fault-tolerant depth that is not practically simulable at ULB/IEEE-CIS scale
on Aer or NISQ hardware. This module provides two things:

1. `classical_pca` — the practical, honest default used throughout the
   comparison experiments (labeled as classical, not quantum).
2. `swap_test_quantum_pca` — a genuine but toy-scale quantum routine that
   estimates pairwise state overlaps (a proxy for the leading principal
   direction) via the swap test on amplitude-encoded vectors. It only scales
   to a handful of qubits/samples and exists to let the research report
   honestly discuss the practicality gap rather than claim scalable qPCA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from sklearn.decomposition import PCA

from src.quantum.noise_models import QuantumPrimitives


@dataclass
class PCAResult:
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    transformed: np.ndarray
    method: str


def classical_pca(X: np.ndarray, n_components: int, random_state: int = 42) -> PCAResult:
    pca = PCA(n_components=n_components, random_state=random_state)
    transformed = pca.fit_transform(X)
    return PCAResult(
        components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        transformed=transformed,
        method="classical_pca",
    )


def _amplitude_encode_pair(vec_a: np.ndarray, vec_b: np.ndarray) -> QuantumCircuit:
    """Build a swap-test circuit estimating |<a|b>|^2 for two unit vectors of
    equal, power-of-two length via amplitude encoding on an ancilla-controlled
    swap. Vectors are zero-padded/truncated to the nearest power of two and
    L2-normalized (required for a valid amplitude-encoded quantum state)."""
    dim = 1
    while dim < max(len(vec_a), len(vec_b)):
        dim *= 2
    a = np.zeros(dim)
    b = np.zeros(dim)
    a[: len(vec_a)] = vec_a
    b[: len(vec_b)] = vec_b
    a = a / (np.linalg.norm(a) or 1.0)
    b = b / (np.linalg.norm(b) or 1.0)

    n_qubits = int(np.log2(dim))
    qc = QuantumCircuit(1 + 2 * n_qubits, 1)
    reg_a = range(1, 1 + n_qubits)
    reg_b = range(1 + n_qubits, 1 + 2 * n_qubits)

    qc.initialize(a, reg_a)
    qc.initialize(b, reg_b)
    qc.h(0)
    for qa, qb in zip(reg_a, reg_b):
        qc.cswap(0, qa, qb)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def swap_test_quantum_pca(
    X: np.ndarray,
    n_components: int,
    primitives: QuantumPrimitives,
    max_samples: int = 8,
    shots: int = 1024,
) -> PCAResult:
    """TOY-SCALE quantum routine: estimates a sample-similarity (Gram) matrix
    via the swap test, then extracts leading eigenvectors classically from
    that quantum-estimated Gram matrix as a stand-in "quantum principal
    component" direction. Restricted to `max_samples` rows because the swap
    test's circuit width grows with feature dimension and its sample
    complexity grows quadratically with the number of vectors compared —
    this does not scale to full fraud datasets and is intended for research
    demonstration / notebooks/03_quantum_feature_selection.ipynb only."""
    if X.shape[0] > max_samples:
        raise ValueError(
            f"swap_test_quantum_pca is toy-scale; got {X.shape[0]} samples > max_samples={max_samples}. "
            f"Use classical_pca for real dataset sizes."
        )

    n = X.shape[0]
    gram = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            qc = _amplitude_encode_pair(X[i], X[j])
            job = primitives.sampler.run([qc], shots=shots)
            counts = job.result()[0].data.c.get_counts()
            p0 = counts.get("0", 0) / shots
            overlap_sq = max(0.0, 2 * p0 - 1)  # swap-test estimator of |<a|b>|^2
            gram[i, j] = gram[j, i] = overlap_sq

    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1][:n_components]
    top_eigvecs = eigvecs[:, order]
    transformed = gram @ top_eigvecs
    explained = eigvals[order] / eigvals.sum() if eigvals.sum() > 0 else np.zeros(n_components)

    return PCAResult(
        components=top_eigvecs.T,
        explained_variance_ratio=explained,
        transformed=transformed,
        method="swap_test_quantum_pca_toy_scale",
    )
