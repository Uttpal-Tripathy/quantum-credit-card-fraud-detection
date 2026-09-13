"""Variational ansatz circuits for VQC/QNN, built with Qiskit 2.x's
function-style circuit-library API."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import efficient_su2, n_local, real_amplitudes

ANSATZ_NAMES = ("real_amplitudes", "efficient_su2", "two_local")


def build_ansatz(
    name: str,
    num_qubits: int,
    reps: int = 3,
    entanglement: str = "linear",
) -> QuantumCircuit:
    """name: 'real_amplitudes' (RY rotations + entanglers, real-valued
    amplitudes — cheapest, good default), 'efficient_su2' (RY+RZ rotations,
    more expressive, deeper), or 'two_local' (fully configurable rotation +
    entanglement blocks via `n_local`)."""
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")

    if name == "real_amplitudes":
        return real_amplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
    if name == "efficient_su2":
        return efficient_su2(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
    if name == "two_local":
        return n_local(
            num_qubits=num_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cx",
            entanglement=entanglement,
            reps=reps,
        )
    raise ValueError(f"Unknown ansatz '{name}'. Choose from {ANSATZ_NAMES}.")
