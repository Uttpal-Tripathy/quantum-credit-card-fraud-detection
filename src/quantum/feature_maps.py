"""Quantum feature maps (data encoding circuits).

Built with Qiskit 2.x's function-style circuit-library API (`zz_feature_map`,
`z_feature_map`, `pauli_feature_map`) rather than the deprecated class-style
`ZZFeatureMap`/`PauliFeatureMap`. Always verify against the installed Qiskit
version (`python -c "import qiskit; print(qiskit.__version__)"`) before
following older tutorials — this project targets Qiskit >=1.3,<3.0.
"""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import pauli_feature_map, z_feature_map, zz_feature_map

FEATURE_MAP_NAMES = ("zz", "z", "pauli")


def build_feature_map(
    name: str,
    num_qubits: int,
    reps: int = 2,
    entanglement: str = "linear",
    paulis: list[str] | None = None,
) -> QuantumCircuit:
    """Build a data-encoding feature map circuit.

    name: 'zz' (ZZFeatureMap-equivalent, 2nd-order Pauli-Z evolution with
      entangling ZZ terms — the standard choice for kernel-based QML),
      'z' (1st-order, no entanglement — a lightweight baseline), or
      'pauli' (custom Pauli-string feature map via `paulis`, e.g. ['Z','ZZ','ZY']).
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")

    if name == "zz":
        return zz_feature_map(feature_dimension=num_qubits, reps=reps, entanglement=entanglement)
    if name == "z":
        return z_feature_map(feature_dimension=num_qubits, reps=reps)
    if name == "pauli":
        return pauli_feature_map(
            feature_dimension=num_qubits,
            reps=reps,
            entanglement=entanglement,
            paulis=paulis or ["Z", "ZZ"],
        )
    raise ValueError(f"Unknown feature map '{name}'. Choose from {FEATURE_MAP_NAMES}.")
