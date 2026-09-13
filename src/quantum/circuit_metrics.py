"""Circuit resource metrics: qubit count, logical depth, transpiled depth,
gate counts, and execution timing. Every quantum experiment record (see
src/utils/experiment_tracker.py) is populated from this module so resource
cost is tracked as rigorously as accuracy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


@dataclass
class CircuitMetrics:
    num_qubits: int
    logical_depth: int
    logical_gate_count: int
    transpiled_depth: int
    transpiled_gate_count: int
    two_qubit_gate_count: int
    optimization_level: int


def compute_circuit_metrics(
    circuit: QuantumCircuit,
    backend=None,
    optimization_level: int = 1,
) -> CircuitMetrics:
    """Transpile `circuit` for `backend` (or a generic basis-gate set if no
    backend is given) and report both logical and transpiled resource costs.
    Transpiled depth is what actually determines wall-clock/noise exposure on
    real hardware, so both numbers are recorded, never just the logical one."""
    logical_depth = circuit.decompose().depth()
    logical_gate_count = sum(circuit.decompose().count_ops().values())

    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        basis_gates=None if backend is not None else ["rz", "sx", "x", "cx"],
    )
    transpiled = pass_manager.run(circuit)
    op_counts = transpiled.count_ops()

    two_qubit_gates = sum(
        count for gate, count in op_counts.items() if gate in {"cx", "cz", "ecr", "swap"}
    )

    return CircuitMetrics(
        num_qubits=circuit.num_qubits,
        logical_depth=logical_depth,
        logical_gate_count=logical_gate_count,
        transpiled_depth=transpiled.depth(),
        transpiled_gate_count=sum(op_counts.values()),
        two_qubit_gate_count=two_qubit_gates,
        optimization_level=optimization_level,
    )


@dataclass
class TimedExecution:
    elapsed_s: float
    result: object


def timed(fn, *args, **kwargs) -> TimedExecution:
    """Run `fn(*args, **kwargs)` and record wall-clock time — used to time
    circuit execution/inference consistently across every quantum model."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return TimedExecution(elapsed_s=elapsed, result=result)
