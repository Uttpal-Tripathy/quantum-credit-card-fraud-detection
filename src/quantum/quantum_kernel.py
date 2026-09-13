"""Quantum kernel construction via fidelity-based state overlap
(Qiskit Machine Learning's `FidelityQuantumKernel` + `ComputeUncompute`).

Compares the standard fidelity-style kernel against custom feature maps —
the feature map IS the kernel definition here, so "comparing kernels" means
swapping `build_feature_map(...)` inputs (see src/quantum/feature_maps.py)
and re-instantiating.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from src.quantum.noise_models import QuantumPrimitives


@dataclass
class QuantumKernelBundle:
    kernel: FidelityQuantumKernel
    feature_map: QuantumCircuit
    backend_info: object


def build_quantum_kernel(
    feature_map: QuantumCircuit,
    primitives: QuantumPrimitives,
    max_circuits_per_job: int | None = None,
) -> QuantumKernelBundle:
    """Wrap a feature map + sampler primitive into a fidelity quantum kernel
    usable by QSVC (kernel-matrix based) or any other kernel method."""
    fidelity = ComputeUncompute(sampler=primitives.sampler)
    kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
        max_circuits_per_job=max_circuits_per_job,
    )
    return QuantumKernelBundle(kernel=kernel, feature_map=feature_map, backend_info=primitives.backend_info)
