"""Noise models and the safe backend manager.

Implements spec section 11 ("Implement a safe backend manager"): the caller
picks backend='simulator' | 'noisy_simulator' | 'ibm_quantum', and this
module returns matching Sampler/Estimator V2 primitives. Real hardware access
is optional everywhere else in the project — if IBM Quantum credentials are
missing or the service is unreachable, `get_primitives` logs a warning and
falls back to the (noisy) Aer simulator instead of raising, so the rest of
the pipeline keeps working without credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
)

from src.config.settings import get_env, load_quantum_config
from src.utils.logging import get_logger

logger = get_logger("quantum.noise_models")

BackendChoice = Literal["simulator", "noisy_simulator", "ibm_quantum"]


def build_noise_model(
    depolarizing_prob_1q: float = 0.001,
    depolarizing_prob_2q: float = 0.01,
    readout_error_prob: float = 0.02,
) -> NoiseModel:
    """Construct a configurable depolarizing + readout-error noise model.
    This is a simplified device-agnostic noise model for controlled
    robustness experiments — not a calibration pull from real hardware."""
    noise_model = NoiseModel()

    error_1q = depolarizing_error(depolarizing_prob_1q, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ["id", "rz", "sx", "x", "ry", "h", "u1", "u2", "u3"])

    error_2q = depolarizing_error(depolarizing_prob_2q, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz", "ecr"])

    readout_error = pauli_error([
        ("X", readout_error_prob),
        ("I", 1 - readout_error_prob),
    ])
    noise_model.add_all_qubit_quantum_error(readout_error, ["measure"])

    return noise_model


@dataclass
class BackendInfo:
    backend_name: str
    backend_choice: str
    is_simulator: bool
    is_noisy: bool
    fallback_reason: str | None = None
    num_qubits_available: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumPrimitives:
    sampler: Any
    estimator: Any
    backend_info: BackendInfo


def _build_simulator_primitives(shots: int, noisy: bool, seed: int, quantum_cfg: dict) -> QuantumPrimitives:
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

    noise_cfg = quantum_cfg.get("noise", {})
    backend_options: dict[str, Any] = {}
    if noisy:
        noise_model = build_noise_model(
            depolarizing_prob_1q=noise_cfg.get("depolarizing_prob_1q", 0.001),
            depolarizing_prob_2q=noise_cfg.get("depolarizing_prob_2q", 0.01),
            readout_error_prob=noise_cfg.get("readout_error_prob", 0.02),
        )
        backend_options["noise_model"] = noise_model

    aer_backend = AerSimulator(**backend_options)
    sampler = AerSamplerV2.from_backend(aer_backend, default_shots=shots, seed=seed)
    estimator = AerEstimatorV2.from_backend(aer_backend)

    info = BackendInfo(
        backend_name="aer_simulator" + ("_noisy" if noisy else ""),
        backend_choice="noisy_simulator" if noisy else "simulator",
        is_simulator=True,
        is_noisy=noisy,
    )
    return QuantumPrimitives(sampler=sampler, estimator=estimator, backend_info=info)


def _try_build_ibm_primitives(shots: int, quantum_cfg: dict) -> QuantumPrimitives | None:
    """Attempt real IBM Quantum backend access. Returns None (never raises)
    if credentials are missing or the service call fails, so the caller can
    fall back to a simulator gracefully."""
    token = get_env("IBM_QUANTUM_TOKEN")
    if not token:
        logger.warning(
            "backend='ibm_quantum' requested but IBM_QUANTUM_TOKEN is not set in .env; "
            "falling back to the noisy simulator."
        )
        return None

    try:
        from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2

        ibm_cfg = quantum_cfg.get("ibm_quantum", {})
        service = QiskitRuntimeService(
            channel=get_env("IBM_QUANTUM_CHANNEL", ibm_cfg.get("channel", "ibm_quantum_platform")),
            token=token,
            instance=get_env("IBM_QUANTUM_INSTANCE") or ibm_cfg.get("instance"),
        )
        backend = service.least_busy(operational=True, simulator=False)

        sampler = RuntimeSamplerV2(mode=backend)
        estimator = RuntimeEstimatorV2(mode=backend)
        sampler.options.default_shots = shots

        info = BackendInfo(
            backend_name=backend.name,
            backend_choice="ibm_quantum",
            is_simulator=False,
            is_noisy=True,
            num_qubits_available=backend.num_qubits,
        )
        return QuantumPrimitives(sampler=sampler, estimator=estimator, backend_info=info)

    except Exception as exc:  # noqa: BLE001 - any failure must degrade gracefully, not crash the run
        logger.warning(f"IBM Quantum backend unavailable ({exc!r}); falling back to noisy simulator.")
        return None


def get_primitives(
    backend: BackendChoice | None = None,
    shots: int | None = None,
    seed: int | None = None,
) -> QuantumPrimitives:
    """The single entry point every quantum model in this project should use
    to obtain (sampler, estimator, backend_info). Handles the
    simulator -> noisy_simulator -> ibm_quantum selection and graceful
    hardware fallback described in spec section 11."""
    quantum_cfg = load_quantum_config()["quantum"]
    backend = backend or quantum_cfg.get("backend", "simulator")
    shots = shots or quantum_cfg.get("shots", 1024)
    seed = seed if seed is not None else quantum_cfg.get("random_seed", 42)

    if backend == "ibm_quantum":
        primitives = _try_build_ibm_primitives(shots, load_quantum_config())
        if primitives is not None:
            return primitives
        fallback = _build_simulator_primitives(shots, noisy=True, seed=seed, quantum_cfg=quantum_cfg)
        fallback.backend_info.fallback_reason = "ibm_quantum unavailable (missing credentials or service error)"
        return fallback

    if backend == "noisy_simulator":
        return _build_simulator_primitives(shots, noisy=True, seed=seed, quantum_cfg=quantum_cfg)

    if backend == "simulator":
        return _build_simulator_primitives(shots, noisy=False, seed=seed, quantum_cfg=quantum_cfg)

    raise ValueError(f"Unknown backend '{backend}'. Choose from: simulator, noisy_simulator, ibm_quantum.")
