from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import services
from api.schemas import QuantumLabRequest

router = APIRouter(prefix="/api/quantum", tags=["quantum"])


@router.get("/summary")
def summary() -> dict:
    return services.demo_quantum_lab_summary()


@router.post("/circuit")
def build_circuit(payload: QuantumLabRequest) -> dict:
    try:
        return services.quantum_circuit_info(
            feature_map=payload.feature_map, ansatz=payload.ansatz, qubits=payload.qubits, reps=payload.reps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
