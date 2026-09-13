from __future__ import annotations

from fastapi import APIRouter

from api import services
from api.schemas import RobustnessRequest

router = APIRouter(prefix="/api/drift", tags=["drift"])


@router.get("")
def get_drift() -> dict:
    return services.drift_report()


@router.post("/robustness")
def robustness(payload: RobustnessRequest) -> dict:
    return services.robustness_perturbation(amount_pct=payload.amount_pct)
