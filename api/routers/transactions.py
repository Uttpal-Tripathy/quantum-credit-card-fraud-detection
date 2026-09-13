from __future__ import annotations

from fastapi import APIRouter, Query

from api import services
from api.schemas import ScoreRequest

router = APIRouter(prefix="/api/transactions", tags=["transactions"])


@router.get("")
def list_transactions(
    limit: int = Query(50, ge=1, le=500),
    decision: str | None = Query(None, description="Comma-separated: APPROVE,REVIEW,BLOCK"),
    routed_only: bool = Query(False),
) -> list[dict]:
    decisions = [d.strip().upper() for d in decision.split(",")] if decision else None
    return services.list_transactions(limit=limit, decisions=decisions, routed_only=routed_only)


@router.post("/score")
def score_transaction(payload: ScoreRequest) -> dict:
    """Draws a fresh synthetic transaction and runs it through the live
    QGFDA pipeline end-to-end (real inference call, not a lookup)."""
    return services.score_new_transaction(amount_multiplier=payload.amount_multiplier)
