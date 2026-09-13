from __future__ import annotations

from fastapi import APIRouter

from api import services

router = APIRouter(prefix="/api/overview", tags=["overview"])


@router.get("")
def get_overview() -> dict:
    return services.overview()
