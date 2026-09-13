from __future__ import annotations

from fastapi import APIRouter

from api import services
from api.schemas import FeatureSelectionRequest

router = APIRouter(prefix="/api/feature-selection", tags=["feature-selection"])


@router.post("")
def run_feature_selection(payload: FeatureSelectionRequest) -> list[dict]:
    return services.feature_selection(candidate_counts=tuple(sorted(payload.candidate_counts)), seed=payload.seed)
