from __future__ import annotations

from fastapi import APIRouter

from linescout_api.deps import State
from linescout_api.errors import service_unavailable
from linescout_api.preprocessing import PREPROCESSING_VERSION
from linescout_api.schemas import ErrorResponse, HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(state: State) -> HealthResponse:
    warnings = list(state.warnings)
    if state.setup_error:
        warnings.insert(0, state.setup_error)
    return HealthResponse(
        ready=state.ready,
        fixture_mode=state.settings.fixture_mode,
        cuda_available=state.device.cuda_available,
        device="cuda" if state.device.is_cuda else "cpu",
        gpu_name=state.device.gpu_name,
        vram_total_mb=state.device.vram_total_mb,
        torch_version=state.device.torch_version,
        api_version=state.api_version,
        schema_version=state.schema_version,
        preprocessing_version=PREPROCESSING_VERSION,
        models=state.models,
        dataset_version=state.gallery.dataset_version if state.gallery else None,
        index_version=state.gallery.manifest_hash[:16] if state.gallery else None,
        gallery_size=len(state.assets),
        disabled_branches=state.disabled_branches,
        warmup=state.warmup,
        warnings=warnings,
        curation_enabled=state.settings.curation_mode,
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ErrorResponse, "description": "Model or gallery is not ready"}},
)
def ready(state: State) -> ReadyResponse:
    if not state.ready:
        raise service_unavailable(
            "not_ready",
            state.setup_error or "the API is not ready",
            details=state.readiness_details(),
        )
    return ReadyResponse()
