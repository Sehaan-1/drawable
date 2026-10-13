from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from time import perf_counter
from typing import Literal, cast
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from drawable_research_contracts.readiness import STAGES, validate_readiness
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from .curation import (
    authorized_preview_path,
    create_crop_asset,
    create_curation_snapshot,
    curation_progress,
    list_candidates,
    list_source_yield_queue,
    review_asset,
    review_history,
    review_view_match,
    viewpoint_rubric,
)
from .database import migrate
from .intake import IntakeError, audit_sources, load_registry
from .prototype_review import prepare_prototype_batch
from .prototype_search import PrototypeSearchRuntime
from .review_workspace import ReviewWorkspace, ReviewWorkspaceError, run_review_preparation
from .schemas import (
    ApiError,
    ApiErrorBody,
    CurationCandidate,
    CurationCandidateList,
    CurationCropRequest,
    CurationCropResponse,
    CurationHistoryItem,
    CurationProgress,
    CurationReviewRequest,
    CurationReviewResponse,
    CurationSnapshotRequest,
    CurationSnapshotResponse,
    CurationViewMatchRequest,
    CurationViewMatchResponse,
    HealthResponse,
    M0WorkspaceResponse,
    ReadinessResponse,
    ReviewCandidatePreparationRequest,
    ReviewDecisionRequest,
    ReviewPreparationJob,
    ReviewPreviewBatchRequest,
    ReviewTermsCaptureRequest,
    ReviewTermsCaptureResponse,
    ReviewWorkSelectionRequest,
    ReviewWorkspaceStatus,
    SearchResponse,
    SearchTimings,
    SourceYieldQueueResponse,
    ViewpointRubricResponse,
)
from .search import (
    QUERY_PREPROCESSING_VERSION,
    SearchInputError,
    accept_generation,
    parse_search_request,
)
from .settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()
    review_workspace: ReviewWorkspace | None = None
    if resolved.curation_mode:
        review_workspace = ReviewWorkspace(resolved)
        resolved = replace(
            resolved,
            source_registry=review_workspace.sources_path,
            anita_work_terms=review_workspace.works_path,
            model_registry=review_workspace.models_path,
        )

    runtime = (
        PrototypeSearchRuntime(resolved)
        if resolved.api_mode == "real" and resolved.local_prototype_mode
        else None
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if resolved.curation_mode:
            migrate(resolved.metadata_db)
            if resolved.local_prototype_mode:
                prepare_prototype_batch(resolved, reuse_pending=True)
        if runtime is not None:
            await runtime.start()
        try:
            yield
        finally:
            if runtime is not None:
                await runtime.close()

    app = FastAPI(
        title="drawable local API",
        version="0.1.0",
        description="Loopback-only API with opt-in, frozen CPU educational prototype search.",
        docs_url=None if resolved.api_mode == "real" else "/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.search_generations = {}
    if runtime is not None:
        runtime.generations = app.state.search_generations
    app.state.review_workspace = review_workspace
    app.state.prototype_search = runtime

    def require_curation(request: Request, *, mutation: bool = False) -> None:
        if not resolved.curation_mode:
            raise HTTPException(
                status_code=404,
                detail="Local curation is disabled; set CURATION_MODE=1 explicitly.",
            )
        allowed_hosts = {
            (urlsplit(origin).netloc or "").casefold()
            for origin in resolved.allowed_origins
            if urlsplit(origin).scheme in {"http", "https"}
        }
        if request.headers.get("host", "").casefold() not in allowed_hosts:
            raise HTTPException(
                status_code=403,
                detail="Curation is available only through a configured local host.",
            )
        origin = request.headers.get("origin")
        if mutation and (not origin or origin.rstrip("/") not in resolved.allowed_origins):
            raise HTTPException(
                status_code=403,
                detail="Curation mutations require an approved local same-origin request.",
            )

    def require_review_workspace(request: Request, *, mutation: bool = False) -> ReviewWorkspace:
        require_curation(request, mutation=mutation)
        workspace = app.state.review_workspace
        if workspace is None:
            raise HTTPException(
                status_code=503, detail="The private local review workspace is unavailable."
            )
        return workspace

    @app.middleware("http")
    async def request_identity(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid4()
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = str(request_id)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, _error: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", uuid4())
        body = ApiError(
            request_id=request_id,
            error=ApiErrorBody(
                code="internal_error", message="The local API could not complete the request."
            ),
        )
        return JSONResponse(status_code=500, content=body.model_dump(mode="json"))

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
    async def health(request: Request) -> HealthResponse:
        fixture = resolved.api_mode == "fixture"
        return HealthResponse(
            request_id=request.state.request_id,
            ready=fixture or (runtime is not None and runtime.ready),
            mode="fixture" if fixture else "cpu",
            models_loaded=runtime is not None and runtime.ready,
            gallery_size=int(runtime.status.get("gallery_size", 0)) if runtime is not None and runtime.ready else 0,
            enabled_branches=runtime.status.get("enabled_branches", [])
            if runtime is not None and runtime.ready
            else [],
            setup_errors=[]
            if fixture or (runtime is not None and runtime.ready)
            else [
                (runtime.error or "prototype_search_worker_stopped")
                if runtime is not None
                else "approved_models_and_gallery_not_installed"
            ],
        )

    @app.get("/api/v1/ready", response_model=ReadinessResponse, tags=["health"])
    async def readiness(request: Request, response: Response) -> ReadinessResponse:
        ready = resolved.api_mode == "fixture" or (runtime is not None and runtime.ready)
        if not ready:
            response.status_code = 503
        return ReadinessResponse(
            request_id=request.state.request_id,
            ready=ready,
            profile=resolved.api_mode,
            reason=None
            if ready
            else (runtime.error or "prototype_search_worker_stopped")
            if runtime is not None
            else "approved_models_and_gallery_not_installed",
        )

    @app.get("/api/v1/search", tags=["search"])
    async def search_not_ready(request: Request) -> JSONResponse:
        error = ApiError(
            request_id=request.state.request_id,
            error=ApiErrorBody(
                code="search_not_configured",
                message="Real search is unavailable until an approved model and gallery build are installed.",
            ),
        )
        return JSONResponse(status_code=503, content=error.model_dump(mode="json"))

    search_multipart_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "session_id",
            "document_id",
            "canvas_revision",
            "query_generation",
            "canvas_width",
            "canvas_height",
            "stroke_count",
            "point_count",
            "counts_approximate",
            "stroke_status",
            "preprocessing_version",
            "image",
        ],
        "properties": {
            "session_id": {"type": "string", "format": "uuid"},
            "document_id": {"type": "string", "format": "uuid"},
            "canvas_revision": {"type": "integer", "minimum": 0, "maximum": 9007199254740991},
            "query_generation": {"type": "integer", "minimum": 0, "maximum": 9007199254740991},
            "canvas_width": {"type": "integer", "const": 2048},
            "canvas_height": {"type": "integer", "const": 2048},
            "stroke_count": {"type": "integer", "minimum": 0, "maximum": 10000},
            "point_count": {"type": "integer", "minimum": 0, "maximum": 1000000},
            "counts_approximate": {"type": "boolean"},
            "stroke_status": {
                "type": "string",
                "enum": [
                    "available",
                    "too_short",
                    "erased_layers",
                    "raster_content",
                    "unsupported_transform",
                    "out_of_bounds",
                    "too_many_boundaries",
                    "disabled",
                ],
            },
            "preprocessing_version": {"type": "string", "maxLength": 100},
            "text_hint": {"type": "string", "maxLength": 120},
            "image": {"type": "string", "format": "binary"},
            "strokes": {"type": "string", "format": "binary"},
        },
    }

    @app.post(
        "/api/v1/search",
        response_model=SearchResponse,
        tags=["search"],
        responses={
            400: {"model": ApiError},
            409: {"model": ApiError},
            413: {"model": ApiError},
            422: {"model": ApiError},
            429: {"model": ApiError},
            503: {"model": ApiError},
            504: {"model": ApiError},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"multipart/form-data": {"schema": search_multipart_schema}},
            }
        },
    )
    async def search(request: Request) -> SearchResponse | JSONResponse:
        started = perf_counter()
        try:
            if runtime is not None:
                allowed_hosts = {
                    (urlsplit(origin).netloc or "").casefold()
                    for origin in resolved.allowed_origins
                }
                if (
                    request.headers.get("host", "").casefold() not in allowed_hosts
                    or request.headers.get("origin", "").rstrip("/") not in resolved.allowed_origins
                ):
                    raise SearchInputError(
                        403,
                        "local_origin_required",
                        "Prototype search requires a configured local same-origin request.",
                    )
            parsed = await parse_search_request(request)
            accept_generation(
                request.app.state.search_generations,
                parsed.fields.session_id,
                parsed.fields.query_generation,
            )
            if not parsed.insufficient and runtime is not None:
                preprocessing_ms = max(0.0, (perf_counter() - started) * 1000)
                result = await runtime.search(parsed, started=started)
                latest = request.app.state.search_generations.get(str(parsed.fields.session_id))
                if latest is not None and latest[0] != parsed.fields.query_generation:
                    raise SearchInputError(
                        409, "superseded_search", "A newer drawing replaced this search."
                    )
                return SearchResponse(
                    request_id=request.state.request_id,
                    session_id=parsed.fields.session_id,
                    document_id=parsed.fields.document_id,
                    canvas_revision=parsed.fields.canvas_revision,
                    query_generation=parsed.fields.query_generation,
                    build_id=result["build_id"],
                    model_version=result["model_version"],
                    gallery_size=result["gallery_size"],
                    scope_coverage=result.get("scope_coverage", {}),
                    style_coverage=result.get("style_coverage", {}),
                    preprocessing_version=QUERY_PREPROCESSING_VERSION,
                    mode=result["mode"],
                    reason=result["reason"],
                    interpretation=result["interpretation"],
                    groups=result["groups"],
                    warnings=result["warnings"],
                    features={
                        "semantic": "frozen",
                        "structural": "unavailable",
                        "stroke": "unavailable",
                        "pose": "unavailable",
                        "text": "unavailable",
                    },
                    timings_ms=SearchTimings(
                        queue_ms=result["queue_ms"],
                        preprocessing_ms=preprocessing_ms,
                        total_ms=max(0.0, (perf_counter() - started) * 1000),
                    ),
                )
        except SearchInputError as error:
            body = ApiError(
                request_id=request.state.request_id,
                error=ApiErrorBody(
                    code=error.code,
                    message=str(error),
                    retryable=error.retryable,
                ),
            )
            headers = {"Retry-After": "1"} if error.retryable else None
            return JSONResponse(
                status_code=error.status_code,
                content=body.model_dump(mode="json"),
                headers=headers,
            )

        elapsed_ms = max(0.0, (perf_counter() - started) * 1000)
        if not parsed.insufficient:
            body = ApiError(
                request_id=request.state.request_id,
                error=ApiErrorBody(
                    code="search_not_configured",
                    message="Real search is unavailable until an approved model and gallery build are installed.",
                ),
            )
            return JSONResponse(status_code=503, content=body.model_dump(mode="json"))
        return SearchResponse(
            request_id=request.state.request_id,
            session_id=parsed.fields.session_id,
            document_id=parsed.fields.document_id,
            canvas_revision=parsed.fields.canvas_revision,
            query_generation=parsed.fields.query_generation,
            gallery_size=int(runtime.status.get("gallery_size", 0)) if runtime is not None and runtime.ready else 0,
            scope_coverage=runtime.status.get("scope_coverage", {}) if runtime is not None and runtime.ready else {},
            style_coverage=runtime.status.get("style_coverage", {}) if runtime is not None and runtime.ready else {},
            preprocessing_version=QUERY_PREPROCESSING_VERSION,
            mode="insufficient",
            reason="insufficient_visible_ink",
            interpretation="Add more visible ink to search for references.",
            features={
                "semantic": "unavailable",
                "structural": "unavailable",
                "stroke": "unavailable",
                "pose": "unavailable",
                "text": "unavailable",
            },
            timings_ms=SearchTimings(
                queue_ms=0.0,
                preprocessing_ms=elapsed_ms,
                total_ms=elapsed_ms,
            ),
        )

    @app.get("/api/v1/assets/{asset_id}/preview", tags=["assets"])
    async def prototype_preview(
        asset_id: str,
        request: Request,
        build_id: str = Query(max_length=100),
        thumbnail: bool = Query(default=False),
    ) -> FileResponse:
        allowed_hosts = {
            (urlsplit(origin).netloc or "").casefold() for origin in resolved.allowed_origins
        }
        if request.headers.get("host", "").casefold() not in allowed_hosts:
            raise HTTPException(
                status_code=403, detail="Reference previews require a configured local host."
            )
        if runtime is None:
            raise HTTPException(status_code=404, detail="Local prototype search is disabled.")
        try:
            path, media_type = runtime.preview(asset_id, build_id, thumbnail=thumbnail)
        except LookupError as error:
            raise HTTPException(
                status_code=404, detail="Reference is absent from the active reviewed batch."
            ) from error
        except (ValueError, OSError) as error:
            raise HTTPException(
                status_code=409, detail="Reference bytes changed; prepare local search again."
            ) from error
        return FileResponse(
            path,
            media_type=media_type,
            headers={"X-Request-ID": str(request.state.request_id), "Cache-Control": "no-store"},
        )

    @app.get("/api/v1/curation/candidates", response_model=CurationCandidateList, tags=["curation"])
    async def curation_candidates(
        request: Request,
        limit: int = Query(default=50, ge=1, le=100),
        source_id: str | None = None,
        state: str = Query(default="pending", pattern="^(pending|candidate|rejected|all)$"),
        work_id: str | None = None,
        candidate_type: str | None = None,
        cursor: str | None = Query(default=None, max_length=2048),
    ) -> CurationCandidateList:
        require_curation(request)
        migrate(resolved.metadata_db)
        try:
            page = list_candidates(
                resolved,
                limit=limit,
                source_id=source_id,
                state=state,
                work_id=work_id,
                candidate_type=candidate_type,
                cursor=cursor,
            )
            return CurationCandidateList(
                items=[CurationCandidate(**item) for item in page["items"]],
                next_cursor=page["next_cursor"],
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/curation/progress", response_model=CurationProgress, tags=["curation"])
    async def curation_progress_route(request: Request) -> CurationProgress:
        require_curation(request)
        migrate(resolved.metadata_db)
        summary = curation_progress(resolved)
        return CurationProgress(
            total=summary["total"],
            pending=summary["pending"],
            candidate=summary["candidate"],
            rejected=summary["rejected"],
            reviewed=summary["reviewed"],
        )

    @app.get(
        "/api/v1/curation/source-yield",
        response_model=SourceYieldQueueResponse,
        tags=["curation"],
    )
    async def curation_source_yield(request: Request) -> SourceYieldQueueResponse:
        require_curation(request)
        migrate(resolved.metadata_db)
        try:
            return SourceYieldQueueResponse(**list_source_yield_queue(resolved))
        except (IntakeError, ValueError, OSError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get(
        "/api/v1/feasibility/m0",
        response_model=M0WorkspaceResponse,
        tags=["feasibility"],
    )
    async def m0_feasibility_workspace(request: Request) -> M0WorkspaceResponse:
        require_curation(request)
        root = resolved.data_root
        default_ledger = root / "reports" / "readiness-ledger-v2.json"
        readiness = validate_readiness(
            root,
            "m0",
            default_ledger
            if default_ledger.is_file() and not default_ledger.is_symlink()
            else None,
        )
        source_registry_path = resolved.source_registry_path
        model_registry_path = resolved.model_registry_path
        try:
            sources = load_registry(source_registry_path)["sources"]
            models_document = json.loads(model_registry_path.read_text(encoding="utf-8"))
            audit = audit_sources(root, registry_path=source_registry_path, verify_archives=False)
            source_archive_status = {
                str(row["source_id"]): str(row["status"]) for row in audit["sources"]
            }
            source_terms_status = {
                str(row["id"]): str(row.get("terms_review", {}).get("status", "pending"))
                for row in sources
            }
            model_terms_status = {
                str(row["id"]): str(row.get("terms_review", {}).get("status", "pending"))
                for row in models_document.get("models", [])
                if isinstance(row, dict)
            }
        except (OSError, ValueError, KeyError, TypeError, IntakeError) as error:
            raise HTTPException(
                status_code=503, detail=f"M0 review registry could not be read: {error}"
            ) from error
        sample_path = root / "reports" / "source-yield-sample-pilot-v2.json"
        sample_status = "not_generated"
        sample_count = 0
        if sample_path.is_file() and not sample_path.is_symlink():
            try:
                sample = json.loads(sample_path.read_text(encoding="utf-8"))
                sample_status = str(sample.get("status", "invalid"))
                sample_count = int(sample.get("selected_count", 0))
            except (OSError, ValueError, TypeError):
                sample_status = "invalid"
        verified = set(readiness["verified_evidence"])
        blockers = readiness["blockers"]
        private_root = root / "reports" / "private"
        effort_directory = private_root / "m0-effort-logs"
        effort_log_available = False
        if not any(
            path.is_symlink() for path in (root / "reports", private_root, effort_directory)
        ):
            try:
                effort_log_available = any(
                    path.is_file() and not path.is_symlink() for path in effort_directory.iterdir()
                )
            except OSError:
                effort_log_available = False
        return M0WorkspaceResponse(
            status="complete" if readiness["status"] == "complete" else "blocked",
            gates={gate: "complete" if gate in verified else "incomplete" for gate in STAGES["m0"]},
            blockers={gate: str(reason) for gate, reason in blockers.items()},
            source_archive_status=source_archive_status,
            source_terms_status=source_terms_status,
            model_terms_status=model_terms_status,
            effort_log_available=effort_log_available,
            source_yield_sample_status=sample_status,
            source_yield_sample_count=max(0, min(250, sample_count)),
        )

    @app.post(
        "/api/v1/curation/prototype/batch", response_model=CurationCandidateList, tags=["curation"]
    )
    async def prototype_next_batch(
        request: Request, limit: int = Query(default=24, ge=1, le=32)
    ) -> CurationCandidateList:
        require_curation(request, mutation=True)
        try:
            prepare_prototype_batch(resolved, limit=limit)
            return CurationCandidateList(**list_candidates(resolved, limit=100))
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except (IntakeError, OSError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get(
        "/api/v1/curation/rubric/viewpoint",
        response_model=ViewpointRubricResponse,
        tags=["curation"],
    )
    async def curation_viewpoint_rubric(request: Request) -> ViewpointRubricResponse:
        require_curation(request)
        try:
            return ViewpointRubricResponse(**viewpoint_rubric())
        except IntakeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post(
        "/api/v1/curation/snapshots", response_model=CurationSnapshotResponse, tags=["curation"]
    )
    async def curation_snapshot(
        request: Request, payload: CurationSnapshotRequest
    ) -> CurationSnapshotResponse:
        require_curation(request, mutation=True)
        migrate(resolved.metadata_db)
        try:
            return create_curation_snapshot(
                resolved,
                payload,
                registry_path=resolved.source_registry_path,
                anita_terms_path=resolved.anita_work_terms_path,
            )
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except (IntakeError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            status = 409 if "idempotency key" in str(error).casefold() else 422
            raise HTTPException(status_code=status, detail=str(error)) from error

    @app.get("/api/v1/curation/assets/{asset_id}/preview", tags=["curation"])
    async def curation_preview(request: Request, asset_id: str) -> FileResponse:
        require_curation(request)
        migrate(resolved.metadata_db)
        try:
            path, media_type = authorized_preview_path(resolved, asset_id)
            return FileResponse(
                path,
                media_type=media_type,
                headers={
                    "Content-Disposition": "inline",
                    "X-Content-Type-Options": "nosniff",
                    "Content-Security-Policy": "default-src 'none'; sandbox",
                },
            )
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (IntakeError, OSError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post(
        "/api/v1/curation/assets/{asset_id}/crops",
        response_model=CurationCropResponse,
        tags=["curation"],
    )
    def curation_crop(
        request: Request, asset_id: str, payload: CurationCropRequest
    ) -> CurationCropResponse:
        require_curation(request, mutation=True)
        migrate(resolved.metadata_db)
        try:
            return create_crop_asset(
                resolved,
                asset_id,
                payload,
                registry_path=resolved.source_registry_path,
                anita_terms_path=resolved.anita_work_terms_path,
            )
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (IntakeError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            status = 409 if "idempotency key" in str(error).casefold() else 422
            raise HTTPException(status_code=status, detail=str(error)) from error

    @app.post(
        "/api/v1/curation/assets/{asset_id}/review",
        response_model=CurationReviewResponse,
        tags=["curation"],
    )
    async def review_curation_asset(
        request: Request, asset_id: str, payload: CurationReviewRequest
    ) -> CurationReviewResponse:
        require_curation(request, mutation=True)
        migrate(resolved.metadata_db)
        try:
            return review_asset(
                resolved,
                asset_id,
                payload,
                registry_path=resolved.source_registry_path,
                anita_terms_path=resolved.anita_work_terms_path,
            )
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except IntakeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            status = 409 if "idempotency key" in str(error).casefold() else 422
            raise HTTPException(status_code=status, detail=str(error)) from error

    @app.post(
        "/api/v1/curation/view-matches", response_model=CurationViewMatchResponse, tags=["curation"]
    )
    async def review_curation_view_match(
        request: Request, payload: CurationViewMatchRequest
    ) -> CurationViewMatchResponse:
        require_curation(request, mutation=True)
        migrate(resolved.metadata_db)
        try:
            return review_view_match(resolved, payload)
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (IntakeError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            status = 409 if "idempotency key" in str(error).casefold() else 422
            raise HTTPException(status_code=status, detail=str(error)) from error

    @app.get(
        "/api/v1/curation/assets/{asset_id}/reviews",
        response_model=list[CurationHistoryItem],
        tags=["curation"],
    )
    async def curation_history(request: Request, asset_id: str) -> list[CurationHistoryItem]:
        require_curation(request)
        migrate(resolved.metadata_db)
        return [CurationHistoryItem(**item) for item in review_history(resolved, asset_id)]

    @app.get("/api/v1/setup", response_model=ReviewWorkspaceStatus, tags=["setup"])
    async def setup_status(request: Request) -> ReviewWorkspaceStatus:
        workspace = require_review_workspace(request)
        try:
            return ReviewWorkspaceStatus(**workspace.status())
        except (ReviewWorkspaceError, IntakeError, OSError, ValueError) as error:
            raise HTTPException(
                status_code=503, detail="Local review status could not be loaded."
            ) from error

    @app.post(
        "/api/v1/setup/terms",
        response_model=ReviewTermsCaptureResponse,
        tags=["setup"],
    )
    async def setup_capture_terms(
        request: Request, payload: ReviewTermsCaptureRequest
    ) -> ReviewTermsCaptureResponse:
        workspace = require_review_workspace(request, mutation=True)
        try:
            result = workspace.capture_terms(payload.kind, payload.id)
            terms_digest = result.get("terms_snapshot_sha256") or result.get("sha256")
            if not isinstance(terms_digest, str):
                raise ReviewWorkspaceError("The terms capture did not return its content hash")
            return ReviewTermsCaptureResponse(
                status=result.get("status", "captured_pending_owner_review"),
                kind=payload.kind,
                id=payload.id if payload.kind == "source" else None,
                model_id=payload.id if payload.kind == "model" else None,
                purpose=result.get("purpose"),
                source_url=result.get("requested_url") or result.get("source_url"),
                final_url=result.get("final_url"),
                snapshot_path=result.get("snapshot_path"),
                terms_snapshot_sha256=terms_digest,
                source_content_sha256=result.get("source_content_sha256"),
                retrieved_at=result.get("retrieved_at"),
                captured_at=result.get("captured_at"),
                byte_length=result.get("byte_length") or result.get("content_bytes"),
                owner_decision_required=True,
                network_used=True,
                model_loaded=False,
                training_started=False,
            )
        except (ReviewWorkspaceError, IntakeError, OSError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/setup/terms/{kind}/{identifier}", tags=["setup"])
    async def setup_terms_document(request: Request, kind: str, identifier: str) -> Response:
        workspace = require_review_workspace(request)
        if kind not in {"source", "model"}:
            raise HTTPException(status_code=404, detail="Terms copy was not found.")
        try:
            content, content_type = workspace.terms_document(kind, identifier)
        except ReviewWorkspaceError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Content-Security-Policy": "sandbox; default-src 'none'; img-src data:; style-src 'unsafe-inline'",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Disposition": "inline",
            },
        )

    @app.post("/api/v1/setup/decisions", tags=["setup"])
    async def setup_record_decision(
        request: Request, payload: ReviewDecisionRequest
    ) -> dict[str, object]:
        workspace = require_review_workspace(request, mutation=True)
        try:
            return workspace.record_decision(
                kind=payload.kind,
                identifier=payload.id,
                expected_revision=payload.expected_revision,
                reviewer_id=payload.reviewer_id,
                decision=payload.decision,
                terms_sha256=payload.terms_snapshot_sha256,
                note=payload.note,
                permissions=payload.permissions.model_dump()
                if payload.permissions is not None
                else None,
                attribution_text=payload.attribution_text,
            )
        except ReviewWorkspaceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.put("/api/v1/setup/works", tags=["setup"])
    async def setup_select_works(
        request: Request, payload: ReviewWorkSelectionRequest
    ) -> dict[str, object]:
        workspace = require_review_workspace(request, mutation=True)
        try:
            return workspace.select_works(
                expected_revision=payload.expected_revision, work_ids=payload.work_ids
            )
        except ReviewWorkspaceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post(
        "/api/v1/setup/prepare-candidates",
        response_model=ReviewPreparationJob,
        tags=["setup"],
    )
    async def setup_prepare_candidates(
        request: Request,
        payload: ReviewCandidatePreparationRequest,
        background_tasks: BackgroundTasks,
    ) -> ReviewPreparationJob:
        workspace = require_review_workspace(request, mutation=True)
        try:
            job_id = workspace.queue_preparation(
                expected_revision=payload.expected_revision,
                reviewer_id=payload.reviewer_id,
            )
        except ReviewWorkspaceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        background_tasks.add_task(run_review_preparation, workspace, job_id)
        return ReviewPreparationJob(
            job_id=UUID(job_id),
            status="queued",
            message="Preparing the verified archive and private candidate queue.",
            progress={"phase": "queued"},
        )

    @app.get(
        "/api/v1/setup/prepare-candidates/{job_id}",
        response_model=ReviewPreparationJob,
        tags=["setup"],
    )
    async def setup_preparation_status(request: Request, job_id: str) -> ReviewPreparationJob:
        workspace = require_review_workspace(request)
        try:
            status = workspace.status()["preparation"]
            if status.get("job_id") != job_id:
                raise HTTPException(status_code=404, detail="Preparation job was not found.")
            phase = str(status.get("phase", status.get("status", "unknown")))
            progress: dict[str, int | str] = {"phase": phase}
            partial = workspace.root / "raw" / "AnitaDataset.zip.part"
            if partial.is_file() and not partial.is_symlink():
                progress["downloaded_bytes"] = partial.stat().st_size
            status_name = str(status.get("status", "blocked"))
            if status_name not in {"queued", "running", "complete", "blocked"}:
                status_name = (
                    "complete" if status_name == "candidates_catalogued_for_preview" else "blocked"
                )
            message = str(status.get("message", "Review candidate preparation status."))
            if str(workspace.root) in message:
                message = message.replace(str(workspace.root), "private data root")
            return ReviewPreparationJob(
                job_id=UUID(job_id),
                status=cast(Literal["queued", "running", "complete", "blocked"], status_name),
                message=message,
                progress=progress,
            )
        except HTTPException:
            raise
        except (ReviewWorkspaceError, IntakeError, OSError, ValueError) as error:
            raise HTTPException(
                status_code=409, detail="Preparation status could not be loaded."
            ) from error

    @app.post("/api/v1/setup/preview-batches", tags=["setup"])
    async def setup_grant_preview_batch(
        request: Request, payload: ReviewPreviewBatchRequest
    ) -> dict[str, object]:
        workspace = require_review_workspace(request, mutation=True)
        try:
            return workspace.grant_preview_batch(
                expected_revision=payload.expected_revision,
                reviewer_id=payload.reviewer_id,
                limit=payload.limit,
            )
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except (ReviewWorkspaceError, IntakeError, OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return app


app = create_app()
