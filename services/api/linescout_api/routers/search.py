"""``POST /api/v1/search`` — multipart contract validation and (fixture) retrieval."""

from __future__ import annotations

import time
from typing import Annotated, NoReturn
from uuid import UUID

from fastapi import APIRouter, File, Form, Request, UploadFile
from linescout_ml.taxonomy import PrimaryStyle

from linescout_api import fixture_ranker
from linescout_api.config import MAX_POINT_COUNT, MAX_REVISION, MAX_STROKE_COUNT
from linescout_api.deps import State
from linescout_api.errors import (
    bad_request,
    resolve_request_id,
    service_unavailable,
    too_large,
    unprocessable,
)
from linescout_api.preferences import compute_affinities, read_preferences
from linescout_api.preprocessing import (
    PREPROCESSING_VERSION,
    SnapshotError,
    decode_snapshot,
    decode_strokes,
    ink_stats,
    is_insufficient,
)
from linescout_api.schemas import (
    Degradation,
    ErrorResponse,
    ScopePrediction,
    SearchMode,
    SearchResponse,
    SearchTiming,
    StrokeSequence,
    StrokeStatus,
    StyleSelection,
)
from linescout_api.state import AppState

router = APIRouter(tags=["search"])


def _raise_snapshot_error(error: SnapshotError, field: str, limit: int | None = None) -> NoReturn:
    if error.code.endswith("too_large"):
        details = None
        if limit is not None and error.received_bytes is not None:
            details = {"max_bytes": limit, "received_bytes": error.received_bytes}
        raise too_large(error.code, error.message, field, details=details)
    if error.code in ("image_dimensions", "image_format"):
        raise unprocessable(error.code, error.message, field)
    raise bad_request(error.code, error.message, field)


def _vector_point_total(stroke_sequence: StrokeSequence | None) -> int | None:
    """Exact number of points carried by a delivered vector payload, if any."""
    if stroke_sequence is None:
        return None
    return sum(len(stroke.points) for stroke in stroke_sequence.strokes)


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Malformed image or strokes"},
        413: {"model": ErrorResponse, "description": "Body too large"},
        422: {"model": ErrorResponse, "description": "Invalid field"},
        503: {"model": ErrorResponse, "description": "Model or gallery is not ready"},
    },
)
async def search(
    request: Request,
    state: State,
    session_id: Annotated[UUID, Form()],
    revision: Annotated[int, Form(ge=1, le=MAX_REVISION)],
    canvas_width: Annotated[int, Form(gt=0)],
    canvas_height: Annotated[int, Form(gt=0)],
    stroke_count: Annotated[int, Form(ge=0, le=MAX_STROKE_COUNT)],
    point_count: Annotated[int, Form(ge=0, le=MAX_POINT_COUNT)],
    image: Annotated[UploadFile, File()],
    strokes: Annotated[UploadFile | None, File()] = None,
    text_hint: Annotated[str | None, Form()] = None,
    selected_style: Annotated[StyleSelection | None, Form()] = None,
) -> SearchResponse:
    settings = state.settings
    started = time.perf_counter()
    request_id = resolve_request_id(request)

    expected_canvas = settings.canvas_logical_size
    if canvas_width != expected_canvas:
        raise unprocessable(
            "canvas_dimensions",
            f"canvas_width must be {expected_canvas}",
            "canvas_width",
        )
    if canvas_height != expected_canvas:
        raise unprocessable(
            "canvas_dimensions",
            f"canvas_height must be {expected_canvas}",
            "canvas_height",
        )

    if text_hint is not None and len(text_hint) > settings.max_text_hint_chars:
        raise unprocessable(
            "text_hint_too_long",
            f"text_hint exceeds {settings.max_text_hint_chars} characters",
            "text_hint",
        )

    # Read with a hard cap so an oversized upload never fully buffers.
    image_bytes = await image.read(settings.max_image_bytes + 1)
    if len(image_bytes) > settings.max_image_bytes:
        raise too_large(
            "image_too_large",
            f"image exceeds {settings.max_image_bytes} bytes",
            "image",
            details={
                "max_bytes": settings.max_image_bytes,
                "received_bytes": len(image_bytes),
            },
        )
    strokes_bytes = (
        await strokes.read(settings.max_strokes_bytes + 1) if strokes is not None else None
    )
    if strokes_bytes is not None and len(strokes_bytes) > settings.max_strokes_bytes:
        raise too_large(
            "strokes_too_large",
            f"strokes exceed {settings.max_strokes_bytes} bytes",
            "strokes",
            details={
                "max_bytes": settings.max_strokes_bytes,
                "received_bytes": len(strokes_bytes),
            },
        )

    try:
        gray = decode_snapshot(image_bytes, settings.max_image_bytes)
    except SnapshotError as error:
        _raise_snapshot_error(error, "image", settings.max_image_bytes)
    expected_snapshot = settings.snapshot_size
    if gray.size != (expected_snapshot, expected_snapshot):
        raise unprocessable(
            "image_dimensions",
            f"snapshot must be {expected_snapshot}x{expected_snapshot}",
            "image",
        )
    try:
        stroke_sequence = decode_strokes(
            strokes_bytes,
            settings.max_strokes_bytes,
            max_expanded_bytes=settings.max_strokes_decompressed_bytes,
        )
    except SnapshotError as error:
        _raise_snapshot_error(error, "strokes", settings.max_strokes_bytes)

    # Canvas dimensions are validated exactly against the vector payload; a
    # logical 2048 request paired with a non-2048 vector canvas is rejected
    # outright (never rescaled or silently reinterpreted).
    if stroke_sequence is not None and (
        stroke_sequence.canvas_width != expected_canvas
        or stroke_sequence.canvas_height != expected_canvas
    ):
        raise unprocessable(
            "canvas_dimensions",
            f"strokes canvas must be {expected_canvas}x{expected_canvas}",
            "strokes",
        )

    # stroke_count is structural (one JSON element per stroke) and must match
    # the delivered payload exactly.
    if stroke_sequence is not None and len(stroke_sequence.strokes) != stroke_count:
        raise unprocessable(
            "stroke_count_mismatch",
            "stroke_count does not match the strokes payload",
            "stroke_count",
        )

    stroke_status = StrokeStatus.PRESENT if stroke_sequence is not None else StrokeStatus.ABSENT
    vector_point_total = _vector_point_total(stroke_sequence)
    # point_count is not structural: the vector payload is the ground truth,
    # and a discrepancy is flagged as approximate rather than silently trusted
    # or used to reject an otherwise well-formed drawing.
    counts_approximate = (
        vector_point_total != point_count
    )  # None != point_count => True when no vector payload

    stats = ink_stats(gray)
    preprocessing_ms = (time.perf_counter() - started) * 1000

    def timing(
        embedding: float = 0.0, retrieval: float = 0.0, reranking: float = 0.0
    ) -> SearchTiming:
        return SearchTiming(
            preprocessing_ms=round(preprocessing_ms, 3),
            embedding_ms=round(embedding, 3),
            retrieval_ms=round(retrieval, 3),
            reranking_ms=round(reranking, 3),
            total_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    dataset_version = state.gallery.dataset_version if state.gallery else None
    index_version = state.gallery.manifest_hash[:16] if state.gallery else None

    def base(mode: SearchMode, degradations: list[Degradation]) -> SearchResponse:
        warning = "; ".join(item.detail for item in degradations) or None
        return SearchResponse(
            request_id=request_id,
            api_version=state.api_version,
            revision=revision,
            canvas_width=expected_canvas,
            canvas_height=expected_canvas,
            stroke_status=stroke_status,
            counts_approximate=counts_approximate,
            mode=mode,
            scope_predictions=[],
            groups=[],
            timing=timing(),
            warning=warning,
            degradations=degradations,
            preprocessing_version=PREPROCESSING_VERSION,
            dataset_version=dataset_version,
            index_version=index_version,
        )

    if is_insufficient(
        stats, point_count, settings.min_points_for_search, settings.min_ink_diagonal_ratio
    ):
        response = base(SearchMode.INSUFFICIENT, _degradations(state, mode=SearchMode.INSUFFICIENT))
        _log_search(state, session_id, response, stroke_count, point_count)
        return response

    if not state.ready:
        raise service_unavailable(
            "not_ready",
            state.setup_error or "the API is not ready",
            details=state.readiness_details(),
        )

    # Row order: explicit style from the request overrides the stored preference for this response.
    stored_selected, learning_enabled = read_preferences(state.connection)
    explicit: PrimaryStyle | None
    if selected_style is None:
        explicit = stored_selected
    elif selected_style is StyleSelection.ALL:
        explicit = None
    else:
        explicit = PrimaryStyle(selected_style.value)
    affinities = compute_affinities(state.connection, settings.preference_half_life_days)
    row_order = fixture_ranker.order_style_rows(explicit, affinities, learning_enabled)

    retrieval_started = time.perf_counter()
    async with state.inference_lock:
        mode, predictions, groups = fixture_ranker.rank(
            state.assets,
            stats,
            stroke_count,
            seed=f"{session_id}:{stroke_count}:{stats.bbox}",
            row_order=row_order,
        )
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

    degradations = _degradations(state, mode=mode)
    response = base(mode, degradations)
    response.scope_predictions = _top_predictions(predictions)
    response.groups = groups
    response.timing = timing(retrieval=retrieval_ms)
    _log_search(state, session_id, response, stroke_count, point_count)
    return response


def _degradations(state: AppState, *, mode: SearchMode) -> list[Degradation]:
    """The structured degradation list for this response (see the contract doc).

    The list is empty when the response is at full quality: real models, GPU,
    every retrieval branch live, and a non-empty gallery.
    """
    items: list[Degradation] = []
    if not state.assets:
        items.append(
            Degradation(kind="gallery_empty", detail="gallery is empty; no references available")
        )
    if state.settings.fixture_mode:
        items.append(
            Degradation(
                kind="fixture_mode", detail="fixture results: retrieval models are not loaded"
            )
        )
    elif not state.device.is_cuda:
        items.append(Degradation(kind="cpu_fallback", detail="CPU fallback — slower search"))
    for branch in state.disabled_branches:
        items.append(
            Degradation(kind="branch_disabled", detail=f"retrieval branch disabled: {branch}")
        )
    return items


def _top_predictions(predictions: list[ScopePrediction], limit: int = 4) -> list[ScopePrediction]:
    return predictions[:limit]


def _log_search(
    state: AppState, session_id: UUID, response: SearchResponse, stroke_count: int, point_count: int
) -> None:
    t = response.timing
    state.connection.execute(
        "INSERT INTO search_log(session_id, revision, mode, stroke_count, point_count,"
        " preprocessing_ms, embedding_ms, retrieval_ms, reranking_ms, total_ms)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            str(session_id),
            response.revision,
            response.mode.value,
            stroke_count,
            point_count,
            t.preprocessing_ms,
            t.embedding_ms,
            t.retrieval_ms,
            t.reranking_ms,
            t.total_ms,
        ),
    )
