"""Pydantic models for every API request/response.

These are the source of the OpenAPI document and therefore of
``packages/contracts``. Field names follow the spec's API contracts section
verbatim so the generated TypeScript matches the document the team reviews.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from linescout_ml.taxonomy import LineArtOrigin, PermissionBasis, PrimaryStyle, ScopeLabel
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Re-exported so the OpenAPI schema names them once and the TS contracts pick
# them up as string-literal unions.
__all__ = [
    "PrimaryStyle",
    "ScopeLabel",
    "LineArtOrigin",
    "PermissionBasis",
    "GalleryKind",
    "SearchMode",
    "StrokeStatus",
    "InteractionEvent",
    "StyleSelection",
]


class SearchMode(StrEnum):
    INSUFFICIENT = "insufficient"
    PROVISIONAL = "provisional"
    CONFIDENT = "confident"


class InteractionEvent(StrEnum):
    OPEN = "open"
    PIN = "pin"
    UNPIN = "unpin"
    TRACE = "trace"


class StyleSelection(StrEnum):
    """``selected_style`` multipart value: a style enum or ``all``."""

    ALL = "all"
    MANGA_ANIME = "manga_anime"
    WESTERN_INK = "western_ink"
    REALISTIC_ACADEMIC = "realistic_academic"
    CARTOON = "cartoon"
    GESTURE_SKETCH = "gesture_sketch"


class StrokeStatus(StrEnum):
    """Whether a vector ``strokes`` payload accompanied the snapshot.

    ``absent`` means the query was raster-only (e.g. an imported image), so the
    reported ``stroke_count``/``point_count`` are client estimates over the
    snapshot with no vector ground truth to verify them against.
    """

    PRESENT = "present"
    ABSENT = "absent"


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


# ---------------------------------------------------------------- errors


class ErrorDetail(ApiModel):
    """One structured error.

    ``code`` is a stable, machine-readable slug (never a filename or path).
    ``field`` names the offending request field when one field is at fault.
    ``details`` is optional, code-specific structured data (e.g. byte limits
    for ``image_too_large``); it must never contain absolute filesystem paths,
    user content, or secrets.
    """

    code: str = Field(description="Stable machine-readable error code, e.g. image_too_large.")
    message: str
    field: str | None = None
    details: dict[str, Any] | None = None


class ErrorResponse(ApiModel):
    """Structured error envelope required on every failure path."""

    schema_version: Literal[1] = 1
    request_id: UUID
    retryable: bool
    error: ErrorDetail


# ---------------------------------------------------------------- health


class ModelVersion(ApiModel):
    name: str
    version: str
    loaded: bool
    device: str | None = None


class ReadyResponse(ApiModel):
    ready: Literal[True] = True


class HealthResponse(ApiModel):
    ready: bool
    fixture_mode: bool
    cuda_available: bool
    device: Literal["cuda", "cpu"]
    gpu_name: str | None
    vram_total_mb: int | None
    torch_version: str | None
    api_version: str
    schema_version: int
    #: Snapshot-preprocessing pipeline identity (see preprocessing.PREPROCESSING_VERSION).
    preprocessing_version: str
    models: list[ModelVersion]
    dataset_version: str | None
    index_version: str | None
    gallery_size: int
    disabled_branches: list[str] = Field(
        description="Optional retrieval branches disabled this session, e.g. pose."
    )
    warmup: Literal["pending", "complete", "skipped"]
    warnings: list[str]
    curation_enabled: bool


# ---------------------------------------------------------------- search


class ScopePrediction(ApiModel):
    label: ScopeLabel
    confidence: float = Field(ge=0.0, le=1.0)


class SearchResult(ApiModel):
    asset_id: str
    thumbnail_url: str
    style: PrimaryStyle
    #: Convenience view: primary first, then secondaries.
    scopes: list[ScopeLabel]
    primary_scope: ScopeLabel
    secondary_scopes: list[ScopeLabel] = Field(default_factory=list)
    origin: LineArtOrigin
    trace_allowed: bool = Field(
        description="Whether this asset may be placed on the trace layer. "
        "Stored per-asset permission (v2); never derived from origin on the wire."
    )
    relevance: float = Field(ge=0.0, le=1.0, description="Calibrated relevance probability.")
    quality: float = Field(ge=0.0, le=1.0)
    asset_url: str = Field(description="Trace-compatible full asset URL.")
    #: ``None`` = not applicable or not assessed; with ``person_count_approximate``
    #: the integer is an estimate (e.g. a crowd), not a headcount.
    person_count: int | None = Field(default=None, ge=0)
    person_count_approximate: bool = False


class SearchGroup(ApiModel):
    id: str
    title: str
    kind: Literal["best_match", "style", "provisional_scope"]
    style: PrimaryStyle | None = None
    scope: ScopeLabel | None = None
    results: list[SearchResult]


class SearchTiming(ApiModel):
    preprocessing_ms: float
    embedding_ms: float
    retrieval_ms: float
    reranking_ms: float
    total_ms: float


class Degradation(ApiModel):
    """One structured way this response is degraded relative to full quality.

    ``degradations`` on the response is the canonical, machine-readable view;
    the top-level ``warning`` string is a convenience join kept for older
    clients and may be removed in a future contract version.
    """

    kind: Literal["fixture_mode", "cpu_fallback", "branch_disabled", "gallery_empty"]
    detail: str


class SearchResponse(ApiModel):
    schema_version: Literal[2] = Field(
        default=2, description="Payload contract version; bump on any breaking response change."
    )
    #: Request identity echoed from ``X-Request-Id`` (matches the error envelope).
    request_id: UUID
    #: Server build/release identifier for this response.
    api_version: str
    #: Request revision echoed unchanged (dedupe / out-of-order guard for clients).
    revision: int = Field(ge=1, description="Echoes the request revision unchanged.")
    #: Logical canvas the query was validated against (always 2048 today).
    canvas_width: int
    canvas_height: int
    #: Whether a vector ``strokes`` payload accompanied this query.
    stroke_status: StrokeStatus
    #: Whether the reported stroke/point counts are exact for this query. True
    #: when no vector payload was present (raster-only) or the reported point
    #: count disagreed with the delivered vector point total.
    counts_approximate: bool
    mode: SearchMode
    scope_predictions: list[ScopePrediction]
    groups: list[SearchGroup]
    timing: SearchTiming
    warning: str | None = None
    degradations: list[Degradation] = Field(default_factory=list)
    #: Snapshot-preprocessing pipeline identity (see preprocessing.PREPROCESSING_VERSION).
    preprocessing_version: str
    #: Gallery the results came from; ``None`` when no gallery is loaded.
    dataset_version: str | None = None
    #: Manifest content hash (index version key); ``None`` when no gallery.
    index_version: str | None = None


class StrokePoint(ApiModel):
    """One sampled pointer position inside the gzipped ``strokes`` field."""

    x: float
    y: float
    p: float = Field(ge=0.0, le=1.0, description="Normalized pressure.")
    t: float = Field(description="Milliseconds since the stroke sequence started.")

    @field_validator("x", "y", "p", "t")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Coordinates must be finite numbers")
        return value


class Stroke(ApiModel):
    tool: Literal["pressure", "monoline", "eraser"]
    pointer: Literal["pen", "mouse", "touch"]
    points: list[StrokePoint]


class StrokeSequence(ApiModel):
    version: Literal[1] = 1
    canvas_width: int = Field(gt=0)
    canvas_height: int = Field(gt=0)
    strokes: list[Stroke]


# ---------------------------------------------------------------- events


class EventRequest(ApiModel):
    #: Client-generated identity for this interaction. Retrying the *same*
    #: request (same UUID, same authoritative payload) replays the original
    #: result instead of writing a second row; reusing the UUID with a
    #: different payload is a ``409 event_uuid_conflict``.
    event_uuid: UUID = Field(
        description="Client-generated UUID identifying this interaction attempt (idempotency key)."
    )
    session_id: UUID
    asset_id: str = Field(min_length=1, max_length=64)
    event: InteractionEvent
    style: PrimaryStyle | None = Field(
        default=None,
        deprecated=True,
        description=(
            "Deprecated and ignored. Style is derived server-side from the gallery row, so this "
            "field is not part of the request's authoritative payload and never affects "
            "idempotency. It is accepted only so older clients keep working."
        ),
    )
    query_revision: int = Field(ge=0)


class EventResponse(ApiModel):
    #: Row id of the stored event, or ``0`` when nothing was stored
    #: (learning disabled).
    id: int
    #: Identity of the stored event. Equals the requested ``event_uuid``
    #: except when the interaction coalesced onto an earlier equivalent event,
    #: in which case it is that event's UUID.
    event_uuid: UUID
    created_at: str
    #: Whether a training event exists for this interaction. ``False`` when
    #: learning is disabled — the call still succeeds, nothing is accumulated.
    recorded: bool = True
    #: ``True`` when this response replays an earlier write (retry with the
    #: same UUID, or a repeat that coalesced onto an existing contribution)
    #: rather than describing a row created by this request.
    replayed: bool = False


# ---------------------------------------------------------------- preferences


class StyleAffinity(ApiModel):
    style: PrimaryStyle
    affinity: float = Field(ge=0.0, le=1.0, description="Laplace-smoothed, decayed share.")


class PreferencesResponse(ApiModel):
    selected_style: PrimaryStyle | None
    learning_enabled: bool
    affinities: list[StyleAffinity]
    row_order: list[PrimaryStyle] = Field(
        description="Style-row order after Best Match: explicit style first, then learned "
        "affinity, then the fixed default order as tie-breaker."
    )


class PreferencesUpdate(ApiModel):
    """Partial update; omitted fields are left unchanged."""

    selected_style: PrimaryStyle | None = Field(
        default=None, description="Explicitly select a style. Ignored unless provided."
    )
    clear_selected_style: bool = Field(default=False, description="Clear the explicit style.")
    learning_enabled: bool | None = None
    reset_affinities: bool = Field(
        default=False,
        description=(
            "Forget learned affinities. Pins are durable application state and are never "
            "cleared by a reset."
        ),
    )


# ---------------------------------------------------------------- pins


class GalleryKind(StrEnum):
    """Namespace an interaction or pin belongs to.

    Stamped server-side from the API's own mode: fixture-gallery state can
    never mix with live-gallery state.
    """

    FIXTURE = "fixture"
    LIVE = "live"


class PinnedAsset(ApiModel):
    """A durably pinned reference, projected against the *current* gallery.

    The projection is recomputed on every read, so a permission or eligibility
    change is reflected immediately: ``trace_allowed`` is the asset's stored
    trace permission (never inferred from ``origin``), and an asset that lost
    display eligibility does not appear here at all.
    """

    asset_id: str
    pinned_at: str
    thumbnail_url: str
    asset_url: str
    style: PrimaryStyle
    primary_scope: ScopeLabel
    scopes: list[ScopeLabel]
    secondary_scopes: list[ScopeLabel] = Field(default_factory=list)
    origin: LineArtOrigin
    trace_allowed: bool = Field(
        description="Stored per-asset trace permission; never derived from origin."
    )
    quality: float = Field(ge=0.0, le=1.0)
    person_count: int | None = Field(default=None, ge=0)
    person_count_approximate: bool = False


class RevokedPin(ApiModel):
    """A pin dropped because its asset is no longer eligible to be shown."""

    asset_id: str
    reasons: list[str] = Field(
        description="Stable machine-readable serving blockers, e.g. display_not_permitted."
    )


class PinsResponse(ApiModel):
    schema_version: Literal[1] = 1
    gallery_kind: GalleryKind
    pins: list[PinnedAsset]
    revoked: list[RevokedPin] = Field(
        default_factory=list,
        description=(
            "Pins removed by this read because revalidation found the asset ineligible "
            "(permission revoked, review changed, derivatives stale, or asset gone)."
        ),
    )


# ---------------------------------------------------------------- assets


class AssetPermissions(ApiModel):
    """Permission metadata a client needs before *using* an asset.

    ``trace_url`` is populated only when tracing is permitted, so a client
    that restores a saved trace layer cannot resurrect an asset whose trace
    permission was revoked.
    """

    asset_id: str
    origin: LineArtOrigin
    permission_basis: PermissionBasis
    attribution: str | None = None
    attribution_required: bool = False
    allowed_display: bool
    allowed_trace: bool
    thumbnail_url: str
    asset_url: str
    trace_url: str | None = None
