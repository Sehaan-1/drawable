from __future__ import annotations

from typing import Literal
from uuid import UUID

from drawable_research_contracts import ViewMatchReviewV1, ViewpointReviewV1
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ApiErrorBody(StrictModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, str | int | bool | None] = Field(default_factory=dict)


class ApiError(StrictModel):
    schema_version: Literal[1] = 1
    error: ApiErrorBody
    request_id: UUID


class HealthResponse(StrictModel):
    schema_version: Literal[1] = 1
    request_id: UUID
    live: Literal[True] = True
    ready: bool
    mode: Literal["fixture", "cpu", "cuda"]
    gpu: Literal["not_probed", "disabled_by_owner", "unavailable", "available"] = "not_probed"
    models_loaded: bool = False
    gallery_size: int = Field(default=0, ge=0)
    enabled_branches: list[str] = Field(default_factory=list)
    setup_errors: list[str] = Field(default_factory=list)


class ReadinessResponse(StrictModel):
    schema_version: Literal[1] = 1
    request_id: UUID
    ready: bool
    profile: Literal["fixture", "real"]
    reason: str | None = None


class SearchTimings(StrictModel):
    queue_ms: float = Field(ge=0)
    preprocessing_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class SearchReferenceAsset(StrictModel):
    id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=1000)
    image_url: str = Field(
        pattern=r"^/api/v1/assets/[^/]+/preview\?build_id=prototype-[0-9a-f]{24}$"
    )
    style: Literal["Manga / anime", "Western ink", "Realistic", "Cartoon", "Gesture"]
    scope: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=2000)
    native: bool
    match: Literal["Tentative"]
    trace_allowed: bool
    visual_score: float = Field(ge=0, le=1, allow_inf_nan=False)


class SearchReferenceGroup(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    tentative: Literal[True] = True
    results: list[SearchReferenceAsset] = Field(min_length=1, max_length=4)


class SearchResponse(StrictModel):
    schema_version: Literal[1] = 1
    request_id: UUID
    session_id: UUID
    document_id: UUID
    canvas_revision: int = Field(ge=0)
    query_generation: int = Field(ge=0)
    build_id: str | None = None
    model_version: str | None = None
    preprocessing_version: str
    mode: Literal["insufficient", "provisional"]
    reason: Literal[
        "insufficient_visible_ink", "no_relevant_candidates", "experimental_visual_matches"
    ]
    interpretation: str
    warnings: list[str] = Field(default_factory=list)
    scope_scores: dict[str, float] = Field(default_factory=dict)
    groups: list[SearchReferenceGroup] = Field(default_factory=list, max_length=3)
    features: dict[str, str] = Field(default_factory=dict)
    timings_ms: SearchTimings
    training_started: Literal[False] = False
    data_uploaded: Literal[False] = False


Scope = Literal[
    "eye",
    "eyebrow",
    "mouth",
    "face_head",
    "hair",
    "hand",
    "foot",
    "upper_body_clothing",
    "full_body",
    "multi_character",
]
Style = Literal["manga_anime", "western_ink", "realistic_academic", "cartoon", "gesture_sketch"]


class CurationLabels(StrictModel):
    style: Style
    primary_scope: Scope
    secondary_scopes: list[Scope] = Field(default_factory=list, max_length=9)
    # Eight is the pose branch's query-time limit, not a curation cap.
    # Retain unknown and larger counts accurately for full compositions.
    person_count: int | None = Field(ge=1)
    sfw: Literal["safe", "unsafe", "uncertain"]
    quality: int = Field(ge=1, le=3)
    character_useful: bool
    text_coverage: float = Field(ge=0, le=1)
    clipped: bool
    damaged: bool
    duplicate_of: str | None = Field(default=None, max_length=200)
    review_seconds: float = Field(ge=0, le=86_400)
    keep_reject_reason: str = Field(min_length=1, max_length=2000)


class CurationUsePermissions(StrictModel):
    local_display: bool = False
    training: bool = False
    evaluation: bool = False
    derive: bool = False
    trace: bool = False
    export: bool = False
    public_demo: bool = False
    redistribute: bool = False
    cloud_storage: bool = False
    cloud_processing: bool = False


class SourceYieldJudgment(StrictModel):
    sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    useful_eye_crop: Literal["yes", "no", "uncertain"]
    review_seconds: float = Field(gt=0, le=86_400)


class CurationReviewRequest(StrictModel):
    expected_revision: int = Field(ge=0)
    reviewer_id: str = Field(pattern=r"^[A-Za-z0-9_-]{3,80}$")
    idempotency_key: UUID
    decision: Literal["keep", "reject", "quarantine"]
    rights_status: Literal["review_required", "approved", "rejected"] = "review_required"
    license_id: str | None = Field(default=None, max_length=200)
    attribution_text: str | None = Field(default=None, max_length=2000)
    terms_snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    use_permissions: CurationUsePermissions = Field(default_factory=CurationUsePermissions)
    labels: CurationLabels | None = None
    learning_split: Literal["unassigned", "pilot_development", "train", "validation"] = "unassigned"
    note: str = Field(default="", max_length=2000)
    viewpoint: ViewpointReviewV1 | None = None
    source_yield_judgment: SourceYieldJudgment | None = None

    @model_validator(mode="after")
    def require_decision_evidence(self) -> CurationReviewRequest:
        if self.decision == "keep" and self.labels is None:
            raise ValueError("Keeping an asset requires explicit, complete review labels")
        if self.labels is None and not self.note.strip():
            raise ValueError(
                "A reject or quarantine decision without labels requires a written reason"
            )
        return self


class CurationCandidate(StrictModel):
    asset_id: str
    source_id: str
    revision: int = Field(ge=0)
    state: str
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    preview_available: bool = False
    preview_blocker: str | None = None
    metadata: dict[str, object]


class CurationCandidateList(StrictModel):
    schema_version: Literal[1] = 1
    items: list[CurationCandidate]
    next_cursor: str | None = None


class SourceYieldQueueResponse(StrictModel):
    schema_version: Literal[1] = 1
    status: str
    sample_sha256: str | None = None
    selected_count: int = Field(ge=0, le=250)
    sources: list[dict[str, object]]
    items: list[CurationCandidate]
    blockers: list[str] = Field(default_factory=list)


class M0WorkspaceResponse(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["blocked", "complete"]
    gpu_status: Literal["not_probed"] = "not_probed"
    training_authorized: Literal[False] = False
    training_started: Literal[False] = False
    gates: dict[str, str]
    blockers: dict[str, str]
    source_archive_status: dict[str, str]
    source_terms_status: dict[str, str]
    model_terms_status: dict[str, str]
    effort_log_available: bool
    source_yield_sample_status: str
    source_yield_sample_count: int = Field(ge=0, le=250)


class ReviewWorkItem(StrictModel):
    id: str
    title: str
    artist: str
    license: str
    decision: str
    permissions: CurationUsePermissions
    attribution_text: str | None = None
    selected: bool = False


class ReviewSourceItem(StrictModel):
    id: str
    title: str
    authority_url: str
    terms_url: str
    decision: str
    terms_snapshot_sha256: str | None = None
    permissions: CurationUsePermissions
    archive: dict[str, object] | None = None
    candidate_count: int = Field(ge=0)


class ReviewModelItem(StrictModel):
    id: str
    title: str
    terms_url: str
    decision: str
    terms_snapshot_sha256: str | None = None
    weights_state: str
    training_authorized: Literal[False] = False


class ReviewWorkspaceStatus(StrictModel):
    schema_version: Literal[1] = 1
    revision: int = Field(ge=0)
    selected_sources: list[ReviewSourceItem]
    selected_works: list[ReviewWorkItem]
    available_works: list[ReviewWorkItem]
    selected_models: list[ReviewModelItem]
    reviewer_id: str
    preparation: dict[str, object]
    curation: CurationProgress
    required_handoff: str
    private_storage: Literal[True] = True
    training_authorized: Literal[False] = False


class ReviewTermsCaptureRequest(StrictModel):
    kind: Literal["source", "model"]
    id: str = Field(min_length=1, max_length=160)


class ReviewTermsCaptureResponse(StrictModel):
    schema_version: Literal[1] = 1
    status: str
    kind: str
    id: str | None = None
    model_id: str | None = None
    purpose: str | None = None
    source_url: str | None = None
    final_url: str | None = None
    snapshot_path: str | None = None
    terms_snapshot_sha256: str
    source_content_sha256: str | None = None
    retrieved_at: str | None = None
    captured_at: str | None = None
    byte_length: int | None = Field(default=None, ge=1)
    owner_decision_required: Literal[True] = True
    network_used: bool = True
    model_loaded: Literal[False] = False
    training_started: Literal[False] = False


class ReviewDecisionRequest(StrictModel):
    expected_revision: int = Field(ge=0)
    kind: Literal["source", "work", "model"]
    id: str = Field(min_length=1, max_length=160)
    reviewer_id: str = Field(pattern=r"^[A-Za-z0-9_-]{3,80}$")
    decision: Literal["approved", "rejected"]
    terms_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    note: str = Field(min_length=12, max_length=1000)
    permissions: CurationUsePermissions | None = None
    attribution_text: str | None = Field(default=None, max_length=1000)


class ReviewWorkSelectionRequest(StrictModel):
    expected_revision: int = Field(ge=0)
    work_ids: list[str] = Field(min_length=1, max_length=14)


class ReviewCandidatePreparationRequest(StrictModel):
    expected_revision: int = Field(ge=0)
    reviewer_id: str = Field(pattern=r"^[A-Za-z0-9_-]{3,80}$")


class ReviewPreviewBatchRequest(StrictModel):
    expected_revision: int = Field(ge=0)
    reviewer_id: str = Field(pattern=r"^[A-Za-z0-9_-]{3,80}$")
    limit: int = Field(default=24, ge=1, le=32)
    confirm_local_preview: Literal[True]


class ReviewPreparationJob(StrictModel):
    schema_version: Literal[1] = 1
    job_id: UUID
    status: Literal["queued", "running", "complete", "blocked"]
    message: str
    progress: dict[str, int | str] = Field(default_factory=dict)


class CurationProgress(StrictModel):
    schema_version: Literal[1] = 1
    total: int = Field(ge=0)
    pending: int = Field(ge=0)
    candidate: int = Field(ge=0)
    rejected: int = Field(ge=0)
    reviewed: int = Field(ge=0)


class CurationCropRequest(StrictModel):
    expected_revision: int = Field(ge=0)
    reviewer_id: str = Field(pattern=r"^[A-Za-z0-9_-]{3,80}$")
    idempotency_key: UUID
    x: float = Field(ge=0, le=1, allow_inf_nan=False)
    y: float = Field(ge=0, le=1, allow_inf_nan=False)
    width: float = Field(gt=0, le=1, allow_inf_nan=False)
    height: float = Field(gt=0, le=1, allow_inf_nan=False)


class CurationCropResponse(StrictModel):
    schema_version: Literal[1] = 1
    asset_id: str
    parent_asset_id: str
    revision: int = Field(ge=0)
    width: int = Field(ge=96)
    height: int = Field(ge=96)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    idempotent_replay: bool = False


class ViewpointRubricResponse(StrictModel):
    schema_version: Literal[1] = 1
    rubric_id: Literal["drawable-viewpoint-v1"]
    rubric_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rubric: dict[str, object]


class CurationViewMatchRequest(StrictModel):
    idempotency_key: UUID
    review: ViewMatchReviewV1


class CurationViewMatchResponse(StrictModel):
    schema_version: Literal[1] = 1
    review_id: UUID
    idempotent_replay: bool = False


class CurationSnapshotRequest(StrictModel):
    expected_rubric_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requester_id: str = Field(pattern=r"^[A-Za-z0-9_-]{3,80}$")
    idempotency_key: UUID
    asset_ids: list[str] = Field(min_length=1, max_length=32)


class CurationSnapshotResponse(StrictModel):
    schema_version: Literal[1] = 1
    snapshot_id: str
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rubric_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_count: int = Field(ge=1)
    status: Literal["created_review_only"] = "created_review_only"
    gallery_activated: Literal[False] = False
    training_started: Literal[False] = False
    idempotent_replay: bool = False


class CurationReviewResponse(StrictModel):
    schema_version: Literal[1] = 1
    review_id: UUID
    asset_id: str
    revision: int = Field(ge=0)
    state: str
    gallery_member: bool
    gallery_eligible_candidate: bool = False
    idempotent_replay: bool = False


class CurationHistoryItem(StrictModel):
    review_id: UUID
    expected_revision: int
    reviewer_id: str
    decision: Literal["keep", "reject", "quarantine"]
    review: dict[str, object]
    created_at: str
