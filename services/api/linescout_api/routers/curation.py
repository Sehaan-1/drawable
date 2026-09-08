"""Local-only curation endpoints. Mounted only when ``LINESCOUT_CURATION_MODE=1``.

The curation loop, end to end and safe under concurrency:

* :func:`next_candidate` returns the next asset the reviewer should look at,
  balancing the five style families and the gallery scope buckets. With a
  ``session_id`` the queue becomes **cursor-based**: each serve advances the
  session cursor in the stable ``asset_id`` order, explicitly skipped assets
  are excluded for the session, and the queue wraps around the non-excluded
  pool — Next/Skip therefore always advance without requiring a label and
  never repeat an item back-to-back.
* :func:`get_candidate` retrieves any candidate **by id** (any review
  state), so Previous re-fetches the *actual* previous candidate — live
  metadata, never a stale copy — and conflict reconciliation has one stable
  read path.
* :func:`write_label` validates a human decision, persists it as an immutable
  audit row in ``curation_labels``, and mirrors the decision into the
  ``assets`` cache. Every write is guarded by the caller's
  ``expected_label_version`` (the per-asset curation version): the write is
  a single **atomic conditional UPDATE inside the audit transaction**, and a
  stale version is a structured 409 carrying the live version and
  reconciliation information — a second curator can never silently overwrite
  the first (lost update).
* :func:`create_crop` turns a reviewer's crop into an **immutable child
  derivative** with parent identity, bounded geometry, fresh files/hashes,
  and its *own* processing and review state; :func:`process_derivative` runs
  the required processing (measurements + thumbnail rebuild from the child's
  own bytes). Nothing can enable the child before processing completes and
  a human approval is recorded.
* SFW adjudication (:func:`list_quarantine`, :func:`reveal_quarantined`,
  :func:`adjudicate_sfw`) is curation-only: quarantined/uncertain records
  are listed without images, previewing them requires a deliberate,
  expiring **reveal grant**, and the public ``/assets`` routes keep failing
  closed on them regardless — there is no public URL bypass.
* :func:`export_snapshot` writes an **immutable full snapshot** from a
  consistent database view (a single ``BEGIN IMMEDIATE`` transaction):
  the latest keep-or-reject label per asset (rejected included), frozen
  metadata, exclusive file publication (``O_CREAT | O_EXCL`` — never an
  existence check followed by an overwrite-capable write), exact
  label↔snapshot linking, and a recorded content hash so the export stays
  verifiable after later edits.
* :func:`progress` aggregates reviewed / accepted / rejected counts overall,
  broken down by style and scope, plus the quarantined backlog.

Curation only touches assets that already live in the ``assets`` cache; the
caller (the dataset pipeline) is expected to have populated the gallery via
:func:`linescout_api.gallery.sync_gallery` first.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
from linescout_ml.manifest import (
    AllowedUses,
    Permissions,
    SfwHumanDecision,
    SfwScreening,
)
from linescout_ml.manifest import (
    CropBox as ManifestCropBox,
)
from linescout_ml.taxonomy import (
    GALLERY_SCOPES,
    CurationBlocker,
    LearningSplit,
    PrimaryStyle,
    ReviewState,
    ScopeLabel,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from linescout_api.deps import State
from linescout_api.derivatives import (
    AWAITING_PROCESSING,
    MIN_CROP_EDGE,
    DerivativeError,
    child_asset_id,
    child_asset_values,
    crop_files_for,
    crop_fits,
    insert_child_asset,
    insert_registry_row,
    process_derivative_files,
    registry_values,
    sync_asset_scopes,
)
from linescout_api.errors import ApiError, conflict, not_found, unprocessable
from linescout_api.gallery import enabled_assets, recompute_enabled, serving_blockers

log = logging.getLogger(__name__)

router = APIRouter(prefix="/curation", tags=["curation"])

_PREVIEW_KINDS: dict[str, str] = {"thumbnail": "thumbnail_path", "line-art": "line_art_path"}

#: SQL fragment matching assets currently **held**: quarantined by review
#: state, screened unsafe/unsure, or flagged by a human. These require an
#: explicit reveal grant to preview through curation routes and can never be
#: served publicly.
_HELD_SQL = (
    "(review_state = 'quarantined' OR sfw_verdict IN ('unsafe', 'unsure') OR sfw_human_safe = 0)"
)


# ----------------------------------------------------------------- taxonomy constants

#: The five style families that drive stratification. ``curation_labels`` may
#: record any value the reviewer picks, but the queue strictly samples from
#: the style×scope grid below.
STYLE_BUCKETS: tuple[PrimaryStyle, ...] = (
    PrimaryStyle.MANGA_ANIME,
    PrimaryStyle.WESTERN_INK,
    PrimaryStyle.REALISTIC_ACADEMIC,
    PrimaryStyle.CARTOON,
    PrimaryStyle.GESTURE_SKETCH,
)

#: Gallery scope buckets. ``unknown`` is excluded — it is a query-only label.
SCOPE_BUCKETS: tuple[ScopeLabel, ...] = tuple(
    scope for scope in ScopeLabel if scope in GALLERY_SCOPES
)

#: Default review target. Matches the Milestone 1 progress payload so old
#: clients still get a sensible "remaining" value when no work has been done.
DEFAULT_TARGET: int = 2000


# ----------------------------------------------------------------- shared helpers


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_ts(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp written by SQLite or Python (``…Z`` form)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@contextmanager
def immediate_transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """A ``BEGIN IMMEDIATE`` transaction that fails closed and clean.

    ``IMMEDIATE`` takes the write lock up front, so the reads inside see a
    consistent view that no other writer can interleave with, and a
    conditional UPDATE cannot race another mutation between read and write.
    If another write is already in flight on the shared connection (or the
    database is locked past the busy timeout), the caller receives a
    *retryable* 503 instead of an interleaved transaction — concurrent
    exports/labels serialize rather than overwrite each other.
    """
    try:
        connection.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as error:
        raise ApiError(
            503,
            "write_in_progress",
            f"another curation write is in flight; retry shortly ({error})",
            retryable=True,
            headers={"Retry-After": "1"},
        ) from error
    try:
        yield connection
    except BaseException:
        with contextlib.suppress(sqlite3.OperationalError):  # pragma: no cover
            connection.execute("ROLLBACK")
        raise
    try:
        connection.execute("COMMIT")
    except sqlite3.OperationalError as error:
        with contextlib.suppress(sqlite3.OperationalError):  # pragma: no cover
            connection.execute("ROLLBACK")
        raise ApiError(
            503,
            "commit_failed",
            f"the write could not be committed; retry ({error})",
            retryable=True,
            headers={"Retry-After": "1"},
        ) from error


def _derivative_api_error(error: DerivativeError) -> ApiError:
    """Map a lifecycle failure to the structured error envelope."""
    if error.retryable:
        return ApiError(
            503,
            error.code,
            error.message,
            retryable=True,
            headers={"Retry-After": "2"},
            details=error.details or None,
        )
    return ApiError(422, error.code, error.message, retryable=False, details=error.details or None)


def _row_held(row: sqlite3.Row) -> bool:
    """Whether an asset is currently held (quarantined / uncertain / flagged)."""
    if row["review_state"] == "quarantined":
        return True
    if row["sfw_verdict"] in ("unsafe", "unsure"):
        return True
    return bool(row["sfw_human_safe"] == 0)


def _reconciliation_details(
    connection: sqlite3.Connection, current: sqlite3.Row, expected_version: int
) -> dict[str, object]:
    """Everything a stale client needs to reconcile after a 409.

    Never includes filesystem paths — identifiers, states, and the latest
    audit decisions only.
    """
    asset_id = str(current["asset_id"])
    latest = connection.execute(
        "SELECT id, decision, reviewer, created_at FROM curation_labels"
        " WHERE asset_id = ? ORDER BY id DESC LIMIT 1",
        (asset_id,),
    ).fetchone()
    adjudication = connection.execute(
        "SELECT safe, reviewer, created_at FROM sfw_adjudications"
        " WHERE asset_id = ? ORDER BY id DESC LIMIT 1",
        (asset_id,),
    ).fetchone()
    current_version = int(current["curation_label_version"])
    latest_reviewer = None
    if latest is not None:
        latest_reviewer = latest["reviewer"]
    elif adjudication is not None:
        latest_reviewer = adjudication["reviewer"]
    return {
        "asset_id": asset_id,
        "expected_label_version": expected_version,
        "current_label_version": current_version,
        "current_review_state": current["review_state"],
        "latest_decision": latest["decision"] if latest else None,
        "latest_label_id": int(latest["id"]) if latest else None,
        "latest_reviewer": latest_reviewer,
        "latest_label_created_at": latest["created_at"] if latest else None,
        "latest_sfw_adjudication": (
            {"safe": bool(adjudication["safe"]), "created_at": adjudication["created_at"]}
            if adjudication
            else None
        ),
        "reconcile": (
            f"GET /api/v1/curation/candidates/{asset_id} returns the live candidate; "
            f"re-apply the decision with expected_label_version={current_version}"
        ),
    }


def _raise_version_conflict(
    connection: sqlite3.Connection,
    asset_id: str,
    expected_version: int,
    expected_state: ReviewState,
) -> None:
    """409 with the live version + reconciliation info (never a lost update)."""
    current = connection.execute(
        "SELECT asset_id, review_state, curation_label_version FROM assets WHERE asset_id = ?",
        (asset_id,),
    ).fetchone()
    if current is None:
        raise not_found("asset_not_found", f"asset {asset_id} is not in the gallery")
    details = _reconciliation_details(connection, current, expected_version)
    if current["review_state"] != expected_state.value:
        raise ApiError(
            409,
            "review_conflict",
            "asset review_state changed since the candidate was loaded",
            field="expected_review_state",
            retryable=False,
            details=details,
        )
    raise ApiError(
        409,
        "label_version_conflict",
        "the asset was changed by another curation write; "
        "reconcile against the current label version",
        field="expected_label_version",
        retryable=False,
        details=details,
    )


def _refresh_serving_cache(state: State) -> None:
    """Reload the in-memory enabled-asset list (index membership/cache state).

    Called after every mutation that can change eligibility, so /search and
    /assets/* observe the new serving flag without a process restart.
    """
    state.assets = enabled_assets(state.connection)


def _advance_session(connection: sqlite3.Connection, session_id: str, asset_id: str) -> None:
    """Move a review session's cursor to ``asset_id`` (its latest served point)."""
    with immediate_transaction(connection) as tx:
        tx.execute(
            "INSERT INTO curation_sessions (session_id, cursor_asset_id, updated_at)"
            " VALUES (?, ?, ?)"
            " ON CONFLICT(session_id) DO UPDATE SET"
            " cursor_asset_id = excluded.cursor_asset_id, updated_at = excluded.updated_at",
            (session_id, asset_id, _iso(_utcnow())),
        )


# ----------------------------------------------------------------- response models


class StyleBreakdown(BaseModel):
    """Reviewed/accepted/rejected counts for a single style family."""

    model_config = ConfigDict(extra="forbid")

    reviewed: int = Field(ge=0)
    accepted: int = Field(ge=0)
    rejected: int = Field(ge=0)
    remaining: int = Field(ge=0)


class ScopeBreakdown(BaseModel):
    """Reviewed/accepted/rejected counts for a single scope bucket."""

    model_config = ConfigDict(extra="forbid")

    reviewed: int = Field(ge=0)
    accepted: int = Field(ge=0)
    rejected: int = Field(ge=0)
    remaining: int = Field(ge=0)


class CurationProgress(BaseModel):
    """Overall review progress plus style/scope breakdowns."""

    model_config = ConfigDict(extra="forbid")

    reviewed: int = Field(ge=0)
    accepted: int = Field(ge=0)
    rejected: int = Field(ge=0)
    #: Assets currently held (quarantined / SFW-uncertain / human-flagged) —
    #: the adjudication backlog.
    quarantined: int = Field(ge=0)
    remaining: int = Field(ge=0)
    target: int = Field(default=DEFAULT_TARGET, ge=0)
    by_style: dict[PrimaryStyle, StyleBreakdown]
    by_scope: dict[ScopeLabel, ScopeBreakdown]


class CropBox(BaseModel):
    """Crop coordinates in source-image pixel space (inclusive-exclusive)."""

    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class CurationCandidate(BaseModel):
    """One candidate asset for review (the v2 wire shape)."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    primary_style: PrimaryStyle
    primary_scope: ScopeLabel
    secondary_scopes: list[ScopeLabel] = Field(default_factory=list)
    person_count: int | None = Field(default=None, ge=0)
    person_count_approximate: bool = False
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    thumbnail_url: str
    line_art_url: str
    origin: Literal["native_line_art", "extracted_line_art"]
    crop: CropBox | None = None
    review_state: Literal["unreviewed", "accepted", "rejected", "quarantined"]
    blockers: list[CurationBlocker] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0)
    #: Automated screen (tri-state) — ``None`` when no screen ever ran.
    sfw_screening: SfwScreening | None = None
    #: Human SFW decision — ``None`` when no human has decided yet.
    sfw_human: SfwHumanDecision | None = None
    #: Permission provenance and the three independent use grants.
    permissions: Permissions
    allowed_uses: AllowedUses
    learning_split: LearningSplit
    gallery_member: bool = True
    gold_member: bool = False
    source_work_id: str
    parent_asset_id: str | None = None
    artist_id: str | None = None
    leakage_group_id: str | None = None
    #: Per-asset curation version. Echo it back as ``expected_label_version``
    #: on the next write; a mismatch is a structured 409, never a lost update.
    label_version: int = Field(ge=0)
    #: Processing lifecycle for curator-created derivatives; ``None`` for
    #: manifest assets (which are validated by the gallery loader instead).
    derivative_processing_state: Literal["pending", "complete", "failed"] | None = None


class LabelRequest(BaseModel):
    """Body of ``POST /curation/labels``."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1, max_length=64)
    #: Review state the client last observed. Compared against the live row
    #: inside the write transaction so a second curator cannot silently
    #: overwrite the first (lost-update).
    expected_review_state: ReviewState
    #: Per-asset curation version the client last observed. **Required**;
    #: the write is a single atomic conditional UPDATE on this version.
    expected_label_version: int = Field(ge=0)
    decision: Literal["keep", "reject"]
    primary_style: PrimaryStyle | None = None
    #: The single best scope; required on ``keep`` when the stored primary is
    #: still ``unknown`` (an unknown primary cannot be accepted).
    primary_scope: ScopeLabel | None = None
    secondary_scopes: list[ScopeLabel] | None = None
    #: Named, use-blocking defects. Blockers force ``reject`` (or quarantine);
    #: a ``keep`` carrying blockers is a 422.
    blockers: list[CurationBlocker] = Field(default_factory=list)
    #: Optional human SFW assertion recorded with this label. ``None`` leaves
    #: any existing human decision untouched.
    sfw_safe: bool | None = None
    quality: int | None = Field(default=None, ge=1, le=3)
    note: str | None = Field(default=None, max_length=2000)
    reviewer: str | None = Field(default=None, max_length=64)
    #: Review session whose queue cursor should advance past this asset.
    session_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("secondary_scopes")
    @classmethod
    def _no_duplicate_scopes(cls, value: list[ScopeLabel] | None) -> list[ScopeLabel] | None:
        if value is None:
            return None
        if len(value) != len(set(value)):
            msg = "secondary_scopes contains duplicates"
            raise ValueError(msg)
        bad = [scope for scope in value if scope not in GALLERY_SCOPES]
        if bad:
            msg = f"secondary_scopes cannot carry query-only or unknown scopes: {bad}"
            raise ValueError(msg)
        return value

    @field_validator("blockers")
    @classmethod
    def _unique_blockers(cls, value: list[CurationBlocker]) -> list[CurationBlocker]:
        if len(set(value)) != len(value):
            msg = "blockers contains duplicates"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _quality_required_on_keep(self) -> LabelRequest:
        if self.decision == "keep" and self.quality is None:
            msg = "quality is required when decision='keep'"
            raise ValueError(msg)
        if self.decision == "keep" and self.blockers:
            msg = "blockers cannot be accepted; reject or quarantine instead"
            raise ValueError(msg)
        if self.decision == "keep" and self.primary_scope is ScopeLabel.UNKNOWN:
            msg = "primary_scope 'unknown' cannot be accepted; pick a scope"
            raise ValueError(msg)
        return self


class LabelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    asset_id: str
    decision: Literal["keep", "reject"]
    review_state: Literal["unreviewed", "accepted", "rejected", "quarantined"]
    review_quality: int | None
    blockers: list[CurationBlocker] = Field(default_factory=list)
    sfw_human_approved: bool | None
    enabled: bool
    #: Why the asset is not servable, as stable machine-readable reasons.
    #: Empty iff ``enabled`` is true.
    serving_blockers: list[str] = Field(default_factory=list)
    #: The new per-asset curation version after this write.
    label_version: int
    created_at: str


class QuarantineCandidate(BaseModel):
    """One held record in the SFW adjudication backlog (metadata only).

    Image URLs stay ``None`` until a deliberate reveal grant exists; the
    preview route enforces the grant independently, so the URLs are never a
    bypass.
    """

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    primary_style: PrimaryStyle
    primary_scope: ScopeLabel
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    review_state: Literal["unreviewed", "accepted", "rejected", "quarantined"]
    blockers: list[CurationBlocker] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0)
    sfw_screening: SfwScreening | None = None
    sfw_human: SfwHumanDecision | None = None
    source_work_id: str
    parent_asset_id: str | None = None
    label_version: int = Field(ge=0)
    derivative_processing_state: Literal["pending", "complete", "failed"] | None = None
    revealed: bool = False
    reveal_expires_at: str | None = None
    thumbnail_url: str | None = None
    line_art_url: str | None = None


class RevealRequest(BaseModel):
    """Body of ``POST /curation/quarantine/{asset_id}/reveal``."""

    model_config = ConfigDict(extra="forbid")

    reviewer: str | None = Field(default=None, max_length=64)


class SfwAdjudicationRequest(BaseModel):
    """Body of ``POST /curation/sfw/{asset_id}/adjudication``."""

    model_config = ConfigDict(extra="forbid")

    safe: bool
    expected_label_version: int = Field(ge=0)
    reviewer: str | None = Field(default=None, max_length=64)
    note: str | None = Field(default=None, max_length=2000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class SfwAdjudicationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    asset_id: str
    safe: bool
    review_state: Literal["unreviewed", "accepted", "rejected", "quarantined"]
    sfw_human: SfwHumanDecision
    label_version: int
    enabled: bool
    serving_blockers: list[str] = Field(default_factory=list)
    created_at: str


class CropRequest(BaseModel):
    """Body of ``POST /curation/assets/{asset_id}/crops``.

    The crop is cut from the parent's *current, verified* line art; the child
    is created pending its own processing and review.
    """

    model_config = ConfigDict(extra="forbid")

    crop: CropBox
    expected_label_version: int = Field(ge=0)
    reviewer: str | None = Field(default=None, max_length=64)
    note: str | None = Field(default=None, max_length=2000)


class CropResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    parent_asset_id: str
    crop: CropBox
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    review_state: Literal["unreviewed", "accepted", "rejected", "quarantined"]
    processing_state: Literal["pending", "complete", "failed"]
    derivatives_current: bool
    label_version: int = Field(ge=0)
    created: bool


class DerivativeMeasurements(BaseModel):
    """Measurements rebuilt from the derivative's own bytes."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(ge=1)
    height: int = Field(ge=1)
    ink_coverage: float = Field(ge=0.0, le=1.0)
    text_coverage: float = Field(ge=0.0, le=1.0)
    background_coverage: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    phash: str


class DerivativeArtifactStamp(BaseModel):
    """Binds embeddings/index entries to the exact derivative bytes.

    Matches the frozen vector contract (``ml/linescout_ml/embeddings.py``):
    an embedding entry is only ``available`` when its recorded stamp equals
    the asset's current stamp. A freshly processed derivative has no vectors
    yet, so its embedding status is ``missing`` until the index build.
    """

    model_config = ConfigDict(extra="forbid")

    processing_revision: int = Field(ge=1)
    line_art_checksum: str


class DerivativeProcessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    parent_asset_id: str
    processing_state: Literal["pending", "complete", "failed"]
    attempts: int = Field(ge=0)
    measurements: DerivativeMeasurements | None = None
    artifact: DerivativeArtifactStamp
    embedding_status: Literal["missing"] = "missing"
    derivatives_current: bool
    label_version: int = Field(ge=0)
    enabled: bool
    serving_blockers: list[str] = Field(default_factory=list)


class SkipRequest(BaseModel):
    """Body of ``POST /curation/queue/skip``."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    asset_id: str = Field(min_length=1, max_length=64)


class SkipResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    cursor_asset_id: str
    excluded_asset_id: str
    #: Eligible candidates left for this session after the skip.
    remaining: int = Field(ge=0)


class SnapshotResponse(BaseModel):
    """Body of ``POST /curation/snapshots``."""

    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    #: Path relative to the API data dir, e.g. ``snapshots/curation_….json``.
    #: Never an absolute host filesystem path.
    path: str
    #: Distinct labeled assets captured (latest label per asset, keeps and rejects).
    label_count: int
    #: Lineage: the previous full snapshot this one supersedes, or ``None``.
    previous_snapshot_id: str | None
    style_breakdown: dict[PrimaryStyle, int]
    #: SHA-256 of the exact published bytes; the file can be re-verified
    #: against it at any time (reproducibility guarantee).
    content_sha256: str
    created_at: str


# ----------------------------------------------------------------- queue logic


def _reviewed_count(connection: sqlite3.Connection) -> int:
    """Number of *distinct* assets with at least one label, regardless of decision."""
    row = connection.execute("SELECT COUNT(DISTINCT asset_id) AS n FROM curation_labels").fetchone()
    return int(row["n"])


_CANDIDATE_SELECT = (
    "SELECT asset_id, primary_style, primary_scope, secondary_scopes_json,"
    " person_count, person_count_approximate, width, height, origin,"
    " crop_json, review_state, blockers_json, quality_score,"
    " sfw_verdict, sfw_confidence, sfw_method,"
    " sfw_human_safe, sfw_human_reviewer, sfw_human_decided_at,"
    " license_id, permission_basis, permission_url, attribution, attribution_required,"
    " allowed_display, allowed_training, allowed_trace,"
    " learning_split, gallery_member, gold_member, source_work_id,"
    " parent_asset_id, artist_id, leakage_group_id, curation_label_version,"
    " (SELECT processing_state FROM curation_derivatives d"
    "  WHERE d.asset_id = assets.asset_id) AS derivative_processing_state"
)


def _candidate_pool_query(
    style: PrimaryStyle | None,
    scope: ScopeLabel | None,
    *,
    session_id: str | None = None,
    after: str | None = None,
) -> tuple[str, list[object]]:
    """Build the SELECT for the candidate pool.

    Stratification happens in :func:`_pick_candidate`; here we just gather
    the row set so the picker can balance across the style×scope grid. The
    queue holds *unreviewed* assets with current derivatives: anything not
    safe per the automated screen was quarantined at ingestion, so no extra
    SFW filter is needed here. Human SFW approval happens through this
    queue, not before it.

    Session semantics: skipped assets are excluded for the session, and with
    a cursor the pool is restricted to ids strictly after the cursor (the
    caller wraps around when that slice is empty).
    """
    sql = [
        _CANDIDATE_SELECT,
        " FROM assets",
        " WHERE review_state = 'unreviewed'",
        # Stale derivatives are not reviewable: a reviewer would accept or
        # reject artifacts that no longer match the current generation.
        " AND derivatives_current = 1",
    ]
    params: list[object] = []
    if style is not None:
        sql.append(" AND primary_style = ?")
        params.append(style.value)
    if scope is not None:
        # Either the asset declares the scope directly, or its primary is
        # still the ``unknown`` placeholder — both are valid review targets.
        sql.append(
            " AND (primary_scope = ? OR secondary_scopes_json LIKE ? OR primary_scope = 'unknown')"
        )
        params.append(scope.value)
        params.append(f'%"{scope.value}"%')
    if session_id is not None:
        sql.append(
            " AND asset_id NOT IN (SELECT asset_id FROM curation_exclusions WHERE session_id = ?)"
        )
        params.append(session_id)
    if after is not None:
        sql.append(" AND asset_id > ?")
        params.append(after)
    sql.append(" ORDER BY asset_id")
    return "".join(sql), params


def _pick_candidate(
    rows: list[sqlite3.Row], style: PrimaryStyle | None, scope: ScopeLabel | None
) -> sqlite3.Row | None:
    """Pick the next asset to surface, balancing style×scope coverage.

    Strategy:
    * If the caller filtered on ``?style`` or ``?scope``, honour it; no need
      to balance — the caller already chose the slice.
    * Otherwise compute the 5×10 coverage matrix from how many candidates each
      cell has in the pool, and pick the cell with the highest *uncovered*
      weight. Within the chosen cell, fall back to the lexicographically
      first ``asset_id`` so the queue is deterministic across restarts.
    """
    if not rows:
        return None
    if style is not None or scope is not None:
        return rows[0]

    # Per-cell coverage count.
    cell_counts: dict[tuple[PrimaryStyle, ScopeLabel], int] = {}
    style_counts: dict[PrimaryStyle, int] = {s: 0 for s in STYLE_BUCKETS}
    scope_counts: dict[ScopeLabel, int] = {s: 0 for s in SCOPE_BUCKETS}
    for row in rows:
        primary = PrimaryStyle(row["primary_style"])
        style_counts[primary] += 1
        for scope_label in _row_scopes(row):
            if scope_label is ScopeLabel.UNKNOWN:
                continue
            scope_counts[scope_label] = scope_counts[scope_label] + 1
            key = (primary, scope_label)
            cell_counts[key] = cell_counts.get(key, 0) + 1

    # Pick the (style, scope) cell with the highest "deficit" relative to the
    # median count of its style row and scope column. Ties go to the lowest
    # asset_id we've already seen, which keeps the order stable.
    chosen: tuple[PrimaryStyle, ScopeLabel] | None = None
    chosen_deficit: float = -1.0
    for primary in STYLE_BUCKETS:
        for scope_label in SCOPE_BUCKETS:
            count = cell_counts.get((primary, scope_label), 0)
            # Deficit: how far this cell is below its row and column median.
            # Higher deficit => the cell is underrepresented => prefer it.
            row_total = style_counts[primary] or 1
            col_total = scope_counts[scope_label] or 1
            expected = (row_total + col_total) / 2.0
            deficit = expected - count
            if deficit > chosen_deficit:
                chosen_deficit = deficit
                chosen = (primary, scope_label)

    if chosen is None:
        return rows[0]
    target_style, target_scope = chosen
    for row in rows:
        if PrimaryStyle(row["primary_style"]) is not target_style:
            continue
        if target_scope in _row_scopes(row):
            return row
    # Cell is empty (deficit was driven by 0 in another column); fall back.
    return rows[0]


def _row_scopes(row: sqlite3.Row) -> list[ScopeLabel]:
    """Primary plus secondary scopes of a queue row; malformed JSON reads empty."""
    try:
        primary = ScopeLabel(row["primary_scope"])
    except ValueError:
        return []
    try:
        secondaries = [ScopeLabel(value) for value in json.loads(row["secondary_scopes_json"])]
    except (ValueError, json.JSONDecodeError):
        secondaries = []
    return [primary, *secondaries]


def _row_blockers(row: sqlite3.Row) -> list[CurationBlocker]:
    try:
        return [CurationBlocker(value) for value in json.loads(row["blockers_json"] or "[]")]
    except (ValueError, json.JSONDecodeError):
        return []


def _build_candidate(row: sqlite3.Row, connection: sqlite3.Connection) -> CurationCandidate:
    """Hydrate a queue row into the wire response."""
    crop: CropBox | None = None
    if row["crop_json"]:
        try:
            data = json.loads(row["crop_json"])
            crop = CropBox.model_validate(data)
        except (ValueError, json.JSONDecodeError):
            crop = None
    screening: SfwScreening | None = None
    if row["sfw_verdict"] is not None:
        screening = SfwScreening(
            verdict=row["sfw_verdict"],
            confidence=row["sfw_confidence"],
            method=row["sfw_method"] or "none",
        )
    sfw_human: SfwHumanDecision | None = None
    if row["sfw_human_safe"] is not None:
        sfw_human = SfwHumanDecision(
            safe=bool(row["sfw_human_safe"]),
            reviewer=row["sfw_human_reviewer"] or "local",
            decided_at=row["sfw_human_decided_at"],
        )
    permissions = Permissions(
        license_id=row["license_id"],
        basis=row["permission_basis"],
        permission_url=row["permission_url"],
        attribution=row["attribution"],
        attribution_required=bool(row["attribution_required"]),
    )
    allowed_uses = AllowedUses(
        display=bool(row["allowed_display"]),
        training=bool(row["allowed_training"]),
        trace=bool(row["allowed_trace"]),
    )
    return CurationCandidate(
        asset_id=row["asset_id"],
        primary_style=PrimaryStyle(row["primary_style"]),
        primary_scope=ScopeLabel(row["primary_scope"]),
        secondary_scopes=[scope for scope in _row_scopes(row)[1:] if scope in GALLERY_SCOPES],
        person_count=row["person_count"],
        person_count_approximate=bool(row["person_count_approximate"]),
        width=int(row["width"]),
        height=int(row["height"]),
        thumbnail_url=f"/api/v1/curation/assets/{row['asset_id']}/thumbnail",
        line_art_url=f"/api/v1/curation/assets/{row['asset_id']}/line-art",
        origin=row["origin"],
        crop=crop,
        review_state=row["review_state"],
        blockers=_row_blockers(row),
        quality_score=float(row["quality_score"]),
        sfw_screening=screening,
        sfw_human=sfw_human,
        permissions=permissions,
        allowed_uses=allowed_uses,
        learning_split=LearningSplit(row["learning_split"]),
        gallery_member=bool(row["gallery_member"]),
        gold_member=bool(row["gold_member"]),
        source_work_id=row["source_work_id"],
        parent_asset_id=row["parent_asset_id"],
        artist_id=row["artist_id"],
        leakage_group_id=row["leakage_group_id"],
        label_version=int(row["curation_label_version"]),
        derivative_processing_state=row["derivative_processing_state"],
    )


# ----------------------------------------------------------------- progress

_TOTAL_SQL = (
    "SELECT COUNT(DISTINCT asset_id) AS reviewed,"
    " COUNT(DISTINCT CASE WHEN decision = 'keep' THEN asset_id END) AS accepted,"
    " COUNT(DISTINCT CASE WHEN decision = 'reject' THEN asset_id END) AS rejected"
    " FROM curation_labels"
)


def _style_breakdown(connection: sqlite3.Connection) -> dict[PrimaryStyle, StyleBreakdown]:
    """Reviewed/accepted/rejected/remaining counts broken down by primary_style."""
    out: dict[PrimaryStyle, StyleBreakdown] = {}
    for style in STYLE_BUCKETS:
        label_row = connection.execute(
            "SELECT COUNT(DISTINCT cl.asset_id) AS reviewed,"
            " COUNT(DISTINCT CASE WHEN cl.decision = 'keep' THEN cl.asset_id END) AS accepted,"
            " COUNT(DISTINCT CASE WHEN cl.decision = 'reject' THEN cl.asset_id END) AS rejected"
            " FROM curation_labels cl"
            " JOIN assets a ON a.asset_id = cl.asset_id"
            " WHERE a.primary_style = ?",
            (style.value,),
        ).fetchone()
        remaining_row = connection.execute(
            "SELECT COUNT(*) AS n FROM assets"
            " WHERE primary_style = ? AND review_state = 'unreviewed'",
            (style.value,),
        ).fetchone()
        out[style] = StyleBreakdown(
            reviewed=int(label_row["reviewed"]),
            accepted=int(label_row["accepted"]),
            rejected=int(label_row["rejected"]),
            remaining=int(remaining_row["n"]),
        )
    return out


def _scope_breakdown(connection: sqlite3.Connection) -> dict[ScopeLabel, ScopeBreakdown]:
    """Reviewed/accepted/rejected/remaining counts broken down by scope bucket.

    An asset appears under every scope it declares, mirroring the search API
    contract. ``unknown`` is excluded from the gallery breakdown because it
    is a query-only label.
    """
    out: dict[ScopeLabel, ScopeBreakdown] = {}
    for scope_label in SCOPE_BUCKETS:
        label_row = connection.execute(
            "SELECT COUNT(DISTINCT cl.asset_id) AS reviewed,"
            " COUNT(DISTINCT CASE WHEN cl.decision = 'keep' THEN cl.asset_id END) AS accepted,"
            " COUNT(DISTINCT CASE WHEN cl.decision = 'reject' THEN cl.asset_id END) AS rejected"
            " FROM curation_labels cl"
            " JOIN asset_scopes s ON s.asset_id = cl.asset_id"
            " WHERE s.scope = ?",
            (scope_label.value,),
        ).fetchone()
        remaining_row = connection.execute(
            "SELECT COUNT(DISTINCT a.asset_id) AS n FROM assets a"
            " JOIN asset_scopes s ON s.asset_id = a.asset_id"
            " WHERE s.scope = ? AND a.review_state = 'unreviewed'",
            (scope_label.value,),
        ).fetchone()
        out[scope_label] = ScopeBreakdown(
            reviewed=int(label_row["reviewed"]),
            accepted=int(label_row["accepted"]),
            rejected=int(label_row["rejected"]),
            remaining=int(remaining_row["n"]),
        )
    return out


@router.get("/progress", response_model=CurationProgress)
def progress(state: State) -> CurationProgress:
    row = state.connection.execute(_TOTAL_SQL).fetchone()
    reviewed = int(row["reviewed"])
    accepted = int(row["accepted"])
    rejected = int(row["rejected"])
    held = state.connection.execute(
        f"SELECT COUNT(*) AS n FROM assets WHERE {_HELD_SQL}"  # noqa: S608
    ).fetchone()
    return CurationProgress(
        reviewed=reviewed,
        accepted=accepted,
        rejected=rejected,
        quarantined=int(held["n"]),
        remaining=max(0, DEFAULT_TARGET - reviewed),
        target=DEFAULT_TARGET,
        by_style=_style_breakdown(state.connection),
        by_scope=_scope_breakdown(state.connection),
    )


# --------------------------------------------------- queue: next / by-id / skip


@router.get("/next", response_model=CurationCandidate)
def next_candidate(
    state: State,
    style: Annotated[PrimaryStyle | None, Query(description="Filter by primary style")] = None,
    scope: Annotated[ScopeLabel | None, Query(description="Filter by scope bucket")] = None,
    session_id: Annotated[
        str | None,
        Query(
            description="Review session id: enables the cursor/skip queue semantics",
            max_length=128,
        ),
    ] = None,
) -> CurationCandidate:
    if state.gallery is None:
        # The dataset pipeline hasn't loaded a gallery yet; refuse rather
        # than serve a synthetic candidate.
        raise not_found(
            "gallery_unavailable",
            "no gallery loaded; set LINESCOUT_GALLERY_MANIFEST and restart the API",
        )
    cursor: str | None = None
    if session_id is not None:
        session = state.connection.execute(
            "SELECT cursor_asset_id FROM curation_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        cursor = str(session["cursor_asset_id"]) if session and session["cursor_asset_id"] else None

    sql, params = _candidate_pool_query(style, scope, session_id=session_id, after=cursor)
    rows = state.connection.execute(sql, params).fetchall()
    if not rows and session_id is not None and cursor is not None:
        # Wrapped past the end of the stable order: serve from the front of
        # the (non-excluded) pool again. Skipped assets never come back.
        sql, params = _candidate_pool_query(style, scope, session_id=session_id, after=None)
        rows = state.connection.execute(sql, params).fetchall()
    chosen = _pick_candidate(rows, style, scope)
    if chosen is None:
        raise not_found("queue_empty", "no candidates are awaiting review")
    if session_id is not None:
        # Serving advances the session cursor, so the next call moves on —
        # with or without a label being written.
        _advance_session(state.connection, session_id, str(chosen["asset_id"]))
    return _build_candidate(chosen, state.connection)


@router.get("/candidates/{asset_id}", response_model=CurationCandidate)
def get_candidate(state: State, asset_id: str) -> CurationCandidate:
    """Stable retrieval of one candidate by id, whatever its review state.

    This is the read path for Previous (re-fetch the *actual* previous
    candidate — live metadata, not a stale copy) and for conflict
    reconciliation after a 409.
    """
    if state.gallery is None:
        raise not_found(
            "gallery_unavailable",
            "no gallery loaded; set LINESCOUT_GALLERY_MANIFEST and restart the API",
        )
    row = state.connection.execute(
        f"{_CANDIDATE_SELECT} FROM assets WHERE asset_id = ?",  # noqa: S608
        (asset_id,),
    ).fetchone()
    if row is None:
        raise not_found("asset_not_found", f"asset {asset_id} is not in the gallery")
    return _build_candidate(row, state.connection)


@router.post("/queue/skip", response_model=SkipResponse)
def skip_candidate(state: State, body: SkipRequest) -> SkipResponse:
    """Advance the session past an asset without a label.

    The asset is excluded from *this session's* queue (it is not mutated:
    its review state, version, and eligibility are untouched) and the cursor
    moves to it, so the next serve is a different item.
    """
    if state.gallery is None:
        raise not_found(
            "gallery_unavailable",
            "no gallery loaded; set LINESCOUT_GALLERY_MANIFEST and restart the API",
        )
    exists = state.connection.execute(
        "SELECT 1 FROM assets WHERE asset_id = ?", (body.asset_id,)
    ).fetchone()
    if exists is None:
        raise not_found("asset_not_found", f"asset {body.asset_id} is not in the gallery")
    with immediate_transaction(state.connection) as tx:
        tx.execute(
            "INSERT INTO curation_exclusions (session_id, asset_id, reason)"
            " VALUES (?, ?, 'skipped')"
            " ON CONFLICT(session_id, asset_id) DO NOTHING",
            (body.session_id, body.asset_id),
        )
        tx.execute(
            "INSERT INTO curation_sessions (session_id, cursor_asset_id, updated_at)"
            " VALUES (?, ?, ?)"
            " ON CONFLICT(session_id) DO UPDATE SET"
            " cursor_asset_id = excluded.cursor_asset_id, updated_at = excluded.updated_at",
            (body.session_id, body.asset_id, _iso(_utcnow())),
        )
    sql, params = _candidate_pool_query(None, None, session_id=body.session_id, after=None)
    remaining = len(state.connection.execute(sql, params).fetchall())
    return SkipResponse(
        session_id=body.session_id,
        cursor_asset_id=body.asset_id,
        excluded_asset_id=body.asset_id,
        remaining=remaining,
    )


# ----------------------------------------------------------------- preview


def _valid_reveal(connection: sqlite3.Connection, asset_id: str) -> datetime | None:
    """The expiry of a live reveal grant for the asset, or ``None``."""
    row = connection.execute(
        "SELECT expires_at FROM sfw_reveals WHERE asset_id = ?", (asset_id,)
    ).fetchone()
    if row is None:
        return None
    expires = _parse_ts(str(row["expires_at"]))
    if expires is None or expires <= _utcnow():
        return None
    return expires


@router.get("/assets/{asset_id}/{kind}", response_class=FileResponse)
def preview_asset(state: State, asset_id: str, kind: str) -> FileResponse:
    """Serve gallery files to the curation UI only, behind reveal controls.

    Public ``/api/v1/assets/{id}/...`` routes stay gated on the derived
    ``enabled = 1`` flag (the full serving predicate) — they never serve
    quarantined/uncertain content, and reveal grants do not apply there.
    Here (curation only), *held* records additionally require a deliberate,
    unexpired reveal grant; without one the preview fails with
    ``reveal_required`` instead of leaking the image.
    """
    column = _PREVIEW_KINDS.get(kind)
    if column is None or state.gallery is None:
        raise not_found("asset_not_found", "asset not found")
    row = state.connection.execute(
        f"SELECT {column} AS path, review_state, sfw_verdict, sfw_human_safe"  # noqa: S608
        " FROM assets WHERE asset_id = ?"
        " AND derivatives_current = 1",
        (asset_id,),
    ).fetchone()
    if row is None:
        raise not_found("asset_not_found", "asset not found")
    if _row_held(row) and _valid_reveal(state.connection, asset_id) is None:
        raise ApiError(
            403,
            "reveal_required",
            "this record is quarantined or SFW-uncertain; issue an explicit reveal"
            " before previewing it",
            retryable=False,
            details={
                "asset_id": asset_id,
                "reveal_endpoint": f"/api/v1/curation/quarantine/{asset_id}/reveal",
            },
        )
    path = (state.gallery.data_root / str(row["path"])).resolve()
    if state.gallery.data_root.resolve() not in path.parents or not path.is_file():
        raise not_found("asset_unavailable", "asset file is missing")
    return FileResponse(
        path, media_type="image/png", headers={"Cache-Control": "private, max-age=60"}
    )


# ----------------------------------------------------------------- labels


@router.post("/labels", response_model=LabelResponse, status_code=201)
def write_label(state: State, body: LabelRequest) -> LabelResponse:
    if state.gallery is None:
        raise not_found(
            "gallery_unavailable",
            "no gallery loaded; set LINESCOUT_GALLERY_MANIFEST and restart the API",
        )
    row = state.connection.execute(
        "SELECT asset_id, review_state, review_quality, enabled, sfw_verdict,"
        " sfw_human_safe, primary_scope, width, height, derivatives_current,"
        " derivative_problems_json, curation_label_version"
        " FROM assets WHERE asset_id = ?",
        (body.asset_id,),
    ).fetchone()
    if row is None:
        raise not_found("asset_not_found", f"asset {body.asset_id} is not in the gallery")
    # Labelling a screened-unsafe asset has no semantic meaning — those are
    # quarantined at ingestion; surface a structured 422 instead of silently
    # accepting the label.
    if row["sfw_verdict"] == "unsafe" and row["sfw_human_safe"] != 1:
        raise unprocessable(
            "asset_not_sfw",
            "cannot label an SFW-unsafe asset; adjudicate it through /curation/sfw first",
        )
    # A label is a human decision about the *current* artifacts. Stale
    # derivatives must be re-processed (and re-screened) before curation.
    if not row["derivatives_current"]:
        problems = json.loads(row["derivative_problems_json"] or "[]")
        raise ApiError(
            422,
            "derivative_stale",
            "cannot label an asset whose derived artifacts are stale, invalid, or"
            " still awaiting processing; re-run processing before curating it",
            field="asset_id",
            retryable=False,
            details={"problems": problems},
        )
    # A keep cannot leave an unknown primary scope in place: the reviewer would
    # be accepting a non-label. Provide one or fix it first.
    if (
        body.decision == "keep"
        and row["primary_scope"] == "unknown"
        and (body.primary_scope is None or body.primary_scope is ScopeLabel.UNKNOWN)
    ):
        raise unprocessable(
            "primary_scope_required",
            "cannot accept an asset whose primary scope is unknown; set primary_scope",
            field="primary_scope",
        )

    review_state, review_quality = _apply_decision_to_asset(body)
    primary_scope = (
        body.primary_scope.value
        if body.primary_scope is not None and body.primary_scope is not ScopeLabel.UNKNOWN
        else None
    )
    secondary_json = (
        json.dumps([scope.value for scope in body.secondary_scopes])
        if body.secondary_scopes is not None
        else None
    )
    blockers_json = json.dumps([blocker.value for blocker in body.blockers])
    reviewer = body.reviewer or "local"
    decided_at = _iso(_utcnow())

    # Single transaction (write-locked up front): append the audit row, then
    # mirror the decision into the assets cache with a **conditional** update
    # on the caller's expected review state *and* label version. If the
    # condition no longer matches, another curator wrote first — the whole
    # transaction rolls back and the caller receives a structured 409 with
    # reconciliation information. The schema's CHECK constraints guard style,
    # scope, and split membership, so a typo from the UI surfaces as a
    # SQLite IntegrityError.
    label_id = 0
    with immediate_transaction(state.connection) as tx:
        cursor = tx.execute(
            "INSERT INTO curation_labels ("
            " asset_id, decision, primary_style, primary_scope, secondary_scopes_json,"
            " blockers_json, sfw_safe, quality, note, reviewer"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                body.asset_id,
                body.decision,
                body.primary_style.value if body.primary_style else None,
                primary_scope,
                secondary_json,
                blockers_json,
                body.sfw_safe,
                body.quality,
                body.note,
                reviewer,
            ),
        )
        label_id = int(cursor.lastrowid or 0)
        cursor = tx.execute(
            "UPDATE assets SET"
            " review_state = ?,"
            " review_quality = ?,"
            " blockers_json = ?,"
            " primary_style = COALESCE(?, primary_style),"
            " primary_scope = COALESCE(?, primary_scope),"
            " secondary_scopes_json = COALESCE(?, secondary_scopes_json),"
            " sfw_human_safe = COALESCE(?, sfw_human_safe),"
            " sfw_human_reviewer = CASE WHEN ? IS NULL THEN sfw_human_reviewer ELSE ? END,"
            " sfw_human_decided_at = CASE WHEN ? IS NULL THEN sfw_human_decided_at ELSE ? END,"
            # ``enabled`` is derived; zero it here (the CHECK constraints
            # evaluate the whole row on UPDATE) and let recompute_enabled
            # restore it below under the new review state.
            " enabled = 0,"
            " curation_label_version = curation_label_version + 1"
            " WHERE asset_id = ? AND review_state = ? AND curation_label_version = ?",
            (
                review_state,
                review_quality,
                blockers_json,
                body.primary_style.value if body.primary_style else None,
                primary_scope,
                secondary_json,
                body.sfw_safe,
                body.sfw_safe,
                reviewer,
                body.sfw_safe,
                decided_at,
                body.asset_id,
                body.expected_review_state.value,
                body.expected_label_version,
            ),
        )
        if cursor.rowcount == 0:
            # Atomic rejection: nothing was written. Report the live state so
            # the client can reconcile (reload by id, re-apply against the
            # current version).
            _raise_version_conflict(
                state.connection,
                body.asset_id,
                body.expected_label_version,
                body.expected_review_state,
            )
        if body.secondary_scopes is not None and primary_scope is not None:
            # Transactional ``asset_scopes`` synchronization: the denormalised
            # scope rows are replaced inside the same transaction as the
            # decision, so they can never drift from ``assets``.
            tx.execute("DELETE FROM asset_scopes WHERE asset_id = ?", (body.asset_id,))
            tx.executemany(
                "INSERT INTO asset_scopes(asset_id, scope) VALUES (?, ?)",
                [
                    (body.asset_id, scope.value)
                    for scope in (ScopeLabel(primary_scope), *body.secondary_scopes)
                    if scope in GALLERY_SCOPES
                ],
            )
        # ``enabled`` is derived (the frozen is_servable predicate); recompute
        # it inside the same transaction so the mirror never drifts.
        enabled = recompute_enabled(tx, body.asset_id)
        blockers = serving_blockers(tx, body.asset_id) if not enabled else []
        # Read the stored row back so the wire response matches what is
        # actually persisted (timestamp and SFW mirror included).
        stamp = tx.execute(
            "SELECT sfw_human_safe, curation_label_version FROM assets WHERE asset_id = ?",
            (body.asset_id,),
        ).fetchone()
        version = int(stamp["curation_label_version"])

    if body.session_id is not None:
        _advance_session(state.connection, body.session_id, body.asset_id)
    # Reload the in-memory enabled-asset list so /search and /assets/* reflect
    # the new serving flag without requiring a process restart.
    _refresh_serving_cache(state)

    sfw_human_approved: bool | None
    if stamp["sfw_human_safe"] is not None:
        sfw_human_approved = bool(stamp["sfw_human_safe"])
    elif row["sfw_human_safe"] is not None:
        sfw_human_approved = bool(row["sfw_human_safe"])
    else:
        sfw_human_approved = None

    return LabelResponse(
        id=label_id,
        asset_id=body.asset_id,
        decision=body.decision,
        review_state=review_state,
        review_quality=review_quality,
        blockers=body.blockers,
        sfw_human_approved=sfw_human_approved,
        enabled=enabled,
        serving_blockers=blockers,
        label_version=version,
        created_at=decided_at,
    )


def _apply_decision_to_asset(body: LabelRequest) -> tuple[str, int | None]:
    """Translate a label into the new (review_state, review_quality) pair.

    * ``keep``   -> ``accepted``. Serving is decided by the derived predicate,
      not here: permission, human SFW approval, and the quality floor all
      still apply, and the response explains any that fail.
    * ``reject`` -> ``rejected``; the asset drops out of search via the same
      derived predicate.
    """
    if body.decision == "keep":
        return "accepted", body.quality
    return "rejected", body.quality


# ------------------------------------------------- quarantine + SFW adjudication


_QUARANTINE_SELECT = (
    "SELECT asset_id, primary_style, primary_scope, width, height, review_state,"
    " blockers_json, quality_score, sfw_verdict, sfw_confidence, sfw_method,"
    " sfw_human_safe, sfw_human_reviewer, sfw_human_decided_at, source_work_id,"
    " parent_asset_id, curation_label_version,"
    " (SELECT processing_state FROM curation_derivatives d"
    "  WHERE d.asset_id = assets.asset_id) AS derivative_processing_state"
    " FROM assets"
)


def _build_quarantine_candidate(
    row: sqlite3.Row, connection: sqlite3.Connection
) -> QuarantineCandidate:
    screening: SfwScreening | None = None
    if row["sfw_verdict"] is not None:
        screening = SfwScreening(
            verdict=row["sfw_verdict"],
            confidence=row["sfw_confidence"],
            method=row["sfw_method"] or "none",
        )
    sfw_human: SfwHumanDecision | None = None
    if row["sfw_human_safe"] is not None:
        sfw_human = SfwHumanDecision(
            safe=bool(row["sfw_human_safe"]),
            reviewer=row["sfw_human_reviewer"] or "local",
            decided_at=row["sfw_human_decided_at"],
        )
    expires = _valid_reveal(connection, str(row["asset_id"]))
    revealed = expires is not None
    return QuarantineCandidate(
        asset_id=row["asset_id"],
        primary_style=PrimaryStyle(row["primary_style"]),
        primary_scope=ScopeLabel(row["primary_scope"]),
        width=int(row["width"]),
        height=int(row["height"]),
        review_state=row["review_state"],
        blockers=_row_blockers(row),
        quality_score=float(row["quality_score"]),
        sfw_screening=screening,
        sfw_human=sfw_human,
        source_work_id=row["source_work_id"],
        parent_asset_id=row["parent_asset_id"],
        label_version=int(row["curation_label_version"]),
        derivative_processing_state=row["derivative_processing_state"],
        revealed=revealed,
        reveal_expires_at=_iso(expires) if expires is not None else None,
        # URLs stay null until a reveal grant exists — the preview route
        # enforces the grant independently, so this is a UX affordance, not
        # the gate itself.
        thumbnail_url=(
            f"/api/v1/curation/assets/{row['asset_id']}/thumbnail" if revealed else None
        ),
        line_art_url=f"/api/v1/curation/assets/{row['asset_id']}/line-art" if revealed else None,
    )


@router.get("/quarantine", response_model=list[QuarantineCandidate])
def list_quarantine(state: State) -> list[QuarantineCandidate]:
    """The SFW adjudication backlog: held records, **metadata only**.

    Quarantined (any reason), screened unsafe/unsure, or human-flagged
    records. No image bytes and no preview URLs until a deliberate reveal.
    """
    if state.gallery is None:
        raise not_found(
            "gallery_unavailable",
            "no gallery loaded; set LINESCOUT_GALLERY_MANIFEST and restart the API",
        )
    rows = state.connection.execute(
        f"{_QUARANTINE_SELECT} WHERE {_HELD_SQL} ORDER BY asset_id"  # noqa: S608
    ).fetchall()
    return [_build_quarantine_candidate(row, state.connection) for row in rows]


@router.post("/quarantine/{asset_id}/reveal", response_model=QuarantineCandidate)
def reveal_quarantined(state: State, asset_id: str, body: RevealRequest) -> QuarantineCandidate:
    """Issue a deliberate, expiring reveal grant for one held record.

    The grant is recorded (who revealed what, when, until when) and is the
    *only* way the curation preview route will serve held content. It never
    applies to the public asset routes, and it changes nothing about the
    asset itself — review state, version, and eligibility are untouched.
    """
    if state.gallery is None:
        raise not_found(
            "gallery_unavailable",
            "no gallery loaded; set LINESCOUT_GALLERY_MANIFEST and restart the API",
        )
    row = state.connection.execute(
        f"{_QUARANTINE_SELECT} WHERE asset_id = ? AND {_HELD_SQL}",  # noqa: S608
        (asset_id,),
    ).fetchone()
    if row is None:
        exists = state.connection.execute(
            "SELECT 1 FROM assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        if exists is None:
            raise not_found("asset_not_found", f"asset {asset_id} is not in the gallery")
        raise unprocessable(
            "asset_not_quarantined",
            "asset is not held; no reveal is required",
            field="asset_id",
        )
    expires = _utcnow() + timedelta(seconds=state.settings.curation_reveal_ttl_seconds)
    with immediate_transaction(state.connection) as tx:
        tx.execute(
            "INSERT INTO sfw_reveals (asset_id, revealed_by, revealed_at, expires_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(asset_id) DO UPDATE SET"
            " revealed_by = excluded.revealed_by, revealed_at = excluded.revealed_at,"
            " expires_at = excluded.expires_at",
            (asset_id, body.reviewer or "local", _iso(_utcnow()), _iso(expires)),
        )
    refreshed = state.connection.execute(
        f"{_QUARANTINE_SELECT} WHERE asset_id = ?",  # noqa: S608
        (asset_id,),
    ).fetchone()
    return _build_quarantine_candidate(refreshed, state.connection)


@router.post(
    "/sfw/{asset_id}/adjudication", response_model=SfwAdjudicationResponse, status_code=201
)
def adjudicate_sfw(
    state: State, asset_id: str, body: SfwAdjudicationRequest
) -> SfwAdjudicationResponse:
    """Record an explicit human SFW adjudication for a held/uncertain record.

    Effects (one transaction, atomic conditional on ``expected_label_version``):

    * ``safe`` — the human decision is mirrored onto the asset and a
      quarantined record returns to ``unreviewed`` so it can be reviewed on
      its merits through the normal queue. Serving still requires every
      other gate (permission, accepted review with quality, no blockers).
    * ``unsafe`` — the record is quarantined and human-flagged; it leaves
      every serving surface until a future adjudication says otherwise.
      The public asset routes were already closed to it and stay closed.

    Both outcomes are append-only audit history in ``sfw_adjudications``.
    """
    if state.gallery is None:
        raise not_found(
            "gallery_unavailable",
            "no gallery loaded; set LINESCOUT_GALLERY_MANIFEST and restart the API",
        )
    row = state.connection.execute(
        "SELECT review_state, sfw_verdict, sfw_human_safe, curation_label_version"
        " FROM assets WHERE asset_id = ?",
        (asset_id,),
    ).fetchone()
    if row is None:
        raise not_found("asset_not_found", f"asset {asset_id} is not in the gallery")
    # Held records are adjudicated here; previously-decided records may be
    # re-adjudicated (e.g. an accepted asset later judged unsafe).
    if not (_row_held(row) or row["sfw_human_safe"] is not None):
        raise unprocessable(
            "asset_not_sfw_pending",
            "asset is not quarantined or SFW-uncertain; record SFW decisions with a label",
            field="asset_id",
        )

    reviewer = body.reviewer or "local"
    decided_at = _iso(_utcnow())
    new_state = (
        ("unreviewed" if row["review_state"] == "quarantined" else row["review_state"])
        if body.safe
        else "quarantined"
    )
    adjudication_id = 0
    with immediate_transaction(state.connection) as tx:
        cursor = tx.execute(
            "INSERT INTO sfw_adjudications"
            " (asset_id, safe, prior_verdict, prior_review_state, reviewer, note)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                asset_id,
                int(body.safe),
                row["sfw_verdict"],
                row["review_state"],
                reviewer,
                body.note,
            ),
        )
        adjudication_id = int(cursor.lastrowid or 0)
        cursor = tx.execute(
            "UPDATE assets SET"
            " sfw_human_safe = ?,"
            " sfw_human_reviewer = ?,"
            " sfw_human_decided_at = ?,"
            " review_state = ?,"
            # Zero the derived flag first (whole-row CHECK constraints), then
            # recompute_enabled restores it if the new state still serves.
            " enabled = 0,"
            " curation_label_version = curation_label_version + 1"
            " WHERE asset_id = ? AND curation_label_version = ?",
            (
                int(body.safe),
                reviewer,
                decided_at,
                new_state,
                asset_id,
                body.expected_label_version,
            ),
        )
        if cursor.rowcount == 0:
            _raise_version_conflict(
                state.connection,
                asset_id,
                body.expected_label_version,
                ReviewState(row["review_state"]),
            )
        enabled = recompute_enabled(tx, asset_id)
        blockers = serving_blockers(tx, asset_id) if not enabled else []
        stamp = tx.execute(
            "SELECT curation_label_version FROM assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        version = int(stamp["curation_label_version"])

    if body.session_id is not None:
        _advance_session(state.connection, body.session_id, asset_id)
    _refresh_serving_cache(state)

    return SfwAdjudicationResponse(
        id=adjudication_id,
        asset_id=asset_id,
        safe=body.safe,
        review_state=new_state,
        sfw_human=SfwHumanDecision(safe=body.safe, reviewer=reviewer, decided_at=decided_at),
        label_version=version,
        enabled=enabled,
        serving_blockers=blockers,
        created_at=decided_at,
    )


# ----------------------------------------------------------------- crops (derivatives)


@router.post("/assets/{asset_id}/crops", response_model=CropResponse, status_code=201)
def create_crop(state: State, asset_id: str, body: CropRequest) -> CropResponse:
    """Cut an immutable child derivative from this asset.

    The child gets its own identity (deterministic from dataset+item+crop),
    its parent's identity links, bounded geometry, **fresh files and
    hashes**, and its own processing/review state — it is created
    ``unreviewed`` with ``derivatives_current = 0`` (awaiting processing)
    and ``enabled = 0``. It can only ever be enabled after its required
    processing completes *and* a human accepts it with all serving gates
    passing; nothing about the parent's review/SFW/gold state is inherited.
    Held (quarantined/uncertain/flagged) parents cannot be cropped — that
    would launder held content into a fresh reviewable asset.
    """
    if state.gallery is None:
        raise not_found(
            "gallery_unavailable",
            "no gallery loaded; set LINESCOUT_GALLERY_MANIFEST and restart the API",
        )
    parent = state.connection.execute(
        "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
    ).fetchone()
    if parent is None:
        raise not_found("asset_not_found", f"asset {asset_id} is not in the gallery")
    if not parent["derivatives_current"]:
        raise ApiError(
            422,
            "derivative_stale",
            "cannot crop an asset whose derived artifacts are stale or invalid;"
            " re-run the pipeline before cropping it",
            field="asset_id",
            retryable=False,
        )
    if _row_held(parent):
        raise unprocessable(
            "crop_source_restricted",
            "cannot crop a quarantined or SFW-uncertain asset; adjudicate it first",
            field="asset_id",
        )
    # The wire crop shape is the API's own contract; convert to the frozen
    # manifest model at the boundary so the derivative pipeline works with
    # the exact geometry type it was frozen against.
    crop = ManifestCropBox.model_validate(body.crop.model_dump())
    if not crop_fits(crop, int(parent["width"]), int(parent["height"])):
        raise unprocessable(
            "crop_out_of_bounds",
            "crop rectangle exceeds the asset's width/height",
            field="crop",
        )
    if min(body.crop.width, body.crop.height) < MIN_CROP_EDGE:
        raise unprocessable(
            "crop_too_small",
            f"each crop edge must be at least {MIN_CROP_EDGE} px",
            field="crop",
        )

    child_id = child_asset_id(parent, crop)
    existing = state.connection.execute(
        "SELECT asset_id FROM assets WHERE asset_id = ?", (child_id,)
    ).fetchone()
    if existing is not None:
        raise conflict(
            "derivative_exists",
            "a derivative for this exact crop already exists",
            field="crop",
        )

    # Filesystem stage: fresh files cut from the parent's verified artifacts.
    # Rolled back entirely on failure — no partial artifacts survive.
    try:
        files, directory = crop_files_for(
            state.gallery.data_root,
            parent,
            child_id,
            crop,
            state.settings.derivative_thumbnail_size,
        )
    except DerivativeError as error:
        raise _derivative_api_error(error) from error

    # Database stage: registry row (durable) + cache row + scopes, with the
    # parent's state re-verified under the write lock so the crop cannot be
    # cut from a parent that changed while the files were being written.
    try:
        with immediate_transaction(state.connection) as tx:
            fresh = tx.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,)).fetchone()
            if fresh is None:
                raise not_found("asset_not_found", f"asset {asset_id} is not in the gallery")
            if int(fresh["curation_label_version"]) != body.expected_label_version:
                _raise_version_conflict(
                    state.connection,
                    asset_id,
                    body.expected_label_version,
                    ReviewState(fresh["review_state"]),
                )
            if not fresh["derivatives_current"] or _row_held(fresh):
                raise unprocessable(
                    "crop_source_restricted",
                    "the source asset changed while the crop was being written;"
                    " reload it and retry",
                    field="asset_id",
                )
            if (
                tx.execute("SELECT 1 FROM assets WHERE asset_id = ?", (child_id,)).fetchone()
                or tx.execute(
                    "SELECT 1 FROM curation_derivatives WHERE asset_id = ?", (child_id,)
                ).fetchone()
            ):
                raise conflict(
                    "derivative_exists",
                    "a derivative for this exact crop already exists",
                    field="crop",
                )
            registry = registry_values(
                fresh,
                child_id,
                crop,
                files,
                created_by=body.reviewer or "local",
                note=body.note,
            )
            insert_registry_row(tx, registry)
            values = child_asset_values(fresh, registry)
            insert_child_asset(tx, values)
            sync_asset_scopes(tx, child_id, values)
    except Exception:
        # The transaction rolled back; the freshly written files must go too.
        try:
            for name in ("original.png", "line_art.png", "thumbnail.png"):
                (directory / name).unlink(missing_ok=True)
            directory.rmdir()
        except OSError:  # pragma: no cover - best-effort cleanup
            log.warning("could not clean up derivative files for %s", child_id)
        raise

    _refresh_serving_cache(state)
    return CropResponse(
        asset_id=child_id,
        parent_asset_id=asset_id,
        crop=body.crop,
        width=body.crop.width,
        height=body.crop.height,
        review_state="unreviewed",
        processing_state="pending",
        derivatives_current=False,
        label_version=0,
        created=True,
    )


def _record_processing_failure(state: State, asset_id: str, error: DerivativeError) -> None:
    """Persist a failed processing attempt (audit) and invalidate the child.

    The asset row is marked not-current with the failure recorded, so a
    previously-complete derivative whose files went missing/tampered is
    pulled from every serving surface until processing succeeds again.
    """
    problems = [AWAITING_PROCESSING]
    raw_problems = (error.details or {}).get("problems")
    if isinstance(raw_problems, (list, tuple)):
        problems.extend(str(problem) for problem in raw_problems)
    with immediate_transaction(state.connection) as tx:
        tx.execute(
            "UPDATE curation_derivatives SET processing_state = 'failed',"
            " processing_attempts = processing_attempts + 1, processing_error = ?"
            " WHERE asset_id = ?",
            (error.message, asset_id),
        )
        tx.execute(
            "UPDATE assets SET derivatives_current = 0,"
            " derivative_problems_json = ?,"
            " enabled = 0,"
            " curation_label_version = curation_label_version + 1"
            " WHERE asset_id = ?",
            (json.dumps(problems), asset_id),
        )
        recompute_enabled(tx, asset_id)
    _refresh_serving_cache(state)


@router.post("/assets/{asset_id}/process", response_model=DerivativeProcessResponse)
def process_derivative(state: State, asset_id: str) -> DerivativeProcessResponse:
    """Run (or re-run) a derivative's required processing.

    Verifies the child's files against their recorded checksums, rebuilds
    every measurement from the child's *own* bytes, regenerates the
    thumbnail, and — only when all of that succeeds — marks the child's
    derivatives current. Index membership (the in-memory serving list) and
    the derived ``enabled`` flag are recomputed in the same transaction, and
    the response carries the artifact stamp any future embedding must bind
    to (per the frozen vector contract, a new derivative has no vectors yet:
    ``embedding_status: "missing"``).
    """
    if state.gallery is None:
        raise not_found(
            "gallery_unavailable",
            "no gallery loaded; set LINESCOUT_GALLERY_MANIFEST and restart the API",
        )
    asset = state.connection.execute(
        "SELECT asset_id, enabled FROM assets WHERE asset_id = ?", (asset_id,)
    ).fetchone()
    if asset is None:
        raise not_found("asset_not_found", f"asset {asset_id} is not in the gallery")
    registry = state.connection.execute(
        "SELECT * FROM curation_derivatives WHERE asset_id = ?", (asset_id,)
    ).fetchone()
    if registry is None:
        raise unprocessable(
            "not_a_derivative",
            "asset is not a curator-created derivative; manifest assets are processed"
            " by the dataset pipeline",
            field="asset_id",
        )

    try:
        measurements, replacement = process_derivative_files(
            state.gallery.data_root, registry, state.settings.derivative_thumbnail_size
        )
    except DerivativeError as error:
        _record_processing_failure(state, asset_id, error)
        raise _derivative_api_error(error) from error

    with immediate_transaction(state.connection) as tx:
        attempts = int(registry["processing_attempts"]) + 1
        if replacement is not None:
            tx.execute(
                "UPDATE curation_derivatives SET processing_state = 'complete',"
                " processing_attempts = ?, processing_error = NULL, measurements_json = ?,"
                " thumbnail_checksum = ? WHERE asset_id = ?",
                (
                    attempts,
                    json.dumps(measurements.as_dict()),
                    replacement.thumbnail_checksum,
                    asset_id,
                ),
            )
        else:
            tx.execute(
                "UPDATE curation_derivatives SET processing_state = 'complete',"
                " processing_attempts = ?, processing_error = NULL, measurements_json = ?"
                " WHERE asset_id = ?",
                (attempts, json.dumps(measurements.as_dict()), asset_id),
            )
        tx.execute(
            "UPDATE assets SET"
            " text_coverage = ?, ink_coverage = ?, phash = ?, quality_score = ?,"
            " derivatives_current = 1, derivative_problems_json = '[]',"
            " curation_label_version = curation_label_version + 1"
            " WHERE asset_id = ?",
            (
                measurements.text_coverage,
                measurements.ink_coverage,
                measurements.phash,
                measurements.quality_score,
                asset_id,
            ),
        )
        enabled = recompute_enabled(tx, asset_id)
        blockers = serving_blockers(tx, asset_id) if not enabled else []
        stamp = tx.execute(
            "SELECT curation_label_version FROM assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        version = int(stamp["curation_label_version"])

    _refresh_serving_cache(state)
    return DerivativeProcessResponse(
        asset_id=asset_id,
        parent_asset_id=str(registry["parent_asset_id"]),
        processing_state="complete",
        attempts=attempts,
        measurements=DerivativeMeasurements(
            width=measurements.width,
            height=measurements.height,
            ink_coverage=measurements.ink_coverage,
            text_coverage=measurements.text_coverage,
            background_coverage=measurements.background_coverage,
            quality_score=measurements.quality_score,
            phash=measurements.phash,
        ),
        artifact=DerivativeArtifactStamp(
            processing_revision=int(registry["processing_revision"]),
            line_art_checksum=str(registry["line_art_checksum"]),
        ),
        embedding_status="missing",
        derivatives_current=True,
        label_version=version,
        enabled=enabled,
        serving_blockers=blockers,
    )


# ----------------------------------------------------------------- snapshots


def _snapshot_dir(state: State) -> Path:
    """Resolve and ensure the snapshots directory exists under the data dir."""
    base = state.settings.data_dir
    if not base.is_absolute():
        base = base.resolve()
    target = base / "snapshots"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _snapshot_timestamp(now: datetime | None = None) -> str:
    """``YYYYMMDD_HHMMSS_ffffff`` form, UTC — microsecond precision so two
    exports in the same second cannot collide on the filename."""
    moment = now or _utcnow()
    return moment.strftime("%Y%m%d_%H%M%S_%f")


def _publish_snapshot_exclusively(
    directory: Path, base_id: str, payload: str
) -> tuple[str, str, str]:
    """Publish the snapshot file with **exclusive** creation semantics.

    ``O_CREAT | O_EXCL`` fails if the target exists, so two concurrent
    exports can never land on (or overwrite) the same file — this is not an
    existence check followed by an overwrite-capable write. On a name
    collision the caller retries under a fresh unique suffix. The bytes are
    fsynced (file and directory) so a crash cannot leave a truncated
    snapshot that the registry believes in.

    Returns ``(snapshot_id, relative_path, content_sha256)``.
    """
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    snapshot_id = base_id
    for _ in range(8):
        path = directory / f"{snapshot_id}.json"
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            snapshot_id = f"{base_id}_{uuid4().hex[:8]}"
            continue
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            Path(path).unlink(missing_ok=True)
            raise
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:  # pragma: no cover - directory fsync is best-effort
            pass
        return snapshot_id, f"snapshots/{path.name}", digest
    raise ApiError(
        503,
        "snapshot_write_failed",
        "could not allocate a unique snapshot filename; retry",
        retryable=True,
        headers={"Retry-After": "1"},
    )


@router.post("/snapshots", response_model=SnapshotResponse, status_code=201)
def export_snapshot(state: State) -> SnapshotResponse:
    """Write an immutable **full** snapshot of the current curation state.

    Frozen semantics: the snapshot captures the *latest* keep-or-reject
    label per asset — rejected assets included — from a **consistent
    database view**. The whole export (label read, lineage read, file
    publication, registry insert, and the exact linking of the exported
    label rows to the snapshot) runs inside one ``BEGIN IMMEDIATE``
    transaction, so a concurrent label write or a concurrent export can
    never interleave: writers serialize, the second export chains to the
    first via ``previous_snapshot_id``, and neither file can overwrite the
    other. Every export is a new file; snapshots are never edited and never
    incremental, and the append-only ``curation_labels`` audit history stays
    complete in the database. The published bytes' SHA-256 is recorded on
    the registry row so the export stays verifiable after later edits.
    """
    if state.gallery is None:
        raise not_found(
            "gallery_unavailable",
            "no gallery loaded; set LINESCOUT_GALLERY_MANIFEST and restart the API",
        )

    target_dir = _snapshot_dir(state)
    with immediate_transaction(state.connection) as tx:
        # Latest label per asset (highest id among keep/reject decisions).
        # Includes rejected assets — quality-model training needs both classes.
        rows = tx.execute(
            "SELECT cl.id, cl.asset_id, cl.decision, cl.primary_style, cl.primary_scope,"
            " cl.secondary_scopes_json, cl.crop_json, cl.blockers_json, cl.sfw_safe,"
            " cl.quality, cl.note, cl.reviewer, cl.created_at,"
            " a.primary_style AS asset_primary_style, a.primary_scope AS asset_primary_scope,"
            " a.secondary_scopes_json AS asset_secondary_scopes_json,"
            " a.review_state, a.review_quality"
            " FROM curation_labels cl"
            " JOIN ("
            "   SELECT asset_id, MAX(id) AS latest_id FROM curation_labels"
            "   WHERE decision IN ('keep', 'reject') GROUP BY asset_id"
            " ) latest ON latest.latest_id = cl.id"
            " JOIN assets a ON a.asset_id = cl.asset_id"
            " ORDER BY cl.asset_id"
        ).fetchall()
        previous = tx.execute(
            "SELECT snapshot_id FROM snapshots ORDER BY created_at DESC, snapshot_id DESC LIMIT 1"
        ).fetchone()
        previous_snapshot_id = str(previous["snapshot_id"]) if previous else None

        breakdown: dict[PrimaryStyle, int] = {style: 0 for style in STYLE_BUCKETS}
        serialized: list[dict[str, object]] = []
        exported_label_ids: list[int] = []
        for label_row in rows:
            primary = PrimaryStyle(label_row["primary_style"] or label_row["asset_primary_style"])
            breakdown[primary] += 1
            exported_label_ids.append(int(label_row["id"]))
            serialized.append(
                {
                    "id": int(label_row["id"]),
                    "asset_id": label_row["asset_id"],
                    "decision": label_row["decision"],
                    "primary_style": primary.value,
                    "primary_scope": label_row["primary_scope"] or label_row["asset_primary_scope"],
                    "secondary_scopes": json.loads(
                        label_row["secondary_scopes_json"]
                        or label_row["asset_secondary_scopes_json"]
                        or "[]"
                    ),
                    "crop": json.loads(label_row["crop_json"]) if label_row["crop_json"] else None,
                    "blockers": json.loads(label_row["blockers_json"] or "[]"),
                    "sfw_human_approved": (
                        bool(label_row["sfw_safe"]) if label_row["sfw_safe"] is not None else None
                    ),
                    "quality": int(label_row["quality"])
                    if label_row["quality"] is not None
                    else None,
                    "note": label_row["note"],
                    "reviewer": label_row["reviewer"],
                    "review_state": label_row["review_state"],
                    "review_quality": int(label_row["review_quality"])
                    if label_row["review_quality"] is not None
                    else None,
                    "created_at": label_row["created_at"],
                }
            )

        created_at = _iso(_utcnow())
        base_id = f"curation_{_snapshot_timestamp()}"
        # Allocate id + payload, then publish exclusively; a name collision
        # (another export in the same microsecond, or a leftover file) gets a
        # fresh unique suffix and a rebuilt payload so file, registry, and
        # payload always agree on the identity.
        snapshot_id = ""
        relative = ""
        content_sha256 = ""
        for _ in range(8):
            payload = {
                "schema_version": 2,
                "snapshot_id": base_id,
                "previous_snapshot_id": previous_snapshot_id,
                "created_at": created_at,
                "gallery": {
                    "dataset_version": state.gallery.dataset_version,
                    "manifest_hash": state.gallery.manifest_hash,
                },
                "label_count": len(serialized),
                "style_breakdown": {style.value: count for style, count in breakdown.items()},
                "labels": serialized,
            }
            document = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            snapshot_id, relative, content_sha256 = _publish_snapshot_exclusively(
                target_dir, base_id, document
            )
            break

        tx.execute(
            "INSERT INTO snapshots (snapshot_id, created_at, label_count,"
            " previous_snapshot_id, path, content_sha256) VALUES (?,?,?,?,?,?)",
            (
                snapshot_id,
                created_at,
                len(serialized),
                previous_snapshot_id,
                relative,
                content_sha256,
            ),
        )
        # Link **exactly** the exported labels to this snapshot: the rows in
        # the file, and only those rows. A superseded label keeps whatever
        # snapshot (if any) first exported it — this column is audit
        # bookkeeping, never a filter on what a future snapshot contains.
        tx.executemany(
            "UPDATE curation_labels SET snapshot_id = ? WHERE id = ? AND snapshot_id IS NULL",
            [(snapshot_id, label_id) for label_id in exported_label_ids],
        )

    log.info("wrote curation snapshot %s (%d labels)", snapshot_id, len(serialized))

    return SnapshotResponse(
        snapshot_id=snapshot_id,
        path=relative,
        label_count=len(serialized),
        previous_snapshot_id=previous_snapshot_id,
        style_breakdown=breakdown,
        content_sha256=content_sha256,
        created_at=created_at,
    )
