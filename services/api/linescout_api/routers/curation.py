"""Local-only curation endpoints. Mounted only when ``LINESCOUT_CURATION_MODE=1``.

Milestone 2 implements the curation loop end-to-end:

* :func:`next_candidate` returns the next asset the reviewer should look at,
  balancing the five style families and the gallery scope buckets so the batch
  stays representative.
* :func:`write_label` validates a human decision, persists it as an immutable
  audit row in ``curation_labels``, and mirrors the decision into the
  ``assets`` cache so the change is observable in search results and asset
  serving immediately.
* :func:`export_snapshot` writes an **immutable full snapshot**: the latest
  keep-or-reject label per asset (rejected assets included) at a point in
  time, chained to the previous snapshot via ``previous_snapshot_id``. Every
  export is a new file; snapshots are never edited and never incremental.
  The append-only ``curation_labels`` audit history stays in the database —
  the snapshot is the state view, the table is the history.
* :func:`progress` aggregates reviewed / accepted / rejected counts overall
  and broken down by style and scope.

Curation only touches assets that already live in the ``assets`` cache; the
caller (the Milestone 2 dataset pipeline) is expected to have populated the
gallery via :func:`linescout_api.gallery.sync_gallery` first.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
from linescout_ml.manifest import AllowedUses, Permissions, SfwHumanDecision, SfwScreening
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
from linescout_api.errors import conflict, not_found, unprocessable
from linescout_api.gallery import enabled_assets, recompute_enabled, serving_blockers

log = logging.getLogger(__name__)

router = APIRouter(prefix="/curation", tags=["curation"])

_PREVIEW_KINDS: dict[str, str] = {"thumbnail": "thumbnail_path", "line-art": "line_art_path"}


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


class LabelRequest(BaseModel):
    """Body of ``POST /curation/labels``."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1, max_length=64)
    #: Review state the client last observed. Compared against the live row
    #: inside the write transaction so a second curator cannot silently
    #: overwrite the first (lost-update).
    expected_review_state: ReviewState
    decision: Literal["keep", "reject"]
    primary_style: PrimaryStyle | None = None
    #: The single best scope; required on ``keep`` when the stored primary is
    #: still ``unknown`` (an unknown primary cannot be accepted).
    primary_scope: ScopeLabel | None = None
    secondary_scopes: list[ScopeLabel] | None = None
    crop: CropBox | None = None
    #: Named, use-blocking defects. Blockers force ``reject`` (or quarantine);
    #: a ``keep`` carrying blockers is a 422.
    blockers: list[CurationBlocker] = Field(default_factory=list)
    #: Optional human SFW assertion recorded with this label. ``None`` leaves
    #: any existing human decision untouched.
    sfw_safe: bool | None = None
    quality: int | None = Field(default=None, ge=1, le=3)
    note: str | None = Field(default=None, max_length=2000)
    reviewer: str | None = Field(default=None, max_length=64)

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
    created_at: str


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
    created_at: str


# ----------------------------------------------------------------- queue logic


def _reviewed_count(connection: sqlite3.Connection) -> int:
    """Number of *distinct* assets with at least one label, regardless of decision."""
    row = connection.execute("SELECT COUNT(DISTINCT asset_id) AS n FROM curation_labels").fetchone()
    return int(row["n"])


def _candidate_base_query(
    style: PrimaryStyle | None, scope: ScopeLabel | None
) -> tuple[str, list[object]]:
    """Build the SELECT for the candidate pool.

    Stratification happens in :func:`_pick_candidate`; here we just gather the
    row set so the picker can balance across the style×scope grid. The queue
    holds *unreviewed* assets: anything not safe per the automated screen was
    quarantined at ingestion, so no extra SFW filter is needed here. Human SFW
    approval happens through this queue, not before it.
    """
    sql = [
        "SELECT asset_id, primary_style, primary_scope, secondary_scopes_json,",
        " person_count, person_count_approximate, width, height, origin,",
        " crop_json, review_state, blockers_json, quality_score,",
        " sfw_verdict, sfw_confidence, sfw_method,",
        " sfw_human_safe, sfw_human_reviewer, sfw_human_decided_at,",
        " license_id, permission_basis, permission_url, attribution, attribution_required,",
        " allowed_display, allowed_training, allowed_trace,",
        " learning_split, gallery_member, gold_member, source_work_id,",
        " parent_asset_id, artist_id, leakage_group_id",
        " FROM assets",
        " WHERE review_state = 'unreviewed'",
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
    sql.append(" ORDER BY asset_id")
    return "\n".join(sql), params


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
    return CurationProgress(
        reviewed=reviewed,
        accepted=accepted,
        rejected=rejected,
        remaining=max(0, DEFAULT_TARGET - reviewed),
        target=DEFAULT_TARGET,
        by_style=_style_breakdown(state.connection),
        by_scope=_scope_breakdown(state.connection),
    )


# ----------------------------------------------------------------- next


@router.get("/next", response_model=CurationCandidate)
def next_candidate(
    state: State,
    style: Annotated[PrimaryStyle | None, Query(description="Filter by primary style")] = None,
    scope: Annotated[ScopeLabel | None, Query(description="Filter by scope bucket")] = None,
) -> CurationCandidate:
    if state.gallery is None:
        # The Milestone 2 dataset pipeline hasn't loaded a gallery yet; refuse
        # rather than serve a synthetic candidate.
        raise not_found(
            "gallery_unavailable",
            "no gallery loaded; set LINESCOUT_GALLERY_MANIFEST and restart the API",
        )
    sql, params = _candidate_base_query(style, scope)
    rows = state.connection.execute(sql, params).fetchall()
    chosen = _pick_candidate(rows, style, scope)
    if chosen is None:
        raise not_found("queue_empty", "no candidates are awaiting review")
    return _build_candidate(chosen, state.connection)


@router.get("/assets/{asset_id}/{kind}", response_class=FileResponse)
def preview_asset(state: State, asset_id: str, kind: str) -> FileResponse:
    """Serve unreviewed (and other) gallery files to the curation UI only.

    Public ``/api/v1/assets/{id}/...`` routes stay gated on the derived
    ``enabled = 1`` flag (the full v2 serving predicate).
    """
    column = _PREVIEW_KINDS.get(kind)
    if column is None or state.gallery is None:
        raise not_found("asset_not_found", "asset not found")
    row = state.connection.execute(
        f"SELECT {column} AS path FROM assets WHERE asset_id = ?"  # noqa: S608
        " AND (sfw_verdict IS NULL OR sfw_verdict != 'unsafe')"
        " AND (sfw_human_safe IS NULL OR sfw_human_safe = 1)",
        (asset_id,),
    ).fetchone()
    if row is None:
        raise not_found("asset_not_found", "asset not found")
    path = (state.gallery.data_root / str(row["path"])).resolve()
    if state.gallery.data_root.resolve() not in path.parents or not path.is_file():
        raise not_found("asset_unavailable", "asset file is missing")
    return FileResponse(
        path, media_type="image/png", headers={"Cache-Control": "private, max-age=60"}
    )


# ----------------------------------------------------------------- labels


def _crop_fits(crop: CropBox, width: int, height: int) -> bool:
    """True when the crop rectangle lies entirely inside the image."""
    return crop.x + crop.width <= width and crop.y + crop.height <= height


@router.post("/labels", response_model=LabelResponse, status_code=201)
def write_label(state: State, body: LabelRequest) -> LabelResponse:
    if state.gallery is None:
        raise not_found(
            "gallery_unavailable",
            "no gallery loaded; set LINESCOUT_GALLERY_MANIFEST and restart the API",
        )
    row = state.connection.execute(
        "SELECT asset_id, review_state, review_quality, enabled, sfw_verdict,"
        " sfw_human_safe, primary_scope, width, height"
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
            "cannot label an SFW-unsafe asset; quarantine happens at ingestion",
        )
    if body.crop is not None and not _crop_fits(body.crop, int(row["width"]), int(row["height"])):
        raise unprocessable(
            "crop_out_of_bounds",
            "crop rectangle exceeds the asset's width/height",
            field="crop",
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
    crop_json = body.crop.model_dump_json() if body.crop else None
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
    decided_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    # Single transaction: write the audit row, then mirror the decision into
    # the assets cache. The schema's CHECK constraints guard style, scope, and
    # split membership, so a typo from the UI surfaces as a SQLite IntegrityError.
    cursor = state.connection.execute("BEGIN")
    try:
        cursor = state.connection.execute(
            "INSERT INTO curation_labels ("
            " asset_id, decision, primary_style, primary_scope, secondary_scopes_json,"
            " crop_json, blockers_json, sfw_safe, quality, note, reviewer"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                body.asset_id,
                body.decision,
                body.primary_style.value if body.primary_style else None,
                primary_scope,
                secondary_json,
                crop_json,
                blockers_json,
                body.sfw_safe,
                body.quality,
                body.note,
                reviewer,
            ),
        )
        label_id = int(cursor.lastrowid or 0)
        cursor = state.connection.execute(
            "UPDATE assets SET"
            " review_state = ?,"
            " review_quality = ?,"
            " blockers_json = ?,"
            " primary_style = COALESCE(?, primary_style),"
            " primary_scope = COALESCE(?, primary_scope),"
            " secondary_scopes_json = COALESCE(?, secondary_scopes_json),"
            " crop_json = COALESCE(?, crop_json),"
            " sfw_human_safe = COALESCE(?, sfw_human_safe),"
            " sfw_human_reviewer = CASE WHEN ? IS NULL THEN sfw_human_reviewer ELSE ? END,"
            " sfw_human_decided_at = CASE WHEN ? IS NULL THEN sfw_human_decided_at ELSE ? END"
            " WHERE asset_id = ? AND review_state = ?",
            (
                review_state,
                review_quality,
                blockers_json,
                body.primary_style.value if body.primary_style else None,
                primary_scope,
                secondary_json,
                crop_json,
                body.sfw_safe,
                body.sfw_safe,
                reviewer,
                body.sfw_safe,
                decided_at,
                body.asset_id,
                body.expected_review_state.value,
            ),
        )
        if cursor.rowcount == 0:
            current = state.connection.execute(
                "SELECT review_state FROM assets WHERE asset_id = ?",
                (body.asset_id,),
            ).fetchone()
            if current is None:
                raise not_found("asset_not_found", f"asset {body.asset_id} is not in the gallery")
            raise conflict(
                "review_conflict",
                "asset review_state changed since the candidate was loaded",
                field="expected_review_state",
            )
        if body.secondary_scopes is not None and primary_scope is not None:
            state.connection.execute(
                "DELETE FROM asset_scopes WHERE asset_id = ?", (body.asset_id,)
            )
            state.connection.executemany(
                "INSERT INTO asset_scopes(asset_id, scope) VALUES (?, ?)",
                [
                    (body.asset_id, scope.value)
                    for scope in (ScopeLabel(primary_scope), *body.secondary_scopes)
                    if scope in GALLERY_SCOPES
                ],
            )
        # ``enabled`` is derived (the frozen is_servable predicate); recompute
        # it inside the same transaction so the mirror never drifts.
        enabled = recompute_enabled(state.connection, body.asset_id)
        blockers = serving_blockers(state.connection, body.asset_id) if not enabled else []
        # Audit row timestamp is the source of truth for clients; read it back
        # so the wire response matches what is stored.
        stamp = state.connection.execute(
            "SELECT created_at, sfw_safe FROM curation_labels WHERE id = ?", (label_id,)
        ).fetchone()
        state.connection.execute("COMMIT")
    except Exception:
        state.connection.execute("ROLLBACK")
        raise

    # Reload the in-memory enabled-asset list so /search and /assets/* reflect
    # the new serving flag without requiring a process restart.
    state.assets = enabled_assets(state.connection)

    return LabelResponse(
        id=label_id,
        asset_id=body.asset_id,
        decision=body.decision,
        review_state=review_state,
        review_quality=review_quality,
        blockers=body.blockers,
        sfw_human_approved=bool(stamp["sfw_safe"])
        if stamp and stamp["sfw_safe"] is not None
        else (bool(row["sfw_human_safe"]) if row["sfw_human_safe"] is not None else None),
        enabled=enabled,
        serving_blockers=blockers,
        created_at=str(stamp["created_at"]) if stamp else "",
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


# ----------------------------------------------------------------- snapshots


def _snapshot_dir(state: State) -> Path:
    """Resolve and ensure the snapshots directory exists under the data dir."""
    settings = state.settings
    base = settings.data_dir
    if not base.is_absolute():
        base = base.resolve()
    target = base / "snapshots"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _snapshot_timestamp(now: datetime | None = None) -> str:
    """``YYYYMMDD_HHMMSS_ffffff`` form, UTC — microsecond precision so two
    exports in the same second cannot collide on the filename."""
    moment = now or datetime.now(UTC)
    return moment.strftime("%Y%m%d_%H%M%S_%f")


def _snapshot_target(directory: Path, now: datetime | None = None) -> tuple[str, Path]:
    """Allocate a unique ``curation_<stamp>.json`` path under ``directory``."""
    stamp = _snapshot_timestamp(now)
    snapshot_id = f"curation_{stamp}"
    path = directory / f"{snapshot_id}.json"
    if path.exists():
        snapshot_id = f"curation_{stamp}_{uuid4().hex[:8]}"
        path = directory / f"{snapshot_id}.json"
    return snapshot_id, path


@router.post("/snapshots", response_model=SnapshotResponse, status_code=201)
def export_snapshot(state: State) -> SnapshotResponse:
    """Write an immutable **full** snapshot of the current curation state.

    Frozen semantics (v2): the snapshot captures the *latest* keep-or-reject
    label per asset — rejected assets included — at the moment of the export.
    It is a state view, not an export cursor: every POST writes a new file and
    never mutates a previous one, and the append-only ``curation_labels``
    audit history stays complete in the database. Lineage is recorded through
    ``previous_snapshot_id`` and the ``snapshots`` registry table.
    """
    if state.gallery is None:
        raise not_found(
            "gallery_unavailable",
            "no gallery loaded; set LINESCOUT_GALLERY_MANIFEST and restart the API",
        )

    # Latest label per asset (highest id among keep/reject decisions). This
    # includes rejected assets — quality-model training needs both classes.
    rows = state.connection.execute(
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

    target_dir = _snapshot_dir(state)
    snapshot_id, target_path = _snapshot_target(target_dir)
    previous = state.connection.execute(
        "SELECT snapshot_id FROM snapshots ORDER BY created_at DESC, snapshot_id DESC LIMIT 1"
    ).fetchone()
    previous_snapshot_id = str(previous["snapshot_id"]) if previous else None

    breakdown: dict[PrimaryStyle, int] = {style: 0 for style in STYLE_BUCKETS}
    serialized: list[dict[str, object]] = []
    for row in rows:
        primary = PrimaryStyle(row["primary_style"] or row["asset_primary_style"])
        breakdown[primary] += 1
        serialized.append(
            {
                "id": int(row["id"]),
                "asset_id": row["asset_id"],
                "decision": row["decision"],
                "primary_style": primary.value,
                "primary_scope": row["primary_scope"] or row["asset_primary_scope"],
                "secondary_scopes": json.loads(
                    row["secondary_scopes_json"] or row["asset_secondary_scopes_json"] or "[]"
                ),
                "crop": json.loads(row["crop_json"]) if row["crop_json"] else None,
                "blockers": json.loads(row["blockers_json"] or "[]"),
                "sfw_human_approved": (
                    bool(row["sfw_safe"]) if row["sfw_safe"] is not None else None
                ),
                "quality": int(row["quality"]) if row["quality"] is not None else None,
                "note": row["note"],
                "reviewer": row["reviewer"],
                "review_state": row["review_state"],
                "review_quality": int(row["review_quality"])
                if row["review_quality"] is not None
                else None,
                "created_at": row["created_at"],
            }
        )

    created_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    payload = {
        "schema_version": 2,
        "snapshot_id": snapshot_id,
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

    target_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("wrote curation snapshot %s (%d labels)", snapshot_id, len(serialized))

    # Register the snapshot for lineage, then stamp every never-exported label
    # with the *first* snapshot that captured it. That column is an audit
    # trail only — it is never used to filter what a future snapshot contains
    # (snapshots are full, not incremental).
    state.connection.execute("BEGIN")
    try:
        state.connection.execute(
            "INSERT INTO snapshots (snapshot_id, created_at, label_count,"
            " previous_snapshot_id, path) VALUES (?,?,?,?,?)",
            (
                snapshot_id,
                created_at,
                len(serialized),
                previous_snapshot_id,
                f"snapshots/{target_path.name}",
            ),
        )
        state.connection.execute(
            "UPDATE curation_labels SET snapshot_id = ?"
            " WHERE decision IN ('keep', 'reject') AND snapshot_id IS NULL",
            (snapshot_id,),
        )
        state.connection.execute("COMMIT")
    except Exception:
        state.connection.execute("ROLLBACK")
        raise

    return SnapshotResponse(
        snapshot_id=snapshot_id,
        path=f"snapshots/{target_path.name}",
        label_count=len(serialized),
        previous_snapshot_id=previous_snapshot_id,
        style_breakdown=breakdown,
        created_at=created_at,
    )
