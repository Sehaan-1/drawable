"""Provenance manifest schema (v3) for every source asset and derived crop.

One :class:`ManifestRecord` exists per gallery/training item. The manifest is
the contract between ``ml`` (which produces it) and ``services/api`` (which
loads it into SQLite and refuses to serve anything that fails validation).

Schema v3 is the *frozen* contract; the decisions and the field/invariant
matrix are written down in ``docs/contracts/manifest-v3.md`` and the mappings
from v1/v2 in ``docs/contracts/migration-manifest.md``. Version history:

* **v2** — separated scopes, learning vs membership, identity, per-use
  permissions, human SFW approval, named blockers, and artifact versioning.
* **v3** — introduces :class:`ArtifactContract` (the dataset's *current*
  artifact generation) and makes the public eligibility predicate one
  canonical, shared policy: accepted human review with quality 2–3, human SFW
  approval, permitted display use, gallery membership, no anatomy/extraction
  blockers, and **current valid derivatives**. Records whose derived artifacts
  (line art, thumbnail, automated labels) belong to an older generation are
  kept in the manifest for audit but are neither servable nor trainable until
  they are re-processed. ``permissions.basis`` is part of the predicate, so a
  record with unknown permission fails closed even if a stale cache row claims
  otherwise.

The policy functions here (``is_servable``, ``is_trainable``, ``is_gold``,
``derivative_reasons``) are the single source of truth; the SQLite layer and
the API mirror them, and ``docs/contracts/`` states the mirroring rules.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Annotated, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from linescout_ml.taxonomy import (
    GALLERY_SCOPES,
    CurationBlocker,
    LearningSplit,
    LineArtOrigin,
    PermissionBasis,
    PrimaryStyle,
    ReviewState,
    ScopeLabel,
    SfwScreeningMethod,
    SfwVerdict,
)

MANIFEST_SCHEMA_VERSION: Literal[3] = 3

AssetId = Annotated[str, StringConstraints(pattern=r"^ls_[a-z0-9]{2,16}_[a-f0-9]{16}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
PHash = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{16}$")]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
RelativePath = Annotated[str, StringConstraints(min_length=1, max_length=512)]
#: Free-form identity strings for artist / leakage group. ``None`` means
#: "unknown" — an identifier is never invented to fill the gap.
IdentityRef = Annotated[str, StringConstraints(min_length=1, max_length=128)]
#: ISO-8601 timestamp (UTC, ``...Z``). ``None`` means the instant is unknown.
Timestamp = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")]


class SplitIdentityRecord(Protocol):
    """The identity surface leak checks need — shared by v2/v3 frozen shapes."""

    asset_id: str
    source_work_id: str
    learning_split: LearningSplit
    leakage_group_id: str | None
    artist_id: str | None
    parent_asset_id: str | None


class CropBox(BaseModel):
    """Crop coordinates in source-image pixel space (inclusive-exclusive)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class AllowedUses(BaseModel):
    """What the asset may be used for. Three independent grants, all opt-in.

    A use is only ``True`` when the permission basis justifies it (or a
    recorded human permission decision does). Anything unknown stays ``False``:
    unknown permission must not imply permission.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: May be served in the app's reference panel (display / retrieval).
    display: bool = False
    #: May be used to train or fine-tune models.
    training: bool = False
    #: May be placed on the trace layer. Requires ``display``.
    trace: bool = False


class Permissions(BaseModel):
    """Permission provenance for one asset.

    ``license_id`` has no default — provenance is the point of the manifest,
    and a silent default would let an unverified licence claim ship. ``basis``
    defaults to ``unknown``, which grants no uses at all.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    license_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    basis: PermissionBasis = PermissionBasis.UNKNOWN
    #: Where the permission is recorded (licence page, consent mail, …).
    permission_url: Annotated[str, StringConstraints(max_length=2048)] | None = None
    #: Credit line to display when the licence requires attribution.
    attribution: Annotated[str, StringConstraints(max_length=256)] | None = None
    attribution_required: bool = False


class SfwScreening(BaseModel):
    """An automated (machine) SFW screen. Never a substitute for approval.

    ``None`` on the record means no screen was ever run. A screen result is
    tri-state: ``unsure`` is a concern to resolve, not a pass.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: SfwVerdict
    #: Probability the content is SFW per the screen, when the method
    #: produces one; ``None`` for method ``source_rating`` / ``none``.
    confidence: UnitInterval | None = None
    method: SfwScreeningMethod = SfwScreeningMethod.NONE


class SfwHumanDecision(BaseModel):
    """A human SFW decision. The only thing that can gate display."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    safe: bool
    reviewer: Annotated[str, StringConstraints(min_length=1, max_length=64)] = "local"
    #: When the decision was made; ``None`` when the instant is unknown
    #: (e.g. carried by migration from a v1 ``manual`` decision).
    decided_at: Timestamp | None = None


class HumanReview(BaseModel):
    """Human curation state plus named, use-blocking defects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: ReviewState = ReviewState.UNREVIEWED
    quality: Literal[1, 2, 3] | None = None
    #: Named blockers (anatomy, extraction). Any blocker blocks serving and
    #: training regardless of state; the curation API refuses ``keep`` with
    #: blockers. Records migrated from v1 may carry accepted+blocker rows —
    #: they stay accepted but are not servable or trainable.
    blockers: list[CurationBlocker] = Field(default_factory=list)
    note: str | None = Field(default=None, max_length=500)

    @field_validator("blockers")
    @classmethod
    def _unique_blockers(cls, value: list[CurationBlocker]) -> list[CurationBlocker]:
        if len(set(value)) != len(value):
            msg = "blockers must be unique"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _quality_only_when_reviewed(self) -> Self:
        if self.state is ReviewState.UNREVIEWED and self.quality is not None:
            msg = "quality cannot be set on an unreviewed asset"
            raise ValueError(msg)
        return self


class ArtifactContract(BaseModel):
    """The dataset's *current* artifact generation (schema v3).

    Every derived artifact on a :class:`ManifestRecord` (line art, thumbnail,
    automated labels) was produced under the ``pipeline_version`` /
    ``label_version`` / ``processing_revision`` stored on that record. This
    contract declares which generation is *current* for the dataset. A record
    whose three version fields differ from the contract has stale derivatives:
    it stays in the manifest for audit, but is excluded from public serving
    and from training until it is re-processed (see :func:`is_servable`).

    A v3 manifest carries the contract as ``Manifest.artifact_contract``; a
    value of ``None`` means "generation unknown" and marks **every** record's
    derivatives as unverified (stale) — conservative migration output. The
    currency gate is thus always enforced inside a ``Manifest``; a bare-record
    predicate call without a contract also fails closed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pipeline_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    label_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    processing_revision: int = Field(default=1, ge=1)

    def describe(self) -> str:
        return f"{self.pipeline_version}/l{self.label_version}/r{self.processing_revision}"


class ManifestRecord(BaseModel):
    """One gallery or training asset (schema v3).

    See ``docs/contracts/manifest-v3.md`` for the full field/invariant matrix.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # ------------------------------------------------------------------
    # Identity and provenance. Unknown values are explicit ``None``;
    # identifiers are never invented.
    # ------------------------------------------------------------------
    asset_id: AssetId
    source_dataset: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    source_item_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    #: The source work (page, sketchbook sheet, one artist's batch). Assets
    #: from one work never cross learning splits.
    source_work_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    #: Derivative identity: the asset this crop/panel was cut from. ``None``
    #: means the asset is not a derivative.
    parent_asset_id: AssetId | None = None
    #: Artist identity as declared by the source. ``None`` = unknown.
    artist_id: IdentityRef | None = None
    #: Leakage group: the finest identity across which splits must not leak.
    #: ``None`` = unknown, in which case ``source_work_id`` is the de-facto
    #: leakage group.
    leakage_group_id: IdentityRef | None = None
    source_url: str | None = Field(default=None, max_length=2048)

    # ------------------------------------------------------------------
    # Permission provenance and allowed uses (independent grants).
    # ------------------------------------------------------------------
    permissions: Permissions
    allowed_uses: AllowedUses = Field(default_factory=AllowedUses)

    # ------------------------------------------------------------------
    # Local files (relative to the data root; never absolute).
    # ------------------------------------------------------------------
    original_path: RelativePath
    line_art_path: RelativePath
    thumbnail_path: RelativePath

    # ------------------------------------------------------------------
    # Line-art origin.
    # ------------------------------------------------------------------
    origin: LineArtOrigin
    extraction_model: str | None = Field(default=None, max_length=64)
    extraction_version: str | None = Field(default=None, max_length=32)

    # ------------------------------------------------------------------
    # Labels.
    # ------------------------------------------------------------------
    primary_style: PrimaryStyle
    #: Exactly one primary scope. ``unknown`` is a legal provisional value
    #: ("no scope confidently determined"); it must be resolved before the
    #: asset can be accepted or gold.
    primary_scope: ScopeLabel
    #: Additional scopes; unique, never ``unknown``, never the primary.
    secondary_scopes: list[ScopeLabel] = Field(default_factory=list)
    #: ``None`` for non-human sketches (still-life doodles, objects) where a
    #: person count is not meaningful, or when not yet assessed.
    #: ``multi_character`` still requires ``>= 2``.
    person_count: int | None = Field(default=None, ge=0, le=50)
    #: ``True`` when ``person_count`` is an estimate rather than an exact
    #: count (e.g. a crowd). Never ``True`` with a null count.
    person_count_approximate: bool = False

    # ------------------------------------------------------------------
    # SFW: automated screen and human approval are separate facts.
    # ------------------------------------------------------------------
    sfw_screening: SfwScreening | None = None
    sfw_human: SfwHumanDecision | None = None

    # ------------------------------------------------------------------
    # Geometry.
    # ------------------------------------------------------------------
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    crop: CropBox | None = None

    # ------------------------------------------------------------------
    # Automated measurements (measured on the line-art view).
    # ------------------------------------------------------------------
    text_coverage: UnitInterval
    ink_coverage: UnitInterval
    phash: PHash
    quality_score: UnitInterval

    # ------------------------------------------------------------------
    # Curation, learning, and membership.
    # ------------------------------------------------------------------
    review: HumanReview = Field(default_factory=HumanReview)
    learning_split: LearningSplit
    #: Included in the gallery the API serves (subject to the other gates).
    gallery_member: bool = True
    #: Human-verified gold labels used for evaluation. Never gained by
    #: default or migration; requires accepted review with a quality grade.
    gold_member: bool = False

    # ------------------------------------------------------------------
    # Pipeline and artifact versioning.
    # ------------------------------------------------------------------
    pipeline_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    #: Generation of derived artifacts for this asset. Must increment when
    #: the line-art or thumbnail bytes are re-derived.
    processing_revision: int = Field(default=1, ge=1)
    #: Version of the labeling contract that produced the automated labels
    #: (prompt sets, models, taxonomy). Human decisions are append-only and
    #: survive a bump; automated labels do not.
    label_version: Annotated[str, StringConstraints(min_length=1, max_length=32)] = "1"
    source_checksum: Sha256
    line_art_checksum: Sha256
    thumbnail_checksum: Sha256

    @field_validator("original_path", "line_art_path", "thumbnail_path")
    @classmethod
    def _relative_posix_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            msg = f"path must be relative to the data root and contain no '..': {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("secondary_scopes")
    @classmethod
    def _secondary_scopes_rules(cls, value: list[ScopeLabel]) -> list[ScopeLabel]:
        if len(set(value)) != len(value):
            msg = "secondary_scopes must be unique"
            raise ValueError(msg)
        bad = [scope for scope in value if scope not in GALLERY_SCOPES]
        if bad:
            msg = f"secondary_scopes cannot carry query-only or unknown scopes: {bad}"
            raise ValueError(msg)
        return value

    @property
    def scopes(self) -> list[ScopeLabel]:
        """Primary first, then secondaries — the order clients display."""
        return [self.primary_scope, *self.secondary_scopes]

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> Self:
        # Line-art provenance.
        if self.origin is LineArtOrigin.EXTRACTED:
            if not (self.extraction_model and self.extraction_version):
                msg = "extracted assets must record extraction_model and extraction_version"
                raise ValueError(msg)
        elif self.extraction_model or self.extraction_version:
            msg = "native assets must not record an extraction model"
            raise ValueError(msg)

        # Scope structure.
        if self.primary_scope in self.secondary_scopes:
            msg = f"primary_scope {self.primary_scope.value} must not repeat in secondary_scopes"
            raise ValueError(msg)
        if ScopeLabel.MULTI_CHARACTER in self.scopes and (
            self.person_count is None or self.person_count < 2
        ):
            msg = "multi_character assets must have person_count >= 2"
            raise ValueError(msg)
        if self.person_count_approximate and self.person_count is None:
            msg = "person_count_approximate requires a person_count"
            raise ValueError(msg)

        # Derivative identity: an asset is never its own parent.
        if self.parent_asset_id is not None and self.parent_asset_id == self.asset_id:
            msg = "parent_asset_id must not equal asset_id"
            raise ValueError(msg)

        # Acceptance implies confirmed labels: an unknown primary scope cannot
        # be accepted (the reviewer would be approving a non-label).
        if self.review.state is ReviewState.ACCEPTED and self.primary_scope is ScopeLabel.UNKNOWN:
            msg = "accepted assets must have a known primary_scope"
            raise ValueError(msg)

        # Gold implies verified labels: accepted, graded 2–3, unblocked, a
        # known (non-unknown) primary scope, and human SFW approval.
        if self.gold_member:
            if self.review.state is not ReviewState.ACCEPTED:
                msg = "gold_member requires an accepted review"
                raise ValueError(msg)
            if self.review.quality is None:
                msg = "gold_member requires a review quality grade"
                raise ValueError(msg)
            if self.review.quality < 2:
                msg = "gold_member requires review quality 2 or 3"
                raise ValueError(msg)
            if self.review.blockers:
                msg = "gold_member cannot carry blockers"
                raise ValueError(msg)
            if self.primary_scope is ScopeLabel.UNKNOWN:
                msg = "gold_member requires a known primary_scope"
                raise ValueError(msg)
            if self.sfw_human is None or not self.sfw_human.safe:
                msg = "gold_member requires human SFW approval"
                raise ValueError(msg)

        # Permission grants must be justified.
        if self.permissions.basis is PermissionBasis.UNKNOWN and (
            self.allowed_uses.display or self.allowed_uses.training or self.allowed_uses.trace
        ):
            msg = "allowed uses require a permission basis other than 'unknown'"
            raise ValueError(msg)
        if self.allowed_uses.trace and not self.allowed_uses.display:
            msg = "trace use requires display use"
            raise ValueError(msg)
        if self.permissions.attribution_required and not self.permissions.attribution:
            msg = "attribution is required by the licence but no credit line is recorded"
            raise ValueError(msg)

        # Geometry.
        if min(self.width, self.height) < 256:
            msg = "short edge must be at least 256 px"
            raise ValueError(msg)
        if self.crop is not None and (
            self.crop.x + self.crop.width > self.width
            or self.crop.y + self.crop.height > self.height
        ):
            msg = "crop box exceeds image bounds"
            raise ValueError(msg)
        return self


def derivative_reasons(record: ManifestRecord, contract: ArtifactContract | None) -> list[str]:
    """Why the record's derived artifacts are not current, as stable reasons.

    Returns an empty list only when the record's generation is verified as
    current. A ``None`` contract means "generation unknown" and fails closed
    (``derivative_generation_unknown``). The codes are stable machine-readable
    strings: ``derivative_generation_unknown``,
    ``derivative_stale:pipeline_version``, ``derivative_stale:label_version``,
    ``derivative_stale:processing_revision``.

    File-level validity (missing file, checksum mismatch) is *not* a manifest
    property — the gallery loader verifies bytes against the recorded hashes
    and reports those as separate reasons (``derivative_file_missing`` /
    ``derivative_checksum_mismatch``).
    """
    if contract is None:
        return ["derivative_generation_unknown"]
    reasons: list[str] = []
    if record.pipeline_version != contract.pipeline_version:
        reasons.append("derivative_stale:pipeline_version")
    if record.label_version != contract.label_version:
        reasons.append("derivative_stale:label_version")
    if record.processing_revision != contract.processing_revision:
        reasons.append("derivative_stale:processing_revision")
    return reasons


def serving_reasons(record: ManifestRecord, contract: ArtifactContract | None) -> list[str]:
    """Every reason a record is not publicly servable, in stable order.

    This is the *canonical* public eligibility policy. A record is publicly
    servable iff this list is empty:

    * ``gallery_member`` is true;
    * ``allowed_uses.display`` is true and ``permissions.basis`` is known
      (unknown permission grants nothing — fail closed);
    * ``review.state`` is ``accepted``;
    * ``review.quality`` is 2 or 3 (quality 1 or a missing grade never serves);
    * no anatomy/extraction blockers;
    * ``sfw_human`` exists and is ``safe`` — an automated ``safe`` screen alone
      never publishes an asset;
    * the derived artifacts are current under ``contract`` (stale derivatives
      cannot be served or searched).
    """
    reasons: list[str] = []
    if not record.gallery_member:
        reasons.append("not_a_gallery_member")
    if not record.allowed_uses.display:
        reasons.append("display_not_permitted")
    if record.permissions.basis is PermissionBasis.UNKNOWN:
        reasons.append("permission_unknown")
    if record.review.state is not ReviewState.ACCEPTED:
        reasons.append(f"review_{record.review.state.value}")
    if record.review.quality is None:
        reasons.append("quality_missing")
    elif record.review.quality < 2:
        reasons.append("quality_below_floor")
    for blocker in record.review.blockers:
        reasons.append(f"blocker_{blocker.value}")
    if record.sfw_human is None:
        reasons.append("sfw_human_unapproved")
    elif not record.sfw_human.safe:
        reasons.append("sfw_human_unsafe")
    reasons.extend(derivative_reasons(record, contract))
    return reasons


def is_servable(record: ManifestRecord, contract: ArtifactContract | None = None) -> bool:
    """The canonical serving predicate (replaces the v1 stored ``enabled`` flag).

    An asset may appear in search results and be served through
    ``/assets/{id}/...`` iff it is a gallery member, display use is permitted
    under a known permission basis, a human accepted it with quality 2 or 3,
    a human approved it as SFW, it carries no blockers, and its derived
    artifacts are current under ``contract``. An automated ``safe`` screening
    verdict is *not* sufficient — only ``sfw_human`` approval gates display.
    """
    return not serving_reasons(record, contract)


def training_reasons(record: ManifestRecord, contract: ArtifactContract | None) -> list[str]:
    """Why a record is not trainable, as stable reasons (mirror of serving)."""
    reasons: list[str] = []
    if not record.allowed_uses.training:
        reasons.append("training_not_permitted")
    if record.permissions.basis is PermissionBasis.UNKNOWN:
        reasons.append("permission_unknown")
    if record.learning_split is not LearningSplit.TRAIN:
        reasons.append(f"split_{record.learning_split.value}")
    if record.review.state is not ReviewState.ACCEPTED:
        reasons.append(f"review_{record.review.state.value}")
    for blocker in record.review.blockers:
        reasons.append(f"blocker_{blocker.value}")
    if record.sfw_screening is not None and record.sfw_screening.verdict is SfwVerdict.UNSAFE:
        reasons.append("sfw_screen_unsafe")
    reasons.extend(derivative_reasons(record, contract))
    return reasons


def is_trainable(record: ManifestRecord, contract: ArtifactContract | None = None) -> bool:
    """The canonical training predicate.

    Training requires the training grant under a known permission basis, a
    ``train`` learning assignment, an accepted review without blockers, an
    automated SFW screen that is not ``unsafe`` (a human accepted the asset;
    the screen just must not object), and current derived artifacts.
    """
    return not training_reasons(record, contract)


def gold_reasons(record: ManifestRecord, contract: ArtifactContract | None) -> list[str]:
    """Why a record is not gold-eligible, as stable reasons.

    Gold membership has its own documented approval/quality conditions: an
    accepted human review graded 2 or 3, no blockers, a known primary scope,
    explicit human SFW approval, and current derived artifacts (gold labels
    are evaluated against the current generation's bytes).
    """
    reasons: list[str] = []
    if record.review.state is not ReviewState.ACCEPTED:
        reasons.append("gold_review_not_accepted")
    if record.review.quality is None:
        reasons.append("gold_quality_missing")
    elif record.review.quality < 2:
        reasons.append("gold_quality_below_floor")
    if record.review.blockers:
        reasons.append("gold_blockers_present")
    if record.primary_scope is ScopeLabel.UNKNOWN:
        reasons.append("gold_scope_unknown")
    if record.sfw_human is None or not record.sfw_human.safe:
        reasons.append("gold_sfw_human_unapproved")
    reasons.extend(
        f"gold_{reason}" if reason.startswith("derivative") else reason
        for reason in derivative_reasons(record, contract)
    )
    return reasons


def is_gold(record: ManifestRecord, contract: ArtifactContract | None = None) -> bool:
    """The canonical gold-membership predicate (schema v3 conditions)."""
    return not gold_reasons(record, contract)


class Manifest(BaseModel):
    """A versioned collection of records plus the current artifact contract.

    Schema v3: the manifest declares the artifact generation that is current
    (``artifact_contract``); a record whose own generation fields differ is
    kept for audit but is not servable or trainable until re-processed.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[3] = MANIFEST_SCHEMA_VERSION
    dataset_version: Annotated[
        str, StringConstraints(pattern=r"^\d{4}\.\d{2}\.\d{2}(-[a-z0-9]+)?$")
    ]
    #: The current artifact generation. ``null`` means the generation is
    #: unknown (e.g. a conservatively converted legacy manifest): no record is
    #: verified current, so nothing servable or trainable until an operator
    #: declares the current generation or re-processes the dataset.
    artifact_contract: ArtifactContract | None
    records: list[ManifestRecord]

    @model_validator(mode="after")
    def _unique_ids_and_integrity(self) -> Self:
        seen_ids: set[str] = set()
        for record in self.records:
            if record.asset_id in seen_ids:
                msg = f"duplicate asset_id {record.asset_id}"
                raise ValueError(msg)
            seen_ids.add(record.asset_id)
        problems = check_split_integrity(self.records)
        if problems:
            msg = "; ".join(problems[:5])
            raise ValueError(msg)
        problems = check_parent_integrity(self.records)
        if problems:
            msg = "; ".join(problems[:5])
            raise ValueError(msg)
        # Gold membership must satisfy the v3 conditions against the manifest
        # contract — a stale generation invalidates gold labels.
        for record in self.records:
            gold_problems = gold_reasons(record, self.artifact_contract)
            if record.gold_member and gold_problems:
                msg = f"gold_member {record.asset_id} violates gold eligibility: " + ", ".join(
                    gold_problems
                )
                raise ValueError(msg)
        return self

    @property
    def servable_records(self) -> list[ManifestRecord]:
        return [record for record in self.records if is_servable(record, self.artifact_contract)]

    def content_hash(self) -> str:
        """Stable hash of the manifest content, used as the index version key."""
        payload = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_split_integrity(records: Iterable[SplitIdentityRecord]) -> list[str]:
    """Violations of the one-group-one-split rule across every identity axis.

    A violation is two *different assigned* splits (train/validation/test)
    sharing any of these pooling keys:

    * ``source_work_id`` — the artistic work (the de-facto leakage group);
    * a non-null ``leakage_group_id`` (finest declared identity);
    * a non-null ``artist_id`` — one artist's works never straddle splits;
    * one parent chain — a derivative and its ancestors depict the same art
      and must never cross splits. Every record is keyed by its chain's root
      asset id (a root record keys by its own id), so a derivative whose
      ``source_work_id`` disagrees with its ancestor is still caught.

    ``none`` never conflicts — an unassigned asset leaks nothing. Two records
    that are *unassigned* (``none``) may share artists/groups freely.
    """
    record_list = list(records)
    by_id = {record.asset_id: record for record in record_list}
    works: dict[str, set[LearningSplit]] = {}
    groups: dict[str, set[LearningSplit]] = {}
    artists: dict[str, set[LearningSplit]] = {}
    chains: dict[str, set[LearningSplit]] = {}

    def record_split(record: SplitIdentityRecord) -> LearningSplit | None:
        return None if record.learning_split is LearningSplit.NONE else record.learning_split

    def split_name(splits: set[LearningSplit]) -> str:
        return ", ".join(sorted(split.value for split in splits))

    for record in record_list:
        split = record_split(record)
        if split is None:
            continue
        works.setdefault(record.source_work_id, set()).add(split)
        if record.leakage_group_id is not None:
            groups.setdefault(record.leakage_group_id, set()).add(split)
        if record.artist_id is not None:
            artists.setdefault(record.artist_id, set()).add(split)
        # Every record is pooled by its chain root — including a root record
        # itself (no parent): a derivative may name a different source work
        # than its ancestor, so the *chain* is the identity that must not
        # straddle splits, not the work fields the two sides happen to carry.
        root = _parent_chain_root(record, by_id)
        chains.setdefault(root if root is not None else record.asset_id, set()).add(split)

    problems = [
        f"source work {work!r} spans splits {split_name(splits)}"
        for work, splits in sorted(works.items())
        if len(splits) > 1
    ]
    problems += [
        f"leakage group {group!r} spans splits {split_name(splits)}"
        for group, splits in sorted(groups.items())
        if len(splits) > 1
    ]
    problems += [
        f"artist {artist!r} spans splits {split_name(splits)}"
        for artist, splits in sorted(artists.items())
        if len(splits) > 1
    ]
    problems += [
        f"parent chain rooted at {root[:12]} spans splits {split_name(splits)}"
        for root, splits in sorted(chains.items())
        if len(splits) > 1
    ]
    return problems


def _parent_chain_root(
    record: SplitIdentityRecord, by_id: dict[str, SplitIdentityRecord]
) -> str | None:
    """The root ancestor's asset id for a record's parent chain, or ``None``.

    Records without a parent are not pooled (their chain is just themselves;
    the work/artist/group keys already cover them). A record whose parent is
    not in the map (a dangling reference) returns ``None`` here —
    :func:`check_parent_integrity` reports that separately.
    """
    if record.parent_asset_id is None:
        return None
    seen: set[str] = set()
    cursor = record
    while cursor.parent_asset_id is not None:
        if cursor.parent_asset_id in seen or cursor.parent_asset_id == cursor.asset_id:
            break
        seen.add(cursor.parent_asset_id)
        parent = by_id.get(cursor.parent_asset_id)
        if parent is None:
            break
        cursor = parent
    return cursor.asset_id


def check_parent_integrity(records: Iterable[SplitIdentityRecord]) -> list[str]:
    """Derivative-graph problems: dangling parents, self-parents, cycles.

    Accepts the shared :class:`SplitIdentityRecord` shape so the frozen v2
    conversion shapes are checked with the same rules as v3 records.
    """
    record_list = list(records)
    by_id = {record.asset_id: record for record in record_list}
    problems: list[str] = []
    for record in record_list:
        parent = record.parent_asset_id
        if parent is None:
            continue
        if parent == record.asset_id:
            problems.append(f"{record.asset_id}: parent_asset_id equals asset_id")
            continue
        if parent not in by_id:
            problems.append(f"{record.asset_id}: parent_asset_id {parent} is not in the manifest")
            continue
        # Walk the parent chain; a revisit means a cycle.
        seen = {record.asset_id}
        cursor = by_id[parent]
        while cursor.parent_asset_id is not None:
            if cursor.parent_asset_id in seen:
                problems.append(
                    f"{record.asset_id}: parent chain contains a cycle at {cursor.parent_asset_id}"
                )
                break
            seen.add(cursor.parent_asset_id)
            ancestor = by_id.get(cursor.parent_asset_id)
            if ancestor is None:
                # Reported already from the dangling side; do not double-report.
                break
            cursor = ancestor
    return problems


def learning_split_report(records: Iterable[ManifestRecord]) -> dict[str, object]:
    """Split proportions over *source works* for the 70/15/15 policy check.

    The policy is a build-time target, not a per-manifest hard invariant, so
    this returns a report (proportions and drift vs the 70/15/15 target) that
    the CLI prints and the ingestion pipeline records in its run report.
    """
    record_list = list(records)
    assigned: dict[LearningSplit, int] = {
        LearningSplit.TRAIN: 0,
        LearningSplit.VALIDATION: 0,
        LearningSplit.TEST: 0,
    }
    works: dict[str, set[LearningSplit]] = {}
    for record in record_list:
        works.setdefault(record.source_work_id, set()).add(record.learning_split)
    for splits in works.values():
        for split in splits:
            if split is not LearningSplit.NONE:
                assigned[split] += 1
    works_total = len(works)
    assigned_total = sum(assigned.values())
    target = {
        LearningSplit.TRAIN: 0.70,
        LearningSplit.VALIDATION: 0.15,
        LearningSplit.TEST: 0.15,
    }
    proportions = {
        split.value: (round(count / assigned_total, 4) if assigned_total else 0.0)
        for split, count in assigned.items()
    }
    drift = {
        split.value: round(proportions[split.value] - share, 4) for split, share in target.items()
    }
    return {
        "works_total": works_total,
        "works_assigned": assigned_total,
        "works_unassigned": works_total - assigned_total,
        "counts": {split.value: count for split, count in assigned.items()},
        "proportions": proportions,
        "policy_target": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "drift": drift,
    }


def make_asset_id(source_dataset: str, source_item_id: str, crop: CropBox | None = None) -> str:
    """Deterministic project asset ID: ``ls_<dataset>_<16 hex>``."""
    dataset_slug = "".join(ch for ch in source_dataset.lower() if ch.isalnum())[:16] or "src"
    material = f"{source_dataset}\x00{source_item_id}"
    if crop is not None:
        material += f"\x00{crop.x},{crop.y},{crop.width},{crop.height}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"ls_{dataset_slug}_{digest}"


def manifest_json_schema() -> dict[str, object]:
    return Manifest.model_json_schema()


def dump_json_schema() -> str:
    return json.dumps(manifest_json_schema(), indent=2, sort_keys=True) + "\n"
