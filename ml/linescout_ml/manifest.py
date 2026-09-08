"""Provenance manifest schema (v2) for every source asset and derived crop.

One :class:`ManifestRecord` exists per gallery/training item. The manifest is
the contract between ``ml`` (which produces it) and ``services/api`` (which
loads it into SQLite and refuses to serve anything that fails validation).

Schema v2 is the *frozen* contract; the decisions and the field/invariant
matrix are written down in ``docs/contracts/manifest-v2.md`` and the mapping
from v1 in ``docs/contracts/migration-v1-to-v2.md``. The headline changes:

* **Scopes** — one explicit ``primary_scope`` plus ``secondary_scopes``
  (``unknown`` is a legal provisional primary, never a secondary).
* **Learning vs membership** — ``learning_split`` (train/validation/test/none)
  is independent of ``gallery_member`` and ``gold_member``.
* **Identity** — ``parent_asset_id``, ``artist_id``, and ``leakage_group_id``
  join the existing source identity; unknown values are explicit ``None``,
  never invented identifiers.
* **Uses** — ``allowed_uses`` models display, training, and tracing
  independently; ``permissions`` records the basis, attribution, and
  provenance. Unknown permission grants nothing.
* **SFW** — automated ``sfw_screening`` (tri-state verdict) is separate from
  ``sfw_human`` approval; only the latter can gate display.
* **Review** — ``blockers`` (anatomy/extraction) are named, use-blocking
  defects; rejection and quarantine have explicit definitions.
* **Artifacts** — ``processing_revision`` and ``label_version`` version the
  derived artifacts and labels; the stored ``enabled`` flag is gone and
  replaced by the derived :func:`is_servable` predicate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

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

MANIFEST_SCHEMA_VERSION: Literal[2] = 2

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


class ManifestRecord(BaseModel):
    """One gallery or training asset (schema v2).

    See ``docs/contracts/manifest-v2.md`` for the full field/invariant matrix.
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
    #: SHA-256 of the checkpoint file the extractor actually loaded, so a gallery
    #: merged across extractor revisions still says which bytes drew each line.
    extraction_sha256: Sha256 | None = None

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

        # Gold implies verified labels: accepted, graded, unblocked.
        if self.gold_member:
            if self.review.state is not ReviewState.ACCEPTED:
                msg = "gold_member requires an accepted review"
                raise ValueError(msg)
            if self.review.quality is None:
                msg = "gold_member requires a review quality grade"
                raise ValueError(msg)
            if self.review.blockers:
                msg = "gold_member cannot carry blockers"
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


class CheckpointProvenance(BaseModel):
    """One pinned model artifact, as verified by the pipeline that used it.

    ``status`` is the whole point: ``verified`` means the bytes matched a
    published digest, ``recorded`` means the digest was captured on first use
    because the publisher exposes none, ``revision`` means only a repository
    commit pinned it, and ``unchecked`` means verification was switched off.
    A dataset that quietly mixed those three is the failure this file exists to
    prevent, so the difference is written down per artifact.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Annotated[str, StringConstraints(min_length=3, max_length=128)]
    group: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    #: Where the bytes came from: ``repo@commit/path`` or an absolute URL.
    locator: Annotated[str, StringConstraints(max_length=512)]
    sha256: Sha256 | None = None
    size_bytes: int | None = Field(default=None, ge=1)
    status: Literal["verified", "recorded", "revision", "unchecked"]
    pinned: bool


class PipelineProvenance(BaseModel):
    """Which code, environment, and weights produced this gallery.

    Identity only — no timestamps, because ``Manifest.content_hash()`` keys the
    derived indexes and must not move when nothing but the clock did.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = MANIFEST_SCHEMA_VERSION
    pipeline_version: Annotated[str, StringConstraints(min_length=1, max_length=32)] = "colab-m2-1"
    #: Repository that supplied ``linescout_ml``, e.g. ``junosapollo/drawable``.
    source_repo: Annotated[str, StringConstraints(max_length=128)] | None = None
    #: The commit HEAD was *verified* to equal — never a branch or a tag.
    source_revision: Annotated[str | None, StringConstraints(pattern=r"^[0-9a-f]{40}$")] = None
    #: The checkout carried local edits, so this run is not fully described by
    #: ``source_revision`` alone. Recorded, never hidden.
    source_dirty: bool | None = None
    source_action: Literal["cloned", "reused", "fetched", "failed"] | None = None
    #: SHA-256 of the pinned requirements file the run installed from.
    environment_sha256: Sha256 | None = None
    #: SHA-256 of ``models.lock.json`` these checkpoints were checked against.
    model_lock_sha256: Sha256 | None = None
    checkpoint_policy: Literal["strict", "record", "off"] = "record"
    #: ``package -> version`` actually importable in the producing runtime.
    runtime: dict[str, Annotated[str, StringConstraints(max_length=64)]] = Field(
        default_factory=dict, max_length=64
    )
    checkpoints: list[CheckpointProvenance] = Field(default_factory=list, max_length=64)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = self.model_dump(mode="json")
        return payload


def is_servable(record: ManifestRecord) -> bool:
    """The frozen serving predicate (replaces the v1 stored ``enabled`` flag).

    An asset may appear in search results and be served through
    ``/assets/{id}/...`` iff it is a gallery member, display use is permitted,
    a human accepted it with at least a quality-2 grade, a human approved it
    as SFW, and it carries no blockers. Note that an automated ``safe``
    screening verdict is *not* sufficient — only ``sfw_human`` approval gates
    display.
    """
    return (
        record.gallery_member
        and record.allowed_uses.display
        and record.review.state is ReviewState.ACCEPTED
        and record.review.quality is not None
        and record.review.quality >= 2
        and not record.review.blockers
        and record.sfw_human is not None
        and record.sfw_human.safe
    )


def is_trainable(record: ManifestRecord) -> bool:
    """The frozen training predicate.

    Training requires the training grant, a ``train`` learning assignment, an
    accepted review without blockers, and an automated SFW screen that is not
    ``unsafe`` (a human accepted the asset; the screen just must not object).
    """
    return (
        record.allowed_uses.training
        and record.learning_split is LearningSplit.TRAIN
        and record.review.state is ReviewState.ACCEPTED
        and not record.review.blockers
        and (record.sfw_screening is None or record.sfw_screening.verdict is not SfwVerdict.UNSAFE)
    )


class Manifest(BaseModel):
    """A versioned collection of records plus the dataset/index version stamp."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[2] = MANIFEST_SCHEMA_VERSION
    dataset_version: Annotated[
        str, StringConstraints(pattern=r"^\d{4}\.\d{2}\.\d{2}(-[a-z0-9]+)?$")
    ]
    records: list[ManifestRecord]
    #: Optional because a manifest is also hand-written and hand-curated: the
    #: pipeline always fills it in, and its absence says "nobody recorded".
    provenance: PipelineProvenance | None = None

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
        return self

    @property
    def servable_records(self) -> list[ManifestRecord]:
        return [record for record in self.records if is_servable(record)]

    def content_hash(self) -> str:
        """Stable hash of the manifest content, used as the index version key."""
        payload = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_split_integrity(records: Iterable[ManifestRecord]) -> list[str]:
    """Violations of the one-group-one-split rule over works and leakage groups.

    A violation is two *different assigned* splits (train/validation/test) on
    the same ``source_work_id`` or the same non-null ``leakage_group_id``.
    ``none`` never conflicts — an unassigned asset leaks nothing.
    """
    works: dict[str, set[LearningSplit]] = {}
    groups: dict[str, set[LearningSplit]] = {}
    for record in records:
        if record.learning_split is LearningSplit.NONE:
            continue
        works.setdefault(record.source_work_id, set()).add(record.learning_split)
        if record.leakage_group_id is not None:
            groups.setdefault(record.leakage_group_id, set()).add(record.learning_split)
    problems = [
        f"source work {work!r} spans splits {sorted(split.value for split in splits)}"
        for work, splits in sorted(works.items())
        if len(splits) > 1
    ]
    problems += [
        f"leakage group {group!r} spans splits {sorted(split.value for split in splits)}"
        for group, splits in sorted(groups.items())
        if len(splits) > 1
    ]
    return problems


def check_parent_integrity(records: Iterable[ManifestRecord]) -> list[str]:
    """Derivative-graph problems: dangling parents, self-parents, cycles."""
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
