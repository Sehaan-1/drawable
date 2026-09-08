"""Migrate manifest schema v1 records to the frozen v2 contract.

The v1 shapes below are **frozen copies** of the v1 schema — they exist so a
v1 manifest can be parsed and migrated long after the live models moved on.
The mapping rules are the normative ones documented in
``docs/contracts/migration-v1-to-v2.md``; the two safety properties this
module guarantees (and its tests assert) are:

1. **No old asset gains permission.** ``allowed_uses`` is all-``False`` after
   migration and ``permissions.basis`` is ``unknown``. A v1 ``enabled`` flag
   was a serving decision made before per-use permissions existed; it does not
   carry over as any v2 grant.
2. **No old asset gains human approval.** ``sfw_human`` is only created from a
   v1 ``sfw.method == "manual"`` decision (which *was* a human decision);
   automated screening methods become ``sfw_screening`` and nothing else.
   ``gold_member`` is always ``False`` after migration.

Run it through the CLI::

    linescout-manifest migrate-v1 manifest.json --out manifest.v2.json
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from linescout_ml.manifest import (
    AllowedUses,
    CropBox,
    HumanReview,
    Manifest,
    ManifestRecord,
    Permissions,
    SfwHumanDecision,
    SfwScreening,
    is_servable,
)
from linescout_ml.taxonomy import (
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

V1_SCREENING_METHODS: tuple[str, ...] = (
    "source_rating",
    "opennsfw2",
    "source_rating+opennsfw2",
)
#: Reviewer name recorded on human SFW decisions carried by migration, so the
#: decision's provenance ("came from a v1 manual entry") stays inspectable.
MIGRATION_REVIEWER = "migrate-v1"
#: v1 automated labels were produced by the first labeling contract.
V1_LABEL_VERSION = "1"


# --------------------------------------------------------------- frozen v1 shapes


class V1SfwDecision(BaseModel):
    """Frozen v1 shape. ``method="manual"`` was a human decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    safe: bool
    confidence: float = Field(ge=0.0, le=1.0)
    method: Literal["source_rating", "opennsfw2", "source_rating+opennsfw2", "manual"]


class V1HumanReview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state: ReviewState = ReviewState.UNREVIEWED
    quality: Literal[1, 2, 3] | None = None
    malformed_anatomy: bool = False
    poor_extraction: bool = False
    note: str | None = Field(default=None, max_length=500)


class V1ManifestRecord(BaseModel):
    """Frozen v1 record shape (``schema_version`` 1 manifests)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    asset_id: str
    source_dataset: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    source_item_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    source_work_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    source_url: str | None = Field(default=None, max_length=2048)
    license_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    original_path: str
    line_art_path: str
    thumbnail_path: str
    origin: LineArtOrigin
    extraction_model: str | None = Field(default=None, max_length=64)
    extraction_version: str | None = Field(default=None, max_length=32)
    primary_style: PrimaryStyle
    scopes: list[ScopeLabel] = Field(min_length=1)
    person_count: int | None = Field(default=None, ge=0, le=50)
    sfw: V1SfwDecision
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    crop: CropBox | None = None
    text_coverage: float = Field(ge=0.0, le=1.0)
    ink_coverage: float = Field(ge=0.0, le=1.0)
    phash: str
    quality_score: float = Field(ge=0.0, le=1.0)
    review: V1HumanReview = Field(default_factory=V1HumanReview)
    split: Literal["train", "validation", "test", "gallery_only"]
    enabled: bool
    pipeline_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    source_checksum: str
    line_art_checksum: str
    thumbnail_checksum: str


class V1Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    dataset_version: str
    records: list[V1ManifestRecord]


# ---------------------------------------------------------------------- report


class MigrationReport(BaseModel):
    """What the migration did — the machine-readable version of the mapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    records_migrated: int
    #: v1 ``sfw.method="manual"`` decisions carried into ``sfw_human``.
    sfw_human_carried: int
    #: v1 records with automated-only SFW evidence; they gained no approval.
    sfw_human_absent: int
    #: v1 ``enabled=true`` records that are *not* servable under v2
    #: (permission basis unknown and/or no human SFW approval).
    serving_eligibility_lost: int
    #: Safety properties — both must stay 0.
    uses_granted: int = 0
    human_approvals_fabricated: int = 0
    gold_members_created: int = 0
    #: old ``split`` value -> migrated record count.
    split_mapping: dict[str, int]
    blockers_created: dict[str, int]
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------- mapping


def _screening_from_v1(sfw: V1SfwDecision) -> SfwScreening | None:
    """Automated methods become a screening; ``manual`` was human, not a screen."""
    if sfw.method == "manual":
        return None
    return SfwScreening(
        verdict=SfwVerdict.SAFE if sfw.safe else SfwVerdict.UNSAFE,
        confidence=sfw.confidence,
        method=SfwScreeningMethod(sfw.method),
    )


def _human_from_v1(sfw: V1SfwDecision) -> SfwHumanDecision | None:
    """Only a v1 ``manual`` decision was made by a human; carry it, dated unknown."""
    if sfw.method != "manual":
        return None
    return SfwHumanDecision(safe=sfw.safe, reviewer=MIGRATION_REVIEWER, decided_at=None)


def _learning_split_from_v1(split: str) -> LearningSplit:
    """``gallery_only`` means "in the gallery, not in learning" → ``none``."""
    if split == "gallery_only":
        return LearningSplit.NONE
    return LearningSplit(split)


def migrate_record(record: V1ManifestRecord) -> ManifestRecord:
    """Map one v1 record to v2 under the frozen rules.

    Every grant-shaped field is deny-by-default; see the module docstring for
    the two safety properties.
    """
    # v1 wrote ``scopes`` in descending labeler score (the zero-shot stage
    # emitted its top-k in score order), so the first element is the primary
    # and the remainder are secondary.
    primary_scope = record.scopes[0]
    secondary_scopes = record.scopes[1:]

    blockers: list[CurationBlocker] = []
    if record.review.malformed_anatomy:
        blockers.append(CurationBlocker.ANATOMY)
    if record.review.poor_extraction:
        blockers.append(CurationBlocker.EXTRACTION)

    return ManifestRecord(
        asset_id=record.asset_id,
        source_dataset=record.source_dataset,
        source_item_id=record.source_item_id,
        source_work_id=record.source_work_id,
        # Unknown identity is explicit None — never invented.
        parent_asset_id=None,
        artist_id=None,
        leakage_group_id=None,
        source_url=record.source_url,
        # Permission provenance: licence carried verbatim; basis unknown, so
        # no uses are granted. v1 ``enabled`` does not become a display grant.
        permissions=Permissions(license_id=record.license_id, basis=PermissionBasis.UNKNOWN),
        allowed_uses=AllowedUses(display=False, training=False, trace=False),
        original_path=record.original_path,
        line_art_path=record.line_art_path,
        thumbnail_path=record.thumbnail_path,
        origin=record.origin,
        extraction_model=record.extraction_model,
        extraction_version=record.extraction_version,
        primary_style=record.primary_style,
        primary_scope=primary_scope,
        secondary_scopes=secondary_scopes,
        person_count=record.person_count,
        # v1 counts were asserted as exact; no approximation flag existed.
        person_count_approximate=False,
        sfw_screening=_screening_from_v1(record.sfw),
        sfw_human=_human_from_v1(record.sfw),
        width=record.width,
        height=record.height,
        crop=record.crop,
        text_coverage=record.text_coverage,
        ink_coverage=record.ink_coverage,
        phash=record.phash,
        quality_score=record.quality_score,
        review=HumanReview(
            state=record.review.state,
            quality=record.review.quality,
            blockers=blockers,
            note=record.review.note,
        ),
        learning_split=_learning_split_from_v1(record.split),
        # Every v1 manifest record was a gallery candidate by construction.
        gallery_member=True,
        # Gold membership never exists in v1 and is never created by migration.
        gold_member=False,
        pipeline_version=record.pipeline_version,
        processing_revision=1,
        label_version=V1_LABEL_VERSION,
        source_checksum=record.source_checksum,
        line_art_checksum=record.line_art_checksum,
        thumbnail_checksum=record.thumbnail_checksum,
    )


def migrate_manifest(data: Mapping[str, Any] | V1Manifest) -> tuple[Manifest, MigrationReport]:
    """Migrate a parsed v1 manifest document into v2 plus a report.

    Raises ``pydantic.ValidationError`` when the input is not a valid v1
    manifest, and ``ValueError`` when a migrated record violates a v2
    invariant (which would indicate a mapping bug, not bad input).
    """
    v1 = data if isinstance(data, V1Manifest) else V1Manifest.model_validate(data)

    records = [migrate_record(record) for record in v1.records]
    manifest = Manifest(dataset_version=v1.dataset_version, records=records)

    split_mapping: dict[str, int] = {}
    blockers_created: dict[str, int] = {blocker.value: 0 for blocker in CurationBlocker}
    sfw_human_carried = 0
    serving_lost = 0
    for old, new in zip(v1.records, records, strict=True):
        split_mapping[old.split] = split_mapping.get(old.split, 0) + 1
        for blocker in new.review.blockers:
            blockers_created[blocker.value] += 1
        if new.sfw_human is not None:
            sfw_human_carried += 1
        if old.enabled and not is_servable(new):
            serving_lost += 1

    notes = [
        "allowed_uses are all false: v1 had no per-use permission evidence",
        "gold_member is false for every migrated record",
        "primary_scope = v1 scopes[0] (v1 emitted scopes in descending labeler score)",
        "artist, parent, and leakage-group identity are unknown (null) for every migrated record",
        "v1 enabled=true does not carry over as a serving grant; "
        f"{serving_lost} previously enabled record(s) are not servable until "
        "permission provenance and human SFW approval are recorded",
    ]
    report = MigrationReport(
        records_migrated=len(records),
        sfw_human_carried=sfw_human_carried,
        sfw_human_absent=len(records) - sfw_human_carried,
        serving_eligibility_lost=serving_lost,
        uses_granted=0,
        human_approvals_fabricated=0,
        gold_members_created=0,
        split_mapping=split_mapping,
        blockers_created=blockers_created,
        notes=notes,
    )
    return manifest, report
