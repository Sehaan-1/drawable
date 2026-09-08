"""Versioned manifest conversion: v1 -> v2 -> v3 (and v2 -> v3).

The v1 and v2 shapes below are **frozen copies** of those schemas — they exist
so an old manifest can be parsed and converted long after the live models moved
on. The mapping rules are the normative ones documented in
``docs/contracts/migration-manifest.md``.

Safety properties this module guarantees (its tests assert them):

1. **No old asset gains permission.** ``allowed_uses`` is all-``False`` after
   any conversion from v1, ``permissions.basis`` is ``unknown``, and a v1
   ``enabled`` flag never becomes a v2/v3 grant.
2. **No old asset gains human approval.** ``sfw_human`` is only created from a
   v1 ``sfw.method == "manual"`` decision; automated screening methods become
   ``sfw_screening`` and nothing else.
3. **No old asset becomes gold.** ``gold_member`` is ``False`` after v1
   conversion; a v2 gold record is only carried into v3 when it satisfies the
   v3 gold conditions (accepted, quality 2–3, no blockers, known scope, human
   SFW approval) — otherwise it is conservatively disabled and reported.
4. **Ambiguous legacy generations are disabled, never promoted.** When v2
   records disagree about their ``pipeline_version`` / ``label_version`` /
   ``processing_revision`` (or the caller supplies no explicit contract), the
   v3 manifest is written with ``artifact_contract: null``. Under ``null``
   every record's derivatives are unverified, so nothing is servable or
   trainable until the operator declares the current generation — and the
   report says exactly that.

Run it through the CLI::

    linescout-manifest convert manifest.v2.json --from 2 --to 3 \\
        --out manifest.v3.json --report conversion.json
    linescout-manifest convert manifest.v1.json --from 1 --to 3 \\
        --out manifest.v3.json --report conversion.json
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from linescout_ml.manifest import (
    AllowedUses,
    ArtifactContract,
    CropBox,
    HumanReview,
    Manifest,
    ManifestRecord,
    Permissions,
    SfwHumanDecision,
    SfwScreening,
    check_parent_integrity,
    check_split_integrity,
    is_gold,
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
#: decision's provenance (\"came from a v1 manual entry\") stays inspectable.
MIGRATION_REVIEWER = "migrate-v1"
#: v1 automated labels were produced by the first labeling contract.
V1_LABEL_VERSION = "1"


class ConversionError(ValueError):
    """The input is not a valid manifestation of the declared source version."""


# --------------------------------------------------------------- frozen v1 shapes


class V1SfwDecision(BaseModel):
    """Frozen v1 shape. ``method=\"manual\"`` was a human decision."""

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


# --------------------------------------------------------------- frozen v2 shapes


class V2ManifestRecord(BaseModel):
    """Frozen v2 record shape (``schema_version`` 2 manifests).

    The v2 contract has no manifest-level artifact contract: each record
    carries its own ``pipeline_version`` / ``label_version`` /
    ``processing_revision``. The conversion decides whether those agree.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    asset_id: str
    source_dataset: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    source_item_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    source_work_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    parent_asset_id: str | None = None
    artist_id: str | None = None
    leakage_group_id: str | None = None
    source_url: str | None = Field(default=None, max_length=2048)
    permissions: Permissions
    allowed_uses: AllowedUses = Field(default_factory=AllowedUses)
    original_path: str
    line_art_path: str
    thumbnail_path: str
    origin: LineArtOrigin
    extraction_model: str | None = Field(default=None, max_length=64)
    extraction_version: str | None = Field(default=None, max_length=32)
    primary_style: PrimaryStyle
    primary_scope: ScopeLabel
    secondary_scopes: list[ScopeLabel] = Field(default_factory=list)
    person_count: int | None = Field(default=None, ge=0, le=50)
    person_count_approximate: bool = False
    sfw_screening: SfwScreening | None = None
    sfw_human: SfwHumanDecision | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    crop: CropBox | None = None
    text_coverage: float = Field(ge=0.0, le=1.0)
    ink_coverage: float = Field(ge=0.0, le=1.0)
    phash: str
    quality_score: float = Field(ge=0.0, le=1.0)
    review: HumanReview = Field(default_factory=HumanReview)
    learning_split: LearningSplit
    gallery_member: bool = True
    gold_member: bool = False
    pipeline_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    processing_revision: int = Field(default=1, ge=1)
    label_version: Annotated[str, StringConstraints(min_length=1, max_length=32)] = "1"
    source_checksum: str
    line_art_checksum: str
    thumbnail_checksum: str


class V2Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    dataset_version: Annotated[
        str, StringConstraints(pattern=r"^\d{4}\.\d{2}\.\d{2}(-[a-z0-9]+)?$")
    ]
    records: list[V2ManifestRecord]

    @model_validator(mode="after")
    def _unique_ids_and_integrity(self) -> V2Manifest:
        seen_ids: set[str] = set()
        for record in self.records:
            if record.asset_id in seen_ids:
                msg = f"duplicate asset_id {record.asset_id}"
                raise ValueError(msg)
            seen_ids.add(record.asset_id)
        problems = check_split_integrity(self.records) + check_parent_integrity(self.records)
        if problems:
            msg = "; ".join(problems[:5])
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------- report


class ConversionReport(BaseModel):
    """What a conversion did — the machine-readable version of the mapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version_from: int
    schema_version_to: int
    records_converted: int
    # v1-era counters (kept for audit parity with the original report).
    sfw_human_carried: int = 0
    sfw_human_absent: int = 0
    serving_eligibility_lost: int = 0
    split_mapping: dict[str, int] = Field(default_factory=dict)
    blockers_created: dict[str, int] = Field(default_factory=dict)
    # Safety properties — all three must stay 0 on every run.
    uses_granted: int = 0
    human_approvals_fabricated: int = 0
    gold_members_created: int = 0
    # v3 conversion counters.
    contract_detected: bool = False
    contract_source: Literal["explicit", "record_consensus", "ambiguous"] | None = None
    artifact_contract: ArtifactContract | None = None
    records_disabled_stale: int = 0
    disabled_reasons: dict[str, int] = Field(default_factory=dict)
    gold_disabled: dict[str, int] = Field(default_factory=dict)
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


def _v1_record_to_v2(record: V1ManifestRecord) -> V2ManifestRecord:
    """Map one v1 record to the frozen v2 shape.

    Every grant-shaped field is deny-by-default; see the module docstring for
    the safety properties.
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

    return V2ManifestRecord(
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
        # Gold membership never exists in v1 and is never created by conversion.
        gold_member=False,
        pipeline_version=record.pipeline_version,
        processing_revision=1,
        label_version=V1_LABEL_VERSION,
        source_checksum=record.source_checksum,
        line_art_checksum=record.line_art_checksum,
        thumbnail_checksum=record.thumbnail_checksum,
    )


def _v2_manifest_from_v1(
    data: Mapping[str, Any] | V1Manifest,
) -> tuple[V2Manifest, dict[str, int], dict[str, int], int, int]:
    """Parse + map a v1 document into a v2 manifest and v1-era counters."""
    v1 = data if isinstance(data, V1Manifest) else V1Manifest.model_validate(data)
    v2_records = [_v1_record_to_v2(record) for record in v1.records]
    # A v1 gallery can already violate the *identity* leakage rules (artist /
    # parent are unknown after mapping, so only work/groups can conflict).
    problems = check_split_integrity(v2_records) + check_parent_integrity(v2_records)
    if problems:
        msg = "cannot convert: " + "; ".join(problems[:5])
        raise ConversionError(msg)
    manifest = V2Manifest(dataset_version=v1.dataset_version, records=v2_records)

    split_mapping: dict[str, int] = {}
    blockers_created: dict[str, int] = {blocker.value: 0 for blocker in CurationBlocker}
    sfw_human_carried = 0
    serving_lost = 0
    for old, new in zip(v1.records, manifest.records, strict=True):
        split_mapping[old.split] = split_mapping.get(old.split, 0) + 1
        for blocker in new.review.blockers:
            blockers_created[blocker.value] += 1
        if new.sfw_human is not None:
            sfw_human_carried += 1
        # The v3 shape carries the same fields; converting here keeps the
        # canonical predicate the single source of eligibility truth.
        if old.enabled and not is_servable(_v2_record_to_v3(new), None):
            serving_lost += 1
    return manifest, split_mapping, blockers_created, sfw_human_carried, serving_lost


def detect_contract(
    records: list[V2ManifestRecord], *, explicit: ArtifactContract | None
) -> tuple[ArtifactContract | None, str]:
    """Decide the v3 artifact contract for a set of v2 records.

    * ``explicit`` — the operator declared the current generation.
    * ``record_consensus`` — every record carries the exact same generation;
      assume that generation is current (deterministic, no promotion).
    * ``ambiguous`` — generations disagree (or a field is missing): return
      ``None`` so every record is conservatively disabled.
    """
    if explicit is not None:
        return explicit, "explicit"
    generations = {(r.pipeline_version, r.label_version, r.processing_revision) for r in records}
    if len(generations) == 1:
        pipeline, label, revision = generations.pop()
        return ArtifactContract(
            pipeline_version=pipeline, label_version=label, processing_revision=revision
        ), "record_consensus"
    return None, "ambiguous"


def _v2_record_to_v3(record: V2ManifestRecord) -> ManifestRecord:
    """Carry one v2 record into the v3 shape (generation fields verbatim)."""
    try:
        return ManifestRecord.model_validate(record.model_dump())
    except ValidationError as error:
        msg = f"v2 record {record.asset_id} cannot be carried to v3: {error}"
        raise ConversionError(msg) from error


def _upgrade_v2_to_v3(
    v2: V2Manifest,
    *,
    explicit_contract: ArtifactContract | None = None,
) -> tuple[Manifest, ConversionReport]:
    """Convert a frozen v2 manifest into a v3 manifest, conservatively."""
    contract, contract_source = detect_contract(v2.records, explicit=explicit_contract)

    records: list[ManifestRecord] = []
    gold_disabled: dict[str, int] = {}
    for record in v2.records:
        data = record.model_dump()
        try:
            records.append(ManifestRecord.model_validate(data))
        except ValidationError as error:
            if not record.gold_member or "gold" not in str(error):
                raise ConversionError(
                    f"v2 record {record.asset_id} cannot be carried to v3: {error}"
                ) from error
            data["gold_member"] = False
            try:
                records.append(ManifestRecord.model_validate(data))
            except ValidationError as retry_error:
                msg = (
                    f"v2 record {record.asset_id} cannot be carried to v3 even "
                    f"after disabling gold membership: {retry_error}"
                )
                raise ConversionError(msg) from retry_error
            reason = "gold_member_dropped_v3_conditions"
            gold_disabled[reason] = gold_disabled.get(reason, 0) + 1

    try:
        manifest = Manifest(
            dataset_version=v2.dataset_version,
            artifact_contract=contract,
            records=records,
        )
    except ValueError as error:
        # The only manifest-level conflict that conversion may fix is gold
        # membership on stale derivatives (conservative downgrade). Anything
        # else — split integrity, duplicate ids — must be fixed by hand.
        message = str(error)
        if "gold_member" not in message:
            raise ConversionError(f"v2 manifest cannot be carried to v3: {message}") from error
        downgraded = 0
        for index, candidate in enumerate(records):
            if candidate.gold_member and not is_gold(candidate, contract):
                records[index] = candidate.model_copy(update={"gold_member": False})
                downgraded += 1
                reason = "gold_member_stale_derivatives"
                gold_disabled[reason] = gold_disabled.get(reason, 0) + 1
        manifest = Manifest(
            dataset_version=v2.dataset_version,
            artifact_contract=contract,
            records=records,
        )
        if downgraded == 0:
            raise ConversionError(f"v2 manifest cannot be carried to v3: {message}") from error

    disabled_reasons: dict[str, int] = {}
    stale = 0
    for v3_record in manifest.records:
        reasons = serving_reasons_for(v3_record, contract)
        if any(reason.startswith("derivative_") for reason in reasons):
            stale += 1
        for reason in reasons:
            disabled_reasons[reason] = disabled_reasons.get(reason, 0) + 1

    notes: list[str] = []
    if contract is None:
        notes.append(
            "artifact_contract is null: v2 records disagree about their "
            "pipeline/label generation, so no derivative is verified current; "
            "every record is disabled until the current generation is declared "
            "or the dataset is re-processed"
        )
    else:
        notes.append(
            f"artifact_contract = {contract.describe()} "
            f"(source: {contract_source}); records from other generations are "
            "kept for audit and disabled until re-processed"
        )
    notes.append(
        "gold membership was carried only where v3 conditions hold; downgrades are counted"
    )
    notes.append(
        "audits (curation labels, events) and source bytes are never touched by conversion"
    )

    report = ConversionReport(
        schema_version_from=2,
        schema_version_to=3,
        records_converted=len(records),
        contract_detected=contract is not None,
        contract_source=contract_source,
        artifact_contract=contract,
        records_disabled_stale=stale,
        disabled_reasons=disabled_reasons,
        gold_disabled=gold_disabled,
        notes=notes,
    )
    return manifest, report


def serving_reasons_for(record: ManifestRecord, contract: ArtifactContract | None) -> list[str]:
    """Thin import surface kept here so conversion code stays policy-agnostic."""
    from linescout_ml.manifest import serving_reasons

    return serving_reasons(record, contract)


def migrate_record(record: V1ManifestRecord) -> ManifestRecord:
    """Map one v1 record to a v3 record under the frozen deny-by-default rules.

    ``artifact_contract`` is a manifest property, so a bare record is returned
    with generation fields verbatim; :func:`convert_manifest` decides currency.
    """
    return _v2_record_to_v3(_v1_record_to_v2(record))


def convert_manifest(
    data: Mapping[str, Any] | V1Manifest | V2Manifest,
    *,
    from_version: int,
    to_version: int,
    explicit_contract: ArtifactContract | None = None,
) -> tuple[Manifest | V2Manifest, ConversionReport]:
    """Convert a parsed or raw manifest document between supported versions.

    Supported paths: 1→2, 1→3, 2→3, 3→3 (revalidate). A v2 conversion takes
    ``explicit_contract`` to declare the current generation; otherwise the
    contract is detected from record consensus or left ``null`` (conservative).
    """
    if from_version == to_version == 3:
        parsed = Manifest.model_validate(data) if not isinstance(data, Manifest) else data
        report = ConversionReport(
            schema_version_from=3,
            schema_version_to=3,
            records_converted=len(parsed.records),
            contract_detected=parsed.artifact_contract is not None,
            contract_source="explicit" if parsed.artifact_contract is not None else "ambiguous",
            artifact_contract=parsed.artifact_contract,
            notes=["v3 manifest revalidated; identity preserved"],
        )
        return parsed, report

    if from_version == 1:
        if isinstance(data, V2Manifest):
            msg = "a schema v2 manifest cannot be converted as v1; use --from 2"
            raise ConversionError(msg)
        v2, split_mapping, blockers, carried, lost = _v2_manifest_from_v1(data)
        if to_version == 2:
            report = ConversionReport(
                schema_version_from=1,
                schema_version_to=2,
                records_converted=len(v2.records),
                sfw_human_carried=carried,
                sfw_human_absent=len(v2.records) - carried,
                serving_eligibility_lost=lost,
                split_mapping=split_mapping,
                blockers_created=blockers,
                notes=[
                    "allowed_uses are all false: v1 had no per-use permission evidence",
                    "gold_member is false for every converted record",
                    "primary_scope = v1 scopes[0] (v1 emitted scopes in descending labeler score)",
                    "artist, parent, and leakage-group identity are unknown (null) "
                    "for every record",
                ],
            )
            return v2, report
        if to_version == 3:
            manifest, report = _upgrade_v2_to_v3(v2, explicit_contract=explicit_contract)
            upgrade = ConversionReport(
                schema_version_from=1,
                schema_version_to=3,
                records_converted=report.records_converted,
                sfw_human_carried=carried,
                sfw_human_absent=len(v2.records) - carried,
                serving_eligibility_lost=lost,
                split_mapping=split_mapping,
                blockers_created=blockers,
                contract_detected=report.contract_detected,
                contract_source=report.contract_source,
                artifact_contract=report.artifact_contract,
                records_disabled_stale=report.records_disabled_stale,
                disabled_reasons=report.disabled_reasons,
                gold_disabled=report.gold_disabled,
                notes=[
                    *report.notes,
                    "allowed_uses are all false and basis is unknown: v1 had no "
                    "per-use permission evidence and nothing was promoted",
                ],
            )
            return manifest, upgrade

    if from_version == 2:
        v2 = data if isinstance(data, V2Manifest) else V2Manifest.model_validate(data)
        if to_version == 2:
            report = ConversionReport(
                schema_version_from=2,
                schema_version_to=2,
                records_converted=len(v2.records),
                notes=["v2 manifest revalidated; identity preserved"],
            )
            return v2, report
        if to_version == 3:
            return _upgrade_v2_to_v3(v2, explicit_contract=explicit_contract)

    msg = f"unsupported conversion from version {from_version} to version {to_version}"
    raise ConversionError(msg)


def migrate_manifest(
    data: Mapping[str, Any] | V1Manifest,
    *,
    explicit_contract: ArtifactContract | None = None,
) -> tuple[Manifest, ConversionReport]:
    """Compatibility entry point: convert a v1 manifest straight to v3."""
    manifest, report = convert_manifest(
        data,
        from_version=1,
        to_version=3,
        explicit_contract=explicit_contract,
    )
    assert isinstance(manifest, Manifest)
    return manifest, report
