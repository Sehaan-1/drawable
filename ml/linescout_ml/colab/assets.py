"""Gallery tree assembly: files on disk, manifest records, manifest slices.

This is where a pile of processed candidates becomes something the API will
actually load. Two rules from ``services/api`` shape the output:

* ``sync_gallery`` refuses to start unless every *enabled* asset's
  ``line_art`` and ``thumbnail`` files exist relative to the manifest, so the
  tree is written first and the manifest last.
* ``GET /assets/{id}/thumbnail`` and ``/line-art`` always answer
  ``image/png``, so *derivatives* are stored as PNG. The original is copied
  byte-for-byte (JPEG stays JPEG) so source hashes remain independent of
  derivative hashes.

Freshly ingested assets are written ``enabled=false`` with
``review.state="unreviewed"`` (unsafe assets instead get
``quarantined`` + ``enabled=false``). Unreviewed assets stay out of
production search until a human accepts them; the curation UI renders them
through dedicated preview routes, not the public asset endpoints. ``enabled``
is true only when the asset is SFW-safe *and* the review state is
``accepted``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from linescout_ml.colab.config import PipelineConfig, SourceSpec
from linescout_ml.colab.label import review_state_for
from linescout_ml.colab.measure import sha256_file
from linescout_ml.colab.sources import AssetLabels, Candidate, Measurements
from linescout_ml.manifest import (
    AllowedUses,
    ArtifactContract,
    Manifest,
    ManifestRecord,
    Permissions,
    check_parent_integrity,
    check_split_integrity,
    derivative_reasons,
    is_servable,
    is_trainable,
    make_asset_id,
)
from linescout_ml.taxonomy import LineArtOrigin, PrimaryStyle, ReviewState, ScopeLabel

ORIGINALS_DIR = "originals"
LINE_ART_DIR = "line_art"
THUMBNAILS_DIR = "thumbnails"
GALLERY_DIRS: tuple[str, ...] = (ORIGINALS_DIR, LINE_ART_DIR, THUMBNAILS_DIR)


class GalleryBuildError(RuntimeError):
    """A record could not be built, or the manifest failed validation."""


@dataclass(frozen=True)
class AssetPaths:
    """The three manifest paths for one asset, relative to the gallery root."""

    original: str
    line_art: str
    thumbnail: str


_ORIGINAL_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def original_suffix_for(path: Path | str) -> str:
    """Preserve the source file's suffix; ``.jpeg`` is normalised to ``.jpg``."""
    suffix = Path(path).suffix.lower()
    if suffix == ".jpeg":
        return ".jpg"
    if suffix in _ORIGINAL_SUFFIXES:
        return suffix
    return ".png"


def asset_paths(asset_id: str, *, original_suffix: str = ".png") -> AssetPaths:
    if not original_suffix.startswith("."):
        original_suffix = f".{original_suffix}"
    return AssetPaths(
        original=f"{ORIGINALS_DIR}/{asset_id}{original_suffix}",
        line_art=f"{LINE_ART_DIR}/{asset_id}.png",
        thumbnail=f"{THUMBNAILS_DIR}/{asset_id}.png",
    )


def copy_original(source: Path, dest: Path) -> Path:
    """Copy source bytes unchanged. Never re-encode."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(source.read_bytes())
    return dest


def asset_id_for(source: SourceSpec, item_id: str) -> str:
    """Deterministic id for a whole source image.

    The crop is *not* folded into the id even though ``make_asset_id`` accepts
    one: here the crop is advisory framing metadata that the curation UI may
    edit, and including it would silently re-identify every asset whenever the
    crop heuristic changed. Cropped derivatives (a panel cut out of a page) are
    separate assets and should pass their crop explicitly.
    """
    return make_asset_id(source.name, item_id)


def write_png(image: Image.Image, path: Path) -> Path:
    """Save as PNG, creating parents. Never overwrites silently in resume mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    return path


def build_record(
    candidate: Candidate,
    source: SourceSpec,
    config: PipelineConfig,
    *,
    review_state: ReviewState | None = None,
) -> ManifestRecord:
    """Assemble one manifest record from a fully processed candidate.

    Raises :class:`GalleryBuildError` naming the first missing stage, because a
    half-processed candidate is a bug in the run, not a reason to write a
    half-valid record.
    """
    for field in (
        "asset_id",
        "width",
        "height",
        "original_path",
        "line_art_path",
        "source_checksum",
        "line_art_checksum",
        "thumbnail_checksum",
    ):
        if getattr(candidate, field) is None:
            msg = f"candidate {candidate.key} is missing {field}; run the extract stage"
            raise GalleryBuildError(msg)
    if candidate.measurements is None:
        msg = f"candidate {candidate.key} has no measurements; run the measure stage"
        raise GalleryBuildError(msg)
    if candidate.labels is None:
        msg = f"candidate {candidate.key} has no labels; run the label stage"
        raise GalleryBuildError(msg)

    measurements: Measurements = candidate.measurements
    labels: AssetLabels = candidate.labels
    paths = asset_paths(
        str(candidate.asset_id),
        original_suffix=Path(str(candidate.original_path)).suffix or ".png",
    )
    if labels.sfw is not None:
        state = review_state or review_state_for(labels.sfw)
    else:
        # A human decision without a screen (operator-asserted source).
        state = review_state or ReviewState.UNREVIEWED
    extracted = source.origin is LineArtOrigin.EXTRACTED

    try:
        return ManifestRecord(
            asset_id=str(candidate.asset_id),
            source_dataset=source.name,
            source_item_id=candidate.item_id,
            source_work_id=candidate.work_id,
            parent_asset_id=None,
            artist_id=candidate.artist_id,
            leakage_group_id=candidate.leakage_group_id,
            source_url=source.source_url(candidate.item_id),
            permissions=Permissions(
                license_id=source.license_id,
                basis=source.permission_basis,
                permission_url=source.permission_url,
                attribution=source.attribution,
                attribution_required=source.attribution_required,
            ),
            allowed_uses=AllowedUses(
                display=source.allowed_display,
                training=source.allowed_training,
                trace=source.allowed_trace,
            ),
            original_path=paths.original,
            line_art_path=paths.line_art,
            thumbnail_path=paths.thumbnail,
            origin=source.origin,
            extraction_model=candidate.extraction_model if extracted else None,
            extraction_version=candidate.extraction_version if extracted else None,
            primary_style=labels.primary_style,
            primary_scope=labels.primary_scope,
            secondary_scopes=list(labels.secondary_scopes),
            person_count=labels.person_count,
            person_count_approximate=labels.person_count_approximate,
            sfw_screening=labels.sfw,
            sfw_human=labels.sfw_human,
            width=int(candidate.width or 0),
            height=int(candidate.height or 0),
            crop=candidate.crop,
            text_coverage=measurements.text_coverage,
            ink_coverage=measurements.ink_coverage,
            phash=measurements.phash,
            quality_score=measurements.quality_score,
            review=review_for(state),
            learning_split=candidate.split,
            gallery_member=True,
            gold_member=False,
            pipeline_version=config.pipeline_version,
            processing_revision=config.processing_revision,
            label_version=config.label_version,
            source_checksum=str(candidate.source_checksum),
            line_art_checksum=str(candidate.line_art_checksum),
            thumbnail_checksum=str(candidate.thumbnail_checksum),
        )
    except ValueError as error:  # pydantic ValidationError and friends
        msg = f"candidate {candidate.key} produced an invalid record: {error}"
        raise GalleryBuildError(msg) from error


def review_for(state: ReviewState) -> dict[str, Any]:
    """A fresh asset has no human quality judgement yet — curation supplies it."""
    return {"state": state, "quality": None, "blockers": []}


def build_manifest(
    records: Sequence[ManifestRecord],
    dataset_version: str,
    *,
    artifact_contract: ArtifactContract | None = None,
) -> Manifest:
    """Validate records as a whole, including the one-group-one-split rule.

    The manifest declares the *current* artifact generation via
    ``artifact_contract`` (required for anything to be servable). Records built
    by this pipeline carry config's generation fields; records merged from an
    older manifest that do not match stay in the output for audit and are
    disabled by :func:`is_servable` until re-processed.
    """
    problems = check_split_integrity(records) + check_parent_integrity(records)
    if problems:
        msg = "split integrity violated: " + "; ".join(problems[:5])
        raise GalleryBuildError(msg)
    try:
        return Manifest(
            dataset_version=dataset_version,
            artifact_contract=artifact_contract,
            records=list(records),
        )
    except ValueError as error:
        msg = f"manifest failed validation: {error}"
        raise GalleryBuildError(msg) from error


def read_manifest(path: Path) -> Manifest | None:
    """Load an existing manifest, or ``None`` when there is nothing to merge into."""
    if not path.is_file():
        return None
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        msg = f"existing manifest {path} is unreadable: {error}"
        raise GalleryBuildError(msg) from error


def merge_records(
    existing: Iterable[ManifestRecord], incoming: Iterable[ManifestRecord]
) -> list[ManifestRecord]:
    """Patch a manifest slice into an existing record list.

    Incoming records win on ``asset_id`` (a re-run may correct labels), existing
    order is preserved, and brand-new records are appended in the order given so
    a merge is reproducible.
    """
    merged: dict[str, ManifestRecord] = {}
    order: list[str] = []
    for record in existing:
        if record.asset_id not in merged:
            order.append(record.asset_id)
        merged[record.asset_id] = record
    for record in incoming:
        if record.asset_id not in merged:
            order.append(record.asset_id)
        merged[record.asset_id] = record
    return [merged[asset_id] for asset_id in order]


def write_manifest(manifest: Manifest, path: Path) -> Path:
    """Write the manifest in the same shape as the committed synthetic fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def missing_files(manifest: Manifest, root: Path) -> list[str]:
    """Assets whose served files are absent — required for search *and* curation preview."""
    return [
        problem
        for problem in derivative_problems(manifest, root)
        if problem.startswith("missing_file")
    ]


def derivative_problems(manifest: Manifest, root: Path) -> list[str]:
    """Every derivative-integrity problem, as stable ``<code> asset_id: …`` strings.

    Checks per record:
    * existence of ``line_art_path`` / ``thumbnail_path`` (served files) and
      ``original_path`` (preserved source bytes);
    * sha256 of each file against the recorded checksum — a mismatch means the
      bytes on disk were modified after the manifest was written;
    * generation currency against ``manifest.artifact_contract``.

    The manifest is authoritative: a problem never rewrites the manifest, it
    only reports what the gallery loader will disable.
    """
    problems: list[str] = []
    for record in manifest.records:
        for label, relative, checksum in (
            ("line_art", record.line_art_path, record.line_art_checksum),
            ("thumbnail", record.thumbnail_path, record.thumbnail_checksum),
            ("original", record.original_path, record.source_checksum),
        ):
            path = root / relative
            if not path.is_file():
                problems.append(f"missing_file {record.asset_id}: {label} {relative}")
                continue
            if sha256_file(path) != checksum:
                problems.append(f"checksum_mismatch {record.asset_id}: {label} {relative}")
        for reason in derivative_reasons(record, manifest.artifact_contract):
            problems.append(f"{reason} {record.asset_id}")
    return problems


def summarise(
    records: Iterable[ManifestRecord],
    *,
    artifact_contract: ArtifactContract | None = None,
) -> dict[str, Any]:
    """Counts the notebook shows as a table and the run report records.

    ``servable``/``trainable`` use the canonical eligibility policy with the
    manifest's artifact contract, so records whose derivatives are stale are
    counted separately and excluded.
    """
    styles: dict[str, int] = {style.value: 0 for style in PrimaryStyle}
    scopes: dict[str, int] = {scope.value: 0 for scope in ScopeLabel if scope.value != "unknown"}
    splits: dict[str, int] = {}
    origins: dict[str, int] = {}
    states: dict[str, int] = {}
    servable = 0
    trainable = 0
    gold = 0
    stale = 0
    approximate_counts = 0
    total = 0
    quality: list[float] = []

    for record in records:
        total += 1
        servable += int(is_servable(record, artifact_contract))
        trainable += int(is_trainable(record, artifact_contract))
        gold += int(record.gold_member)
        stale += int(bool(derivative_reasons(record, artifact_contract)))
        approximate_counts += int(record.person_count_approximate)
        styles[record.primary_style.value] += 1
        scopes[record.primary_scope.value] = scopes.get(record.primary_scope.value, 0) + 1
        for scope in record.secondary_scopes:
            scopes[scope.value] = scopes.get(scope.value, 0) + 1
        splits[record.learning_split.value] = splits.get(record.learning_split.value, 0) + 1
        origins[record.origin.value] = origins.get(record.origin.value, 0) + 1
        states[record.review.state.value] = states.get(record.review.state.value, 0) + 1
        quality.append(record.quality_score)

    ordered_quality = sorted(quality)
    return {
        "total": total,
        "servable": servable,
        "trainable": trainable,
        "gold": gold,
        "stale_derivatives": stale,
        "approximate_person_counts": approximate_counts,
        "by_style": styles,
        "by_scope": scopes,
        "by_learning_split": splits,
        "by_origin": origins,
        "by_review_state": states,
        "quality_min": round(ordered_quality[0], 4) if ordered_quality else None,
        "quality_median": (
            round(ordered_quality[len(ordered_quality) // 2], 4) if ordered_quality else None
        ),
        "quality_max": round(ordered_quality[-1], 4) if ordered_quality else None,
    }


def dump_summary(
    records: Iterable[ManifestRecord],
    *,
    artifact_contract: ArtifactContract | None = None,
) -> str:
    """Pretty JSON summary, handy for a notebook cell's last line."""
    return json.dumps(
        summarise(records, artifact_contract=artifact_contract), indent=2, sort_keys=True
    )
