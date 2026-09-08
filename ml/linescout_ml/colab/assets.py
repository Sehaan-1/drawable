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

Freshly ingested assets are written ``review.state="unreviewed"`` with no
``sfw_human`` approval and no quality grade, and assets that fail the SFW screen
are written ``quarantined``. Under the v2 contract there is no stored ``enabled``
flag to flip: serving is the derived ``is_servable`` predicate — gallery member,
display grant, human acceptance with a quality grade, human SFW approval, no
blockers — so an unreviewed asset is invisible to search by construction rather
than by a field the pipeline could have set wrongly. The curation UI renders such
assets through dedicated preview routes, not the public endpoints.
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
from linescout_ml.colab.sources import AssetLabels, Candidate, Measurements
from linescout_ml.manifest import (
    AllowedUses,
    Manifest,
    ManifestRecord,
    Permissions,
    PipelineProvenance,
    check_parent_integrity,
    check_split_integrity,
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
            extraction_sha256=candidate.extraction_sha256 if extracted else None,
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
            processing_revision=1,
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
    provenance: PipelineProvenance | None = None,
) -> Manifest:
    """Validate records as a whole: one work one split, one parent one family.

    ``provenance`` names the code, environment, and model checkpoints that produced
    the records. It is part of the manifest rather than only the run report because the
    report travels with one *run* while the manifest travels with the *dataset*: a
    gallery merged over three months still has to say which pins the oldest records
    were built from.
    """
    problems = check_split_integrity(records) + check_parent_integrity(records)
    if problems:
        msg = "split integrity violated: " + "; ".join(problems[:5])
        raise GalleryBuildError(msg)
    try:
        return Manifest(
            dataset_version=dataset_version, records=list(records), provenance=provenance
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
    problems: list[str] = []
    for record in manifest.records:
        for label, relative in (
            ("line_art", record.line_art_path),
            ("thumbnail", record.thumbnail_path),
        ):
            if not (root / relative).is_file():
                problems.append(f"{record.asset_id}: missing {label} file {relative}")
    return problems


def summarise(records: Iterable[ManifestRecord]) -> dict[str, Any]:
    """Counts the notebook shows as a table and the run report records."""
    styles: dict[str, int] = {style.value: 0 for style in PrimaryStyle}
    scopes: dict[str, int] = {scope.value: 0 for scope in ScopeLabel if scope.value != "unknown"}
    splits: dict[str, int] = {}
    origins: dict[str, int] = {}
    states: dict[str, int] = {}
    servable = 0
    trainable = 0
    gold = 0
    approximate_counts = 0
    total = 0
    quality: list[float] = []

    for record in records:
        total += 1
        servable += int(is_servable(record))
        trainable += int(is_trainable(record))
        gold += int(record.gold_member)
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


def dump_summary(records: Iterable[ManifestRecord]) -> str:
    """Pretty JSON summary, handy for a notebook cell's last line."""
    return json.dumps(summarise(records), indent=2, sort_keys=True)
