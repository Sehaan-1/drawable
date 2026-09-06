"""Gallery tree assembly: files on disk, manifest records, manifest slices.

This is where a pile of processed candidates becomes something the API will
actually load. Two rules from ``services/api`` shape the output:

* ``sync_gallery`` refuses to start unless every *enabled* asset's
  ``line_art`` and ``thumbnail`` files exist relative to the manifest, so the
  tree is written first and the manifest last.
* ``GET /assets/{id}/thumbnail`` and ``/line-art`` always answer
  ``image/png``, so every file in the gallery — including a JPEG original — is
  re-encoded as PNG. Keeping the source bytes would be smaller but would put a
  lie in the response header.

Freshly ingested assets are written ``enabled=true`` with
``review.state="unreviewed"`` (unsafe assets instead get
``quarantined`` + ``enabled=false``). That is deliberate: the curation UI can
only render assets the API will serve, and rejection is what flips
``enabled`` off. The manifest's own invariants keep this honest — an enabled
asset must be SFW and must not be rejected or quarantined.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from linescout_ml.colab.config import PipelineConfig, SourceSpec
from linescout_ml.colab.sources import AssetLabels, Candidate, Measurements
from linescout_ml.manifest import (
    Manifest,
    ManifestRecord,
    check_split_integrity,
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


def asset_paths(asset_id: str) -> AssetPaths:
    return AssetPaths(
        original=f"{ORIGINALS_DIR}/{asset_id}.png",
        line_art=f"{LINE_ART_DIR}/{asset_id}.png",
        thumbnail=f"{THUMBNAILS_DIR}/{asset_id}.png",
    )


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
    for field in ("asset_id", "width", "height", "line_art_path", "checksum"):
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
    paths = asset_paths(str(candidate.asset_id))
    state = review_state or (ReviewState.UNREVIEWED if labels.sfw.safe else ReviewState.QUARANTINED)
    extracted = source.origin is LineArtOrigin.EXTRACTED

    try:
        return ManifestRecord(
            asset_id=str(candidate.asset_id),
            source_dataset=source.name,
            source_item_id=candidate.item_id,
            source_work_id=candidate.work_id,
            source_url=source.source_url(candidate.item_id),
            license_id=source.license_id,
            original_path=paths.original,
            line_art_path=paths.line_art,
            thumbnail_path=paths.thumbnail,
            origin=source.origin,
            extraction_model=candidate.extraction_model if extracted else None,
            extraction_version=candidate.extraction_version if extracted else None,
            primary_style=labels.primary_style,
            scopes=list(labels.scopes),
            person_count=labels.person_count,
            sfw=labels.sfw,
            width=int(candidate.width or 0),
            height=int(candidate.height or 0),
            crop=candidate.crop,
            text_coverage=measurements.text_coverage,
            ink_coverage=measurements.ink_coverage,
            phash=measurements.phash,
            quality_score=measurements.quality_score,
            review=review_for(state),
            split=candidate.split,
            enabled=labels.sfw.safe and state is not ReviewState.REJECTED,
            pipeline_version=config.pipeline_version,
            checksum=str(candidate.checksum),
        )
    except ValueError as error:  # pydantic ValidationError and friends
        msg = f"candidate {candidate.key} produced an invalid record: {error}"
        raise GalleryBuildError(msg) from error


def review_for(state: ReviewState) -> dict[str, Any]:
    """A fresh asset has no human quality judgement yet — curation supplies it."""
    return {"state": state, "quality": None, "malformed_anatomy": False, "poor_extraction": False}


def build_manifest(records: Sequence[ManifestRecord], dataset_version: str) -> Manifest:
    """Validate records as a whole, including the one-work-one-split rule."""
    problems = check_split_integrity(records)
    if problems:
        msg = "split integrity violated: " + "; ".join(problems[:5])
        raise GalleryBuildError(msg)
    try:
        return Manifest(dataset_version=dataset_version, records=list(records))
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
    """Enabled assets whose served files are absent — what ``sync_gallery`` rejects."""
    problems: list[str] = []
    for record in manifest.enabled_records:
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
    enabled = 0
    total = 0
    quality: list[float] = []

    for record in records:
        total += 1
        enabled += int(record.enabled)
        styles[record.primary_style.value] += 1
        for scope in record.scopes:
            scopes[scope.value] = scopes.get(scope.value, 0) + 1
        splits[record.split.value] = splits.get(record.split.value, 0) + 1
        origins[record.origin.value] = origins.get(record.origin.value, 0) + 1
        states[record.review.state.value] = states.get(record.review.state.value, 0) + 1
        quality.append(record.quality_score)

    ordered_quality = sorted(quality)
    return {
        "total": total,
        "enabled": enabled,
        "by_style": styles,
        "by_scope": scopes,
        "by_split": splits,
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
