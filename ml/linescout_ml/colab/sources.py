"""Source discovery, candidate state, and the split rule.

Discovery turns a pile of image files into :class:`Candidate` rows that every
later stage fills in. Candidates are persisted as JSON Lines under
``<output_root>/_pipeline/candidates.jsonl`` so a Colab runtime that dies
mid-run resumes instead of starting over — the single most annoying failure
mode on a free GPU tier.

``source_item_id`` is the path relative to the source root (no extension) and
``source_work_id`` is either the same value or its parent directory, depending
on ``SourceSpec.work_grouping``. Splits are then derived from the *work* id by
hashing, which is what makes the manifest's "assets from one source work never
cross splits" rule true by construction rather than by post-hoc checking.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from linescout_ml.colab.config import PipelineConfig, SourceSpec, SplitFractions
from linescout_ml.manifest import CropBox, SfwHumanDecision, SfwScreening
from linescout_ml.taxonomy import GALLERY_SCOPES, LearningSplit, PrimaryStyle, ScopeLabel

#: Split order used by the cumulative-fraction lookup in :func:`split_for_work`.
SPLIT_ORDER: tuple[LearningSplit, ...] = (
    LearningSplit.TRAIN,
    LearningSplit.VALIDATION,
    LearningSplit.TEST,
    LearningSplit.NONE,
)


class Measurements(BaseModel):
    """Pixel measurements taken from the analysis view of an asset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ink_coverage: float = Field(ge=0.0, le=1.0)
    text_coverage: float = Field(ge=0.0, le=1.0)
    background_coverage: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    #: 64-bit DCT perceptual hash, ``imagehash.phash``-compatible.
    phash: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{16}$")]


class AssetLabels(BaseModel):
    """Provisional labels written before human curation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    primary_style: PrimaryStyle
    #: The single best scope. ``unknown`` is legal here: when no scope clears
    #: the confidence floor the pipeline records "not determined" instead of
    #: inventing a label, and curation resolves it before acceptance.
    primary_scope: ScopeLabel = ScopeLabel.UNKNOWN
    #: Additional scopes below the primary, in descending score order.
    secondary_scopes: list[ScopeLabel] = Field(default_factory=list)
    person_count: int | None = Field(default=None, ge=0, le=50)
    #: Pipeline person counts are heuristics (multi -> 2, human scope -> 1),
    #: so the label stages record them as approximate; a human count is exact.
    person_count_approximate: bool = False
    #: Automated SFW screen only. A human SFW decision is recorded through
    #: curation, or — for whole sources the operator asserts by hand — as the
    #: ``sfw_human`` batch decision below.
    sfw: SfwScreening | None = None
    #: A human SFW decision made at ingestion time (operator-asserted source).
    #: ``None`` for every automated method; curation writes the rest.
    sfw_human: SfwHumanDecision | None = None
    #: ``zero_shot`` when a CLIP text encoder ranked the labels, otherwise the
    #: source defaults were used and curation must confirm them.
    labelled_by: Literal["zero_shot", "source_default"] = "source_default"
    style_scores: dict[str, float] | None = None
    scope_scores: dict[str, float] | None = None

    @property
    def scopes(self) -> list[ScopeLabel]:
        """Primary first, then secondaries (mirrors the manifest record)."""
        return [self.primary_scope, *self.secondary_scopes]

    @model_validator(mode="after")
    def _gallery_scopes_only(self) -> Self:
        bad = [scope for scope in self.secondary_scopes if scope not in GALLERY_SCOPES]
        if bad:
            msg = f"secondary scopes cannot carry query-only scopes: {bad}"
            raise ValueError(msg)
        if self.primary_scope in self.secondary_scopes:
            msg = "primary_scope must not repeat in secondary_scopes"
            raise ValueError(msg)
        if ScopeLabel.MULTI_CHARACTER in self.scopes and (
            self.person_count is None or self.person_count < 2
        ):
            msg = "multi_character assets must have person_count >= 2"
            raise ValueError(msg)
        if self.person_count_approximate and self.person_count is None:
            msg = "person_count_approximate requires a person_count"
            raise ValueError(msg)
        return self


class Candidate(BaseModel):
    """One image on its way to becoming a manifest record."""

    model_config = ConfigDict(extra="forbid")

    #: Stable identity across runs: ``<source>/<item_id>``.
    key: str
    source_name: str
    item_id: str
    work_id: str
    relative_path: str
    source_path: str
    split: LearningSplit
    #: Leakage group id derived from the source's ``leakage_grouping``.
    #: ``None`` when unknown (the work id is then the de-facto group).
    leakage_group_id: str | None = None
    #: Artist identity when the source declares one. ``None`` = unknown.
    artist_id: str | None = None

    # Filled by the extract stage
    asset_id: str | None = None
    width: int | None = None
    height: int | None = None
    crop: CropBox | None = None
    original_path: str | None = None
    line_art_path: str | None = None
    extraction_model: str | None = None
    extraction_version: str | None = None
    #: SHA-256 of the checkpoint the extractor loaded (see models.lock.json).
    extraction_sha256: str | None = None
    source_checksum: str | None = None
    line_art_checksum: str | None = None
    thumbnail_checksum: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_checksum(cls, data: Any) -> Any:
        """Accept older candidate rows that stored a single ``checksum``."""
        if isinstance(data, dict) and "checksum" in data and "line_art_checksum" not in data:
            data = dict(data)
            data["line_art_checksum"] = data.pop("checksum")
        return data

    # Filled by the measure / label stages
    measurements: Measurements | None = None
    labels: AssetLabels | None = None

    # Terminal states
    duplicate_of: str | None = None
    skip_reason: str | None = None

    @property
    def is_active(self) -> bool:
        """Still a candidate for the gallery (not skipped, not a duplicate)."""
        return self.skip_reason is None and self.duplicate_of is None

    def resolve_source_file(self, source: SourceSpec) -> Path:
        """Best available path for the original file.

        Drive mounts change between sessions (``/content/drive`` vs a shortcut
        path), so the stored absolute path is a cache and the source-relative
        path is the truth.
        """
        stored = Path(self.source_path)
        if stored.is_file():
            return stored
        return source.root / self.relative_path


@dataclass(frozen=True)
class DiscoveredImage:
    """A file found on disk, before any processing."""

    source: SourceSpec
    path: Path
    relative_path: str
    item_id: str
    work_id: str


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def item_id_for(relative_path: str) -> str:
    """``source_item_id``: the relative path without its extension."""
    item = PurePosixPath(relative_path).with_suffix("").as_posix()
    return item or PurePosixPath(relative_path).name


def work_id_for(source: SourceSpec, relative_path: str) -> str:
    """``source_work_id``: the unit that must never straddle two splits."""
    if source.work_grouping == "parent_dir":
        parent = PurePosixPath(relative_path).parent.as_posix()
        return source.name if parent in {".", "/"} else f"{source.name}/{parent}"
    return f"{source.name}/{item_id_for(relative_path)}"


def leakage_group_for(source: SourceSpec, work_id: str) -> str | None:
    """Leakage group id under the source's ``leakage_grouping`` policy.

    ``work`` keeps the one-work-one-split rule (group key = work id, reported
    as ``None`` so the work id itself remains the de-facto group);
    ``artist`` groups by the source's declared artist; ``source`` puts the
    entire source in one group.
    """
    if source.leakage_grouping == "source":
        return f"source:{source.name}"
    if source.leakage_grouping == "artist":
        if source.default_artist_id is None:
            return None
        return f"artist:{source.default_artist_id}"
    return None


def split_for_work(work_id: str, seed: int, fractions: SplitFractions) -> LearningSplit:
    """Deterministic split assignment from a hash of the work id.

    Hashing (rather than shuffling a list) means the answer does not depend on
    how many images this run happened to see: re-running on a subset, or adding
    a second source later, cannot move an existing work to another split.
    """
    digest = hashlib.sha256(f"{seed}\x00{work_id}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    cumulative = 0.0
    for split, share in zip(
        SPLIT_ORDER,
        (fractions.train, fractions.validation, fractions.test, fractions.unassigned),
        strict=True,
    ):
        cumulative += share
        if unit < cumulative:
            return split
    return SPLIT_ORDER[-1]


def discover_source(source: SourceSpec, limit: int | None = None) -> list[DiscoveredImage]:
    """List the images under ``source.root``, sorted for reproducibility."""
    root = source.root.expanduser()
    if not root.is_dir():
        msg = f"source {source.name!r} root does not exist: {root}"
        raise FileNotFoundError(msg)

    found: list[DiscoveredImage] = []
    for pattern in source.patterns:
        for path in root.rglob(pattern):
            if not path.is_file() or path.name.startswith("."):
                continue
            relative = _relative_posix(path.resolve(), root.resolve())
            found.append(
                DiscoveredImage(
                    source=source,
                    path=path,
                    relative_path=relative,
                    item_id=item_id_for(relative),
                    work_id=work_id_for(source, relative),
                )
            )

    # rglob per pattern can surface the same file twice on case-insensitive
    # mounts (*.JPG vs *.jpg); de-duplicate on the resolved path.
    unique = {image.path.resolve(): image for image in found}
    ordered = sorted(unique.values(), key=lambda image: image.relative_path)
    return ordered[:limit] if limit else ordered


def discover(config: PipelineConfig) -> list[Candidate]:
    """Discover every configured source into fresh :class:`Candidate` rows."""
    candidates: list[Candidate] = []
    for source in config.sources:
        for image in discover_source(source, config.limit_per_source):
            candidates.append(
                Candidate(
                    key=f"{source.name}/{image.item_id}",
                    source_name=source.name,
                    item_id=image.item_id,
                    work_id=image.work_id,
                    relative_path=image.relative_path,
                    source_path=str(image.path.resolve()),
                    split=split_for_work(image.work_id, config.seed, config.splits),
                    leakage_group_id=leakage_group_for(source, image.work_id),
                    artist_id=source.default_artist_id,
                )
            )
    return candidates


class CandidateStore:
    """JSON Lines persistence for candidates.

    ``save`` writes atomically (temp file + ``os.replace``) so an interrupted
    Colab cell cannot leave a truncated state file behind.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._candidates: dict[str, Candidate] = {}

    def __len__(self) -> int:
        return len(self._candidates)

    def __iter__(self) -> Iterator[Candidate]:
        return iter(self._candidates.values())

    @property
    def candidates(self) -> list[Candidate]:
        return list(self._candidates.values())

    @property
    def active(self) -> list[Candidate]:
        return [candidate for candidate in self._candidates.values() if candidate.is_active]

    def get(self, key: str) -> Candidate | None:
        return self._candidates.get(key)

    def upsert(self, candidates: Iterable[Candidate]) -> int:
        """Insert or update candidates, keeping existing stage results.

        Re-discovery must not throw away work: a candidate that already has a
        line-art path keeps it, and only discovery-owned fields are refreshed.
        """
        updated = 0
        for candidate in candidates:
            existing = self._candidates.get(candidate.key)
            if existing is None:
                self._candidates[candidate.key] = candidate
            else:
                merged = candidate.model_copy(
                    update={
                        "asset_id": existing.asset_id,
                        "width": existing.width,
                        "height": existing.height,
                        "crop": existing.crop,
                        "original_path": existing.original_path,
                        "line_art_path": existing.line_art_path,
                        "extraction_model": existing.extraction_model,
                        "extraction_version": existing.extraction_version,
                        "extraction_sha256": existing.extraction_sha256,
                        "source_checksum": existing.source_checksum,
                        "line_art_checksum": existing.line_art_checksum,
                        "thumbnail_checksum": existing.thumbnail_checksum,
                        "measurements": existing.measurements,
                        "labels": existing.labels,
                        "duplicate_of": existing.duplicate_of,
                        "skip_reason": existing.skip_reason,
                    }
                )
                self._candidates[candidate.key] = merged
            updated += 1
        return updated

    def load(self) -> int:
        """Read a previous run's state. Missing file is not an error."""
        self._candidates = {}
        if not self.path.is_file():
            return 0
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                candidate = Candidate.model_validate(json.loads(line))
                self._candidates[candidate.key] = candidate
        return len(self._candidates)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for candidate in sorted(self._candidates.values(), key=lambda item: item.key):
                handle.write(candidate.model_dump_json() + "\n")
        os.replace(temporary, self.path)
        return self.path
