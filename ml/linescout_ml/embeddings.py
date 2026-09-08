"""Vector payload contract: how missing and unsupported embeddings are represented.

Feature shards (``indexes/<version>/<encoder>/``) are the handoff from
ingestion to the Milestone 4 index build. The frozen rules for per-asset
vector availability are:

* A vector payload is **never fabricated**: no zero vectors, no nulls standing
  in for real features, no silently re-used vectors from an older artifact.
* Availability is explicit and per (asset, embedder). :class:`VectorStatus`
  distinguishes ``available`` / ``missing`` / ``stale`` / ``unsupported``:
    - ``available``   — an entry exists whose embedder key matches, and whose
      ``processing_revision`` and ``line_art_checksum`` match the asset's
      current derived artifacts.
    - ``stale``       — an entry exists but the artifact binding mismatches
      (the asset was re-extracted). Stale entries must be re-embedded; they
      are never served or counted as coverage.
    - ``missing``     — no entry for that embedder.
    - ``unsupported`` — the embedder key is not part of the loaded/relevant
      set for this session or index build.
* An asset without an ``available`` vector for a branch is excluded from that
  retrieval branch (degrading the response, never ranking it on a fabricated
  vector), and the exclusion is reportable.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class VectorStatus(StrEnum):
    """Explicit per-(asset, embedder) vector availability."""

    AVAILABLE = "available"
    MISSING = "missing"
    STALE = "stale"
    UNSUPPORTED = "unsupported"


class ArtifactStamp(BaseModel):
    """Binds one embedding entry to the derived artifact it was computed from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    processing_revision: int = Field(ge=1)
    line_art_checksum: Sha256


class EmbeddingEntry(BaseModel):
    """One per-asset entry in a shard ``index.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact: ArtifactStamp
    shard: Annotated[str, StringConstraints(min_length=1, max_length=128)]


class EmbeddingIndex(BaseModel):
    """The per-asset entry map of one embedder's shard index."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    embedder: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    entries: dict[str, EmbeddingEntry] = Field(default_factory=dict)

    @property
    def covered_ids(self) -> set[str]:
        """Asset ids with a stored entry (stale or not) — not "available" ids."""
        return set(self.entries)


def vector_status(
    *,
    embedder: str,
    supported_embedders: frozenset[str] | set[str],
    entry: EmbeddingEntry | None,
    artifact: ArtifactStamp,
) -> VectorStatus:
    """Classify one (asset, embedder) pair under the frozen rules.

    ``supported_embedders`` is the set of embedder keys relevant to this
    session/index build (e.g. the branches the API loaded). An unsupported
    embedder is reported even when an entry exists, because serving from it
    would mix model versions.
    """
    if embedder not in supported_embedders:
        return VectorStatus.UNSUPPORTED
    if entry is None:
        return VectorStatus.MISSING
    if (
        entry.artifact.processing_revision != artifact.processing_revision
        or entry.artifact.line_art_checksum != artifact.line_art_checksum
    ):
        return VectorStatus.STALE
    return VectorStatus.AVAILABLE


def coverage_report(
    *,
    asset_ids: list[str],
    artifacts: dict[str, ArtifactStamp],
    indexes: dict[str, EmbeddingIndex],
    supported_embedders: frozenset[str] | set[str],
) -> dict[str, dict[str, int]]:
    """Per-embedder status counts over a set of assets.

    The shape is ``{embedder: {status: count}}``; the API surfaces aggregate
    warnings from this (e.g. "mobileclip2_s2: 3 stale") rather than letting
    missing vectors degrade silently.
    """
    report: dict[str, dict[str, int]] = {}
    for embedder, index in indexes.items():
        counts = {status.value: 0 for status in VectorStatus}
        for asset_id in asset_ids:
            status = vector_status(
                embedder=embedder,
                supported_embedders=supported_embedders,
                entry=index.entries.get(asset_id),
                artifact=artifacts[asset_id],
            )
            counts[status.value] += 1
        report[embedder] = counts
    return report
