"""Tests for the frozen vector-payload status contract."""

from __future__ import annotations

from linescout_ml.embeddings import (
    ArtifactStamp,
    EmbeddingEntry,
    EmbeddingIndex,
    VectorStatus,
    coverage_report,
    vector_status,
)

CHECKSUM_A = "a" * 64
CHECKSUM_B = "b" * 64


def _stamp(checksum: str = CHECKSUM_A, revision: int = 1) -> ArtifactStamp:
    return ArtifactStamp(processing_revision=revision, line_art_checksum=checksum)


def _entry(checksum: str = CHECKSUM_A, revision: int = 1) -> EmbeddingEntry:
    return EmbeddingEntry(artifact=_stamp(checksum, revision), shard="shard-000000.npz")


def test_available_when_artifact_binding_matches() -> None:
    assert (
        vector_status(
            embedder="mobileclip2_s2",
            supported_embedders={"mobileclip2_s2", "dinov2_vits14"},
            entry=_entry(),
            artifact=_stamp(),
        )
        is VectorStatus.AVAILABLE
    )


def test_stale_when_revision_or_checksum_changed() -> None:
    # Re-extracted line art (new checksum) invalidates the old vector.
    assert (
        vector_status(
            embedder="mobileclip2_s2",
            supported_embedders={"mobileclip2_s2"},
            entry=_entry(checksum=CHECKSUM_A),
            artifact=_stamp(checksum=CHECKSUM_B),
        )
        is VectorStatus.STALE
    )
    # A new processing revision invalidates it too, even with equal checksums.
    assert (
        vector_status(
            embedder="mobileclip2_s2",
            supported_embedders={"mobileclip2_s2"},
            entry=_entry(revision=1),
            artifact=_stamp(revision=2),
        )
        is VectorStatus.STALE
    )


def test_missing_when_no_entry_exists() -> None:
    assert (
        vector_status(
            embedder="mobileclip2_s2",
            supported_embedders={"mobileclip2_s2"},
            entry=None,
            artifact=_stamp(),
        )
        is VectorStatus.MISSING
    )


def test_unsupported_beats_everything() -> None:
    # An entry for an embedder this session does not load is unsupported,
    # never silently served.
    assert (
        vector_status(
            embedder="dinov2_vitb14_reg",
            supported_embedders={"mobileclip2_s2"},
            entry=_entry(),
            artifact=_stamp(),
        )
        is VectorStatus.UNSUPPORTED
    )
    assert (
        vector_status(
            embedder="dinov2_vitb14_reg",
            supported_embedders={"mobileclip2_s2"},
            entry=None,
            artifact=_stamp(),
        )
        is VectorStatus.UNSUPPORTED
    )


def test_coverage_report_counts_every_status() -> None:
    index = EmbeddingIndex(
        embedder="mobileclip2_s2",
        entries={
            "asset-1": _entry(),
            "asset-2": _entry(checksum=CHECKSUM_B),
        },
    )
    report = coverage_report(
        asset_ids=["asset-1", "asset-2", "asset-3"],
        artifacts={
            "asset-1": _stamp(),
            "asset-2": _stamp(),
            "asset-3": _stamp(),
        },
        indexes={"mobileclip2_s2": index},
        supported_embedders={"mobileclip2_s2"},
    )
    assert report["mobileclip2_s2"] == {
        "available": 1,
        "stale": 1,
        "missing": 1,
        "unsupported": 0,
    }
    assert index.covered_ids == {"asset-1", "asset-2"}
