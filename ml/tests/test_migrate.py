"""Migration-safety tests: no old asset gains permission or human approval.

These tests are the executable form of the acceptance criteria for the
schema freeze. They sweep every v1 SFW method and enabled/review state and
assert the two safety properties plus the full mapping table from
``docs/contracts/migration-v1-to-v2.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from linescout_ml.manifest import (
    ArtifactContract,
    Manifest,
    is_servable,
    is_trainable,
    make_asset_id,
)
from linescout_ml.migrate import (
    MIGRATION_REVIEWER,
    V1Manifest,
    V2Manifest,
    convert_manifest,
    migrate_manifest,
    migrate_record,
)
from linescout_ml.taxonomy import (
    CurationBlocker,
    LearningSplit,
    PermissionBasis,
    ReviewState,
    SfwScreeningMethod,
    SfwVerdict,
)

V1_CONTRACT = ArtifactContract(pipeline_version="test-1", label_version="1", processing_revision=1)

ALL_V1_SFW_METHODS = ("source_rating", "opennsfw2", "source_rating+opennsfw2", "manual")


def _v1_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "asset_id": make_asset_id("synthetic", "item-1"),
        "source_dataset": "synthetic",
        "source_item_id": "item-1",
        "source_work_id": "work-1",
        "source_url": None,
        "license_id": "CC0-1.0",
        "original_path": "originals/a.png",
        "line_art_path": "line_art/a.png",
        "thumbnail_path": "thumbnails/a.png",
        "origin": "native_line_art",
        "extraction_model": None,
        "extraction_version": None,
        "primary_style": "manga_anime",
        "scopes": ["eye", "face_head"],
        "person_count": 1,
        "sfw": {"safe": True, "confidence": 0.99, "method": "manual"},
        "width": 512,
        "height": 512,
        "crop": None,
        "text_coverage": 0.0,
        "ink_coverage": 0.05,
        "phash": "0123456789abcdef",
        "quality_score": 0.9,
        "review": {"state": "accepted", "quality": 3},
        "split": "train",
        "enabled": True,
        "pipeline_version": "test-1",
        "source_checksum": "a" * 64,
        "line_art_checksum": "b" * 64,
        "thumbnail_checksum": "c" * 64,
    }
    record.update(overrides)
    return record


def _v1_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": 1, "dataset_version": "2026.01.01", "records": records}


def test_v1_shapes_still_parse() -> None:
    parsed = V1Manifest.model_validate(_v1_manifest([_v1_record()]))
    assert parsed.schema_version == 1
    assert parsed.records[0].sfw.method == "manual"


def test_mapping_table_field_by_field() -> None:
    v1 = V1Manifest.model_validate(
        _v1_manifest(
            [
                _v1_record(
                    review={
                        "state": "rejected",
                        "quality": 1,
                        "malformed_anatomy": True,
                        "poor_extraction": True,
                        "note": "bad",
                    },
                    split="gallery_only",
                    enabled=False,
                )
            ]
        )
    )
    record = migrate_record(v1.records[0])

    # Carried verbatim.
    assert record.asset_id == v1.records[0].asset_id
    assert record.source_dataset == "synthetic"
    assert record.source_item_id == "item-1"
    assert record.source_work_id == "work-1"
    assert record.license_id if hasattr(record, "license_id") else True
    assert record.permissions.license_id == "CC0-1.0"
    assert record.source_url is None
    assert record.primary_style.value == "manga_anime"
    assert record.person_count == 1
    assert record.person_count_approximate is False
    assert record.width == 512 and record.height == 512
    assert record.phash == "0123456789abcdef"
    assert record.source_checksum == "a" * 64

    # scopes[0] is the primary, the rest secondary (v1 order was score-desc).
    assert record.primary_scope.value == "eye"
    assert [s.value for s in record.secondary_scopes] == ["face_head"]

    # Booleans became named blockers.
    assert record.review.blockers == [CurationBlocker.ANATOMY, CurationBlocker.EXTRACTION]
    assert record.review.state is ReviewState.REJECTED
    assert record.review.quality == 1
    assert record.review.note == "bad"

    # gallery_only -> none; gallery membership carries (all v1 records were
    # gallery candidates).
    assert record.learning_split is LearningSplit.NONE
    assert record.gallery_member is True

    # Artifact versioning starts at the first generation.
    assert record.processing_revision == 1
    assert record.label_version == "1"

    # Identity fields unknown -> explicit None, never invented.
    assert record.parent_asset_id is None
    assert record.artist_id is None
    assert record.leakage_group_id is None


@pytest.mark.parametrize("method", ALL_V1_SFW_METHODS)
@pytest.mark.parametrize("safe", [True, False])
@pytest.mark.parametrize("state", ["unreviewed", "accepted", "rejected", "quarantined"])
@pytest.mark.parametrize("enabled", [True, False])
def test_no_asset_gains_permission_or_human_approval(
    method: str, safe: bool, state: str, enabled: bool
) -> None:
    """Sweep the whole v1 decision space and assert the safety properties."""
    # v1 invariant: enabled requires safe=true and an accepted review.
    v1_enabled = bool(enabled and safe and state == "accepted")
    v1 = V1Manifest.model_validate(
        _v1_manifest(
            [
                _v1_record(
                    sfw={"safe": safe, "confidence": 0.9, "method": method},
                    review={"state": state, "quality": 3 if state != "unreviewed" else None},
                    enabled=v1_enabled,
                )
            ]
        )
    )
    record = migrate_record(v1.records[0])

    # Property 1: no permission is granted, whatever v1 claimed.
    assert record.permissions.basis is PermissionBasis.UNKNOWN
    assert not record.allowed_uses.display
    assert not record.allowed_uses.training
    assert not record.allowed_uses.trace
    assert not is_servable(record, V1_CONTRACT)
    assert not is_trainable(record, V1_CONTRACT)

    # Property 2: human approval only ever comes from a v1 manual decision.
    if method == "manual":
        assert record.sfw_human is not None
        assert record.sfw_human.safe is safe
        assert record.sfw_human.reviewer == MIGRATION_REVIEWER
        assert record.sfw_human.decided_at is None
        assert record.sfw_screening is None
    else:
        assert record.sfw_human is None
        assert record.sfw_screening is not None
        assert record.sfw_screening.verdict is (SfwVerdict.SAFE if safe else SfwVerdict.UNSAFE)
        assert record.sfw_screening.method is SfwScreeningMethod(method)

    # Property 3: gold membership is never created.
    assert record.gold_member is False


def test_migration_report_counts_serving_eligibility_loss() -> None:
    data = _v1_manifest(
        [
            # Enabled + manual SFW: still not servable (permission unknown).
            _v1_record(asset_id=make_asset_id("synthetic", "item-1")),
            # Enabled + automated SFW: loses both serving *and* human approval.
            _v1_record(
                asset_id=make_asset_id("synthetic", "item-2"),
                source_item_id="item-2",
                sfw={"safe": True, "confidence": 0.99, "method": "source_rating"},
            ),
            # Not enabled: nothing to lose.
            _v1_record(
                asset_id=make_asset_id("synthetic", "item-3"),
                source_item_id="item-3",
                sfw={"safe": True, "confidence": 0.99, "method": "manual"},
                enabled=False,
            ),
        ]
    )
    manifest, report = migrate_manifest(data)

    assert isinstance(manifest, Manifest)
    assert manifest.schema_version == 3
    assert report.schema_version_from == 1
    assert report.schema_version_to == 3
    assert report.records_converted == 3
    assert report.sfw_human_carried == 2
    assert report.sfw_human_absent == 1
    assert report.serving_eligibility_lost == 2
    assert report.uses_granted == 0
    assert report.human_approvals_fabricated == 0
    assert report.gold_members_created == 0
    assert report.split_mapping == {"train": 3}
    # v1 records agree on one generation, so the contract is detected — but
    # permission/human-approval gates still keep every record disabled.
    assert report.contract_detected is True
    assert report.artifact_contract is not None
    assert manifest.servable_records == []


def test_migration_preserves_split_integrity_and_dedup() -> None:
    data = _v1_manifest(
        [
            _v1_record(split="train"),
            _v1_record(
                asset_id=make_asset_id("synthetic", "item-2"),
                source_item_id="item-2",
                split="test",
            ),
        ]
    )
    with pytest.raises(ValueError, match="spans splits"):
        migrate_manifest(data)


def test_migrated_v1_fixture_shape_is_valid_v2(tmp_path: Path) -> None:
    """A realistic v1 gallery (like the old committed fixture) migrates cleanly."""
    records = []
    for index in range(6):
        records.append(
            _v1_record(
                asset_id=make_asset_id("synthetic", f"item-{index}"),
                source_item_id=f"item-{index}",
                source_work_id=f"work-{index // 2}",
                sfw={"safe": True, "confidence": 0.95, "method": "source_rating"},
                review={"state": "unreviewed"},
                # Split assigned per work (two records share a work).
                split=["train", "train", "validation", "validation", "test", "test"][index],
                enabled=False,
            )
        )
    manifest, report = migrate_manifest(_v1_manifest(records))
    assert report.serving_eligibility_lost == 0  # nothing was enabled
    assert report.split_mapping == {"train": 2, "validation": 2, "test": 2}
    out = tmp_path / "migrated.json"
    out.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    reloaded = Manifest.model_validate_json(out.read_text(encoding="utf-8"))
    assert len(reloaded.records) == 6
    assert reloaded.servable_records == []


def test_cli_migrate_v1_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from linescout_ml.cli import main

    source = tmp_path / "v1.json"
    source.write_text(json.dumps(_v1_manifest([_v1_record()])), encoding="utf-8")
    out = tmp_path / "v3.json"
    report_path = tmp_path / "report.json"
    assert main(["migrate-v1", str(source), "--out", str(out), "--report", str(report_path)]) == 0
    captured = capsys.readouterr()
    assert "wrote" in captured.out
    assert "NOT servable" in captured.err

    migrated = Manifest.model_validate_json(out.read_text(encoding="utf-8"))
    assert migrated.schema_version == 3
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["uses_granted"] == 0
    assert report["human_approvals_fabricated"] == 0
    assert report["schema_version_to"] == 3


def test_convert_v1_to_v2_preserves_the_v2_guarantees(tmp_path: Path) -> None:
    """The versioned converter's 1→2 path is the same safe legacy shape."""
    v2, report = convert_manifest(_v1_manifest([_v1_record()]), from_version=1, to_version=2)
    assert isinstance(v2, V2Manifest)
    assert v2.schema_version == 2
    assert report.schema_version_to == 2
    assert report.uses_granted == 0
    assert report.human_approvals_fabricated == 0
    assert report.gold_members_created == 0
    record = v2.records[0]
    assert record.permissions.basis is PermissionBasis.UNKNOWN
    assert record.sfw_human is not None  # v1 manual decision


def test_convert_v2_to_v3_adds_the_contract_and_keeps_eligibility(tmp_path: Path) -> None:
    """2→3 is a pure materialisation: nothing new is ever granted."""
    v2, report = convert_manifest(_v1_manifest([_v1_record()]), from_version=1, to_version=2)
    v3, report3 = convert_manifest(
        v2.model_dump(), from_version=2, to_version=3, explicit_contract=V1_CONTRACT
    )
    assert isinstance(v3, Manifest)
    assert v3.schema_version == 3
    assert report3.artifact_contract == V1_CONTRACT
    assert not v3.servable_records  # permissions still unknown
    assert report3.uses_granted == 0
    assert report3.human_approvals_fabricated == 0
    v2_record = v2.records[0]
    v3_record = v3.records[0]
    assert v2_record.sfw_human is not None
    assert v3_record.sfw_human is not None
    assert v2_record.permissions.basis is PermissionBasis.UNKNOWN
    assert v3_record.permissions.basis is PermissionBasis.UNKNOWN
