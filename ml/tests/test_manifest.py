from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from linescout_ml.cli import main
from linescout_ml.manifest import (
    AllowedUses,
    CropBox,
    Manifest,
    ManifestRecord,
    SfwHumanDecision,
    SfwScreening,
    check_parent_integrity,
    check_split_integrity,
    is_servable,
    is_trainable,
    learning_split_report,
    make_asset_id,
    manifest_json_schema,
)
from linescout_ml.synthetic import write_synthetic_dataset
from linescout_ml.taxonomy import (
    DEFAULT_STYLE_ORDER,
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

FIXTURE = Path(__file__).parent.parent / "fixtures" / "synthetic" / "manifest.json"


def _base_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "asset_id": make_asset_id("synthetic", "item-1"),
        "source_dataset": "synthetic",
        "source_item_id": "item-1",
        "source_work_id": "work-1",
        "original_path": "originals/a.png",
        "line_art_path": "line_art/a.png",
        "thumbnail_path": "thumbnails/a.png",
        "origin": "native_line_art",
        "primary_style": "manga_anime",
        "primary_scope": "eye",
        "secondary_scopes": ["face_head"],
        "person_count": 1,
        "sfw_human": {"safe": True, "reviewer": "test", "decided_at": None},
        "width": 512,
        "height": 512,
        "text_coverage": 0.0,
        "ink_coverage": 0.05,
        "phash": "0123456789abcdef",
        "quality_score": 0.9,
        "review": {"state": "accepted", "quality": 3},
        "learning_split": "train",
        "gallery_member": True,
        "gold_member": False,
        "permissions": {
            "license_id": "CC0-1.0",
            "basis": "public_domain",
        },
        "allowed_uses": {"display": True, "training": True, "trace": True},
        "pipeline_version": "test-1",
        "processing_revision": 1,
        "label_version": "1",
        "source_checksum": "a" * 64,
        "line_art_checksum": "b" * 64,
        "thumbnail_checksum": "c" * 64,
    }
    record.update(overrides)
    return record


def test_taxonomy_matches_spec() -> None:
    assert [s.value for s in ScopeLabel] == [
        "eye",
        "eyebrow",
        "mouth",
        "face_head",
        "hair",
        "hand",
        "foot",
        "upper_body_clothing",
        "full_body",
        "multi_character",
        "unknown",
    ]
    # eyebrow and mouth are retained as first-class detail scopes.
    assert "eyebrow" in {s.value for s in ScopeLabel}
    assert "mouth" in {s.value for s in ScopeLabel}
    assert [s.value for s in PrimaryStyle] == [
        "manga_anime",
        "western_ink",
        "realistic_academic",
        "cartoon",
        "gesture_sketch",
    ]
    assert [s.value for s in LineArtOrigin] == ["native_line_art", "extracted_line_art"]
    assert [s.value for s in LearningSplit] == ["train", "validation", "test", "none"]
    assert [s.value for s in CurationBlocker] == ["anatomy", "extraction"]
    assert [s.value for s in SfwVerdict] == ["safe", "unsafe", "unsure"]
    assert "manual" not in {s.value for s in SfwScreeningMethod}
    # Fixed default row order: manga, realistic, Western comic, cartoon, gesture.
    assert DEFAULT_STYLE_ORDER == (
        PrimaryStyle.MANGA_ANIME,
        PrimaryStyle.REALISTIC_ACADEMIC,
        PrimaryStyle.WESTERN_INK,
        PrimaryStyle.CARTOON,
        PrimaryStyle.GESTURE_SKETCH,
    )


def test_asset_id_is_deterministic_and_crop_sensitive() -> None:
    plain = make_asset_id("Manga109", "ARMS/012")
    assert plain == make_asset_id("Manga109", "ARMS/012")
    assert plain.startswith("ls_manga109_")
    assert plain != make_asset_id("Manga109", "ARMS/012", CropBox(x=0, y=0, width=10, height=10))


def test_valid_record_round_trips() -> None:
    record = ManifestRecord.model_validate(_base_record())
    assert record.origin is LineArtOrigin.NATIVE
    assert ManifestRecord.model_validate_json(record.model_dump_json()) == record


def test_person_count_may_be_null_for_non_human_sketches() -> None:
    record = ManifestRecord.model_validate(
        _base_record(person_count=None, primary_scope="full_body", secondary_scopes=[])
    )
    assert record.person_count is None
    assert record.person_count_approximate is False


def test_unknown_primary_scope_is_legal_but_not_acceptable() -> None:
    provisional = ManifestRecord.model_validate(
        _base_record(primary_scope="unknown", review={"state": "unreviewed"})
    )
    assert provisional.primary_scope is ScopeLabel.UNKNOWN
    with pytest.raises(ValidationError, match="known primary_scope"):
        ManifestRecord.model_validate(
            _base_record(
                primary_scope="unknown",
                review={"state": "accepted", "quality": 2},
            )
        )


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"origin": "extracted_line_art"}, "extraction_model"),
        ({"extraction_model": "anime2sketch", "extraction_version": "1"}, "native assets must not"),
        # Secondary scopes: unique, gallery-only, not the primary.
        ({"secondary_scopes": ["eye", "eye"]}, "unique"),
        ({"secondary_scopes": ["unknown"]}, "unknown"),
        ({"secondary_scopes": ["eye"]}, "must not repeat"),
        # multi_character requires a person count >= 2 (primary or secondary).
        ({"primary_scope": "multi_character", "secondary_scopes": [], "person_count": 1}, ">= 2"),
        (
            {
                "primary_scope": "full_body",
                "secondary_scopes": ["multi_character"],
                "person_count": None,
            },
            ">= 2",
        ),
        ({"person_count_approximate": True, "person_count": None}, "requires a person_count"),
        # Gold requires accepted, graded, unblocked review.
        (
            {"gold_member": True, "review": {"state": "rejected", "quality": 1}},
            "accepted review",
        ),
        ({"gold_member": True, "review": {"state": "accepted", "quality": None}}, "quality grade"),
        (
            {
                "gold_member": True,
                "review": {"state": "accepted", "quality": 2, "blockers": ["anatomy"]},
            },
            "blockers",
        ),
        # Permission grants must be justified.
        (
            {
                "permissions": {"license_id": "CC0-1.0", "basis": "unknown"},
                "allowed_uses": {"display": True, "training": False, "trace": False},
            },
            "permission basis",
        ),
        (
            {"allowed_uses": {"display": False, "training": False, "trace": True}},
            "trace use requires display",
        ),
        (
            {
                "permissions": {
                    "license_id": "CC-BY-4.0",
                    "basis": "license_terms",
                    "attribution_required": True,
                }
            },
            "credit line",
        ),
        # Parent identity.
        ({"parent_asset_id": make_asset_id("synthetic", "item-1")}, "must not equal asset_id"),
        # Legacy quality rule.
        ({"review": {"state": "unreviewed", "quality": 2}}, "quality cannot be set"),
        ({"width": 200}, "256"),
        ({"original_path": "/abs/path.png"}, "relative"),
        ({"original_path": "../escape.png"}, "relative"),
        ({"crop": {"x": 500, "y": 0, "width": 100, "height": 100}}, "exceeds"),
        ({"asset_id": "not-an-id"}, "pattern"),
        ({"extra_field": 1}, "extra"),
    ],
)
def test_invalid_records_are_rejected(overrides: dict[str, object], fragment: str) -> None:
    with pytest.raises(ValidationError) as info:
        ManifestRecord.model_validate(_base_record(**overrides))
    assert fragment in str(info.value)


def test_servable_and_trainable_predicates() -> None:
    base = ManifestRecord.model_validate(_base_record())
    assert is_servable(base)
    assert is_trainable(base)

    # Human SFW approval gates display: a safe automated screen is not enough.
    screening_only = base.model_copy(
        update={
            "sfw_human": None,
            "sfw_screening": SfwScreening(
                verdict=SfwVerdict.SAFE, confidence=0.99, method=SfwScreeningMethod.OPENNSFW2
            ),
        }
    )
    assert not is_servable(screening_only)
    # … and neither is an unsure or unsafe human decision.
    unsafe_human = base.model_copy(update={"sfw_human": SfwHumanDecision(safe=False, reviewer="t")})
    assert not is_servable(unsafe_human)

    # Gallery membership is independent of the learning split.
    not_a_member = base.model_copy(update={"gallery_member": False})
    assert not is_servable(not_a_member)
    assert is_trainable(not_a_member)

    # Training requires the train assignment plus the grant.
    val_asset = base.model_copy(update={"learning_split": LearningSplit.VALIDATION})
    assert is_servable(val_asset)
    assert not is_trainable(val_asset)
    no_training_grant = base.model_copy(
        update={"allowed_uses": AllowedUses(display=True, training=False, trace=False)}
    )
    assert is_servable(no_training_grant)
    assert not is_trainable(no_training_grant)

    # Blockers block both uses regardless of review state.
    blocked = base.model_copy(
        update={"review": base.review.model_copy(update={"blockers": [CurationBlocker.EXTRACTION]})}
    )
    assert not is_servable(blocked)
    assert not is_trainable(blocked)

    # An unsafe automated screen blocks training but not serving.
    unsafe_screen = base.model_copy(
        update={
            "sfw_screening": SfwScreening(
                verdict=SfwVerdict.UNSAFE, confidence=0.1, method=SfwScreeningMethod.OPENNSFW2
            )
        }
    )
    assert is_servable(unsafe_screen)
    assert not is_trainable(unsafe_screen)


def test_unknown_permission_grants_nothing() -> None:
    record = ManifestRecord.model_validate(
        _base_record(
            permissions={"license_id": "CC0-1.0", "basis": "unknown"},
            allowed_uses={"display": False, "training": False, "trace": False},
        )
    )
    assert record.permissions.basis is PermissionBasis.UNKNOWN
    assert not is_servable(record) and not is_trainable(record)


def test_manifest_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate asset_id"):
        Manifest.model_validate(
            {
                "dataset_version": "2026.09.08",
                "records": [_base_record(), _base_record()],
            }
        )


def test_split_integrity_flags_cross_split_works_and_leakage_groups() -> None:
    a = ManifestRecord.model_validate(_base_record())
    b = ManifestRecord.model_validate(
        _base_record(
            asset_id=make_asset_id("synthetic", "item-2"),
            source_item_id="item-2",
            learning_split="test",
        )
    )
    problems = check_split_integrity([a, b])
    assert len(problems) == 1 and "work-1" in problems[0]

    # `none` is not a conflict: unassigned assets leak nothing.
    b_none = b.model_copy(update={"learning_split": LearningSplit.NONE})
    assert check_split_integrity([a, b_none]) == []

    # A leakage group must not span assigned splits either.
    grouped_a = a.model_copy(update={"leakage_group_id": "artist-7"})
    grouped_b = b.model_copy(
        update={
            "leakage_group_id": "artist-7",
            "source_work_id": "work-2",
            "source_item_id": "item-3",
        }
    )
    problems = check_split_integrity([grouped_a, grouped_b])
    assert any("leakage group" in problem for problem in problems)


def test_manifest_rejects_dangling_parents_and_cycles() -> None:
    child = _base_record(
        asset_id=make_asset_id("synthetic", "item-2"),
        source_item_id="item-2",
        parent_asset_id=make_asset_id("synthetic", "missing-parent"),
    )
    # Dangling parent reference.
    with pytest.raises(ValidationError, match="parent_asset_id .* is not in the manifest"):
        Manifest.model_validate(
            {"dataset_version": "2026.09.08", "records": [_base_record(), child]}
        )

    # Self-referencing parent chain (cycle of two).
    base_dict = _base_record()
    child["parent_asset_id"] = base_dict["asset_id"]
    base_dict["parent_asset_id"] = child["asset_id"]
    with pytest.raises(ValidationError, match="cycle"):
        Manifest.model_validate({"dataset_version": "2026.09.08", "records": [base_dict, child]})

    assert check_parent_integrity([]) == []


def test_learning_split_report_targets_70_15_15() -> None:
    report = learning_split_report(
        Manifest.model_validate_json(FIXTURE.read_text(encoding="utf-8")).records
    )
    assert report["policy_target"] == {"train": 0.70, "validation": 0.15, "test": 0.15}
    assert set(report["proportions"]) == {"train", "validation", "test"}
    # `none` works are counted separately, never as a split.
    assert report["works_total"] == report["works_assigned"] + report["works_unassigned"]


def test_committed_fixture_is_valid_and_deterministic(tmp_path: Path) -> None:
    committed = Manifest.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    assert committed.schema_version == 2
    assert check_split_integrity(committed.records) == []
    assert any(r.review.state is ReviewState.REJECTED for r in committed.records)
    assert any(r.review.blockers for r in committed.records)
    assert any(r.gold_member for r in committed.records)
    assert {r.primary_style for r in committed.servable_records} == set(PrimaryStyle)

    regenerated_path = write_synthetic_dataset(tmp_path, count=len(committed.records), seed=7)
    regenerated = Manifest.model_validate_json(regenerated_path.read_text(encoding="utf-8"))
    assert regenerated.content_hash() == committed.content_hash(), (
        "synthetic fixture drifted; regenerate with `linescout-manifest synth`"
    )


def test_cli_validate_with_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = write_synthetic_dataset(tmp_path / "ds", count=9, seed=1)
    assert main(["validate", str(manifest_path), "--require-files"]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "servable" in out

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["records"][0]["thumbnail_path"] = "thumbnails/missing.png"
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(data), encoding="utf-8")
    assert (
        main(["validate", str(broken), "--data-root", str(tmp_path / "ds"), "--require-files"]) == 1
    )
    assert "missing thumbnail" in capsys.readouterr().err


def test_cli_rejects_v1_manifests_with_migration_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    v1 = {
        "schema_version": 1,
        "dataset_version": "2026.01.01",
        "records": [
            {
                "asset_id": make_asset_id("synthetic", "item-1"),
                "source_dataset": "synthetic",
                "source_item_id": "item-1",
                "source_work_id": "work-1",
                "license_id": "CC0-1.0",
                "original_path": "originals/a.png",
                "line_art_path": "line_art/a.png",
                "thumbnail_path": "thumbnails/a.png",
                "origin": "native_line_art",
                "primary_style": "manga_anime",
                "scopes": ["eye"],
                "person_count": 1,
                "sfw": {"safe": True, "confidence": 0.99, "method": "manual"},
                "width": 512,
                "height": 512,
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
        ],
    }
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(v1), encoding="utf-8")
    assert main(["validate", str(path)]) == 1
    assert "migrate-v1" in capsys.readouterr().err


def test_json_schema_exposes_enums() -> None:
    schema = manifest_json_schema()
    defs = schema["$defs"]
    assert set(defs["ScopeLabel"]["enum"]) == {s.value for s in ScopeLabel}
    assert set(defs["PrimaryStyle"]["enum"]) == {s.value for s in PrimaryStyle}
    assert set(defs["LearningSplit"]["enum"]) == {s.value for s in LearningSplit}
    assert set(defs["CurationBlocker"]["enum"]) == {s.value for s in CurationBlocker}
    assert set(defs["PermissionBasis"]["enum"]) == {s.value for s in PermissionBasis}
    assert set(defs["SfwVerdict"]["enum"]) == {s.value for s in SfwVerdict}
