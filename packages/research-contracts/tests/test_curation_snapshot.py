import base64
import hashlib
from datetime import UTC, datetime

import pytest

from drawable_research_contracts import (
    CurationSnapshotV1,
    SnapshotAssetV1,
    validate_curation_snapshot_files,
)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/rWQAAAAASUVORK5CYII="
)


def make_asset(path: str = "references/asset.png", person_count: int | None = 1) -> SnapshotAssetV1:
    return SnapshotAssetV1(
        asset_id="asset-1",
        source_id="source-1",
        relative_path=path,
        sha256=hashlib.sha256(PNG).hexdigest(),
        byte_length=len(PNG),
        media_type="image/png",
        width=256,
        height=256,
        source_family="work-1",
        learning_split="pilot_development",
        rights_status="approved",
        review_status="approved",
        gallery_eligible_candidate=True,
        terms_snapshot_sha256="a" * 64,
        source_registry_sha256="b" * 64,
        license_id="CC0",
        attribution="Artist / source attribution.",
        permissions={"local_display": True},
        labels={
            "style": "western_ink",
            "primary_scope": "eye",
            "secondary_scopes": [],
            "person_count": person_count,
            "quality": 2,
            "sfw": "safe",
            "character_useful": True,
            "text_coverage": 0,
            "clipped": False,
            "damaged": False,
            "duplicate_of": None,
            "review_seconds": 30,
            "keep_reject_reason": "Useful reviewed test asset.",
        },
        asset_revision=1,
        review_id="review-1",
        reviewer_id="reviewer-1",
    )


def test_snapshot_validates_review_only_scope_and_exact_local_bytes(tmp_path) -> None:
    asset_path = tmp_path / "references" / "asset.png"
    asset_path.parent.mkdir()
    asset_path.write_bytes(PNG)
    snapshot = CurationSnapshotV1(
        created_at=datetime.now(UTC),
        created_by="reviewer-1",
        viewpoint_rubric_sha256="c" * 64,
        items=[make_asset()],
    )
    result = validate_curation_snapshot_files(snapshot, tmp_path)
    assert result == {
        "status": "verified",
        "items": 1,
        "training_started": False,
        "data_uploaded": False,
    }
    asset_path.write_bytes(PNG + b"changed")
    with pytest.raises(ValueError, match="byte length changed"):
        validate_curation_snapshot_files(snapshot, tmp_path)


def test_snapshot_rejects_paths_outside_private_root_and_permission_gaps() -> None:
    with pytest.raises(ValueError, match="normalized and relative"):
        make_asset("../outside.png")
    asset = make_asset().model_dump()
    with pytest.raises(ValueError, match="local-display permission"):
        SnapshotAssetV1.model_validate({**asset, "permissions": {"local_display": False}})


def test_person_count_can_be_unknown_and_is_not_pose_capped() -> None:
    assert make_asset(person_count=None).labels.person_count is None
    assert make_asset(person_count=12).labels.person_count == 12
