import base64
import hashlib
import json
import sys
from datetime import UTC, datetime

from drawable_research_contracts import CurationSnapshotV1, SnapshotAssetV1
from drawable_research_contracts.readiness import REQUIREMENTS_REVISION

from drawable_ml.cli import main

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/rWQAAAAASUVORK5CYII="
)


def make_snapshot(path: str) -> CurationSnapshotV1:
    item = SnapshotAssetV1(
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
        attribution="Recorded attribution.",
        permissions={"local_display": True},
        labels={
            "style": "western_ink",
            "primary_scope": "eye",
            "secondary_scopes": [],
            "person_count": 1,
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
    return CurationSnapshotV1(
        created_at=datetime.now(UTC),
        created_by="reviewer-1",
        viewpoint_rubric_sha256="c" * 64,
        items=[item],
    )


def test_cli_validates_snapshot_files_without_training(tmp_path, monkeypatch, capsys) -> None:
    data_root = tmp_path / "data"
    asset_path = data_root / "references" / "asset.png"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(PNG)
    snapshot_path = data_root / "reports" / "private" / "curation-snapshots" / "snapshot.json"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(
        json.dumps(make_snapshot("references/asset.png").model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drawable-ml",
            "curation",
            "validate-snapshot",
            "--data-root",
            str(data_root),
            "--snapshot",
            "reports/private/curation-snapshots/snapshot.json",
        ],
    )
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["training_started"] is False
    assert report["data_uploaded"] is False
    assert report["items"] == 1

    asset_path.write_bytes(PNG + b"tampered")
    assert main() == 2


def test_train_prepare_fails_closed_when_dataset_manifest_is_missing(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drawable-ml",
            "train",
            "prepare",
            "--data-root",
            str(tmp_path),
            "--manifest",
            "manifests/not-created.json",
            "--output",
            "reports/readiness.json",
        ],
    )

    assert main() == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked"
    assert report["training_authorized"] is False
    assert report["training_started"] is False
    assert report["optimizer_created"] is False
    assert report["backward_passes_run"] is False
    assert report["data_uploaded"] is False
    assert any(
        blocker.startswith("dataset_manifest_not_validated:") for blocker in report["blockers"]
    )


def test_cli_attests_hash_bound_readiness_report_without_authorizing_training(
    tmp_path, monkeypatch, capsys
) -> None:
    data_root = tmp_path / "private-data"
    evidence_path = data_root / "manifests" / "source.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text('{"sample":"fixed-v1"}\n', encoding="utf-8")
    report_path = data_root / "reports" / "source-yield.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                    "schema_version": 2,
                    "requirements_revision": REQUIREMENTS_REVISION,
                "artifact_type": "drawable-gate-report",
                "gate_id": "source_yield_review",
                "producer": "owner-source-yield-report",
                "produced_at": datetime.now(UTC).isoformat(),
                "checks": {
                    "sample_identity_verified": True,
                    "owner_judgments_complete": True,
                    "review_time_measured": True,
                },
                    "measurements": {
                    "reviewed_candidates": 12,
                    "median_review_seconds_per_candidate": 42,
                    "useful_eye_crop_candidates": 3,
                    "viewpoint_identifiable_candidates": 8,
                        "ambiguous_viewpoint_candidates": 1,
                    },
                    "profile_hashes": {
                        "source_selection": hashlib.sha256(b"selected-profile").hexdigest(),
                    },
                "inputs": [
                    {
                        "path": "manifests/source.json",
                        "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                    }
                ],
                "summary": "Reviewed the fixed source-yield sample and recorded active review time.",
            }
        ),
        encoding="utf-8",
    )
    ledger_path = "reports/readiness-ledger-v2.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drawable-ml",
            "readiness",
            "attest",
            "--data-root",
            str(data_root),
            "--gate",
            "source_yield_review",
            "--report",
            "reports/source-yield.json",
            "--reviewer-id",
            "owner-local",
            "--review-note",
            "Reviewed the fixed sample identities, labels and timing.",
            "--output",
            ledger_path,
        ],
    )

    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    ledger = json.loads((data_root / ledger_path).read_text(encoding="utf-8"))
    assert result["status"] == "owner_attested_pending_stage_validation"
    assert ledger["attestations"]["source_yield_review"]["reviewer_id"] == "owner-local"
    assert result["training_authorized"] is False
    assert result["training_started"] is False
