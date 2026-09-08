"""Eligibility enforcement through every public path (acceptance tests).

These tests mutate a copy of the committed synthetic gallery one violation at
a time and prove the canonical policy holds end-to-end:

* a quality-1 or missing-quality asset is never publicly served or searched;
* an unknown permission basis fails closed;
* an automated ``safe`` screen alone never publishes an asset;
* stale or byte-invalid derivatives never enter search and are 404 on the
  asset routes (disable-and-report, never a silent promotion);
* a legacy (v1) manifest converted to v3 loads with zero servable assets.

The SQLite ``enabled`` column is asserted only indirectly (through the public
API), since it is a derived cache of the manifest policy.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from linescout_ml.manifest import Manifest

from linescout_api.main import create_app
from tests.conftest import SYNTHETIC_MANIFEST, draw_figure, make_settings, png_bytes, post_search


def copy_gallery(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Copy the synthetic fixture's files and manifest into ``tmp_path``.

    Returns ``(manifest_path, parsed_manifest)``. ``tmp_path`` becomes the
    data root, so mutated manifests resolve the same relative image paths.
    """
    source_root = SYNTHETIC_MANIFEST.parent
    for subdir in ("originals", "line_art", "thumbnails"):
        shutil.copytree(source_root / subdir, tmp_path / subdir)
    manifest_path = tmp_path / "manifest.json"
    shutil.copyfile(SYNTHETIC_MANIFEST, manifest_path)
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def write_manifest(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    # The manifest must survive the same validation the API applies.
    Manifest.model_validate(data)
    return path


def enabled_asset_ids(client: TestClient) -> set[str]:
    """Collect the asset ids a *real* search response is allowed to surface.

    A blank canvas returns ``mode=insufficient`` with no groups, so this helper
    always searches a sufficient figure drawing. When no assets are eligible
    the ranker returns an empty result set (mode ``insufficient``/``empty``) —
    which is exactly the outcome the exclusion assertions check.
    """
    _, body = post_search(
        client,
        "00000000-0000-0000-0000-000000000000",
        png_bytes(draw_figure),
        stroke_count=14,
        point_count=900,
    )
    ids: set[str] = set()
    for group in body["groups"]:
        for result in group["results"]:
            ids.add(result["asset_id"])
    return ids


def test_search_control_records_are_surfaced(client: TestClient) -> None:
    """The search-exclusion tests below are meaningful (not vacuous): with the
    committed fixture, a healthy search response contains servable assets."""
    actual = enabled_asset_ids(client)
    assert actual, "a healthy gallery search must return results"
    committed = json.loads(SYNTHETIC_MANIFEST.read_text(encoding="utf-8"))
    # At least one accepted, quality-2/3, human-approved fixture asset is found.
    assert actual <= {record["asset_id"] for record in committed["records"]}
    assert len(actual) > 0


def test_quality_one_and_missing_quality_are_never_served(
    tmp_path: Path,
) -> None:
    _, data = copy_gallery(tmp_path)
    # Index 1 is accepted with quality 2 or 3; make it quality 1. Index 2 is
    # accepted; drop its review quality.
    data["records"][1]["review"] = {
        "state": "accepted",
        "quality": 1,
        "blockers": [],
        "note": None,
    }
    data["records"][2]["review"] = {
        "state": "accepted",
        "quality": None,
        "blockers": [],
        "note": None,
    }
    write_manifest(tmp_path, data)

    with TestClient(
        create_app(make_settings(tmp_path, gallery_manifest=tmp_path / "manifest.json"))
    ) as client:
        health = client.get("/api/v1/health").json()
        assert health["gallery_size"] == 21  # 24 - 1 rejected - 2 new disables
        for index in (1, 2):
            asset_id = data["records"][index]["asset_id"]
            # Neither path serves the file.
            assert client.get(f"/api/v1/assets/{asset_id}/thumbnail").status_code == 404
            assert client.get(f"/api/v1/assets/{asset_id}/line-art").status_code == 404
        surfaced = enabled_asset_ids(client)
        assert surfaced, "other fixture assets must remain searchable"
        assert not ({data["records"][1]["asset_id"], data["records"][2]["asset_id"]} & surfaced)


def test_unknown_permission_basis_fails_closed(tmp_path: Path) -> None:
    _, data = copy_gallery(tmp_path)
    record = data["records"][1]
    record["permissions"] = {
        "license_id": "CC0-1.0",
        "basis": "unknown",
        "permission_url": None,
        "attribution": None,
        "attribution_required": False,
    }
    # The record validator forbids unknown + display; the API-level rule that
    # matters is that *no* use is granted without a known basis.
    record["allowed_uses"] = {"display": False, "training": False, "trace": False}
    write_manifest(tmp_path, data)

    with TestClient(
        create_app(make_settings(tmp_path, gallery_manifest=tmp_path / "manifest.json"))
    ) as client:
        asset_id = data["records"][1]["asset_id"]
        assert client.get(f"/api/v1/assets/{asset_id}/thumbnail").status_code == 404
        surfaced = enabled_asset_ids(client)
        assert surfaced, "other fixture assets must remain searchable"
        assert asset_id not in surfaced


def test_automated_sfw_screen_alone_cannot_publish(tmp_path: Path) -> None:
    _, data = copy_gallery(tmp_path)
    record = data["records"][1]
    record["sfw_human"] = None
    record["sfw_screening"] = {
        "verdict": "safe",
        "confidence": 0.99,
        "method": "opennsfw2",
    }
    write_manifest(tmp_path, data)

    with TestClient(
        create_app(make_settings(tmp_path, gallery_manifest=tmp_path / "manifest.json"))
    ) as client:
        asset_id = data["records"][1]["asset_id"]
        assert client.get(f"/api/v1/assets/{asset_id}/thumbnail").status_code == 404
        surfaced = enabled_asset_ids(client)
        assert surfaced, "other fixture assets must remain searchable"
        assert asset_id not in surfaced


def test_unreviewed_and_quarantined_records_are_disabled(tmp_path: Path) -> None:
    _, data = copy_gallery(tmp_path)
    data["records"][1]["review"] = {
        "state": "unreviewed",
        "quality": None,
        "blockers": [],
        "note": None,
    }
    data["records"][2]["review"] = {
        "state": "quarantined",
        "quality": None,
        "blockers": [],
        "note": None,
    }
    write_manifest(tmp_path, data)
    with TestClient(
        create_app(make_settings(tmp_path, gallery_manifest=tmp_path / "manifest.json"))
    ) as client:
        for index in (1, 2):
            asset_id = data["records"][index]["asset_id"]
            assert client.get(f"/api/v1/assets/{asset_id}/thumbnail").status_code == 404
        surfaced = enabled_asset_ids(client)
        assert surfaced, "other fixture assets must remain searchable"
        assert not ({data["records"][1]["asset_id"], data["records"][2]["asset_id"]} & surfaced)


def test_stale_derivatives_never_enter_search(tmp_path: Path) -> None:
    _, data = copy_gallery(tmp_path)
    # Record 1 passes every serving gate but was produced by an older label
    # version than the manifest's artifact_contract.
    stale_id = data["records"][1]["asset_id"]
    data["records"][1]["label_version"] = "ancient-1"
    write_manifest(tmp_path, data)

    with TestClient(
        create_app(make_settings(tmp_path, gallery_manifest=tmp_path / "manifest.json"))
    ) as client:
        health = client.get("/api/v1/health").json()
        assert health["gallery_size"] == 22  # 24 - 1 rejected - 1 stale
        assert any(
            "disabled because their derivatives" in warning for warning in health["warnings"]
        )
        assert client.get(f"/api/v1/assets/{stale_id}/thumbnail").status_code == 404
        surfaced = enabled_asset_ids(client)
        assert surfaced, "other fixture assets must remain searchable"
        assert stale_id not in surfaced


def test_missing_or_tampered_file_disables_and_never_serves(tmp_path: Path) -> None:
    manifest_path, data = copy_gallery(tmp_path)
    # Tamper one thumbnail on disk *before* load: the loader reports it and
    # disables the row; everything else stays up.
    tampered_id = data["records"][1]["asset_id"]
    tampered_rel = data["records"][1]["thumbnail_path"]
    (tmp_path / tampered_rel).write_bytes(b"not the approved thumbnail")
    # Remove one line-art file entirely.
    missing_id = data["records"][2]["asset_id"]
    missing_rel = data["records"][2]["line_art_path"]
    (tmp_path / missing_rel).unlink()

    with TestClient(create_app(make_settings(tmp_path, gallery_manifest=manifest_path))) as client:
        health = client.get("/api/v1/health").json()
        assert health["gallery_size"] == 21  # 24 - 1 rejected - 2 derivative problems
        for asset_id in (tampered_id, missing_id):
            assert client.get(f"/api/v1/assets/{asset_id}/thumbnail").status_code in (404,)
            assert client.get(f"/api/v1/assets/{asset_id}/line-art").status_code in (404,)
        surfaced = enabled_asset_ids(client)
        assert surfaced, "other fixture assets must remain searchable"
        assert not ({tampered_id, missing_id} & surfaced)


def test_file_changed_after_startup_is_not_served(tmp_path: Path) -> None:
    manifest_path, data = copy_gallery(tmp_path)
    with TestClient(create_app(make_settings(tmp_path, gallery_manifest=manifest_path))) as client:
        asset_id = data["records"][1]["asset_id"]
        thumb = client.get(f"/api/v1/assets/{asset_id}/thumbnail")
        assert thumb.status_code == 200
        # Replace the file the manifest vouches for; the route must fail
        # closed on the recorded hash rather than serve the new bytes.
        (tmp_path / data["records"][1]["thumbnail_path"]).write_bytes(b"attacker-controlled bytes")
        after = client.get(f"/api/v1/assets/{asset_id}/thumbnail")
        assert after.status_code == 404
        assert after.json()["error"]["code"] == "asset_unavailable"


def test_converted_v1_manifest_loads_with_nothing_served(tmp_path: Path) -> None:
    """Legacy data migrates without silently promoting a single asset."""
    from linescout_ml.migrate import convert_manifest

    v1 = {
        "schema_version": 1,
        "dataset_version": "2025.01.01",
        "records": [
            {
                "asset_id": f"ls_synthetic_{index:016d}",
                "source_dataset": "synthetic",
                "source_item_id": f"item-{index}",
                "source_work_id": "work-1",
                "source_url": None,
                "license_id": "CC0-1.0",
                "original_path": f"originals/ls_synthetic_{index:016d}.png",
                "line_art_path": f"line_art/ls_synthetic_{index:016d}.png",
                "thumbnail_path": f"thumbnails/ls_synthetic_{index:016d}.png",
                "origin": "native_line_art",
                "extraction_model": None,
                "extraction_version": None,
                "primary_style": "cartoon",
                "scopes": ["eye", "face_head"],
                "person_count": 1,
                "sfw": {"safe": True, "confidence": 0.9, "method": "manual"},
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
                # Deliberately disagreeing generations: the conversion cannot
                # verify which one is current, so it must fail closed.
                "pipeline_version": "legacy-a" if index % 2 == 0 else "legacy-b",
                "source_checksum": "a" * 64,
                "line_art_checksum": "b" * 64,
                "thumbnail_checksum": "c" * 64,
            }
            for index in range(4)
        ],
    }

    v3, report = convert_manifest(v1, from_version=1, to_version=3)
    assert report.uses_granted == 0
    assert report.human_approvals_fabricated == 0
    assert report.contract_detected is False
    assert report.artifact_contract is None
    assert not v3.servable_records

    for record in v3.records:
        for rel in (record.original_path, record.line_art_path, record.thumbnail_path):
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\x89PNG\r\n\x1a\n")
    manifest_path = write_manifest(tmp_path, json.loads(v3.model_dump_json()))

    with TestClient(create_app(make_settings(tmp_path, gallery_manifest=manifest_path))) as client:
        health = client.get("/api/v1/health").json()
        # Ambiguous generations: contract is null, so nothing is verified
        # current — every derivative is conservatively disabled.
        assert health["gallery_size"] == 0
        assert any("artifact_contract is unknown" in warning for warning in health["warnings"])
        assert enabled_asset_ids(client) == set()
        for record in v3.records:
            assert client.get(f"/api/v1/assets/{record.asset_id}/thumbnail").status_code == 404
