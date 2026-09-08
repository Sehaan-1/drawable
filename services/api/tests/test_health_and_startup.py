from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient

from linescout_api.config import DevicePolicy
from linescout_api.db import connect, list_migrations, migrate, schema_version, split_statements
from linescout_api.device import detect_device
from linescout_api.main import create_app
from tests.conftest import SYNTHETIC_MANIFEST, make_settings, png_bytes, post_search


def test_invalid_host_header_is_rejected(client: TestClient) -> None:
    response = client.get("http://evil.example/api/v1/health")
    assert response.status_code == 403
    body = response.json()
    assert body["error"]["code"] == "invalid_host"
    assert body["schema_version"] == 1
    assert body["retryable"] is False
    UUID(body["request_id"])
    assert response.headers["x-request-id"] == body["request_id"]


def test_cross_origin_mutation_is_forbidden(client: TestClient, session_id: str) -> None:
    response = client.post(
        "/api/v1/events",
        json={
            "session_id": session_id,
            "asset_id": "ls_synthetic_0000000000000000",
            "event": "open",
            "style": "cartoon",
            "query_revision": 1,
        },
        headers={"origin": "http://evil.example"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "cross_origin_mutation_forbidden"
    # Same-origin (or no Origin, as in TestClient) mutations still work.
    status, _ = post_search(client, session_id, png_bytes(None), stroke_count=0, point_count=0)
    assert status == 200


def test_ready_probe_is_200_when_healthy(client: TestClient) -> None:
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}
    UUID(response.headers["x-request-id"])


def test_ready_probe_is_503_when_not_ready(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, gallery_manifest=tmp_path / "missing.json")
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["error"]["code"] == "not_ready"
        assert body["retryable"] is True
        assert body["schema_version"] == 1
        UUID(body["request_id"])
        assert response.headers["retry-after"] == "5"
        assert response.headers["x-request-id"] == body["request_id"]


def test_health_reports_every_spec_field(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    for key in (
        "ready",
        "cuda_available",
        "device",
        "gpu_name",
        "models",
        "dataset_version",
        "index_version",
        "gallery_size",
        "disabled_branches",
        "warmup",
        "warnings",
    ):
        assert key in body, key
    assert body["ready"] is True
    assert body["fixture_mode"] is True
    assert body["gallery_size"] == 23  # 24 synthetic records, one disabled
    assert body["dataset_version"] == "2026.09.08-synthetic"
    assert body["schema_version"] == 2  # the v2 contract migration
    assert {model["name"] for model in body["models"]} == {
        "semantic",
        "structural",
        "stroke",
        "scope",
        "pose",
    }


def test_cpu_fallback_is_reported_not_fatal(tmp_path: Path) -> None:
    info = detect_device(DevicePolicy.AUTO)
    # The CI/sandbox machine has no CUDA; the API must still come up with a warning.
    if not info.cuda_available:
        with TestClient(create_app(make_settings(tmp_path))) as client:
            body = client.get("/api/v1/health").json()
            assert body["device"] == "cpu"
            assert body["ready"] is True
            assert any("CPU fallback" in warning for warning in body["warnings"])


def test_forced_cuda_without_gpu_fails_readiness(tmp_path: Path) -> None:
    info = detect_device(DevicePolicy.CUDA)
    if info.cuda_available:
        return  # nothing to assert on a GPU machine
    with TestClient(create_app(make_settings(tmp_path, device=DevicePolicy.CUDA))) as client:
        body = client.get("/api/v1/health").json()
        assert body["ready"] is False
        assert body["device"] == "cpu"


def test_missing_manifest_fails_readiness_with_setup_error(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, gallery_manifest=tmp_path / "nope" / "manifest.json")
    with TestClient(create_app(settings)) as client:
        body = client.get("/api/v1/health").json()
        assert body["ready"] is False
        assert body["gallery_size"] == 0
        assert "manifest not found" in body["warnings"][0]


def test_invalid_manifest_fails_readiness(tmp_path: Path) -> None:
    data = json.loads(SYNTHETIC_MANIFEST.read_text(encoding="utf-8"))
    # A use granted without a permission basis is a hard v2 violation.
    data["records"][0]["permissions"]["basis"] = "unknown"
    data["records"][0]["allowed_uses"]["display"] = True
    broken = tmp_path / "manifest.json"
    broken.write_text(json.dumps(data), encoding="utf-8")
    with TestClient(create_app(make_settings(tmp_path, gallery_manifest=broken))) as client:
        body = client.get("/api/v1/health").json()
        assert body["ready"] is False
        assert "failed validation" in body["warnings"][0]


def test_no_manifest_in_fixture_mode_is_ready_but_empty(tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path, gallery_manifest=None))) as client:
        body = client.get("/api/v1/health").json()
        assert body["ready"] is True
        assert body["gallery_size"] == 0
        assert body["dataset_version"] is None


def test_fixture_mode_defaults_to_the_synthetic_gallery(tmp_path: Path) -> None:
    from linescout_api.config import SYNTHETIC_MANIFEST, Settings

    settings = Settings(_env_file=None, db_path=tmp_path / "d.sqlite3")  # type: ignore[call-arg]
    assert settings.gallery_manifest == SYNTHETIC_MANIFEST
    assert Settings(_env_file=None, gallery_manifest="").gallery_manifest is None  # type: ignore[call-arg]
    assert Settings(_env_file=None, fixture_mode=False).gallery_manifest is None  # type: ignore[call-arg]
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/health").json()["gallery_size"] == 23


def test_migrations_are_idempotent_and_wal(tmp_path: Path) -> None:
    connection = connect(tmp_path / "m.sqlite3")
    first = migrate(connection)
    assert first == [name for _, name, _ in list_migrations()]
    assert migrate(connection) == []
    assert schema_version(connection) == len(first)
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {
        "assets",
        "asset_scopes",
        "events",
        "preferences",
        "curation_labels",
        "search_log",
        "gallery_versions",
    } <= tables
    connection.close()


def test_split_statements_respects_semicolons_in_literals() -> None:
    sql = (
        "CREATE TABLE t (a TEXT CHECK (a IN ('x;y', 'z')));\n"
        "-- comment; with semicolon\n"
        "INSERT INTO t VALUES ('a;b');\n"
    )
    statements = split_statements(sql)
    assert len(statements) == 2
    assert statements[0].startswith("CREATE TABLE")
    assert statements[1].startswith("INSERT")


def test_gallery_reload_is_skipped_when_hash_unchanged(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)):
        pass
    with TestClient(create_app(settings)) as client:
        connection = connect(settings.db_path)
        loaded = connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        assert loaded == 24
        assert connection.execute("SELECT COUNT(*) FROM asset_scopes").fetchone()[0] > 24
        assert client.get("/api/v1/health").json()["gallery_size"] == 23
        connection.close()


def test_manifest_ids_and_sqlite_ids_are_one_to_one(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)):
        manifest = json.loads(SYNTHETIC_MANIFEST.read_text(encoding="utf-8"))
        connection = connect(settings.db_path)
        db_ids = {row[0] for row in connection.execute("SELECT asset_id FROM assets")}
        assert db_ids == {record["asset_id"] for record in manifest["records"]}
        connection.close()


def _insert_v2_asset(connection: sqlite3.Connection, **overrides: object) -> None:
    """Insert a minimal, fully-granted v2 asset row, applying named overrides."""
    columns = (
        "asset_id, source_dataset, source_item_id, source_work_id, license_id,"
        " permission_basis, allowed_display, allowed_training, allowed_trace,"
        " original_path, line_art_path, thumbnail_path, origin, primary_style,"
        " primary_scope, secondary_scopes_json, person_count, width, height,"
        " text_coverage, ink_coverage, phash, quality_score, review_state,"
        " review_quality, blockers_json, learning_split, gallery_member, gold_member,"
        " pipeline_version, processing_revision, label_version,"
        " source_checksum, line_art_checksum, thumbnail_checksum, sfw_human_safe, enabled"
    )
    values: list[object] = [
        "ls_x_0000000000000000",
        "s",
        "i",
        "w",
        "l",
        "first_party",
        1,
        1,
        1,
        "o",
        "la",
        "t",
        "native_line_art",
        "cartoon",
        "eye",
        "[]",
        1,
        300,
        300,
        0,
        0.1,
        "0000000000000000",
        0.5,
        "accepted",
        3,
        "[]",
        "train",
        1,
        0,
        "p",
        1,
        "1",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        1,
        1,
    ]
    names = [name.strip() for name in columns.split(",")]
    row = dict(zip(names, values, strict=True))
    row.update(overrides)
    ordered = [row[name] for name in names]
    placeholders = ",".join("?" * len(names))
    connection.execute(
        f"INSERT INTO assets ({columns}) VALUES ({placeholders})",
        ordered,  # noqa: S608
    )


def test_sqlite_refuses_enabled_without_human_sfw_approval(tmp_path: Path) -> None:
    """enabled=1 with no human SFW decision must not insert (no gain by default)."""
    import pytest

    connection = connect(tmp_path / "c.sqlite3")
    migrate(connection)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_v2_asset(connection, sfw_human_safe=None)
    connection.close()


def test_sqlite_refuses_enabled_unreviewed_assets(tmp_path: Path) -> None:
    import pytest

    connection = connect(tmp_path / "c.sqlite3")
    migrate(connection)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_v2_asset(connection, review_state="unreviewed", review_quality=None)
    connection.close()


def test_sqlite_refuses_enabled_blocked_assets(tmp_path: Path) -> None:
    import pytest

    connection = connect(tmp_path / "c.sqlite3")
    migrate(connection)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_v2_asset(connection, blockers_json='["anatomy"]')
    connection.close()


def test_sqlite_refuses_display_without_known_permission(tmp_path: Path) -> None:
    import pytest

    connection = connect(tmp_path / "c.sqlite3")
    migrate(connection)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_v2_asset(connection, permission_basis="unknown")
    connection.close()
