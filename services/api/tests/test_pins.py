"""Acceptance tests for durable pins (``/api/v1/pins``).

Pins are state, not learning:

* they survive an API restart (SQLite-backed, hydrated on the next read);
* they survive ``learning_enabled = false`` and ``reset_affinities``;
* fixture-gallery pins and live-gallery pins never mix;
* every read revalidates against the current gallery, so a pin whose asset
  loses permission or eligibility is revoked with machine-readable reasons;
* ``trace_allowed`` in the projection is the asset's stored trace permission.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from linescout_api.config import Settings
from linescout_api.main import create_app
from tests.conftest import make_settings
from tests.test_events_and_preferences import event_body

TRACEABLE = "ls_synthetic_ee05835d4e94f4f3"  # native cartoon, trace permitted
NON_TRACEABLE = "ls_synthetic_f1becf0b9d67dcc3"  # extracted cartoon, trace forbidden
DISABLED = "ls_synthetic_dd4dbf8306f50951"  # rejected at review


def pin(client: TestClient, asset_id: str) -> dict[str, Any]:
    response = client.put(f"/api/v1/pins/{asset_id}")
    assert response.status_code == 200, response.text
    return dict(response.json())


def unpin(client: TestClient, asset_id: str) -> dict[str, Any]:
    response = client.delete(f"/api/v1/pins/{asset_id}")
    assert response.status_code == 200, response.text
    return dict(response.json())


def pinned_ids(body: dict[str, Any]) -> list[str]:
    return [item["asset_id"] for item in body["pins"]]


def test_pin_is_durable_across_restart(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        body = pin(client, TRACEABLE)
        assert pinned_ids(body) == [TRACEABLE]
        pinned_at = body["pins"][0]["pinned_at"]

    # A completely new process against the same database hydrates the pin.
    with TestClient(create_app(settings)) as client:
        body = client.get("/api/v1/pins").json()
        assert pinned_ids(body) == [TRACEABLE]
        assert body["pins"][0]["pinned_at"] == pinned_at
        assert body["revoked"] == []


def test_pinning_is_idempotent_and_keeps_the_original_timestamp(client: TestClient) -> None:
    first = pin(client, TRACEABLE)
    again = pin(client, TRACEABLE)
    assert pinned_ids(again) == [TRACEABLE]
    assert again["pins"][0]["pinned_at"] == first["pins"][0]["pinned_at"]


def test_unpin_is_idempotent_and_never_fails(client: TestClient) -> None:
    pin(client, TRACEABLE)
    assert pinned_ids(unpin(client, TRACEABLE)) == []
    # Unpinning something that is not pinned is already true, so it succeeds.
    assert pinned_ids(unpin(client, TRACEABLE)) == []
    assert pinned_ids(unpin(client, "ls_synthetic_0000000000000000")) == []


def test_ineligible_assets_cannot_be_pinned(client: TestClient) -> None:
    assert client.put(f"/api/v1/pins/{DISABLED}").status_code == 404
    assert client.put("/api/v1/pins/ls_synthetic_0000000000000000").status_code == 404
    assert client.get("/api/v1/pins").json()["pins"] == []


def test_projection_carries_stored_trace_permission(client: TestClient) -> None:
    pin(client, TRACEABLE)
    pin(client, NON_TRACEABLE)
    by_id = {item["asset_id"]: item for item in client.get("/api/v1/pins").json()["pins"]}
    assert by_id[TRACEABLE]["trace_allowed"] is True
    assert by_id[TRACEABLE]["origin"] == "native_line_art"
    assert by_id[NON_TRACEABLE]["trace_allowed"] is False
    assert by_id[NON_TRACEABLE]["origin"] == "extracted_line_art"
    # Display is still permitted for the non-traceable asset.
    assert client.get(by_id[NON_TRACEABLE]["thumbnail_url"]).status_code == 200


def test_pins_survive_learning_disabled_and_affinity_reset(
    client: TestClient, session_id: str
) -> None:
    client.put("/api/v1/preferences", json={"learning_enabled": False})
    # State changes are still allowed while learning is off...
    assert pinned_ids(pin(client, TRACEABLE)) == [TRACEABLE]
    # ...but no training event is accumulated for them.
    response = client.post(
        "/api/v1/events", json=event_body(session_id, "pin", "cartoon", asset_id=TRACEABLE)
    )
    assert response.status_code == 201
    assert response.json()["recorded"] is False

    client.put("/api/v1/preferences", json={"learning_enabled": True})
    client.put("/api/v1/preferences", json={"reset_affinities": True})
    # Forgetting learned affinities never throws away what the artist kept.
    assert pinned_ids(client.get("/api/v1/pins").json()) == [TRACEABLE]


def test_fixture_and_live_pins_never_mix(tmp_path: Path) -> None:
    """The namespace is stamped from the API's own mode, not from the client."""
    shared_db = tmp_path / "shared.sqlite3"
    fixture_settings = make_settings(tmp_path, db_path=shared_db, fixture_mode=True)
    live_settings = make_settings(tmp_path, db_path=shared_db, fixture_mode=False)

    with TestClient(create_app(fixture_settings)) as fixture_client:
        pin(fixture_client, TRACEABLE)
        assert fixture_client.get("/api/v1/pins").json()["gallery_kind"] == "fixture"

    with TestClient(create_app(live_settings)) as live_client:
        body = live_client.get("/api/v1/pins").json()
        assert body["gallery_kind"] == "live"
        assert body["pins"] == []
        pin(live_client, NON_TRACEABLE)
        assert pinned_ids(live_client.get("/api/v1/pins").json()) == [NON_TRACEABLE]

    with TestClient(create_app(fixture_settings)) as fixture_client:
        assert pinned_ids(fixture_client.get("/api/v1/pins").json()) == [TRACEABLE]


# ------------------------------------------------------- revalidation


def copy_gallery(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    from tests.conftest import SYNTHETIC_MANIFEST

    source_root = SYNTHETIC_MANIFEST.parent
    for subdir in ("originals", "line_art", "thumbnails"):
        shutil.copytree(source_root / subdir, tmp_path / subdir, dirs_exist_ok=True)
    manifest_path = tmp_path / "manifest.json"
    shutil.copyfile(SYNTHETIC_MANIFEST, manifest_path)
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


@pytest.fixture
def gallery(tmp_path: Path) -> Iterator[tuple[Settings, dict[str, Any], Path]]:
    manifest_path, data = copy_gallery(tmp_path)
    settings = make_settings(tmp_path, gallery_manifest=manifest_path, data_dir=tmp_path)
    yield settings, data, manifest_path


def _write(manifest_path: Path, data: dict[str, Any]) -> None:
    # A changed manifest hash is what triggers the cache rebuild.
    data["dataset_version"] = "2026.09.09-synthetic"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")


def test_pin_is_revoked_when_display_permission_is_withdrawn(
    gallery: tuple[Settings, dict[str, Any], Path],
) -> None:
    settings, data, manifest_path = gallery
    with TestClient(create_app(settings)) as client:
        assert pinned_ids(pin(client, TRACEABLE)) == [TRACEABLE]

    for record in data["records"]:
        if record["asset_id"] == TRACEABLE:
            record["allowed_uses"] = {"display": False, "training": False, "trace": False}
    _write(manifest_path, data)

    with TestClient(create_app(settings)) as client:
        body = client.get("/api/v1/pins").json()
        assert body["pins"] == []
        assert [item["asset_id"] for item in body["revoked"]] == [TRACEABLE]
        assert "display_not_permitted" in body["revoked"][0]["reasons"]
        # Revalidation is durable: the revoked pin does not come back.
        assert client.get("/api/v1/pins").json()["revoked"] == []
        assert client.get("/api/v1/pins").json()["pins"] == []


def test_pin_keeps_display_but_loses_trace_when_trace_permission_is_withdrawn(
    gallery: tuple[Settings, dict[str, Any], Path],
) -> None:
    settings, data, manifest_path = gallery
    with TestClient(create_app(settings)) as client:
        assert pin(client, TRACEABLE)["pins"][0]["trace_allowed"] is True

    for record in data["records"]:
        if record["asset_id"] == TRACEABLE:
            record["allowed_uses"]["trace"] = False
    _write(manifest_path, data)

    with TestClient(create_app(settings)) as client:
        body = client.get("/api/v1/pins").json()
        # Still displayable, so still pinned — but no longer traceable.
        assert pinned_ids(body) == [TRACEABLE]
        assert body["pins"][0]["trace_allowed"] is False
        assert body["revoked"] == []
        # And the trace event for it is now refused.
        assert (
            client.post(
                "/api/v1/events",
                json=event_body(
                    "0f0f0f0f-0f0f-4f0f-8f0f-0f0f0f0f0f0f",
                    "trace",
                    "cartoon",
                    asset_id=TRACEABLE,
                ),
            ).status_code
            == 403
        )


def test_pin_is_revoked_when_review_is_withdrawn(
    gallery: tuple[Settings, dict[str, Any], Path],
) -> None:
    settings, data, manifest_path = gallery
    with TestClient(create_app(settings)) as client:
        pin(client, TRACEABLE)

    for record in data["records"]:
        if record["asset_id"] == TRACEABLE:
            record["review"] = {
                "state": "quarantined",
                "quality": None,
                "blockers": [],
                "note": None,
            }
            record["gold_member"] = False
    _write(manifest_path, data)

    with TestClient(create_app(settings)) as client:
        body = client.get("/api/v1/pins").json()
        assert body["pins"] == []
        assert body["revoked"][0]["asset_id"] == TRACEABLE
        assert "review_quarantined" in body["revoked"][0]["reasons"]
