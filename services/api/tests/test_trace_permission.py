"""Acceptance tests for trace permission.

Trace permission comes from the recorded **source permission metadata**
(``allowed_uses.trace``), never from whether the line art is native or
extracted. The two cases that prove it are deliberately the inverse of the
usual correlation:

* a **native** asset whose source forbids tracing, and
* an **extracted** asset whose source explicitly permits tracing.

Both directions are enforced on the wire (search projection, pin projection,
the permissions endpoint) and in the server operations a client could try to
forge (``POST /events`` with ``event=trace``).
"""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from linescout_api.main import create_app
from tests.conftest import SYNTHETIC_MANIFEST, draw_figure, make_settings, png_bytes, post_search

# In the committed fixture these two correlate origin with trace permission;
# the fixture below inverts exactly that for these ids.
NATIVE = "ls_synthetic_ee05835d4e94f4f3"  # native_line_art, trace: true -> false
EXTRACTED = "ls_synthetic_f1becf0b9d67dcc3"  # extracted_line_art, trace: false -> true


@pytest.fixture
def inverted_client(tmp_path: Path) -> Iterator[TestClient]:
    """A gallery where trace permission is the inverse of the origin."""
    source_root = SYNTHETIC_MANIFEST.parent
    for subdir in ("originals", "line_art", "thumbnails"):
        shutil.copytree(source_root / subdir, tmp_path / subdir, dirs_exist_ok=True)
    manifest_path = tmp_path / "manifest.json"
    data: dict[str, Any] = json.loads(SYNTHETIC_MANIFEST.read_text(encoding="utf-8"))
    for record in data["records"]:
        if record["asset_id"] == NATIVE:
            assert record["origin"] == "native_line_art"
            record["allowed_uses"]["trace"] = False
        if record["asset_id"] == EXTRACTED:
            assert record["origin"] == "extracted_line_art"
            record["allowed_uses"]["trace"] = True
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    settings = make_settings(tmp_path, gallery_manifest=manifest_path, data_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        yield client


def _trace_event(client: TestClient, asset_id: str) -> Any:
    return client.post(
        "/api/v1/events",
        json={
            "event_uuid": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "asset_id": asset_id,
            "event": "trace",
            "query_revision": 1,
        },
    )


def test_permissions_endpoint_reports_the_stored_permission(inverted_client: TestClient) -> None:
    native = inverted_client.get(f"/api/v1/assets/{NATIVE}/permissions").json()
    assert native["origin"] == "native_line_art"
    assert native["allowed_display"] is True
    assert native["allowed_trace"] is False
    # No trace source is offered for an asset that may not be traced.
    assert native["trace_url"] is None

    extracted = inverted_client.get(f"/api/v1/assets/{EXTRACTED}/permissions").json()
    assert extracted["origin"] == "extracted_line_art"
    assert extracted["allowed_trace"] is True
    assert extracted["trace_url"] == f"/api/v1/assets/{EXTRACTED}/line-art"


def test_permissions_endpoint_hides_ineligible_assets(inverted_client: TestClient) -> None:
    assert (
        inverted_client.get("/api/v1/assets/ls_synthetic_dd4dbf8306f50951/permissions").status_code
        == 404
    )
    assert (
        inverted_client.get("/api/v1/assets/ls_synthetic_0000000000000000/permissions").status_code
        == 404
    )


def test_search_projection_never_infers_trace_from_origin(inverted_client: TestClient) -> None:
    _, body = post_search(
        inverted_client,
        str(uuid.uuid4()),
        png_bytes(draw_figure),
        stroke_count=14,
        point_count=900,
    )
    seen = {result["asset_id"]: result for group in body["groups"] for result in group["results"]}
    assert seen, "the fixture ranker returned no results to inspect"
    for result in seen.values():
        if result["asset_id"] == NATIVE:
            assert result["origin"] == "native_line_art"
            assert result["trace_allowed"] is False
        if result["asset_id"] == EXTRACTED:
            assert result["origin"] == "extracted_line_art"
            assert result["trace_allowed"] is True


def test_forged_trace_event_on_a_forbidden_asset_is_rejected(
    inverted_client: TestClient,
) -> None:
    response = _trace_event(inverted_client, NATIVE)
    assert response.status_code == 403
    payload = response.json()
    assert payload["error"]["code"] == "trace_not_permitted"
    assert payload["retryable"] is False
    # Nothing was learned from the refused interaction.
    connection = inverted_client.app.state.linescout.connection  # type: ignore[attr-defined]
    assert int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]) == 0
    # Non-trace interactions with the same asset remain fine (it is displayable).
    assert (
        inverted_client.post(
            "/api/v1/events",
            json={
                "event_uuid": str(uuid.uuid4()),
                "session_id": str(uuid.uuid4()),
                "asset_id": NATIVE,
                "event": "open",
                "query_revision": 1,
            },
        ).status_code
        == 201
    )


def test_explicitly_permitted_extracted_asset_can_be_traced(inverted_client: TestClient) -> None:
    assert _trace_event(inverted_client, EXTRACTED).status_code == 201
    pinned = inverted_client.put(f"/api/v1/pins/{EXTRACTED}").json()
    assert pinned["pins"][0]["trace_allowed"] is True
    assert inverted_client.get(f"/api/v1/assets/{EXTRACTED}/line-art").status_code == 200


def test_pin_projection_never_infers_trace_from_origin(inverted_client: TestClient) -> None:
    inverted_client.put(f"/api/v1/pins/{NATIVE}")
    inverted_client.put(f"/api/v1/pins/{EXTRACTED}")
    by_id = {item["asset_id"]: item for item in inverted_client.get("/api/v1/pins").json()["pins"]}
    assert by_id[NATIVE]["trace_allowed"] is False
    assert by_id[EXTRACTED]["trace_allowed"] is True


def test_trace_event_on_an_ineligible_asset_is_404_not_403(inverted_client: TestClient) -> None:
    """Eligibility is checked before permission; neither leaks the other."""
    assert _trace_event(inverted_client, "ls_synthetic_dd4dbf8306f50951").status_code == 404
