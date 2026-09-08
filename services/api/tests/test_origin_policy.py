"""Host and Origin enforcement policy tests.

Explicit policy (see ``LoopbackSecurityMiddleware``):

* Host: loopback names only, plus any test-only names opted into via
  ``LINESCOUT_ADDITIONAL_ALLOWED_HOSTS``. Production settings leave that
  list empty, so Starlette's ``testserver`` must be rejected there.
* Origin, on mutations (POST/PUT/PATCH/DELETE) only: a present ``Origin``
  must exactly match a configured browser origin; an absent ``Origin`` is
  allowed (browsers always attach it cross-site, so Origin-less mutations
  cannot come from a cross-site browser context).
* Reads (GET) are not Origin-gated at this layer; CORS governs what a
  browser may read back.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from linescout_api.config import Settings
from linescout_api.gallery import enabled_assets
from linescout_api.main import create_app
from tests.conftest import SYNTHETIC_MANIFEST, make_settings

EVIL_ORIGIN = "http://evil.example"
ALLOWED_ORIGIN = "http://127.0.0.1:5173"  # default browser origin


def assert_forbidden_envelope(response: httpx.Response, code: str) -> None:
    assert response.status_code == 403
    body = response.json()
    assert body["error"]["code"] == code
    assert body["schema_version"] == 1
    assert body["retryable"] is False
    UUID(body["request_id"])
    assert response.headers["x-request-id"] == body["request_id"]


# ----------------------------------------------------------------- test-host isolation


def test_testserver_host_is_isolated_from_production(tmp_path: Path) -> None:
    """Without an explicit opt-in, the TestClient hostname is a DNS-rebinding reject."""
    settings = make_settings(tmp_path, additional_allowed_hosts=[])
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        assert client.get("/api/v1/health").status_code == 200
        response = client.get("/api/v1/health", headers={"host": "testserver"})
        assert_forbidden_envelope(response, "invalid_host")


def test_opted_in_testserver_host_is_allowed(client: TestClient) -> None:
    # The standard client fixture opts in through Settings, not middleware code.
    assert client.get("/api/v1/health").status_code == 200


def test_additional_allowed_hosts_must_be_bare_hostnames() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, additional_allowed_hosts=["http://testserver"])  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Settings(_env_file=None, additional_allowed_hosts=["testserver:8000"])  # type: ignore[call-arg]
    settings = Settings(_env_file=None, additional_allowed_hosts=["TestServer"])  # type: ignore[call-arg]
    assert settings.additional_allowed_hosts == ["testserver"]


# ----------------------------------------------------------------- origin configuration validation


@pytest.mark.parametrize(
    "origin",
    [
        "*",  # wildcards never match an Origin header exactly; refuse them
        "https://example.com/app",  # path
        "https://example.com/?q=1",  # query
        "https://example.com/#frag",
        "https://user@example.com",  # credentials
        "ftp://example.com",  # scheme
        "example.com",  # no scheme
        "",
    ],
)
def test_invalid_cors_origins_are_rejected(origin: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cors_origins=[origin])  # type: ignore[call-arg]


def test_cors_origins_are_normalized_and_deduplicated() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        cors_origins=["HTTP://LOCALHOST:5173/", "http://localhost:5173", "http://127.0.0.1:5173"],
    )
    assert settings.cors_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]


# ----------------------------------------------------------------- mutation origin matrix


def _event_body(session_id: str) -> dict[str, object]:
    return {
        # A fresh idempotency key per attempt: the origin matrix is about the
        # security middleware, not about replay behaviour.
        "event_uuid": str(uuid.uuid4()),
        "session_id": session_id,
        "asset_id": "ls_synthetic_f1becf0b9d67dcc3",
        "event": "open",
        "query_revision": 1,
    }


@pytest.fixture
def curation_client(tmp_path: Path) -> TestClient:
    manifest = json.loads(SYNTHETIC_MANIFEST.read_text(encoding="utf-8"))
    for record in manifest["records"]:
        for key in ("line_art_path", "thumbnail_path"):
            target = SYNTHETIC_MANIFEST.parent / record[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                Image.new("RGBA", (16, 16), (0, 0, 0, 0)).save(target, "PNG")
    with TestClient(create_app(make_settings(tmp_path, curation_mode=True))) as client:
        state = client.app.state.linescout
        # gold members must stay 'accepted' per the schema CHECK constraint, so
        # demote everyone to unreviewed *and* strip the gold flag together.
        state.connection.execute(
            "UPDATE assets SET review_state = 'unreviewed', review_quality = NULL, "
            "enabled = 0, gold_member = 0"
        )
        state.assets = enabled_assets(state.connection)
        yield client  # type: ignore[misc]


def test_events_mutation_origin_matrix(client: TestClient, session_id: str) -> None:
    evil = client.post(
        "/api/v1/events", json=_event_body(session_id), headers={"origin": EVIL_ORIGIN}
    )
    assert_forbidden_envelope(evil, "cross_origin_mutation_forbidden")
    allowed = client.post(
        "/api/v1/events", json=_event_body(session_id), headers={"origin": ALLOWED_ORIGIN}
    )
    assert allowed.status_code == 201
    # Origin-less (non-browser client): allowed by policy.
    originless = client.post("/api/v1/events", json=_event_body(session_id))
    assert originless.status_code == 201


def test_preferences_mutation_origin_matrix(client: TestClient) -> None:
    payload = {"learning_enabled": False}
    evil = client.put("/api/v1/preferences", json=payload, headers={"origin": EVIL_ORIGIN})
    assert_forbidden_envelope(evil, "cross_origin_mutation_forbidden")
    allowed = client.put("/api/v1/preferences", json=payload, headers={"origin": ALLOWED_ORIGIN})
    assert allowed.status_code == 200
    originless = client.put("/api/v1/preferences", json=payload)
    assert originless.status_code == 200


def _label_body(client: TestClient) -> dict[str, object]:
    candidate = client.get("/api/v1/curation/next").json()
    return {
        "asset_id": candidate["asset_id"],
        "expected_review_state": "unreviewed",
        "expected_label_version": candidate["label_version"],
        "decision": "keep",
        "quality": 3,
    }


def test_curation_labels_mutation_origin_matrix(curation_client: TestClient) -> None:
    evil = curation_client.post(
        "/api/v1/curation/labels",
        json=_label_body(curation_client),
        headers={"origin": EVIL_ORIGIN},
    )
    assert_forbidden_envelope(evil, "cross_origin_mutation_forbidden")
    allowed = curation_client.post(
        "/api/v1/curation/labels",
        json=_label_body(curation_client),
        headers={"origin": ALLOWED_ORIGIN},
    )
    assert allowed.status_code == 201
    originless = curation_client.post("/api/v1/curation/labels", json=_label_body(curation_client))
    assert originless.status_code == 201


def test_curation_snapshots_mutation_origin_matrix(curation_client: TestClient) -> None:
    evil = curation_client.post("/api/v1/curation/snapshots", headers={"origin": EVIL_ORIGIN})
    assert_forbidden_envelope(evil, "cross_origin_mutation_forbidden")
    allowed = curation_client.post("/api/v1/curation/snapshots", headers={"origin": ALLOWED_ORIGIN})
    assert allowed.status_code == 201
    originless = curation_client.post("/api/v1/curation/snapshots")
    assert originless.status_code == 201


def test_reads_are_not_origin_gated(client: TestClient) -> None:
    # CORS still governs what the browser may read back; the guard only
    # blocks Host spoofing and cross-origin mutations.
    assert client.get("/api/v1/health", headers={"origin": EVIL_ORIGIN}).status_code == 200
    assert client.get("/api/v1/preferences", headers={"origin": EVIL_ORIGIN}).status_code == 200
    assert (
        TestClient(client.app)
        .get("/api/v1/health", headers={"origin": EVIL_ORIGIN, "host": "evil.example"})
        .status_code
        == 403
    )


def test_origin_check_applies_to_search_algorithm_too(client: TestClient, session_id: str) -> None:
    from tests.conftest import png_bytes, post_search

    files = {"image": ("snapshot.png", png_bytes(None), "image/png")}
    evil = client.post(
        "/api/v1/search",
        data={
            "session_id": session_id,
            "revision": 1,
            "canvas_width": 2048,
            "canvas_height": 2048,
            "stroke_count": 0,
            "point_count": 0,
        },
        files=files,
        headers={"origin": EVIL_ORIGIN},
    )
    assert_forbidden_envelope(evil, "cross_origin_mutation_forbidden")
    status, _ = post_search(client, session_id, png_bytes(None), stroke_count=0, point_count=0)
    assert status == 200
