"""Acceptance tests for interaction identity (POST /api/v1/events).

The contract under test:

* a **retry** (same ``event_uuid``, same authoritative payload) replays the
  original result and never writes a second row;
* a **conflicting reuse** (same ``event_uuid``, different payload) is a
  documented ``409 event_uuid_conflict``;
* ``style`` is not authoritative, so replaying with a different style is a
  replay, not a conflict;
* repeated interactions with *fresh* UUIDs coalesce onto the first
  contribution — identical id, identical ``created_at`` — so neither the
  decayed weight nor the contribution timestamp can be refreshed;
* with learning disabled nothing is accumulated at all.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from linescout_api.config import Settings
from linescout_api.db import connect
from linescout_api.main import create_app
from linescout_api.preferences import compute_affinities
from tests.conftest import make_settings
from tests.test_events_and_preferences import event_body

TRACEABLE_CARTOON = "ls_synthetic_ee05835d4e94f4f3"


@pytest.fixture
def live_client(tmp_path: Path) -> Iterator[TestClient]:
    """Learning only accumulates for the live gallery namespace."""
    with TestClient(create_app(make_settings(tmp_path, fixture_mode=False))) as client:
        yield client


def _rows(settings: Settings) -> list[dict[str, object]]:
    connection = connect(settings.db_path)
    try:
        return [
            dict(row) for row in connection.execute("SELECT * FROM events ORDER BY id").fetchall()
        ]
    finally:
        connection.close()


def test_retry_with_the_same_uuid_replays_the_original_result(
    client: TestClient, session_id: str, settings: Settings
) -> None:
    body = event_body(session_id, "open", "cartoon", asset_id=TRACEABLE_CARTOON)
    first = client.post("/api/v1/events", json=body)
    assert first.status_code == 201
    assert first.json()["replayed"] is False

    for _ in range(3):
        retry = client.post("/api/v1/events", json=body)
        assert retry.status_code == 201
        replay = retry.json()
        assert replay["id"] == first.json()["id"]
        assert replay["created_at"] == first.json()["created_at"]
        assert replay["event_uuid"] == body["event_uuid"]
        assert replay["recorded"] is True
        assert replay["replayed"] is True

    assert len(_rows(settings)) == 1


def test_ignored_style_does_not_make_a_replay_a_conflict(
    client: TestClient, session_id: str, settings: Settings
) -> None:
    identity = str(uuid.uuid4())
    first = client.post(
        "/api/v1/events",
        json=event_body(
            session_id,
            "open",
            "cartoon",
            asset_id=TRACEABLE_CARTOON,
            event_uuid=identity,
            include_style=True,
        ),
    )
    assert first.status_code == 201
    # Same interaction, different (ignored) client style: still a replay.
    replay = client.post(
        "/api/v1/events",
        json=event_body(
            session_id,
            "open",
            "manga_anime",
            asset_id=TRACEABLE_CARTOON,
            event_uuid=identity,
            include_style=True,
        ),
    )
    assert replay.status_code == 201
    assert replay.json()["replayed"] is True
    assert replay.json()["id"] == first.json()["id"]
    rows = _rows(settings)
    assert len(rows) == 1
    assert rows[0]["style"] == "cartoon"  # from the gallery, never the client


@pytest.mark.parametrize(
    "changed",
    [
        {"asset_id": "ls_synthetic_ac1f55b7390698a7"},
        {"event": "pin"},
        {"query_revision": 9},
        {"session_id": str(uuid.uuid4())},
    ],
)
def test_conflicting_reuse_of_a_uuid_is_a_documented_409(
    client: TestClient, session_id: str, settings: Settings, changed: dict[str, object]
) -> None:
    identity = str(uuid.uuid4())
    body = event_body(
        session_id, "open", "cartoon", asset_id=TRACEABLE_CARTOON, event_uuid=identity
    )
    assert client.post("/api/v1/events", json=body).status_code == 201

    conflicting = {**body, **changed}
    response = client.post("/api/v1/events", json=conflicting)
    assert response.status_code == 409
    payload = response.json()
    assert payload["error"]["code"] == "event_uuid_conflict"
    assert payload["error"]["field"] == "event_uuid"
    assert payload["retryable"] is False
    # The original write is untouched.
    assert len(_rows(settings)) == 1


def test_repeats_with_fresh_uuids_coalesce_onto_the_first_contribution(
    client: TestClient, session_id: str, settings: Settings
) -> None:
    first = client.post(
        "/api/v1/events",
        json=event_body(session_id, "trace", "cartoon", asset_id=TRACEABLE_CARTOON),
    ).json()
    for _ in range(4):
        repeat = client.post(
            "/api/v1/events",
            json=event_body(session_id, "trace", "cartoon", asset_id=TRACEABLE_CARTOON),
        )
        assert repeat.status_code == 201
        body = repeat.json()
        # Same row, same timestamp: the contribution was neither duplicated
        # nor refreshed, and the response points at the surviving identity.
        assert body["id"] == first["id"]
        assert body["created_at"] == first["created_at"]
        assert body["event_uuid"] == first["event_uuid"]
        assert body["replayed"] is True
    assert len(_rows(settings)) == 1


def test_replays_cannot_move_the_decay_clock(
    live_client: TestClient, session_id: str, tmp_path: Path
) -> None:
    """A retry storm must not make an old interaction look fresh."""
    settings: Settings = live_client.app.state.linescout.settings  # type: ignore[attr-defined]
    body = event_body(session_id, "trace", "cartoon", asset_id=TRACEABLE_CARTOON)
    assert live_client.post("/api/v1/events", json=body).status_code == 201

    connection = live_client.app.state.linescout.connection  # type: ignore[attr-defined]
    connection.execute("UPDATE events SET created_at = '2026-01-01T00:00:00.000Z'")
    # A fixed clock, so any difference is the stored contribution moving.
    frozen = datetime(2026, 3, 1, tzinfo=UTC)
    before = compute_affinities(connection, settings.preference_half_life_days, now=frozen)

    # Both a retry (same uuid) and a repeat (fresh uuid) must be inert.
    assert live_client.post("/api/v1/events", json=body).status_code == 201
    assert (
        live_client.post(
            "/api/v1/events",
            json=event_body(session_id, "trace", "cartoon", asset_id=TRACEABLE_CARTOON),
        ).status_code
        == 201
    )
    assert compute_affinities(connection, settings.preference_half_life_days, now=frozen) == before
    assert int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]) == 1
    assert (
        connection.execute("SELECT created_at FROM events").fetchone()["created_at"]
        == "2026-01-01T00:00:00.000Z"
    )


def test_learning_disabled_accumulates_nothing_and_keeps_no_identity(
    client: TestClient, session_id: str, settings: Settings
) -> None:
    client.put("/api/v1/preferences", json={"learning_enabled": False})
    body = event_body(session_id, "open", "cartoon", asset_id=TRACEABLE_CARTOON)
    response = client.post("/api/v1/events", json=body)
    assert response.status_code == 201
    payload = response.json()
    assert payload["recorded"] is False
    assert payload["id"] == 0
    assert payload["event_uuid"] == body["event_uuid"]
    assert _rows(settings) == []

    # Re-enabling learning does not resurrect the dropped event, and the same
    # UUID may still be used for the first real write.
    client.put("/api/v1/preferences", json={"learning_enabled": True})
    stored = client.post("/api/v1/events", json=body)
    assert stored.status_code == 201
    assert stored.json()["recorded"] is True
    assert len(_rows(settings)) == 1
