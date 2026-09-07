from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from sqlite3 import Connection

from fastapi.testclient import TestClient
from linescout_ml.taxonomy import DEFAULT_STYLE_ORDER, PrimaryStyle

from linescout_api.config import Settings
from linescout_api.db import connect, migrate
from linescout_api.fixture_ranker import order_style_rows
from linescout_api.preferences import compute_affinities

DEFAULT = [style.value for style in DEFAULT_STYLE_ORDER]

# Enabled synthetic assets, one per style. Events look the style up from the gallery.
_STYLE_ASSETS = {
    "manga_anime": "ls_synthetic_ac1f55b7390698a7",
    "western_ink": "ls_synthetic_4822d3e3a4cefde7",
    "realistic_academic": "ls_synthetic_34579de628de2f89",
    "cartoon": "ls_synthetic_f1becf0b9d67dcc3",
    "gesture_sketch": "ls_synthetic_fa19d028df6f073c",
}


def _event(
    client: TestClient,
    session_id: str,
    event: str,
    style: str,
    *,
    asset_id: str | None = None,
    query_revision: int = 4,
) -> int:
    response = client.post(
        "/api/v1/events",
        json={
            "session_id": session_id,
            "asset_id": asset_id or _STYLE_ASSETS.get(style, _STYLE_ASSETS["cartoon"]),
            "event": event,
            "style": style,
            "query_revision": query_revision,
        },
    )
    return response.status_code


def _event_count(settings: Settings) -> int:
    connection = connect(settings.db_path)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
    finally:
        connection.close()


def test_new_profile_uses_fixed_default_order(client: TestClient) -> None:
    body = client.get("/api/v1/preferences").json()
    assert body["selected_style"] is None
    assert body["learning_enabled"] is True
    assert body["row_order"] == DEFAULT
    # Laplace smoothing: no events means uniform affinities.
    assert len({affinity["affinity"] for affinity in body["affinities"]}) == 1


def test_events_are_recorded_with_server_timestamp(client: TestClient, session_id: str) -> None:
    response = client.post(
        "/api/v1/events",
        json={
            "session_id": session_id,
            "asset_id": _STYLE_ASSETS["cartoon"],
            "event": "pin",
            "style": "cartoon",
            "query_revision": 1,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["id"] >= 1
    datetime.fromisoformat(body["created_at"].replace("Z", "+00:00"))  # parses


def test_invalid_event_payloads_are_422(client: TestClient, session_id: str) -> None:
    assert _event(client, session_id, "like", "cartoon") == 422
    assert _event(client, session_id, "pin", "oil") == 422
    assert _event(client, "nope", "pin", "cartoon") == 422


def test_unknown_asset_is_404(client: TestClient, session_id: str) -> None:
    response = client.post(
        "/api/v1/events",
        json={
            "session_id": session_id,
            "asset_id": "ls_synthetic_0000000000000000",
            "event": "open",
            "style": "cartoon",
            "query_revision": 1,
        },
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "asset_not_found"


def test_disabled_asset_is_404(client: TestClient, session_id: str) -> None:
    assert (
        _event(
            client,
            session_id,
            "open",
            "cartoon",
            asset_id="ls_synthetic_dd4dbf8306f50951",
        )
        == 404
    )


def test_event_style_comes_from_gallery_not_client(
    client: TestClient, session_id: str, settings: Settings
) -> None:
    response = client.post(
        "/api/v1/events",
        json={
            "session_id": session_id,
            "asset_id": _STYLE_ASSETS["cartoon"],
            "event": "open",
            "style": "manga_anime",  # spoof
            "query_revision": 1,
        },
    )
    assert response.status_code == 201
    connection = connect(settings.db_path)
    try:
        row = connection.execute("SELECT style FROM events").fetchone()
        assert row["style"] == "cartoon"
    finally:
        connection.close()


def test_learned_affinity_reorders_rows(client: TestClient, session_id: str) -> None:
    assert _event(client, session_id, "trace", "cartoon") == 201
    assert _event(client, session_id, "pin", "cartoon") == 201
    assert _event(client, session_id, "open", "gesture_sketch") == 201
    order = client.get("/api/v1/preferences").json()["row_order"]
    assert order[0] == "cartoon"
    assert order[1] == "gesture_sketch"
    assert order[2:] == ["manga_anime", "realistic_academic", "western_ink"]  # default tie-break


def test_unpin_cancels_pin(client: TestClient, session_id: str) -> None:
    _event(client, session_id, "pin", "western_ink")
    _event(client, session_id, "unpin", "western_ink")
    assert client.get("/api/v1/preferences").json()["row_order"] == DEFAULT


def test_unpin_without_pin_does_not_penalize(client: TestClient, session_id: str) -> None:
    _event(client, session_id, "unpin", "cartoon")
    assert client.get("/api/v1/preferences").json()["row_order"] == DEFAULT


def test_repeated_clicks_do_not_write_duplicate_rows(
    client: TestClient, session_id: str, settings: Settings
) -> None:
    assert _event(client, session_id, "trace", "cartoon") == 201
    assert _event(client, session_id, "trace", "cartoon") == 201
    assert _event(client, session_id, "trace", "cartoon") == 201
    assert _event_count(settings) == 1


def test_events_while_learning_disabled_are_not_stored(
    client: TestClient, session_id: str, settings: Settings
) -> None:
    client.put("/api/v1/preferences", json={"learning_enabled": False})
    assert _event(client, session_id, "trace", "cartoon") == 201
    assert _event_count(settings) == 0
    client.put("/api/v1/preferences", json={"learning_enabled": True})
    assert client.get("/api/v1/preferences").json()["row_order"] == DEFAULT


def test_explicit_style_goes_first_then_learned(client: TestClient, session_id: str) -> None:
    _event(client, session_id, "trace", "cartoon")
    body = client.put("/api/v1/preferences", json={"selected_style": "realistic_academic"}).json()
    assert body["selected_style"] == "realistic_academic"
    assert body["row_order"][:2] == ["realistic_academic", "cartoon"]


def test_disable_learning_keeps_explicit_style_only(client: TestClient, session_id: str) -> None:
    _event(client, session_id, "trace", "cartoon")
    client.put("/api/v1/preferences", json={"selected_style": "western_ink"})
    body = client.put("/api/v1/preferences", json={"learning_enabled": False}).json()
    assert body["learning_enabled"] is False
    assert body["row_order"] == [
        "western_ink",
        "manga_anime",
        "realistic_academic",
        "cartoon",
        "gesture_sketch",
    ]


def test_reset_and_clear(client: TestClient, session_id: str) -> None:
    _event(client, session_id, "trace", "cartoon")
    client.put("/api/v1/preferences", json={"selected_style": "cartoon"})
    body = client.put(
        "/api/v1/preferences", json={"clear_selected_style": True, "reset_affinities": True}
    ).json()
    assert body["selected_style"] is None
    assert body["row_order"] == DEFAULT
    # Events after a reset count again.
    _event(client, session_id, "trace", "gesture_sketch")
    assert client.get("/api/v1/preferences").json()["row_order"][0] == "gesture_sketch"


def test_partial_update_leaves_other_fields_alone(client: TestClient) -> None:
    client.put("/api/v1/preferences", json={"selected_style": "cartoon"})
    body = client.put("/api/v1/preferences", json={"learning_enabled": False}).json()
    assert body["selected_style"] == "cartoon"
    assert client.put("/api/v1/preferences", json={"selected_style": "oil"}).status_code == 422
    assert client.put("/api/v1/preferences", json={"bogus": 1}).status_code == 422


def _insert_event(
    connection: Connection,
    *,
    session_id: str,
    asset_id: str,
    event: str,
    style: str,
    created_at: str | None = None,
    query_revision: int = 1,
) -> None:
    if created_at is None:
        connection.execute(
            "INSERT INTO events(session_id, asset_id, event, style, query_revision)"
            " VALUES (?,?,?,?,?)",
            (session_id, asset_id, event, style, query_revision),
        )
        return
    connection.execute(
        "INSERT INTO events(session_id, asset_id, event, style, query_revision, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (session_id, asset_id, event, style, query_revision, created_at),
    )


def test_thirty_day_half_life_decays_old_events(tmp_path: Path) -> None:
    connection = connect(tmp_path / "p.sqlite3")
    migrate(connection)
    now = datetime.now(UTC)
    old = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    # Distinct assets so coalescing does not collapse the two traces into one.
    _insert_event(
        connection,
        session_id="s",
        asset_id="a",
        event="trace",
        style="cartoon",
        created_at=old,
    )
    _insert_event(
        connection,
        session_id="s",
        asset_id="b",
        event="trace",
        style="gesture_sketch",
    )
    affinities = compute_affinities(connection, half_life_days=30.0, now=now)
    # Same event weight, but the 30-day-old one is worth half.
    fresh = affinities[PrimaryStyle.GESTURE_SKETCH] - 1 / (4 + 2 + 5)
    aged = affinities[PrimaryStyle.CARTOON] - 1 / (4 + 2 + 5)
    assert abs(aged / fresh - 0.5) < 0.02
    connection.close()


def test_repeated_traces_on_one_asset_match_a_single_trace(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    spam = connect(tmp_path / "spam.sqlite3")
    migrate(spam)
    once = connect(tmp_path / "once.sqlite3")
    migrate(once)
    for hours in range(8):
        ts = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        _insert_event(
            spam,
            session_id="s",
            asset_id="a",
            event="trace",
            style="cartoon",
            created_at=ts,
        )
    _insert_event(
        once,
        session_id="s",
        asset_id="a",
        event="trace",
        style="cartoon",
        created_at=now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    )
    spam_aff = compute_affinities(spam, half_life_days=30.0, now=now)
    once_aff = compute_affinities(once, half_life_days=30.0, now=now)
    assert spam_aff == once_aff
    spam.close()
    once.close()


def test_distinct_assets_still_accumulate(tmp_path: Path) -> None:
    connection = connect(tmp_path / "p.sqlite3")
    migrate(connection)
    now = datetime.now(UTC)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    _insert_event(
        connection, session_id="s", asset_id="a", event="open", style="cartoon", created_at=stamp
    )
    single = compute_affinities(connection, half_life_days=30.0, now=now)
    _insert_event(
        connection, session_id="s", asset_id="b", event="open", style="cartoon", created_at=stamp
    )
    doubled = compute_affinities(connection, half_life_days=30.0, now=now)
    assert doubled[PrimaryStyle.CARTOON] > single[PrimaryStyle.CARTOON]
    connection.close()


def test_order_style_rows_never_changes_membership() -> None:
    for selected in (None, *PrimaryStyle):
        order = order_style_rows(selected, {PrimaryStyle.CARTOON: 0.9}, learning_enabled=True)
        assert sorted(order) == sorted(PrimaryStyle)
        if selected is not None:
            assert order[0] is selected
