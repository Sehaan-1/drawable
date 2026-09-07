from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter

from linescout_api.deps import State
from linescout_api.errors import not_found
from linescout_api.preferences import read_preferences
from linescout_api.schemas import EventRequest, EventResponse
from linescout_api.state import AppState

router = APIRouter(tags=["events"])

# Double-clicks on the same card must not write a second row.
EVENT_DEBOUNCE = timedelta(seconds=2)


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _gallery_style(state: AppState, asset_id: str) -> str:
    """Style is taken from the enabled gallery row, never from the client."""
    row = state.connection.execute(
        "SELECT primary_style FROM assets"
        " WHERE asset_id = ? AND enabled = 1 AND review_state = 'accepted' AND sfw_safe = 1",
        (asset_id,),
    ).fetchone()
    if row is None:
        raise not_found("asset_not_found", "asset not found")
    return str(row["primary_style"])


@router.post("/events", response_model=EventResponse, status_code=201)
def record_event(state: State, body: EventRequest) -> EventResponse:
    """Record an interaction. Timestamp and style are generated server-side."""
    style = _gallery_style(state, body.asset_id)
    _, learning_enabled = read_preferences(state.connection)
    if not learning_enabled:
        # Accept the call so the UI stays fire-and-forget, but do not accumulate rows.
        return EventResponse(
            id=0,
            created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        )

    last = state.connection.execute(
        "SELECT id, created_at FROM events"
        " WHERE session_id = ? AND asset_id = ? AND event = ? AND query_revision = ?"
        " ORDER BY id DESC LIMIT 1",
        (str(body.session_id), body.asset_id, body.event.value, body.query_revision),
    ).fetchone()
    if last is not None:
        created = _parse_ts(str(last["created_at"]))
        if datetime.now(UTC) - created < EVENT_DEBOUNCE:
            return EventResponse(id=int(last["id"]), created_at=str(last["created_at"]))

    cursor = state.connection.execute(
        "INSERT INTO events(session_id, asset_id, event, style, query_revision) VALUES (?,?,?,?,?)"
        " RETURNING id, created_at",
        (
            str(body.session_id),
            body.asset_id,
            body.event.value,
            style,
            body.query_revision,
        ),
    )
    row = cursor.fetchone()
    return EventResponse(id=int(row["id"]), created_at=str(row["created_at"]))
