"""``POST /api/v1/events`` — idempotent interaction recording.

Contract highlights (see ``docs/contracts/api-contract.md``):

* The client generates an ``event_uuid`` per interaction attempt. The write is
  a single atomic ``INSERT ... ON CONFLICT DO NOTHING``: a retry with the same
  UUID and the same authoritative payload replays the original result, and the
  same UUID with a *different* payload is a ``409 event_uuid_conflict``.
* The authoritative payload is ``(session_id, asset_id, event, query_revision,
  gallery_kind)``. ``style`` is not part of it: it is derived from the gallery
  row, so a client cannot assert (or conflict on) it.
* ``open``/``trace`` interactions coalesce per (session, asset, revision,
  gallery) in the *database*, so repeated clicks can neither append a row,
  inflate the decayed weight, nor refresh the contribution timestamp.
* ``trace`` requires the asset's stored trace permission. A forged trace event
  on a non-traceable asset is a ``403 trace_not_permitted`` — permission comes
  from the recorded source permission metadata, never from native/extracted
  origin.
* When learning is disabled nothing is accumulated: the call is accepted with
  ``recorded=false`` and no row (and therefore no idempotency record) is
  written. Pin state changes go through ``/api/v1/pins`` and are unaffected.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter

from linescout_api.deps import State
from linescout_api.errors import ApiError, conflict, not_found
from linescout_api.preferences import read_preferences
from linescout_api.schemas import ErrorResponse, EventRequest, EventResponse, InteractionEvent
from linescout_api.state import AppState

router = APIRouter(tags=["events"])


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def payload_hash(
    *, session_id: UUID, asset_id: str, event: InteractionEvent, query_revision: int, kind: str
) -> str:
    """Hash of the *authoritative* request payload.

    Only fields the server actually honours take part, so replaying a request
    with a different (ignored) ``style`` is a replay, not a conflict.
    """
    material = "\x1f".join(
        (str(session_id), asset_id, event.value, str(query_revision), kind)
    ).encode()
    return hashlib.sha256(material).hexdigest()


def _asset_row(state: AppState, asset_id: str) -> sqlite3.Row:
    """The enabled gallery row for ``asset_id``; style/permission come from here."""
    row = state.connection.execute(
        "SELECT primary_style, allowed_trace FROM assets WHERE asset_id = ? AND enabled = 1",
        (asset_id,),
    ).fetchone()
    if row is None:
        raise not_found("asset_not_found", "asset not found")
    return row  # type: ignore[no-any-return]


def trace_forbidden(asset_id: str) -> ApiError:
    return ApiError(
        403,
        "trace_not_permitted",
        "this asset's source permissions do not allow placing it on the trace layer",
        field="asset_id",
        retryable=False,
        details={"asset_id": asset_id},
    )


@router.post(
    "/events",
    response_model=EventResponse,
    status_code=201,
    responses={
        403: {"model": ErrorResponse, "description": "Tracing this asset is not permitted"},
        404: {"model": ErrorResponse, "description": "Unknown or ineligible asset"},
        409: {"model": ErrorResponse, "description": "event_uuid reused with a different payload"},
    },
)
def record_event(state: State, body: EventRequest) -> EventResponse:
    """Record an interaction. Timestamp, style, and namespace are server-side."""
    asset = _asset_row(state, body.asset_id)
    style = str(asset["primary_style"])
    if body.event is InteractionEvent.TRACE and not bool(asset["allowed_trace"]):
        raise trace_forbidden(body.asset_id)

    # The gallery namespace is stamped server-side from the API's own mode:
    # fixture-mode events can never influence live-gallery preferences.
    gallery_kind = "fixture" if state.settings.fixture_mode else "live"
    digest = payload_hash(
        session_id=body.session_id,
        asset_id=body.asset_id,
        event=body.event,
        query_revision=body.query_revision,
        kind=gallery_kind,
    )

    _, learning_enabled = read_preferences(state.connection)
    if not learning_enabled:
        # Accept the call so the UI stays fire-and-forget, but accumulate
        # nothing: no row, and therefore no idempotency record either.
        return EventResponse(
            id=0,
            event_uuid=body.event_uuid,
            created_at=_now_stamp(),
            recorded=False,
            replayed=False,
        )

    # One atomic statement. `ON CONFLICT DO NOTHING` (no conflict target)
    # covers both unique indexes: the event identity and the open/trace
    # coalescing key. Nothing is read-then-written, so concurrent retries
    # cannot both insert.
    inserted = state.connection.execute(
        "INSERT INTO events(event_uuid, payload_hash, session_id, asset_id, event, style,"
        " query_revision, gallery_kind)"
        " VALUES (?,?,?,?,?,?,?,?)"
        " ON CONFLICT DO NOTHING"
        " RETURNING id, event_uuid, created_at",
        (
            str(body.event_uuid),
            digest,
            str(body.session_id),
            body.asset_id,
            body.event.value,
            style,
            body.query_revision,
            gallery_kind,
        ),
    ).fetchone()
    if inserted is not None:
        return EventResponse(
            id=int(inserted["id"]),
            event_uuid=UUID(str(inserted["event_uuid"])),
            created_at=str(inserted["created_at"]),
            recorded=True,
            replayed=False,
        )

    # Identity conflict first: the same UUID must always describe the same
    # interaction, whatever else it might also collide with.
    existing = state.connection.execute(
        "SELECT id, event_uuid, created_at, payload_hash FROM events WHERE event_uuid = ?",
        (str(body.event_uuid),),
    ).fetchone()
    if existing is not None:
        if existing["payload_hash"] != digest:
            raise conflict(
                "event_uuid_conflict",
                "this event_uuid was already used for a different interaction",
                "event_uuid",
            )
        return EventResponse(
            id=int(existing["id"]),
            event_uuid=UUID(str(existing["event_uuid"])),
            created_at=str(existing["created_at"]),
            recorded=True,
            replayed=True,
        )

    # Otherwise this is a fresh attempt at an interaction that already has a
    # contribution (a repeated open/trace). It coalesces onto the original
    # row — same id, same created_at, so the decayed weight and the
    # contribution timestamp are untouched.
    coalesced = state.connection.execute(
        "SELECT id, event_uuid, created_at FROM events"
        " WHERE session_id = ? AND asset_id = ? AND event = ? AND query_revision = ?"
        " AND gallery_kind = ?"
        " ORDER BY id LIMIT 1",
        (
            str(body.session_id),
            body.asset_id,
            body.event.value,
            body.query_revision,
            gallery_kind,
        ),
    ).fetchone()
    if coalesced is None:  # pragma: no cover - defensive; one of the two must exist
        raise conflict("event_write_conflict", "the event could not be recorded", "event_uuid")
    return EventResponse(
        id=int(coalesced["id"]),
        event_uuid=UUID(str(coalesced["event_uuid"])),
        created_at=str(coalesced["created_at"]),
        recorded=True,
        replayed=True,
    )
