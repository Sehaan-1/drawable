"""Local, account-free style preference learning.

Event weights: open 1, pin 3, trace 4, **unpin 0**.

Coalescing and timestamp stability
----------------------------------
Repeated ``open``/``trace`` clicks on the same asset (same session, query
revision, and gallery) coalesce to exactly one contribution. The contribution
is dated by the *earliest* matching event, so a retry, a double-click, or a
replayed request can neither add weight nor refresh the decay clock — an
interaction is worth what it was worth when it first happened. (The events
table also enforces this with a partial unique index; the aggregator keeps the
rule so historic rows behave the same way.)

Zero-weight unpin
-----------------
Pins are durable application state (see :mod:`linescout_api.pins`), not a
negative learning event. ``unpin`` carries weight ``0``: it never subtracts
from a style and never penalises anything. What it *does* do is end the
matching pin's contribution — the +3 exists only while the asset is actually
pinned — and re-pinning later starts a new contribution dated at the earliest
pin of that active run, so pin/unpin drumming cannot inflate or refresh
anything either.

Affinity per style is the Laplace-smoothed share of exponentially decayed
weight (30-day half-life). Preferences only control style-row order; they
never touch relevance or Best Match.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime

from linescout_ml.taxonomy import PrimaryStyle

from linescout_api.schemas import InteractionEvent, StyleAffinity

EVENT_WEIGHTS: dict[InteractionEvent, float] = {
    InteractionEvent.OPEN: 1.0,
    InteractionEvent.PIN: 3.0,
    InteractionEvent.TRACE: 4.0,
    # Zero, deliberately: an unpin ends the pin's contribution, it is never a
    # penalty of its own.
    InteractionEvent.UNPIN: 0.0,
}
LAPLACE_ALPHA = 1.0


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def compute_affinities(
    connection: sqlite3.Connection,
    half_life_days: float,
    now: datetime | None = None,
) -> dict[PrimaryStyle, float]:
    now = now or datetime.now(UTC)
    prefs = connection.execute("SELECT affinity_reset_at FROM preferences WHERE id = 1").fetchone()
    reset_at = prefs["affinity_reset_at"] if prefs else None

    # Only live-gallery interactions feed preferences: fixture namespaces are
    # separate, so fixture-mode usage never shapes real retrieval (v2 freeze).
    query = (
        "SELECT session_id, asset_id, query_revision, style, event, created_at FROM events"
        " WHERE gallery_kind = 'live'"
    )
    params: tuple[object, ...] = ()
    if reset_at:
        query += " AND created_at > ?"
        params = (reset_at,)

    decay = math.log(2) / max(half_life_days, 1e-6)
    # Open/trace: one contribution per (session, asset, revision), dated by the
    # FIRST matching event so replays and repeat clicks cannot refresh it.
    # Pin: net toggle per (session, asset). Unpins never contribute a weight of
    # their own (see EVENT_WEIGHTS).
    opens: dict[tuple[str, str, int], tuple[PrimaryStyle, datetime]] = {}
    traces: dict[tuple[str, str, int], tuple[PrimaryStyle, datetime]] = {}
    pin_events: dict[tuple[str, str], list[tuple[datetime, InteractionEvent, PrimaryStyle]]] = {}

    for row in connection.execute(query, params):
        try:
            style = PrimaryStyle(row["style"])
            event = InteractionEvent(row["event"])
        except ValueError:
            continue
        ts = _parse_ts(row["created_at"])
        session_id = str(row["session_id"])
        asset_id = str(row["asset_id"])
        revision = int(row["query_revision"])
        if event in (InteractionEvent.OPEN, InteractionEvent.TRACE):
            bucket = opens if event is InteractionEvent.OPEN else traces
            key = (session_id, asset_id, revision)
            previous = bucket.get(key)
            if previous is None or ts < previous[1]:
                bucket[key] = (style, ts)
        elif event in (InteractionEvent.PIN, InteractionEvent.UNPIN):
            pin_events.setdefault((session_id, asset_id), []).append((ts, event, style))

    weights: dict[PrimaryStyle, float] = dict.fromkeys(PrimaryStyle, 0.0)

    def accumulate(style: PrimaryStyle, ts: datetime, weight: float) -> None:
        age_days = max(0.0, (now - ts).total_seconds() / 86400)
        weights[style] += weight * math.exp(-decay * age_days)

    for style, ts in opens.values():
        accumulate(style, ts, EVENT_WEIGHTS[InteractionEvent.OPEN])
    for style, ts in traces.values():
        accumulate(style, ts, EVENT_WEIGHTS[InteractionEvent.TRACE])
    for sequence in pin_events.values():
        # Sort by (timestamp, id order is already stable) and walk the toggle.
        # ``pin_at`` is the earliest pin of the *currently active* run, so a
        # repeated pin does not refresh the contribution's decay clock.
        sequence.sort(key=lambda item: item[0])
        pin_style: PrimaryStyle | None = None
        pin_at: datetime | None = None
        for ts, event, style in sequence:
            if event is InteractionEvent.PIN:
                if pin_at is None:
                    pin_style, pin_at = style, ts
            else:
                # Unpin: weight 0. It contributes nothing itself and ends the
                # active pin's contribution.
                pin_style, pin_at = None, None
        if pin_style is not None and pin_at is not None:
            accumulate(pin_style, pin_at, EVENT_WEIGHTS[InteractionEvent.PIN])

    clipped = {style: max(0.0, weight) for style, weight in weights.items()}
    total = sum(clipped.values()) + LAPLACE_ALPHA * len(PrimaryStyle)
    return {style: (clipped[style] + LAPLACE_ALPHA) / total for style in PrimaryStyle}


def affinities_response(affinities: dict[PrimaryStyle, float]) -> list[StyleAffinity]:
    return [
        StyleAffinity(style=style, affinity=round(value, 4)) for style, value in affinities.items()
    ]


def read_preferences(connection: sqlite3.Connection) -> tuple[PrimaryStyle | None, bool]:
    row = connection.execute(
        "SELECT selected_style, learning_enabled FROM preferences WHERE id = 1"
    ).fetchone()
    selected = PrimaryStyle(row["selected_style"]) if row and row["selected_style"] else None
    return selected, bool(row["learning_enabled"]) if row else True


def write_preferences(
    connection: sqlite3.Connection,
    *,
    selected_style: PrimaryStyle | None = None,
    clear_selected_style: bool = False,
    learning_enabled: bool | None = None,
    reset_affinities: bool = False,
) -> None:
    updates: list[str] = []
    params: list[object] = []
    if clear_selected_style:
        updates.append("selected_style = NULL")
    elif selected_style is not None:
        updates.append("selected_style = ?")
        params.append(selected_style.value)
    if learning_enabled is not None:
        updates.append("learning_enabled = ?")
        params.append(int(learning_enabled))
    if reset_affinities:
        # Only *learned* weight is forgotten. Pins live in their own table and
        # are deliberately untouched: they are state the artist chose, not an
        # inference the app made.
        updates.append("affinity_reset_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')")
    if not updates:
        return
    updates.append("updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')")
    connection.execute(f"UPDATE preferences SET {', '.join(updates)} WHERE id = 1", params)  # noqa: S608
