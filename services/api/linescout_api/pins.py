"""Durable pinned references.

Pins are *application state*, not a learning signal:

* they live in SQLite (``pins``), so they survive a restart of the API;
* they are namespaced by gallery kind — a pin made against the fixture
  gallery can never appear in a live-gallery session, and vice versa;
* they are untouched by ``learning_enabled = 0`` and by
  ``reset_affinities`` — forgetting learned style affinities never throws
  away what the artist deliberately kept.

Every read re-projects the pin against the current gallery, so a permission
or eligibility change is applied immediately: an asset that is no longer
servable is revoked (and the row removed, with machine-readable reasons), and
``trace_allowed`` always reflects the asset's *stored trace permission*, never
its native/extracted origin.
"""

from __future__ import annotations

import json
import sqlite3

from linescout_ml.taxonomy import LineArtOrigin, PrimaryStyle, ScopeLabel

from linescout_api.config import Settings
from linescout_api.gallery import serving_blockers
from linescout_api.schemas import GalleryKind, PinnedAsset, PinsResponse, RevokedPin

#: Columns the pin projection needs from the assets cache.
_ASSET_PROJECTION = (
    "primary_style, primary_scope, secondary_scopes_json, origin, person_count,"
    " person_count_approximate, quality_score, allowed_trace, enabled"
)


def gallery_kind(settings: Settings) -> GalleryKind:
    """Namespace this API instance writes pins to (never client-asserted)."""
    return GalleryKind.FIXTURE if settings.fixture_mode else GalleryKind.LIVE


def thumbnail_url(asset_id: str) -> str:
    return f"/api/v1/assets/{asset_id}/thumbnail"


def asset_url(asset_id: str) -> str:
    return f"/api/v1/assets/{asset_id}/line-art"


def _project(asset_id: str, pinned_at: str, row: sqlite3.Row) -> PinnedAsset:
    secondary = [ScopeLabel(scope) for scope in json.loads(row["secondary_scopes_json"])]
    primary = ScopeLabel(row["primary_scope"])
    return PinnedAsset(
        asset_id=asset_id,
        pinned_at=pinned_at,
        thumbnail_url=thumbnail_url(asset_id),
        asset_url=asset_url(asset_id),
        style=PrimaryStyle(row["primary_style"]),
        primary_scope=primary,
        scopes=[primary, *secondary],
        secondary_scopes=secondary,
        origin=LineArtOrigin(row["origin"]),
        # Permission metadata, full stop: an extracted asset whose source
        # explicitly permits tracing is traceable, and a native asset whose
        # source forbids it is not.
        trace_allowed=bool(row["allowed_trace"]),
        quality=float(row["quality_score"]),
        person_count=row["person_count"],
        person_count_approximate=bool(row["person_count_approximate"]),
    )


def is_pinnable(connection: sqlite3.Connection, asset_id: str) -> bool:
    """Only an asset that is currently eligible to be shown may be pinned."""
    row = connection.execute(
        "SELECT 1 FROM assets WHERE asset_id = ? AND enabled = 1 AND derivatives_current = 1",
        (asset_id,),
    ).fetchone()
    return row is not None


def list_pins(connection: sqlite3.Connection, kind: GalleryKind) -> PinsResponse:
    """Current pins for ``kind``, revalidated against the live gallery."""
    rows = connection.execute(
        "SELECT asset_id, pinned_at FROM pins WHERE gallery_kind = ? ORDER BY pinned_at DESC,"
        " asset_id",
        (kind.value,),
    ).fetchall()

    pins: list[PinnedAsset] = []
    revoked: list[RevokedPin] = []
    for row in rows:
        asset_id = str(row["asset_id"])
        asset = connection.execute(
            f"SELECT {_ASSET_PROJECTION} FROM assets WHERE asset_id = ?",  # noqa: S608
            (asset_id,),
        ).fetchone()
        if asset is None or not bool(asset["enabled"]):
            revoked.append(
                RevokedPin(asset_id=asset_id, reasons=serving_blockers(connection, asset_id))
            )
            continue
        pins.append(_project(asset_id, str(row["pinned_at"]), asset))

    if revoked:
        # Revalidation is a write: a pin whose asset lost permission must not
        # come back on the next read.
        connection.executemany(
            "DELETE FROM pins WHERE gallery_kind = ? AND asset_id = ?",
            [(kind.value, item.asset_id) for item in revoked],
        )
    return PinsResponse(gallery_kind=kind, pins=pins, revoked=revoked)


def add_pin(connection: sqlite3.Connection, kind: GalleryKind, asset_id: str) -> None:
    """Pin ``asset_id``. Idempotent: re-pinning keeps the original timestamp."""
    connection.execute(
        "INSERT INTO pins(gallery_kind, asset_id) VALUES (?, ?)"
        " ON CONFLICT(gallery_kind, asset_id) DO NOTHING",
        (kind.value, asset_id),
    )


def remove_pin(connection: sqlite3.Connection, kind: GalleryKind, asset_id: str) -> None:
    """Unpin ``asset_id``. Idempotent, and never a negative learning signal."""
    connection.execute(
        "DELETE FROM pins WHERE gallery_kind = ? AND asset_id = ?",
        (kind.value, asset_id),
    )


def is_pinned(connection: sqlite3.Connection, kind: GalleryKind, asset_id: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM pins WHERE gallery_kind = ? AND asset_id = ?",
        (kind.value, asset_id),
    ).fetchone()
    return row is not None
