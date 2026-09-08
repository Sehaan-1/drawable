"""Load a validated manifest into SQLite and answer gallery queries.

The manifest file is authoritative. At startup we validate it, refuse to start
if it is malformed, and then (re)populate the ``assets`` cache table only when
its content hash differs from what is already loaded.

Schema v2 notes:

* Only manifests with ``schema_version`` 2 load. A v1 manifest fails
  validation with an error naming the ``linescout-manifest migrate-v1``
  command (see ``docs/contracts/migration-v1-to-v2.md``).
* The ``enabled`` column is a *derived* cache of the manifest's
  :func:`is_servable` predicate (gallery membership ∧ display permission ∧
  accepted review ∧ human SFW approval ∧ no blockers). It is recomputed on
  sync and on curation writes; it is never hand-edited and a missing file on
  disk does not flip it — runtime availability is in-memory state only.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from linescout_ml.manifest import (
    Manifest,
    ManifestRecord,
    check_parent_integrity,
    check_split_integrity,
    is_servable,
)
from linescout_ml.taxonomy import CurationBlocker, LineArtOrigin, PrimaryStyle

from linescout_api.db import transaction

log = logging.getLogger(__name__)


class GalleryLoadError(RuntimeError):
    """The manifest is missing, malformed, or violates an integrity rule."""


@dataclass(frozen=True)
class GalleryInfo:
    dataset_version: str
    manifest_hash: str
    manifest_path: Path
    data_root: Path
    asset_count: int
    enabled_count: int


def load_manifest(path: Path) -> Manifest:
    if not path.is_file():
        msg = f"gallery manifest not found: {path}"
        raise GalleryLoadError(msg)
    raw = path.read_text(encoding="utf-8")
    try:
        manifest = Manifest.model_validate_json(raw)
    except ValueError as error:
        hint = _v1_hint(raw)
        msg = (
            f"gallery manifest failed validation: {hint}{error}"
            if hint
            else (f"gallery manifest failed validation: {error}")
        )
        raise GalleryLoadError(msg) from error
    problems = check_split_integrity(manifest.records) + check_parent_integrity(manifest.records)
    if problems:
        msg = "gallery manifest violates split integrity: " + "; ".join(problems[:5])
        raise GalleryLoadError(msg)
    return manifest


def _v1_hint(raw: str) -> str:
    """Name the migration command when the document is a schema v1 manifest."""
    try:
        version = json.loads(raw).get("schema_version")
    except json.JSONDecodeError:
        return ""
    if version == 1:
        return (
            "schema v1 manifests are not loadable (schema_version must be 2); "
            "run `linescout-manifest migrate-v1 <path> --out <v2-path>` first — "
        )
    return ""


# One source of truth for the assets row: the column order here must match the
# tuple built by _record_row(); _insert_asset_sql is generated from it so the
# placeholder count can never drift from the column count.
_ASSET_COLUMNS = (
    "asset_id",
    "source_dataset",
    "source_item_id",
    "source_work_id",
    "parent_asset_id",
    "artist_id",
    "leakage_group_id",
    "source_url",
    "license_id",
    "permission_basis",
    "permission_url",
    "attribution",
    "attribution_required",
    "allowed_display",
    "allowed_training",
    "allowed_trace",
    "original_path",
    "line_art_path",
    "thumbnail_path",
    "origin",
    "extraction_model",
    "extraction_version",
    "primary_style",
    "primary_scope",
    "secondary_scopes_json",
    "person_count",
    "person_count_approximate",
    "sfw_verdict",
    "sfw_confidence",
    "sfw_method",
    "sfw_human_safe",
    "sfw_human_reviewer",
    "sfw_human_decided_at",
    "width",
    "height",
    "crop_json",
    "text_coverage",
    "ink_coverage",
    "phash",
    "quality_score",
    "review_state",
    "review_quality",
    "blockers_json",
    "learning_split",
    "gallery_member",
    "gold_member",
    "pipeline_version",
    "processing_revision",
    "label_version",
    "source_checksum",
    "line_art_checksum",
    "thumbnail_checksum",
    "enabled",
)

_INSERT_ASSET = (
    f"INSERT INTO assets ({', '.join(_ASSET_COLUMNS)})"
    f" VALUES ({', '.join('?' * len(_ASSET_COLUMNS))})"
)


def _record_row(record: ManifestRecord) -> tuple[object, ...]:
    values = (
        record.asset_id,
        record.source_dataset,
        record.source_item_id,
        record.source_work_id,
        record.parent_asset_id,
        record.artist_id,
        record.leakage_group_id,
        record.source_url,
        record.permissions.license_id,
        record.permissions.basis.value,
        record.permissions.permission_url,
        record.permissions.attribution,
        int(record.permissions.attribution_required),
        int(record.allowed_uses.display),
        int(record.allowed_uses.training),
        int(record.allowed_uses.trace),
        record.original_path,
        record.line_art_path,
        record.thumbnail_path,
        record.origin.value,
        record.extraction_model,
        record.extraction_version,
        record.primary_style.value,
        record.primary_scope.value,
        json.dumps([scope.value for scope in record.secondary_scopes]),
        record.person_count,
        int(record.person_count_approximate),
        record.sfw_screening.verdict.value if record.sfw_screening else None,
        record.sfw_screening.confidence if record.sfw_screening else None,
        record.sfw_screening.method.value if record.sfw_screening else None,
        record.sfw_human.safe if record.sfw_human else None,
        record.sfw_human.reviewer if record.sfw_human else None,
        record.sfw_human.decided_at if record.sfw_human else None,
        record.width,
        record.height,
        record.crop.model_dump_json() if record.crop else None,
        record.text_coverage,
        record.ink_coverage,
        record.phash,
        record.quality_score,
        record.review.state.value,
        record.review.quality,
        json.dumps([blocker.value for blocker in record.review.blockers]),
        record.learning_split.value,
        int(record.gallery_member),
        int(record.gold_member),
        record.pipeline_version,
        record.processing_revision,
        record.label_version,
        record.source_checksum,
        record.line_art_checksum,
        record.thumbnail_checksum,
        int(is_servable(record)),
    )
    assert len(values) == len(_ASSET_COLUMNS), "assets row out of sync with _ASSET_COLUMNS"
    return values


def sync_gallery(connection: sqlite3.Connection, manifest_path: Path) -> GalleryInfo:
    """Validate ``manifest_path`` and make the ``assets`` table match it."""
    manifest = load_manifest(manifest_path)
    manifest_hash = manifest.content_hash()
    data_root = manifest_path.parent
    servable = manifest.servable_records

    for record in servable:
        for rel in (record.line_art_path, record.thumbnail_path):
            if not (data_root / rel).is_file():
                msg = f"servable asset {record.asset_id} is missing file {rel}"
                raise GalleryLoadError(msg)

    current = connection.execute(
        "SELECT manifest_hash FROM gallery_versions WHERE id = 1"
    ).fetchone()
    if current is not None and current["manifest_hash"] == manifest_hash:
        log.info(
            "gallery %s already loaded (%d assets)", manifest.dataset_version, len(manifest.records)
        )
    else:
        log.info(
            "loading gallery %s (%d assets, %d servable)",
            manifest.dataset_version,
            len(manifest.records),
            len(servable),
        )
        with transaction(connection) as tx:
            tx.execute("DELETE FROM asset_scopes")
            tx.execute("DELETE FROM assets")
            tx.executemany(_INSERT_ASSET, (_record_row(record) for record in manifest.records))
            tx.executemany(
                "INSERT INTO asset_scopes(asset_id, scope) VALUES (?, ?)",
                (
                    (record.asset_id, scope.value)
                    for record in manifest.records
                    for scope in record.scopes
                    if scope.value != "unknown"
                ),
            )
            tx.execute("DELETE FROM gallery_versions")
            tx.execute(
                "INSERT INTO gallery_versions"
                " (id, dataset_version, manifest_hash, manifest_path, asset_count, enabled_count)"
                " VALUES (1, ?, ?, ?, ?, ?)",
                (
                    manifest.dataset_version,
                    manifest_hash,
                    str(manifest_path),
                    len(manifest.records),
                    len(servable),
                ),
            )

    return GalleryInfo(
        dataset_version=manifest.dataset_version,
        manifest_hash=manifest_hash,
        manifest_path=manifest_path,
        data_root=data_root,
        asset_count=len(manifest.records),
        enabled_count=len(servable),
    )


#: SQL mirror of the manifest's ``is_servable`` predicate. Used to recompute
#: the derived ``enabled`` flag after a curation write without a reload.
ENABLED_SQL = (
    "gallery_member = 1 AND allowed_display = 1 AND review_state = 'accepted'"
    " AND COALESCE(sfw_human_safe, 0) = 1 AND blockers_json = '[]'"
    " AND review_quality IN (2, 3)"
)


def recompute_enabled(connection: sqlite3.Connection, asset_id: str) -> bool:
    """Recompute the derived serving flag for one asset; returns the new value."""
    connection.execute(
        f"UPDATE assets SET enabled = ({ENABLED_SQL}) WHERE asset_id = ?",  # noqa: S608
        (asset_id,),
    )
    row = connection.execute(
        "SELECT enabled FROM assets WHERE asset_id = ?", (asset_id,)
    ).fetchone()
    return bool(row["enabled"]) if row else False


def serving_blockers(connection: sqlite3.Connection, asset_id: str) -> list[str]:
    """Why the asset is not servable, as stable machine-readable reasons."""
    row = connection.execute(
        "SELECT gallery_member, allowed_display, review_state, review_quality,"
        " sfw_human_safe, blockers_json FROM assets WHERE asset_id = ?",
        (asset_id,),
    ).fetchone()
    if row is None:
        return ["asset_not_in_gallery"]
    reasons: list[str] = []
    if not row["gallery_member"]:
        reasons.append("not_a_gallery_member")
    if not row["allowed_display"]:
        reasons.append("display_not_permitted")
    if row["review_state"] != "accepted":
        reasons.append(f"review_{row['review_state']}")
    if row["review_quality"] is None or row["review_quality"] < 2:
        reasons.append("quality_below_floor")
    if row["sfw_human_safe"] is None:
        reasons.append("sfw_human_unapproved")
    elif not row["sfw_human_safe"]:
        reasons.append("sfw_human_unsafe")
    try:
        blockers = [CurationBlocker(value) for value in json.loads(row["blockers_json"] or "[]")]
    except ValueError:
        blockers = []
    reasons.extend(f"blocker_{blocker.value}" for blocker in blockers)
    return reasons


@dataclass(frozen=True)
class GalleryAsset:
    asset_id: str
    primary_style: PrimaryStyle
    primary_scope: str
    secondary_scopes: tuple[str, ...]
    origin: LineArtOrigin
    person_count: int | None
    person_count_approximate: bool
    quality_score: float
    trace_allowed: bool
    line_art_path: str
    thumbnail_path: str

    @property
    def scopes(self) -> tuple[str, ...]:
        return (self.primary_scope, *self.secondary_scopes)


def enabled_assets(connection: sqlite3.Connection) -> list[GalleryAsset]:
    """Every asset eligible to appear in a response (the derived ``enabled`` flag)."""
    rows = connection.execute(
        "SELECT asset_id, primary_style, primary_scope, secondary_scopes_json, origin,"
        " person_count, person_count_approximate, quality_score, allowed_trace,"
        " line_art_path, thumbnail_path"
        " FROM assets WHERE enabled = 1 ORDER BY asset_id"
    ).fetchall()
    return [
        GalleryAsset(
            asset_id=row["asset_id"],
            primary_style=PrimaryStyle(row["primary_style"]),
            primary_scope=row["primary_scope"],
            secondary_scopes=tuple(json.loads(row["secondary_scopes_json"])),
            origin=LineArtOrigin(row["origin"]),
            person_count=row["person_count"],
            person_count_approximate=bool(row["person_count_approximate"]),
            quality_score=float(row["quality_score"]),
            trace_allowed=bool(row["allowed_trace"]),
            line_art_path=row["line_art_path"],
            thumbnail_path=row["thumbnail_path"],
        )
        for row in rows
    ]


def asset_file(connection: sqlite3.Connection, asset_id: str, kind: str) -> str | None:
    """Relative path for a servable asset's ``thumbnail`` or ``line_art`` file, else ``None``."""
    column = {"thumbnail": "thumbnail_path", "line_art": "line_art_path"}[kind]
    row = connection.execute(
        f"SELECT {column} AS path FROM assets"  # noqa: S608
        f" WHERE asset_id = ? AND enabled = 1",
        (asset_id,),
    ).fetchone()
    return str(row["path"]) if row else None
