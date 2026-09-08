"""Curator-created crop derivatives: creation, processing, rehydration.

A crop made in curation is an **immutable child derivative**:

* it records its parent's identity (``parent_asset_id``) and the bounded
  crop geometry (in the *parent's* pixel space);
* it owns **fresh files and hashes** — the child's line art, thumbnail, and
  original are new bytes cut from the parent's *verified* artifacts, never
  references to the parent's files;
* it carries **its own processing and review state** — it starts
  ``unreviewed`` with ``derivatives_current = 0`` ("awaiting processing"),
  and nothing can enable it before the required processing completes *and*
  an explicit human approval is recorded (the assets table's CHECK
  constraints enforce this structurally);
* its provenance (identity, permissions, learning split, membership) follows
  the parent, but nothing about the parent's review, SFW, or gold status is
  ever inherited — a crop must earn its own approval.

Durability: the ``curation_derivatives`` registry is the source of truth for
the crop itself. The ``assets`` row is a cache materialized from the parent
row + registry + the child's latest audit label. When a new manifest
rebuilds the ``assets`` cache, :func:`rehydrate_derivatives` re-materializes
every child from the registry: identity/permission/membership/split follow
the parent's *current* row (so a revoked licence or changed split propagates
— no stale grants), while geometry, files, processing, and review come from
the registry and the audit trail. A child whose frozen artifact generation
no longer matches the manifest contract is re-marked stale, and a child
whose parent disappeared is disabled with ``derivative_parent_missing``.

Failure/rollback model: filesystem work happens *before* the database
transaction and is fully rolled back (the child directory is removed) if
either stage fails, so a failed attempt leaves no orphan artifacts and no
registry row. A crash between the two stages can leave at most an orphan
directory, which the next attempt for the same crop clears before writing —
and which nothing references until a registry row exists.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from linescout_ml.manifest import ArtifactContract, CropBox, make_asset_id
from linescout_ml.taxonomy import GALLERY_SCOPES, ScopeLabel
from PIL import Image, UnidentifiedImageError

from linescout_api.derivative_processing import (
    DerivativeImageError,
    Measurements,
    decode_gray,
    make_thumbnail,
    measure_line_art,
    sha256_file,
)
from linescout_api.gallery import recompute_enabled

log = logging.getLogger(__name__)

#: Smallest crop edge the pipeline will materialise (a degenerate 1-px crop
#: is never a useful training or review target).
MIN_CROP_EDGE = 16

#: Placeholder pHash for an unprocessed child (hex zeros, 16 chars).
PENDING_PHASH = "0" * 16

#: Derivative problem recorded while the child's own processing has not
#: successfully completed.
AWAITING_PROCESSING = "derivative_awaiting_processing"

_DERIVATIVE_ROOT = "derivatives"


class DerivativeError(RuntimeError):
    """A structured derivative-lifecycle failure.

    ``code`` is a stable machine-readable string; ``retryable`` says whether
    repeating the same request can succeed (transient I/O) or not (bad
    input/data). Messages never embed absolute filesystem paths.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}


@dataclass(frozen=True)
class CropFiles:
    """The child's fresh files, relative to the data root, with their hashes."""

    original_path: str
    line_art_path: str
    thumbnail_path: str
    source_checksum: str
    line_art_checksum: str
    thumbnail_checksum: str


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def crop_fits(crop: CropBox, width: int, height: int) -> bool:
    """True when the crop rectangle lies entirely inside the parent image."""
    return crop.x + crop.width <= width and crop.y + crop.height <= height


def child_asset_id(parent_row: sqlite3.Row, crop: CropBox) -> str:
    """Deterministic child id: the same dataset/item/crop always re-derives it."""
    return make_asset_id(parent_row["source_dataset"], parent_row["source_item_id"], crop)


def _child_directory(data_root: Path, asset_id: str) -> Path:
    directory = (data_root / _DERIVATIVE_ROOT / asset_id).resolve()
    root = data_root.resolve()
    if root not in directory.parents:
        msg = "derivative directory escapes the data root"
        raise DerivativeError("derivative_path_invalid", msg, retryable=False)
    return directory


def _clear_directory(directory: Path) -> None:
    """Remove any leftovers from a failed earlier attempt (retry safety)."""
    if directory.exists():
        shutil.rmtree(directory)


def crop_files_for(
    data_root: Path, parent_row: sqlite3.Row, child_id: str, crop: CropBox, thumbnail_size: int
) -> tuple[CropFiles, Path]:
    """Cut the child's fresh files from the parent's *verified* artifacts.

    The parent's line art and original are read and re-hashed first: if the
    bytes drifted from what the manifest recorded, the parent's artifacts are
    stale and no derivative may be cut from them (fail closed —
    ``parent_artifact_invalid``). All writes go to a fresh child directory;
    any failure removes the directory again so a failed attempt leaves no
    partial artifacts behind. Returns the file descriptors plus the child
    directory (so the caller can roll the writes back if the database stage
    fails).
    """
    line_art = data_root / str(parent_row["line_art_path"])
    original = data_root / str(parent_row["original_path"])
    for label, path, expected in (
        ("line_art", line_art, parent_row["line_art_checksum"]),
        ("original", original, parent_row["source_checksum"]),
    ):
        if not path.is_file():
            raise DerivativeError(
                "parent_artifact_invalid",
                f"parent {label} file is missing; re-process the parent before cropping",
                retryable=False,
                details={
                    "parent_asset_id": parent_row["asset_id"],
                    "problems": [f"derivative_file_missing:{label}"],
                },
            )
        if sha256_file(path) != expected:
            raise DerivativeError(
                "parent_artifact_invalid",
                f"parent {label} file no longer matches its recorded checksum",
                retryable=False,
                details={
                    "parent_asset_id": parent_row["asset_id"],
                    "problems": [f"derivative_checksum_mismatch:{label}"],
                },
            )

    directory = _child_directory(data_root, child_id)
    try:
        _clear_directory(directory)
        directory.mkdir(parents=True, exist_ok=False)
        box = (crop.x, crop.y, crop.x + crop.width, crop.y + crop.height)
        names: dict[str, Path] = {}
        for label, source in (("original", original), ("line_art", line_art)):
            with Image.open(source) as handle:
                handle.load()
                handle.crop(box).save(directory / f"{label}.png", "PNG")
            names[label] = directory / f"{label}.png"
        thumb = make_thumbnail(decode_gray(names["line_art"]), thumbnail_size)
        names["thumbnail"] = directory / "thumbnail.png"
        thumb.save(names["thumbnail"], "PNG")
    except (OSError, ValueError, UnidentifiedImageError) as error:
        _clear_directory(directory)
        raise DerivativeError(
            "derivative_write_failed",
            f"could not write the derivative files: {error}",
            retryable=True,
        ) from error

    files = CropFiles(
        original_path=f"{_DERIVATIVE_ROOT}/{child_id}/original.png",
        line_art_path=f"{_DERIVATIVE_ROOT}/{child_id}/line_art.png",
        thumbnail_path=f"{_DERIVATIVE_ROOT}/{child_id}/thumbnail.png",
        source_checksum=sha256_file(names["original"]),
        line_art_checksum=sha256_file(names["line_art"]),
        thumbnail_checksum=sha256_file(names["thumbnail"]),
    )
    return files, directory


_REGISTRY_COLUMNS = (
    "asset_id",
    "parent_asset_id",
    "crop_json",
    "original_path",
    "line_art_path",
    "thumbnail_path",
    "source_checksum",
    "line_art_checksum",
    "thumbnail_checksum",
    "width",
    "height",
    "pipeline_version",
    "label_version",
    "processing_revision",
    "processing_state",
    "processing_attempts",
    "processing_error",
    "measurements_json",
    "created_by",
    "note",
    "created_at",
)


def registry_values(
    parent_row: sqlite3.Row,
    child_id: str,
    crop: CropBox,
    files: CropFiles,
    *,
    created_by: str,
    note: str | None,
) -> dict[str, object]:
    """The ``curation_derivatives`` registry row for a fresh crop."""
    return {
        "asset_id": child_id,
        "parent_asset_id": parent_row["asset_id"],
        "crop_json": crop.model_dump_json(),
        "original_path": files.original_path,
        "line_art_path": files.line_art_path,
        "thumbnail_path": files.thumbnail_path,
        "source_checksum": files.source_checksum,
        "line_art_checksum": files.line_art_checksum,
        "thumbnail_checksum": files.thumbnail_checksum,
        "width": crop.width,
        "height": crop.height,
        # Frozen artifact generation: the parent's generation at cut time. A
        # later manifest that changes the generation makes the crop stale.
        "pipeline_version": parent_row["pipeline_version"],
        "label_version": parent_row["label_version"],
        "processing_revision": parent_row["processing_revision"],
        "processing_state": "pending",
        "processing_attempts": 0,
        "processing_error": None,
        "measurements_json": None,
        "created_by": created_by,
        "note": note,
        "created_at": _utcnow_iso(),
    }


def insert_registry_row(connection: sqlite3.Connection, values: dict[str, object]) -> None:
    connection.execute(
        f"INSERT INTO curation_derivatives ({', '.join(_REGISTRY_COLUMNS)})"
        f" VALUES ({', '.join('?' * len(_REGISTRY_COLUMNS))})",
        tuple(values[column] for column in _REGISTRY_COLUMNS),
    )


_ASSET_INSERT_COLUMNS = (
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
    "faiss_row",
    "derivatives_current",
    "derivative_problems_json",
    "enabled",
    "curation_label_version",
)


#: Columns ``child_asset_values`` reads off the parent row (everything the
#: child follows rather than owns).
_PARENT_FOLLOWED_COLUMNS = (
    "source_dataset",
    "source_item_id",
    "source_work_id",
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
    "origin",
    "extraction_model",
    "extraction_version",
    "primary_style",
    "learning_split",
    "gallery_member",
)


def child_asset_values(
    parent_row: sqlite3.Row | dict[str, object],
    registry: dict[str, object] | sqlite3.Row,
    *,
    review: dict[str, object] | None = None,
    problems: list[str] | None = None,
    curation_label_version: int = 0,
) -> dict[str, object]:
    """Build the ``assets`` cache row for a crop child.

    Identity, permission, membership, split, and line-art provenance follow
    the parent row. Geometry, files, hashes, and the frozen artifact
    generation come from the registry. Review fields come from the child's
    latest audit events (``review``) when any exist; otherwise the child is
    a fresh ``unreviewed`` record. The parent's review, SFW-human, gold, and
    blocker state are **never** inherited.
    """
    values: dict[str, object] = {column: parent_row[column] for column in _PARENT_FOLLOWED_COLUMNS}
    values.update(
        {
            # the crop's own facts
            "asset_id": registry["asset_id"],
            "parent_asset_id": registry["parent_asset_id"],
            "crop_json": registry["crop_json"],
            "original_path": registry["original_path"],
            "line_art_path": registry["line_art_path"],
            "thumbnail_path": registry["thumbnail_path"],
            "source_checksum": registry["source_checksum"],
            "line_art_checksum": registry["line_art_checksum"],
            "thumbnail_checksum": registry["thumbnail_checksum"],
            "width": registry["width"],
            "height": registry["height"],
            "pipeline_version": registry["pipeline_version"],
            "label_version": registry["label_version"],
            "processing_revision": registry["processing_revision"],
            # own review state: nothing human is inherited
            "review_state": "unreviewed",
            "review_quality": None,
            "blockers_json": "[]",
            "primary_scope": "unknown",
            "secondary_scopes_json": "[]",
            "person_count": None,
            "person_count_approximate": 0,
            "sfw_verdict": None,
            "sfw_confidence": None,
            "sfw_method": None,
            "sfw_human_safe": None,
            "sfw_human_reviewer": None,
            "sfw_human_decided_at": None,
            "gold_member": 0,
            # measurements are rebuilt from the child's own bytes by processing
            "text_coverage": 0.0,
            "ink_coverage": 0.0,
            "phash": PENDING_PHASH,
            "quality_score": 0.0,
            "faiss_row": None,
            # None = a fresh crop that has not been processed yet; an explicit
            # empty list means "verified, no problems" (the only state the
            # schema allows together with derivatives_current = 1).
            "derivatives_current": 0,
            "derivative_problems_json": json.dumps(
                problems if problems is not None else [AWAITING_PROCESSING]
            ),
            "enabled": 0,
            "curation_label_version": curation_label_version,
        }
    )
    if review:
        values.update(review)
        # ``enabled`` is derived; the caller recomputes it in-transaction.
        values["enabled"] = 0
    return values


def insert_child_asset(connection: sqlite3.Connection, values: dict[str, object]) -> None:
    """Insert the child's ``assets`` row (cache) inside the caller's transaction."""
    connection.execute(
        f"INSERT INTO assets ({', '.join(_ASSET_INSERT_COLUMNS)})"
        f" VALUES ({', '.join('?' * len(_ASSET_INSERT_COLUMNS))})",
        tuple(values.get(column) for column in _ASSET_INSERT_COLUMNS),
    )


def sync_asset_scopes(
    connection: sqlite3.Connection, asset_id: str, values: dict[str, object]
) -> None:
    """Mirror primary+secondary scopes into ``asset_scopes`` (transactional).

    The caller wraps this in the same transaction as the asset row write, so
    the denormalised scope table can never drift from ``assets`` — the same
    contract ``write_label`` follows.
    """
    primary = values.get("primary_scope")
    try:
        raw_scopes = str(values.get("secondary_scopes_json") or "[]")
        secondary = [ScopeLabel(value) for value in json.loads(raw_scopes)]
    except (ValueError, json.JSONDecodeError):
        secondary = []
    scopes: list[str] = []
    if primary is not None and primary != "unknown":
        try:
            primary_scope = ScopeLabel(str(primary))
        except ValueError:
            primary_scope = None
        if primary_scope is not None and primary_scope in GALLERY_SCOPES:
            scopes.append(primary_scope.value)
    scopes.extend(scope.value for scope in secondary if scope in GALLERY_SCOPES)
    connection.execute("DELETE FROM asset_scopes WHERE asset_id = ?", (asset_id,))
    if scopes:
        connection.executemany(
            "INSERT INTO asset_scopes(asset_id, scope) VALUES (?, ?)",
            [(asset_id, scope) for scope in scopes],
        )


def _label_review_values(connection: sqlite3.Connection, asset_id: str) -> dict[str, object] | None:
    """The ``assets``-mirror review values from the asset's latest audit label.

    Mirrors exactly what ``write_label`` writes, so a re-hydrated child keeps
    the curation decisions recorded for it. Returns ``None`` when the asset
    has never been labelled.
    """
    row = connection.execute(
        "SELECT id, decision, primary_style, primary_scope, secondary_scopes_json,"
        " blockers_json, sfw_safe, quality, reviewer, created_at"
        " FROM curation_labels WHERE asset_id = ? ORDER BY id DESC LIMIT 1",
        (asset_id,),
    ).fetchone()
    if row is None:
        return None
    values: dict[str, object] = {
        "review_state": "accepted" if row["decision"] == "keep" else "rejected",
        "review_quality": row["quality"],
        "blockers_json": row["blockers_json"] or "[]",
        "primary_scope": row["primary_scope"] or "unknown",
        "secondary_scopes_json": row["secondary_scopes_json"] or "[]",
    }
    if row["primary_style"]:
        values["primary_style"] = row["primary_style"]
    if row["sfw_safe"] is not None:
        values["sfw_human_safe"] = int(bool(row["sfw_safe"]))
        values["sfw_human_reviewer"] = row["reviewer"]
        values["sfw_human_decided_at"] = row["created_at"]
    return values


def _adjudication_values(
    connection: sqlite3.Connection, asset_id: str, *, not_before: str | None = None
) -> dict[str, object] | None:
    """Latest SFW adjudication for an asset, optionally only at/after a moment.

    ``not_before`` (the latest label's timestamp, if any) is compared as a
    parsed instant: label and adjudication timestamps may come from SQLite's
    ``strftime`` default (millisecond precision) or from Python (second
    precision), so lexicographic comparison would be wrong across formats.
    """
    rows = connection.execute(
        "SELECT safe, reviewer, created_at, prior_review_state FROM sfw_adjudications"
        " WHERE asset_id = ? ORDER BY id DESC",
        (asset_id,),
    ).fetchall()
    if not rows:
        return None
    threshold = _parse_ts(not_before)
    row = None
    for candidate in rows:
        decided = _parse_ts(str(candidate["created_at"]))
        if threshold is None or decided is None or decided >= threshold:
            row = candidate
            break
    if row is None:
        return None
    values: dict[str, object] = {
        "sfw_human_safe": int(bool(row["safe"])),
        "sfw_human_reviewer": row["reviewer"],
        "sfw_human_decided_at": row["created_at"],
    }
    if row["safe"]:
        if row["prior_review_state"] == "quarantined":
            values["review_state"] = "unreviewed"
    else:
        values["review_state"] = "quarantined"
    return values


def _parse_ts(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp written by SQLite or Python (``…Z`` form)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _generation_problems(registry: sqlite3.Row, contract: ArtifactContract | None) -> list[str]:
    """Stale-generation reasons for the child's frozen artifact generation."""
    if contract is None:
        return ["derivative_generation_unknown"]
    reasons: list[str] = []
    if registry["pipeline_version"] != contract.pipeline_version:
        reasons.append("derivative_stale:pipeline_version")
    if registry["label_version"] != contract.label_version:
        reasons.append("derivative_stale:label_version")
    if registry["processing_revision"] != contract.processing_revision:
        reasons.append("derivative_stale:processing_revision")
    return reasons


def _file_problems(data_root: Path, registry: sqlite3.Row) -> list[str]:
    problems: list[str] = []
    for label, relative, expected in (
        ("line_art", registry["line_art_path"], registry["line_art_checksum"]),
        ("thumbnail", registry["thumbnail_path"], registry["thumbnail_checksum"]),
        ("original", registry["original_path"], registry["source_checksum"]),
    ):
        path = data_root / str(relative)
        if not path.is_file():
            problems.append(f"derivative_file_missing:{label}")
        elif sha256_file(path) != expected:
            problems.append(f"derivative_checksum_mismatch:{label}")
    return problems


def _audit_event_count(connection: sqlite3.Connection, asset_id: str) -> int:
    """Curation events recorded for the asset (labels + adjudications).

    The stored ``curation_label_version`` is the optimistic-concurrency
    guard; after a cache rebuild it is re-derived from the durable audit
    tables so a client that saw an older state fails closed with a 409
    rather than silently writing onto a rebuilt row.
    """
    labels = connection.execute(
        "SELECT COUNT(*) AS n FROM curation_labels WHERE asset_id = ?", (asset_id,)
    ).fetchone()["n"]
    adjudications = connection.execute(
        "SELECT COUNT(*) AS n FROM sfw_adjudications WHERE asset_id = ?", (asset_id,)
    ).fetchone()["n"]
    return int(labels) + int(adjudications)


def audit_review_values(connection: sqlite3.Connection, asset_id: str) -> dict[str, object] | None:
    """Combined review mirror: latest label, then any later adjudication."""
    review = _label_review_values(connection, asset_id)
    label_row = connection.execute(
        "SELECT created_at FROM curation_labels WHERE asset_id = ? ORDER BY id DESC LIMIT 1",
        (asset_id,),
    ).fetchone()
    not_before = str(label_row["created_at"]) if label_row is not None else None
    adjudication = _adjudication_values(connection, asset_id, not_before=not_before)
    if adjudication:
        review = {**(review or {}), **adjudication}
    return review


#: Minimal stand-in parent for an orphaned child (parent removed from the
#: manifest). Everything permission-shaped stays deny/unknown, and the child
#: is disabled by its ``derivative_parent_missing`` problem.
_ORPHAN_PARENT: dict[str, object] = {
    "source_dataset": "unknown",
    "source_item_id": "unknown",
    "source_work_id": "unknown",
    "artist_id": None,
    "leakage_group_id": None,
    "source_url": None,
    "license_id": "unknown",
    "permission_basis": "unknown",
    "permission_url": None,
    "attribution": None,
    "attribution_required": 0,
    "allowed_display": 0,
    "allowed_training": 0,
    "allowed_trace": 0,
    "origin": "native_line_art",
    "extraction_model": None,
    "extraction_version": None,
    "primary_style": "manga_anime",
    "learning_split": "none",
    "gallery_member": 0,
}


def rehydrate_derivatives(
    connection: sqlite3.Connection,
    contract: ArtifactContract | None,
    data_root: Path,
) -> int:
    """Re-materialize every registered crop child after a gallery rebuild.

    Called only when ``sync_gallery`` rebuilt the ``assets`` cache from a new
    manifest. Children are re-inserted from the registry (never from the old
    cache rows): the parent's *current* row supplies identity, permissions,
    membership, and split (so a revoked permission or changed split in the
    new manifest propagates to the crop — no stale grants), while geometry,
    files, generation, processing, and review come from the registry and the
    audit trail. Returns the number of children re-hydrated.
    """
    registries = connection.execute(
        "SELECT * FROM curation_derivatives ORDER BY asset_id"
    ).fetchall()
    if not registries:
        return 0
    count = 0
    for registry in registries:
        child_id = str(registry["asset_id"])
        parent = connection.execute(
            "SELECT * FROM assets WHERE asset_id = ?", (registry["parent_asset_id"],)
        ).fetchone()
        problems: list[str] = []
        if parent is None:
            problems.append("derivative_parent_missing")
        else:
            problems.extend(_generation_problems(registry, contract))
        if registry["processing_state"] != "complete":
            problems.append(AWAITING_PROCESSING)
        try:
            problems.extend(_file_problems(data_root, registry))
        except OSError as error:
            problems.append(f"derivative_file_unreadable:{error.__class__.__name__}")
        problems = list(dict.fromkeys(problems))  # stable de-duplicate

        review = audit_review_values(connection, child_id)
        if parent is None:
            # Keep the child visible but inert: it cannot inherit provenance
            # from a parent that no longer exists.
            values = child_asset_values(
                _ORPHAN_PARENT,
                dict(registry),
                review=review,
                problems=problems,
                curation_label_version=_audit_event_count(connection, child_id),
            )
        else:
            values = child_asset_values(
                parent,
                registry,
                review=review,
                problems=problems,
                curation_label_version=_audit_event_count(connection, child_id),
            )
        # The child's derivative validity was re-verified above: current iff
        # the frozen generation matches, processing completed, and every
        # file still matches its recorded checksum.
        values["derivatives_current"] = int(not problems)
        connection.execute("BEGIN IMMEDIATE")
        try:
            insert_child_asset(connection, values)
            sync_asset_scopes(connection, child_id, values)
            recompute_enabled(connection, child_id)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        count += 1
    if count:
        log.info("re-hydrated %d curation derivative(s) after gallery rebuild", count)
    return count


def process_derivative_files(
    data_root: Path,
    registry: sqlite3.Row,
    thumbnail_size: int,
) -> tuple[Measurements, CropFiles | None]:
    """Run the child's required processing from its own bytes.

    Verifies every file against its recorded checksum (integrity), decodes
    the line art, rebuilds the measurements from the child's own pixels, and
    regenerates the thumbnail. Returns the measurements plus a replacement
    :class:`CropFiles` when the thumbnail bytes drifted and had to be rebuilt
    (the registry checksum must follow); ``None`` when the recorded thumbnail
    still matches. Raises :class:`DerivativeError` on any failure.
    """
    problems = _file_problems(data_root, registry)
    if problems:
        raise DerivativeError(
            "derivative_files_invalid",
            "derivative files are missing or do not match their recorded checksums",
            retryable=False,
            details={"problems": problems},
        )
    line_art_path = data_root / str(registry["line_art_path"])
    try:
        gray = decode_gray(line_art_path)
    except DerivativeImageError as error:
        raise DerivativeError(error.code, error.message, retryable=False) from error
    if (gray.width, gray.height) != (int(registry["width"]), int(registry["height"])):
        raise DerivativeError(
            "derivative_geometry_mismatch",
            "the derivative's line art does not match its recorded crop geometry",
            retryable=False,
        )
    measurements = measure_line_art(gray)

    thumb_path = data_root / str(registry["thumbnail_path"])
    try:
        regenerated = make_thumbnail(gray, thumbnail_size)
        regenerated.save(thumb_path, "PNG")
        new_checksum = sha256_file(thumb_path)
    except (OSError, ValueError) as error:
        raise DerivativeError(
            "derivative_write_failed",
            f"could not rebuild the derivative thumbnail: {error}",
            retryable=True,
        ) from error
    replacement: CropFiles | None = None
    if new_checksum != registry["thumbnail_checksum"]:
        replacement = CropFiles(
            original_path=str(registry["original_path"]),
            line_art_path=str(registry["line_art_path"]),
            thumbnail_path=str(registry["thumbnail_path"]),
            source_checksum=str(registry["source_checksum"]),
            line_art_checksum=str(registry["line_art_checksum"]),
            thumbnail_checksum=new_checksum,
        )
    return measurements, replacement
