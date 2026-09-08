"""Load a validated manifest into SQLite and answer gallery queries.

The manifest file is authoritative. At startup we validate it, refuse to start
if it is malformed, and then (re)populate the ``assets`` cache table only when
its content hash differs from what is already loaded.

Schema v3 notes:

* Only manifests with ``schema_version`` 3 load. A v1/v2 manifest fails
  validation with an error naming the ``linescout-manifest convert`` command
  (see ``docs/contracts/migration-manifest.md``).
* The ``enabled`` column is a *derived* cache of the canonical eligibility
  policy (gallery membership ∧ permitted display use under a known permission
  basis ∧ accepted review with quality 2–3 ∧ human SFW approval ∧ no blockers
  ∧ current valid derivatives). It is recomputed on sync and on curation
  writes; it is never hand-edited.
* Derivative validity is computed here: a record whose generation differs from
  the manifest's ``artifact_contract`` is stale, and a record that passes every
  other serving gate is further checked for missing files and checksum
  mismatches against the recorded hashes. Problems are stored per row
  (``derivative_problems_json``), the row is disabled, and the loader reports
  the count — this is a *disable-and-report*, never a silent promotion.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from linescout_ml.manifest import (
    ArtifactContract,
    Manifest,
    ManifestRecord,
    check_parent_integrity,
    check_split_integrity,
    derivative_reasons,
    is_servable,
    serving_reasons,
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
    #: Current artifact generation declared by the manifest (None = unknown).
    artifact_contract: ArtifactContract | None
    #: Count of rows disabled specifically because of derivative problems.
    derivative_problem_count: int


@dataclass(frozen=True)
class DerivativeCheck:
    """Per-record derivative validity (generation + on-disk bytes)."""

    current: bool
    problems: tuple[str, ...]


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
    """Name the conversion command when the document is an old schema version."""
    try:
        version = json.loads(raw).get("schema_version")
    except json.JSONDecodeError:
        return ""
    if version == 1:
        return (
            "schema v1 manifests are not loadable (schema_version must be 3); "
            "run `linescout-manifest convert <path> --from 1 --to 3 --out <v3-path>` first — "
        )
    if version == 2:
        return (
            "schema v2 manifests are not loadable (schema_version must be 3); "
            "run `linescout-manifest convert <path> --from 2 --to 3 --out <v3-path>` first — "
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
    "derivatives_current",
    "derivative_problems_json",
    "enabled",
)

_INSERT_ASSET = (
    f"INSERT INTO assets ({', '.join(_ASSET_COLUMNS)})"
    f" VALUES ({', '.join('?' * len(_ASSET_COLUMNS))})"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derivative_check(
    record: ManifestRecord,
    manifest: Manifest,
    data_root: Path,
) -> DerivativeCheck:
    """Compute one record's derivative validity under the canonical policy.

    Generation currency is checked for every record (cheap). On-disk file
    presence and bytes are verified only for records that pass every *other*
    serving gate — an ineligible record is disabled by policy regardless, and
    verifying unreviewed/quarantined bytes on every startup would cost a full
    gallery hash for no serving benefit.
    """
    problems: list[str] = list(derivative_reasons(record, manifest.artifact_contract))
    other_gates = [
        reason
        for reason in serving_reasons(record, manifest.artifact_contract)
        if not reason.startswith("derivative_")
    ]
    # Gold is a curated statement about the *current* artifact, so its bytes
    # are also verified even when the record is not otherwise displayable.
    if not other_gates or record.gold_member:
        for label, relative, checksum in (
            ("line_art", record.line_art_path, record.line_art_checksum),
            ("thumbnail", record.thumbnail_path, record.thumbnail_checksum),
            ("original", record.original_path, record.source_checksum),
        ):
            path = data_root / relative
            if not path.is_file():
                problems.append(f"derivative_file_missing:{label}")
                continue
            if _sha256(path) != checksum:
                problems.append(f"derivative_checksum_mismatch:{label}")
    return DerivativeCheck(current=not problems, problems=tuple(problems))


def _record_row(record: ManifestRecord, manifest: Manifest, data_root: Path) -> tuple[object, ...]:
    derivative = derivative_check(record, manifest, data_root)
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
        # The cache may only carry gold status it has verified: a gold record
        # whose bytes are missing/tampered is conservatively downgraded here
        # (reported via derivative_problem_count) — never silently un-golded,
        # and never kept gold while its artifact is unverifiable.
        int(record.gold_member and derivative.current),
        record.pipeline_version,
        record.processing_revision,
        record.label_version,
        record.source_checksum,
        record.line_art_checksum,
        record.thumbnail_checksum,
        int(derivative.current),
        json.dumps(derivative.problems),
        int(is_servable(record, manifest.artifact_contract) and derivative.current),
    )
    assert len(values) == len(_ASSET_COLUMNS), "assets row out of sync with _ASSET_COLUMNS"
    return values


def sync_gallery(connection: sqlite3.Connection, manifest_path: Path) -> GalleryInfo:
    """Validate ``manifest_path`` and make the ``assets`` table match it.

    Derivative problems (stale generation, missing file, checksum mismatch)
    never fail the load: they are recorded per row, the row is disabled, and
    counts are returned for the health warning. Manifest *structure* problems
    (wrong schema version, split/parent integrity) still fail loudly.
    """
    manifest = load_manifest(manifest_path)
    manifest_hash = manifest.content_hash()
    data_root = manifest_path.parent

    checks = {
        record.asset_id: derivative_check(record, manifest, data_root)
        for record in manifest.records
    }
    derivative_problem_count = sum(1 for check in checks.values() if not check.current)
    enabled = [
        record.asset_id
        for record in manifest.records
        if is_servable(record, manifest.artifact_contract) and checks[record.asset_id].current
    ]

    current = connection.execute(
        "SELECT manifest_hash FROM gallery_versions WHERE id = 1"
    ).fetchone()
    if current is not None and current["manifest_hash"] == manifest_hash:
        log.info(
            "gallery %s already loaded (%d assets)", manifest.dataset_version, len(manifest.records)
        )
    else:
        log.info(
            "loading gallery %s (%d assets, %d enabled, %d derivative problems)",
            manifest.dataset_version,
            len(manifest.records),
            len(enabled),
            derivative_problem_count,
        )
        with transaction(connection) as tx:
            tx.execute("DELETE FROM asset_scopes")
            tx.execute("DELETE FROM assets")
            tx.executemany(
                _INSERT_ASSET,
                (_record_row(record, manifest, data_root) for record in manifest.records),
            )
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
                " (id, dataset_version, manifest_hash, manifest_path, asset_count,"
                " enabled_count, current_pipeline_version, current_label_version,"
                " current_processing_revision, derivative_problem_count)"
                " VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    manifest.dataset_version,
                    manifest_hash,
                    str(manifest_path),
                    len(manifest.records),
                    len(enabled),
                    manifest.artifact_contract.pipeline_version
                    if manifest.artifact_contract
                    else None,
                    manifest.artifact_contract.label_version
                    if manifest.artifact_contract
                    else None,
                    manifest.artifact_contract.processing_revision
                    if manifest.artifact_contract
                    else None,
                    derivative_problem_count,
                ),
            )

    return GalleryInfo(
        dataset_version=manifest.dataset_version,
        manifest_hash=manifest_hash,
        manifest_path=manifest_path,
        data_root=data_root,
        asset_count=len(manifest.records),
        enabled_count=len(enabled),
        artifact_contract=manifest.artifact_contract,
        derivative_problem_count=derivative_problem_count,
    )


#: SQL mirror of the canonical eligibility policy. Used to recompute the
#: derived ``enabled`` flag after a curation write without a reload.
ENABLED_SQL = (
    "gallery_member = 1 AND allowed_display = 1"
    " AND permission_basis != 'unknown'"
    " AND review_state = 'accepted'"
    " AND COALESCE(sfw_human_safe, 0) = 1 AND blockers_json = '[]'"
    " AND review_quality IN (2, 3)"
    " AND derivatives_current = 1"
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
        "SELECT gallery_member, allowed_display, permission_basis, review_state,"
        " review_quality, sfw_human_safe, blockers_json, derivatives_current,"
        " derivative_problems_json FROM assets WHERE asset_id = ?",
        (asset_id,),
    ).fetchone()
    if row is None:
        return ["asset_not_in_gallery"]
    reasons: list[str] = []
    if not row["gallery_member"]:
        reasons.append("not_a_gallery_member")
    if not row["allowed_display"]:
        reasons.append("display_not_permitted")
    if row["permission_basis"] == "unknown":
        reasons.append("permission_unknown")
    if row["review_state"] != "accepted":
        reasons.append(f"review_{row['review_state']}")
    if row["review_quality"] is None:
        reasons.append("quality_missing")
    elif row["review_quality"] < 2:
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
    try:
        problems = json.loads(row["derivative_problems_json"] or "[]")
    except json.JSONDecodeError:
        problems = []
    reasons.extend(problems)
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
    #: Derived-validity flag; the ranker never lets a stale asset into a
    #: response even if it somehow reaches the in-memory list.
    derivatives_current: bool = True

    @property
    def scopes(self) -> tuple[str, ...]:
        return (self.primary_scope, *self.secondary_scopes)


def enabled_assets(connection: sqlite3.Connection) -> list[GalleryAsset]:
    """Every asset eligible to appear in a search response (canonical policy).

    Defence in depth: the query repeats the ``enabled = 1`` AND
    ``derivatives_current = 1`` gates so an unintended edit to one column
    cannot leak a stale or otherwise ineligible asset into search.
    """
    rows = connection.execute(
        "SELECT asset_id, primary_style, primary_scope, secondary_scopes_json, origin,"
        " person_count, person_count_approximate, quality_score, allowed_trace,"
        " line_art_path, thumbnail_path, derivatives_current"
        " FROM assets WHERE enabled = 1 AND derivatives_current = 1 ORDER BY asset_id"
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
            derivatives_current=bool(row["derivatives_current"]),
        )
        for row in rows
    ]


@dataclass(frozen=True)
class AssetFile:
    """A servable asset's on-disk file plus the checksum it must match."""

    relative_path: str
    sha256: str


def asset_file(connection: sqlite3.Connection, asset_id: str, kind: str) -> AssetFile | None:
    """Relative path + recorded sha256 for a public asset file, else ``None``.

    Public serving requires the full canonical policy: ``enabled = 1`` (which
    includes current derivatives) *and* ``derivatives_current = 1`` — a stale
    or corrupt derivative is never served through any public path.
    """
    column = {"thumbnail": "thumbnail_path", "line_art": "line_art_path"}[kind]
    checksum = {"thumbnail": "thumbnail_checksum", "line_art": "line_art_checksum"}[kind]
    row = connection.execute(
        f"SELECT {column} AS path, {checksum} AS sha256 FROM assets"  # noqa: S608
        " WHERE asset_id = ? AND enabled = 1 AND derivatives_current = 1",
        (asset_id,),
    ).fetchone()
    if row is None:
        return None
    return AssetFile(relative_path=str(row["path"]), sha256=str(row["sha256"]))
