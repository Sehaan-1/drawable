from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .database import connect, migrate
from .intake import (
    IntakeError,
    _anita_work_records,
    ensure_inside,
    load_registry,
    sha256_file,
    utc_now,
    verify_archive_receipt,
)
from .settings import REPOSITORY_ROOT


def catalog_manifest(
    data_root: Path,
    database_path: Path,
    manifest_path: Path,
    *,
    registry_path: Path | None = None,
    anita_terms_path: Path | None = None,
) -> dict[str, Any]:
    root = data_root.resolve()
    manifest_file = manifest_path.resolve()
    if (
        root not in manifest_file.parents
        or not manifest_file.is_file()
        or manifest_file.is_symlink()
    ):
        raise IntakeError("Catalog manifest must be a regular file inside the private data root")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError("Catalog manifest is invalid") from error
    registry_file = registry_path or REPOSITORY_ROOT / "config" / "sources.json"
    registry = load_registry(registry_file)
    registry_hash = sha256_file(registry_file)
    if manifest.get("source_registry_sha256") != registry_hash:
        raise IntakeError("Catalog manifest source-registry hash is stale")
    source_id = manifest.get("source_id")
    source = next((row for row in registry["sources"] if row["id"] == source_id), None)
    if not source:
        raise IntakeError("Catalog manifest source is absent from the reviewed source registry")
    terms_config_hash: str | None = None
    if source_id == "anita-dataset":
        terms_file = anita_terms_path or REPOSITORY_ROOT / "config" / "anita-work-terms.json"
        terms_config_hash = sha256_file(terms_file)
        if manifest.get("anita_work_terms_sha256") != terms_config_hash:
            raise IntakeError("Catalog manifest Anita work-terms hash is stale")
        _anita_work_records(terms_file)
    relative_root = manifest.get("relative_root")
    if not isinstance(relative_root, str):
        raise IntakeError("Catalog manifest has no relative source root")
    tree_root = ensure_inside(root, relative_root)
    if not tree_root.is_dir() or tree_root.is_symlink():
        raise IntakeError("Catalog source tree is missing or unsafe")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise IntakeError("Catalog manifest entries must be a list")
    seen_paths: set[str] = set()
    seen_asset_ids: set[str] = set()
    tree_digest = hashlib.sha256()
    verified_entries: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda row: str(row.get("relative_path", ""))):
        relative = entry.get("relative_path")
        if not isinstance(relative, str) or relative in seen_paths:
            raise IntakeError("Catalog manifest contains a malformed or duplicate relative path")
        seen_paths.add(relative)
        asset_id = entry.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id or asset_id in seen_asset_ids:
            raise IntakeError("Catalog manifest contains a malformed or duplicate asset ID")
        seen_asset_ids.add(asset_id)
        path = ensure_inside(tree_root, relative)
        if path.is_symlink() or not path.is_file():
            raise IntakeError(f"Catalog asset is missing or unsafe: {relative}")
        byte_length = path.stat().st_size
        digest = sha256_file(path)
        if entry.get("byte_length") != byte_length or entry.get("sha256") != digest:
            raise IntakeError(f"Catalog asset no longer matches its manifest: {relative}")
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(byte_length).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\n")
        metadata = dict(entry)
        metadata["relative_root"] = relative_root
        metadata["manifest_sha256"] = sha256_file(manifest_file)
        metadata["source_archive_sha256"] = manifest.get("source_archive_sha256")
        metadata["source_registry_sha256"] = registry_hash
        metadata["anita_work_terms_sha256"] = terms_config_hash
        review_basis = {
            "source_id": source_id,
            "relative_path": relative,
            "sha256": digest,
            "source_registry_sha256": registry_hash,
            "anita_work_terms_sha256": terms_config_hash,
            "work_id": metadata.get("work_id"),
            "group_id": metadata.get("group_id"),
            "artist_group_id": metadata.get("artist_group_id"),
            "candidate_type": metadata.get("candidate_type"),
            "publication_partition": metadata.get("publication_partition"),
            "book_identity_verified": metadata.get("book_identity_verified"),
            "annotation_source_file": metadata.get("annotation_source_file"),
            "annotation_source_sha256": metadata.get("annotation_source_sha256"),
            "character_annotations": metadata.get("character_annotations"),
            "book_mapping_sha256": metadata.get("book_mapping_sha256"),
            "official_book_index_sha256": metadata.get("official_book_index_sha256"),
        }
        metadata["review_basis_sha256"] = hashlib.sha256(
            json.dumps(
                review_basis, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        locked_test = entry.get("publication_partition") == "source_test_locked" or entry.get(
            "learning_split"
        ) in {"test", "source_test_locked"}
        verified_entries.append(
            {
                "id": asset_id,
                "path": relative,
                "hash": digest,
                "bytes": byte_length,
                "metadata": metadata,
                "locked_test": locked_test,
            }
        )
    if source_id == "emoreccom-panels":
        mapping_relative = manifest.get("emoreccom_book_map_path")
        mapping_hash = manifest.get("emoreccom_book_map_sha256")
        if bool(mapping_relative) != bool(mapping_hash):
            raise IntakeError("EmoRecCom book-map path and hash must be present together")
        mapped_books: dict[str, dict[str, Any]] = {}
        if mapping_relative:
            from .emoreccom_map import load_approved_book_mapping

            if not isinstance(mapping_relative, str) or not isinstance(mapping_hash, str):
                raise IntakeError("EmoRecCom manifest book-map reference is malformed")
            map_path = ensure_inside(root, mapping_relative)
            if (
                map_path.is_symlink()
                or not map_path.is_file()
                or sha256_file(map_path) != mapping_hash
            ):
                raise IntakeError(
                    "EmoRecCom manifest book-map file no longer matches its recorded checksum"
                )
            mapped_books = load_approved_book_mapping(root, map_path)
        for item in verified_entries:
            metadata = item["metadata"]
            prefix = metadata.get("provisional_book_prefix")
            book = mapped_books.get(prefix) if isinstance(prefix, str) else None
            if metadata.get("book_identity_verified") is True:
                if (
                    not book
                    or metadata.get("comic_no") != book.get("comic_no")
                    or metadata.get("group_id") != f"emoreccom:book:{book['comic_no']}"
                    or metadata.get("book_mapping_sha256") != book.get("_mapping_sha256")
                ):
                    raise IntakeError(
                        "EmoRecCom manifest book identity differs from the reviewed official crosswalk"
                    )
            elif book is not None or metadata.get("group_id") is not None:
                raise IntakeError(
                    "EmoRecCom manifest has an unmapped or falsely unverified source family"
                )
    if tree_digest.hexdigest() != manifest.get("tree_sha256"):
        raise IntakeError("Catalog manifest tree checksum is stale or malformed")
    archive_receipt = None
    if source.get("archive_path"):
        receipt_path = root / "manifests" / "source-receipts" / f"{source_id}.json"
        if receipt_path.is_file() and not receipt_path.is_symlink():
            try:
                archive_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise IntakeError("Catalog source acquisition receipt is corrupt") from error
            if archive_receipt.get("sha256") != manifest.get("source_archive_sha256"):
                raise IntakeError(
                    "Catalog archive receipt does not match the manifest source archive"
                )

    migrate(database_path)
    db = connect(database_path)
    inserted = updated = preserved = locked = 0
    invalidated_assignments = 0
    seen_asset_ids = {item["id"] for item in verified_entries}
    try:
        db.execute("BEGIN IMMEDIATE")
        if archive_receipt:
            archive_path = ensure_inside(root, str(archive_receipt.get("archive_path", "")))
            verify_archive_receipt(root, source, archive_path, archive_receipt)
            db.execute(
                "INSERT OR IGNORE INTO source_receipts(receipt_id, source_id, archive_path, byte_length, sha256, retrieved_at, terms_sha256, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    archive_receipt.get("receipt_id") or str(uuid4()),
                    source_id,
                    archive_receipt.get("archive_path", ""),
                    archive_receipt.get("byte_length", 0),
                    archive_receipt["sha256"],
                    archive_receipt.get("retrieved_at", utc_now()),
                    archive_receipt.get("terms_snapshot_sha256"),
                    json.dumps(archive_receipt, sort_keys=True, separators=(",", ":")),
                ),
            )
        changed_groups: set[str] = set()
        for item in verified_entries:
            existing = db.execute(
                "SELECT sha256, revision, state, metadata_json FROM assets WHERE asset_id = ?",
                (item["id"],),
            ).fetchone()
            metadata = item["metadata"]
            metadata["catalog_presence"] = "present"
            revision = 0
            state = "locked_test" if item["locked_test"] else "quarantined"
            if existing:
                revision = int(existing["revision"])
                old_metadata = json.loads(existing["metadata_json"])
                same_review_basis = existing["sha256"] == item["hash"] and old_metadata.get(
                    "review_basis_sha256"
                ) == metadata.get("review_basis_sha256")
                if same_review_basis:
                    for key in (
                        "curation_review",
                        "rights_status",
                        "use_permissions",
                        "labels",
                        "gallery_member",
                        "gallery_eligible_candidate",
                        "review_status",
                        "viewpoint",
                        "learning_split",
                        "terms_snapshot_sha256",
                        "license_id",
                        "attribution_text",
                    ):
                        if key in old_metadata:
                            metadata[key] = old_metadata[key]
                    state = str(existing["state"])
                    preserved += 1
                else:
                    for group in (old_metadata.get("group_id"), metadata.get("group_id")):
                        if isinstance(group, str) and group:
                            changed_groups.add(group)
                    revision += 1
                    updated += 1
            else:
                inserted += 1
            if item["locked_test"]:
                state = "locked_test"
                locked += 1
            encoded = json.dumps(
                metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            db.execute(
                "INSERT INTO assets(asset_id, source_id, relative_path, sha256, byte_length, media_type, width, height, group_id, rights_status, state, revision, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(asset_id) DO UPDATE SET source_id=excluded.source_id, relative_path=excluded.relative_path, sha256=excluded.sha256, byte_length=excluded.byte_length, media_type=excluded.media_type, width=excluded.width, height=excluded.height, group_id=excluded.group_id, rights_status=excluded.rights_status, state=excluded.state, revision=excluded.revision, metadata_json=excluded.metadata_json",
                (
                    item["id"],
                    source_id,
                    item["path"],
                    item["hash"],
                    item["bytes"],
                    metadata.get("media_type"),
                    metadata.get("width"),
                    metadata.get("height"),
                    metadata.get("group_id"),
                    metadata.get("rights_status", "review_required"),
                    state,
                    revision,
                    encoded,
                ),
            )
        # A recatalog must not leave approvals attached to files that disappeared
        # from the current source tree. Keep bytes on disk, but quarantine and age
        # the stale catalog rows so they cannot be previewed or promoted.
        source_rows = db.execute(
            "SELECT asset_id, revision, state, group_id, metadata_json FROM assets WHERE source_id = ?",
            (source_id,),
        ).fetchall()
        for row in source_rows:
            if row["asset_id"] in seen_asset_ids:
                continue
            old_metadata = json.loads(row["metadata_json"])
            if (
                old_metadata.get("parent_asset_id")
                or old_metadata.get("candidate_type") == "manual_detail_crop"
            ):
                continue
            if (
                old_metadata.get("catalog_presence") == "missing_from_latest_manifest"
                and row["state"] == "quarantined"
            ):
                continue
            old_metadata["catalog_presence"] = "missing_from_latest_manifest"
            old_metadata["review_status"] = "stale"
            old_metadata["gallery_member"] = False
            old_metadata["gallery_eligible_candidate"] = False
            db.execute(
                "UPDATE assets SET state = 'quarantined', revision = revision + 1, metadata_json = ? WHERE asset_id = ? AND revision = ?",
                (
                    json.dumps(
                        old_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ),
                    row["asset_id"],
                    row["revision"],
                ),
            )
            if isinstance(row["group_id"], str) and row["group_id"]:
                changed_groups.add(row["group_id"])

        for group_id in sorted(changed_groups):
            changed = db.execute(
                "UPDATE family_split_assignments SET status = 'invalidated', invalidated_at = ?, invalidation_reason = ? "
                "WHERE source_id = ? AND group_id = ? AND status = 'active'",
                (utc_now(), "source_bytes_or_review_basis_changed", source_id, group_id),
            )
            invalidated_assignments += changed.rowcount

        # Derivative additions (for example a newly approved crop) can be
        # legitimate family members. Reconcile active assignments against the
        # full catalog after the source rows have been updated.
        active_assignments = db.execute(
            "SELECT assignment_id, group_id, member_count, membership_sha256 FROM family_split_assignments "
            "WHERE source_id = ? AND status = 'active'",
            (source_id,),
        ).fetchall()
        for assignment in active_assignments:
            members = db.execute(
                "SELECT asset_id, sha256 FROM assets WHERE source_id = ? AND group_id = ? ORDER BY asset_id",
                (source_id, assignment["group_id"]),
            ).fetchall()
            membership = [
                {"asset_id": member["asset_id"], "sha256": member["sha256"]} for member in members
            ]
            membership_sha256 = hashlib.sha256(
                json.dumps(membership, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if (
                len(members) != assignment["member_count"]
                or membership_sha256 != assignment["membership_sha256"]
            ):
                db.execute(
                    "UPDATE family_split_assignments SET status = 'invalidated', invalidated_at = ?, invalidation_reason = ? "
                    "WHERE assignment_id = ? AND status = 'active'",
                    (utc_now(), "source_family_membership_changed", assignment["assignment_id"]),
                )
                invalidated_assignments += 1
        db.execute("COMMIT")
    except Exception:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise
    finally:
        db.close()
    return {
        "status": "cataloged_quarantined_pending_human_review",
        "source_id": source_id,
        "manifest_sha256": sha256_file(manifest_file),
        "assets_seen": len(verified_entries),
        "inserted": inserted,
        "updated": updated,
        "review_preserved": preserved,
        "locked_test_assets": locked,
        "invalidated_family_splits": invalidated_assignments,
        "gallery_assets_activated": 0,
    }
