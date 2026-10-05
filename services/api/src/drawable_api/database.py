from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 3


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _backup_before_upgrade(connection: sqlite3.Connection, path: Path, version: int) -> Path:
    backup_dir = path.parent / "reports" / "private" / "metadata-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = (
        backup_dir
        / f"metadata-v{version}-before-v{SCHEMA_VERSION}-{timestamp}-{uuid.uuid4().hex[:8]}.sqlite3"
    )
    descriptor = os.open(backup_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        target = sqlite3.connect(backup_path)
        try:
            connection.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite backup failed its integrity check")
        finally:
            target.close()
        os.chmod(backup_path, 0o600)
        return backup_path
    except Exception:
        backup_path.unlink(missing_ok=True)
        raise


def migrate(path: Path) -> int:
    """Apply forward-only, versioned local metadata migrations."""
    db = connect(path)
    try:
        version = int(db.execute("PRAGMA user_version").fetchone()[0])
        initial_version = version
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Metadata schema {version} is newer than supported schema {SCHEMA_VERSION}"
            )
        if version == 0:
            try:
                db.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE source_receipts (
                        receipt_id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        archive_path TEXT NOT NULL,
                        byte_length INTEGER NOT NULL CHECK(byte_length >= 0),
                        sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
                        retrieved_at TEXT NOT NULL,
                        terms_sha256 TEXT,
                        metadata_json TEXT NOT NULL
                    );
                    CREATE TABLE assets (
                        asset_id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
                        byte_length INTEGER NOT NULL CHECK(byte_length >= 0),
                        media_type TEXT,
                        width INTEGER,
                        height INTEGER,
                        group_id TEXT,
                        rights_status TEXT NOT NULL DEFAULT 'review_required',
                        state TEXT NOT NULL DEFAULT 'quarantined',
                        revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
                        metadata_json TEXT NOT NULL,
                        UNIQUE(source_id, relative_path)
                    );
                    CREATE INDEX assets_source_state ON assets(source_id, state);
                    CREATE TABLE reviews (
                        review_id TEXT PRIMARY KEY,
                        asset_id TEXT NOT NULL REFERENCES assets(asset_id),
                        expected_revision INTEGER NOT NULL,
                        reviewer_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL UNIQUE,
                        decision TEXT NOT NULL CHECK(decision IN ('keep', 'reject', 'quarantine')),
                        labels_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE gallery_builds (
                        build_id TEXT PRIMARY KEY,
                        manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
                        state TEXT NOT NULL CHECK(state IN ('staging', 'validated', 'active', 'retired', 'failed')),
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE service_state (
                        name TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    PRAGMA user_version = 1;
                    COMMIT;
                    """
                )
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise
            version = 1
        if version == 1:
            if initial_version == 1:
                _backup_before_upgrade(db, path, version)
            try:
                db.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE curation_actions (
                        idempotency_key TEXT PRIMARY KEY,
                        request_sha256 TEXT NOT NULL CHECK(length(request_sha256) = 64),
                        response_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE viewpoint_pair_reviews (
                        review_id TEXT PRIMARY KEY,
                        query_asset_id TEXT NOT NULL REFERENCES assets(asset_id),
                        reference_asset_id TEXT NOT NULL REFERENCES assets(asset_id),
                        reviewer_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL UNIQUE,
                        rubric_sha256 TEXT NOT NULL CHECK(length(rubric_sha256) = 64),
                        review_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        CHECK(query_asset_id != reference_asset_id)
                    );
                    CREATE INDEX viewpoint_pair_query ON viewpoint_pair_reviews(query_asset_id, created_at);
                    CREATE TABLE curation_snapshots (
                        snapshot_id TEXT PRIMARY KEY,
                        snapshot_sha256 TEXT NOT NULL UNIQUE CHECK(length(snapshot_sha256) = 64),
                        rubric_sha256 TEXT NOT NULL CHECK(length(rubric_sha256) = 64),
                        item_count INTEGER NOT NULL CHECK(item_count >= 0),
                        manifest_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE family_split_assignments (
                        assignment_id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        group_id TEXT NOT NULL,
                        learning_split TEXT NOT NULL CHECK(learning_split IN ('pilot_development', 'train', 'validation')),
                        member_count INTEGER NOT NULL CHECK(member_count > 0),
                        membership_sha256 TEXT NOT NULL CHECK(length(membership_sha256) = 64),
                        reviewer_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(source_id, group_id)
                    );
                    PRAGMA user_version = 2;
                    COMMIT;
                    """
                )
                version = 2
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise
        if version == 2:
            if initial_version == 2:
                _backup_before_upgrade(db, path, version)
            try:
                db.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE family_split_assignments_v3 (
                        assignment_id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        group_id TEXT NOT NULL,
                        learning_split TEXT NOT NULL CHECK(learning_split IN ('pilot_development', 'train', 'validation')),
                        member_count INTEGER NOT NULL CHECK(member_count > 0),
                        membership_sha256 TEXT NOT NULL CHECK(length(membership_sha256) = 64),
                        reviewer_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'invalidated')),
                        invalidated_at TEXT,
                        invalidation_reason TEXT
                    );
                    INSERT INTO family_split_assignments_v3(
                        assignment_id, source_id, group_id, learning_split, member_count,
                        membership_sha256, reviewer_id, created_at, status
                    )
                    SELECT assignment_id, source_id, group_id, learning_split, member_count,
                           membership_sha256, reviewer_id, created_at, 'active'
                    FROM family_split_assignments;
                    DROP TABLE family_split_assignments;
                    ALTER TABLE family_split_assignments_v3 RENAME TO family_split_assignments;
                    CREATE UNIQUE INDEX family_split_active_group
                        ON family_split_assignments(source_id, group_id) WHERE status = 'active';
                    CREATE INDEX family_split_history ON family_split_assignments(source_id, group_id, created_at);
                    PRAGMA user_version = 3;
                    COMMIT;
                    """
                )
            except Exception:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise
        return int(db.execute("PRAGMA user_version").fetchone()[0])
    finally:
        db.close()
