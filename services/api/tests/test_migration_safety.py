"""Migration-safety tests for the v2 SQL migration (0002_contract_v2).

The acceptance criteria for the schema freeze, applied to the local database:
no old asset gains permission or human approval through migration. The assets
cache is dropped (it is a cache of the manifest, and manifests are migrated by
``linescout-manifest migrate-v1`` under the same deny-by-default rules), while
user data — events, curation labels — is preserved with new deny-by-default
columns added.
"""

from __future__ import annotations

import importlib.resources
import sqlite3
from pathlib import Path

import pytest

from linescout_api.db import applied_versions, connect, migrate, split_statements


@pytest.fixture
def v1_database(tmp_path: Path) -> sqlite3.Connection:
    """A v1-shaped database with one asset, one event, and one curation label."""
    connection = connect(tmp_path / "v1.sqlite3")
    package = importlib.resources.files("linescout_api") / "migrations"
    # The v1 schema by name, not by directory iteration order: iterdir() is
    # filesystem-dependent and must never decide which migration "v1" is.
    sql1 = (package / "0001_initial.sql").read_text(encoding="utf-8")
    connection.execute("BEGIN")
    for statement in split_statements(sql1):
        connection.execute(statement)
    connection.execute("COMMIT")
    applied_versions(connection)
    connection.execute(
        "INSERT INTO schema_migrations(version, name) VALUES (1, '0001_initial.sql')"
    )
    connection.execute("BEGIN")
    connection.execute(
        """
        INSERT INTO assets (asset_id, source_dataset, source_item_id, source_work_id, license_id,
         original_path, line_art_path, thumbnail_path, origin, primary_style, scopes_json,
         person_count, sfw_safe, sfw_confidence, sfw_method, width, height, text_coverage,
         ink_coverage, phash, quality_score, review_state, split, enabled, pipeline_version,
         source_checksum, line_art_checksum, thumbnail_checksum)
        VALUES ('ls_x_0000000000000001','s','i','w','l','o','la','t','native_line_art','cartoon',
         '["eye","face_head"]',1,1,0.9,'source_rating',512,512,0.0,0.05,'0123456789abcdef',0.9,
         'accepted','train',1,'p','a','b','c')
        """
    )
    connection.execute(
        "INSERT INTO events (session_id, asset_id, event, style, query_revision)"
        " VALUES ('s','ls_x_0000000000000001','pin','cartoon',1)"
    )
    connection.execute(
        """
        INSERT INTO curation_labels (asset_id, decision, primary_style, scopes_json,
         malformed_anatomy, poor_extraction, quality, note, reviewer)
        VALUES ('ls_x_0000000000000001','keep','cartoon','["eye","face_head"]',1,0,2,'n','me')
        """
    )
    connection.execute("COMMIT")
    return connection


def test_migration_applies_and_is_idempotent(v1_database: sqlite3.Connection) -> None:
    assert migrate(v1_database) == [
        "0002_contract_v2.sql",
        "0003_eligibility_v3.sql",
        "0004_curation_safety.sql",
    ]
    assert migrate(v1_database) == []


def test_assets_cache_is_dropped_not_carried(v1_database: sqlite3.Connection) -> None:
    """No old asset keeps (or gains) any serving state: the cache is rebuilt
    from a v2 manifest whose grants are explicit and deny-by-default."""
    migrate(v1_database)
    assert v1_database.execute("SELECT COUNT(*) FROM assets").fetchone()[0] == 0


def test_events_preserved_and_stamped_live(v1_database: sqlite3.Connection) -> None:
    """Legacy events keep feeding preferences (they came from the real gallery);
    new fixture-mode events are stamped server-side and never count."""
    migrate(v1_database)
    row = v1_database.execute("SELECT gallery_kind FROM events").fetchone()
    assert row["gallery_kind"] == "live"


def test_curation_labels_preserved_with_blockers_and_split_scopes(
    v1_database: sqlite3.Connection,
) -> None:
    """Human work survives verbatim; the two booleans become named blockers
    and legacy scopes split primary/secondary with the frozen ordering rule."""
    migrate(v1_database)
    row = v1_database.execute(
        "SELECT decision, quality, note, reviewer, malformed_anatomy, poor_extraction,"
        " blockers_json, primary_scope, secondary_scopes_json FROM curation_labels"
    ).fetchone()
    assert row["decision"] == "keep"
    assert row["quality"] == 2
    assert row["note"] == "n"
    assert row["reviewer"] == "me"
    # Historical columns preserved; new blockers column carries the semantics.
    assert row["malformed_anatomy"] == 1
    assert row["poor_extraction"] == 0
    assert row["blockers_json"] == '["anatomy"]'
    assert row["primary_scope"] == "eye"
    assert row["secondary_scopes_json"] == '["face_head"]'


def test_new_columns_deny_by_default(v1_database: sqlite3.Connection) -> None:
    """A v2 row with no explicit grants cannot be enabled, traced, or displayed."""
    migrate(v1_database)
    base = {
        "asset_id": "'ls_x_0000000000000002'",
        "source_dataset": "'s'",
        "source_item_id": "'i'",
        "source_work_id": "'w'",
        "license_id": "'l'",
        "original_path": "'o'",
        "line_art_path": "'la'",
        "thumbnail_path": "'t'",
        "origin": "'native_line_art'",
        "primary_style": "'cartoon'",
        "primary_scope": "'eye'",
        "width": "512",
        "height": "512",
        "text_coverage": "0.0",
        "ink_coverage": "0.05",
        "phash": "'0123456789abcdef'",
        "quality_score": "0.9",
        "review_state": "'accepted'",
        "review_quality": "3",
        "learning_split": "'train'",
        "label_version": "'1'",
        "pipeline_version": "'p'",
        "source_checksum": "'a'",
        "line_art_checksum": "'b'",
        "thumbnail_checksum": "'c'",
    }

    def insert(**extra: str) -> None:
        columns = {**base, **extra}
        names = ", ".join(columns)
        values = ", ".join(columns.values())
        v1_database.execute(f"INSERT INTO assets ({names}) VALUES ({values})")

    # enabled=1 requires the full predicate: human SFW approval is missing.
    with pytest.raises(sqlite3.IntegrityError):
        insert(allowed_display="1", enabled="1", sfw_human_safe="NULL")
    # trace without display is rejected.
    with pytest.raises(sqlite3.IntegrityError):
        insert(allowed_display="0", allowed_trace="1")
    # display without a known permission basis is rejected.
    with pytest.raises(sqlite3.IntegrityError):
        insert(allowed_display="1", permission_basis="'unknown'")
    # gold without acceptance is rejected.
    with pytest.raises(sqlite3.IntegrityError):
        insert(review_state="'rejected'", gold_member="1")
    # A fully-granted, human-approved, quality-2 row is the only servable
    # shape, and it must also have current derivatives (schema v3).
    with pytest.raises(sqlite3.IntegrityError):
        insert(
            permission_basis="'first_party'",
            allowed_display="1",
            allowed_training="1",
            allowed_trace="1",
            sfw_human_safe="1",
            enabled="1",
            derivatives_current="0",
        )
    insert(
        permission_basis="'first_party'",
        allowed_display="1",
        allowed_training="1",
        allowed_trace="1",
        sfw_human_safe="1",
        enabled="1",
        derivatives_current="1",
    )
    # Gold (schema v3) needs its own conditions: unknown scope or stale
    # derivatives are rejected even when everything serving-related holds.
    with pytest.raises(sqlite3.IntegrityError):
        insert(gold_member="1", primary_scope="'unknown'")
    with pytest.raises(sqlite3.IntegrityError):
        insert(gold_member="1", derivatives_current="0")
