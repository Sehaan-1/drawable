-- 0004_curation_safety.sql
-- Schema v4: safe end-to-end curation.
--
-- Safety properties (see docs/contracts/api-contract.md, "Curation"):
--   * Every human curation write is guarded by a per-asset optimistic
--     concurrency version (``assets.curation_label_version``). The write is a
--     single atomic conditional UPDATE inside the audit transaction; a stale
--     version becomes a structured 409 carrying the live version and
--     reconciliation information — never a silent lost update.
--   * Curator-created crops are durable, immutable child derivatives. The
--     ``curation_derivatives`` registry survives gallery cache rebuilds (the
--     ``assets`` cache is rebuilt from the manifest; derivatives are
--     re-hydrated from this registry and their latest audit label), so a
--     dataset reload neither loses a derivative nor resurrects stale parent
--     artifacts. A child is never enabled before its own processing completes
--     and its own review approval is recorded.
--   * Quarantined/uncertain SFW records are adjudicated only through
--     curation routes behind a deliberate, expiring reveal grant
--     (``sfw_reveals``); public asset routes keep failing closed on them.
--     Every adjudication is append-only audit history (``sfw_adjudications``).
--   * The review queue is cursor-based per review session
--     (``curation_sessions``) with explicit skip exclusions
--     (``curation_exclusions``), so Next/Skip advance without a label and
--     Previous can re-fetch the actual previous candidate by id.
--
-- User data (events, preferences, curation labels, search log, snapshots) is
-- preserved verbatim. Source bytes on disk are never touched by a migration.

ALTER TABLE assets ADD COLUMN curation_label_version INTEGER NOT NULL DEFAULT 0
    CHECK (curation_label_version >= 0);

-- Durable registry of curator-created crop derivatives. The ``assets`` row is
-- a cache materialized from the parent row + this record (+ the latest audit
-- label); this table is the source of truth for the crop itself.
CREATE TABLE curation_derivatives (
    asset_id            TEXT PRIMARY KEY,
    parent_asset_id     TEXT NOT NULL CHECK (parent_asset_id != asset_id),
    crop_json           TEXT NOT NULL,               -- region in the parent's pixel space
    original_path       TEXT NOT NULL,               -- fresh files, relative to the data root
    line_art_path       TEXT NOT NULL,
    thumbnail_path      TEXT NOT NULL,
    source_checksum     TEXT NOT NULL,               -- sha256 of the fresh files
    line_art_checksum   TEXT NOT NULL,
    thumbnail_checksum  TEXT NOT NULL,
    width               INTEGER NOT NULL CHECK (width > 0),
    height              INTEGER NOT NULL CHECK (height > 0),
    -- Artifact generation the crop was cut under (the parent's generation at
    -- creation time). Frozen here: a later manifest that changes the current
    -- generation makes every crop stale until it is re-cut/re-processed, and
    -- a re-hydrated child must never claim the new generation for old bytes.
    pipeline_version    TEXT NOT NULL,
    label_version       TEXT NOT NULL,
    processing_revision INTEGER NOT NULL CHECK (processing_revision >= 1),
    -- The child's own processing lifecycle. ``pending`` until the required
    -- processing runs; ``failed`` records a retryable/terminal error;
    -- ``complete`` only after measurements were rebuilt from the child's own
    -- bytes and every file was verified against its checksum.
    processing_state    TEXT NOT NULL DEFAULT 'pending'
                        CHECK (processing_state IN ('pending', 'complete', 'failed')),
    processing_attempts INTEGER NOT NULL DEFAULT 0 CHECK (processing_attempts >= 0),
    processing_error    TEXT,
    measurements_json   TEXT,                        -- NULL until processing completes
    created_by          TEXT NOT NULL DEFAULT 'local',
    note                TEXT,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX curation_derivatives_parent_idx ON curation_derivatives (parent_asset_id);

-- Append-only audit trail of explicit SFW adjudications (human decisions on
-- quarantined/uncertain records). The assets row mirrors the latest decision;
-- this table is the history.
CREATE TABLE sfw_adjudications (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id            TEXT NOT NULL,
    safe                INTEGER NOT NULL CHECK (safe IN (0, 1)),
    prior_verdict       TEXT,                        -- automated verdict at decision time, if any
    prior_review_state  TEXT,
    reviewer            TEXT NOT NULL DEFAULT 'local',
    note                TEXT,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX sfw_adjudications_asset_idx ON sfw_adjudications (asset_id, created_at);

-- Deliberate reveal grants for quarantined/uncertain content. Viewing such a
-- record through the curation preview route requires a fresh, unexpired
-- grant; grants expire and are never issued implicitly. Public asset routes
-- ignore grants entirely and keep failing closed.
CREATE TABLE sfw_reveals (
    asset_id    TEXT PRIMARY KEY,
    revealed_by TEXT NOT NULL DEFAULT 'local',
    revealed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    expires_at  TEXT NOT NULL
);

-- Per-session queue cursor. ``cursor_asset_id`` is the last asset served to
-- (or skipped by) the session; the next call serves strictly after it in the
-- stable asset_id order, wrapping around the non-excluded pool.
CREATE TABLE curation_sessions (
    session_id      TEXT PRIMARY KEY,
    cursor_asset_id TEXT,
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Assets a session has explicitly skipped: excluded from that session's
-- Next until the pool is otherwise empty (the exclusion is per session and
-- never mutates the asset itself).
CREATE TABLE curation_exclusions (
    session_id  TEXT NOT NULL,
    asset_id    TEXT NOT NULL,
    reason      TEXT NOT NULL CHECK (reason IN ('skipped')),
    excluded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (session_id, asset_id)
);
CREATE INDEX curation_exclusions_session_idx ON curation_exclusions (session_id);

-- Snapshots record the sha256 of the exact bytes published, so a snapshot
-- file can be re-verified after the fact (reproducibility guarantee).
ALTER TABLE snapshots ADD COLUMN content_sha256 TEXT;

INSERT INTO migration_reports (version, name, notes_json)
VALUES (4, '0004_curation_safety.sql',
 '{"summary":"curation concurrency versions, durable crop-derivative registry, SFW adjudication with reveal grants, session queue cursors","detail":"assets gains curation_label_version (optimistic concurrency, default 0); curation_derivatives/sfw_adjudications/sfw_reveals/curation_sessions/curation_exclusions are new; snapshots gain content_sha256; user data is preserved verbatim and no source bytes are touched"}');
