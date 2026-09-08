-- 0003_eligibility_v3.sql
-- Schema v3: canonical eligibility + current derivative validation.
--
-- Safety properties (see docs/contracts/migration-manifest.md):
--   * `assets` / `asset_scopes` / `gallery_versions` are caches of the v3
--     manifest, so they are rebuilt with the v3 columns rather than altered.
--     No legacy row is promoted: every rebuilt row starts `enabled = 0`,
--     `derivatives_current = 0`, `derivative_problems_json` records why.
--   * Events, preferences, curation labels (the audit trail), search log, and
--     snapshots are preserved verbatim. Source bytes on disk are never
--     touched by a migration.
--   * The `enabled = 1` CHECK now requires the full canonical policy,
--     including `permission_basis != 'unknown'` and `derivatives_current = 1`.
--   * `gold_member = 1` now requires accepted + quality 2-3 + no blockers +
--     known primary scope + human SFW approval + current derivatives.

DROP TABLE IF EXISTS asset_scopes;
DROP TABLE IF EXISTS assets;
DROP TABLE IF EXISTS gallery_versions;

CREATE TABLE assets (
    asset_id            TEXT PRIMARY KEY,
    source_dataset      TEXT NOT NULL,
    source_item_id      TEXT NOT NULL,
    source_work_id      TEXT NOT NULL,
    parent_asset_id     TEXT,                     -- derivative identity; NULL = not a derivative
    artist_id           TEXT,                     -- NULL = unknown, never invented
    leakage_group_id    TEXT,                     -- NULL = work is the de-facto leakage group
    source_url          TEXT,
    -- Permission provenance (deny by default).
    license_id          TEXT NOT NULL,
    permission_basis    TEXT NOT NULL DEFAULT 'unknown' CHECK (permission_basis IN
                            ('license_terms', 'public_domain', 'explicit_consent',
                             'first_party', 'unknown')),
    permission_url      TEXT,
    attribution         TEXT,
    attribution_required INTEGER NOT NULL DEFAULT 0 CHECK (attribution_required IN (0, 1)),
    allowed_display     INTEGER NOT NULL DEFAULT 0 CHECK (allowed_display IN (0, 1)),
    allowed_training    INTEGER NOT NULL DEFAULT 0 CHECK (allowed_training IN (0, 1)),
    allowed_trace       INTEGER NOT NULL DEFAULT 0 CHECK (allowed_trace IN (0, 1)),
    original_path       TEXT NOT NULL,
    line_art_path       TEXT NOT NULL,
    thumbnail_path      TEXT NOT NULL,
    origin              TEXT NOT NULL CHECK (origin IN ('native_line_art', 'extracted_line_art')),
    extraction_model    TEXT,
    extraction_version  TEXT,
    primary_style       TEXT NOT NULL CHECK (primary_style IN
                            ('manga_anime', 'western_ink', 'realistic_academic', 'cartoon', 'gesture_sketch')),
    -- 'unknown' is a legal provisional primary scope (resolved before acceptance).
    primary_scope       TEXT NOT NULL CHECK (primary_scope IN
                            ('eye', 'eyebrow', 'mouth', 'face_head', 'hair', 'hand', 'foot',
                             'upper_body_clothing', 'full_body', 'multi_character', 'unknown')),
    secondary_scopes_json TEXT NOT NULL DEFAULT '[]',
    person_count        INTEGER CHECK (person_count IS NULL OR person_count >= 0),
    person_count_approximate INTEGER NOT NULL DEFAULT 0 CHECK (person_count_approximate IN (0, 1)),
    -- Automated screen (tri-state) vs human approval (NULL = no human decision).
    sfw_verdict         TEXT CHECK (sfw_verdict IS NULL OR sfw_verdict IN ('safe', 'unsafe', 'unsure')),
    sfw_confidence      REAL,
    sfw_method          TEXT CHECK (sfw_method IS NULL OR sfw_method IN
                            ('none', 'source_rating', 'opennsfw2', 'source_rating+opennsfw2')),
    sfw_human_safe      INTEGER CHECK (sfw_human_safe IS NULL OR sfw_human_safe IN (0, 1)),
    sfw_human_reviewer  TEXT,
    sfw_human_decided_at TEXT,
    width               INTEGER NOT NULL CHECK (width > 0),
    height              INTEGER NOT NULL CHECK (height > 0),
    crop_json           TEXT,                     -- JSON {x,y,width,height} or NULL
    text_coverage       REAL NOT NULL,
    ink_coverage        REAL NOT NULL,
    phash               TEXT NOT NULL,
    quality_score       REAL NOT NULL,
    review_state        TEXT NOT NULL CHECK (review_state IN
                            ('unreviewed', 'accepted', 'rejected', 'quarantined')),
    review_quality      INTEGER CHECK (review_quality IN (1, 2, 3)),
    blockers_json       TEXT NOT NULL DEFAULT '[]',   -- JSON array of blocker names
    -- Learning assignment is independent of gallery/gold membership.
    learning_split      TEXT NOT NULL CHECK (learning_split IN ('train', 'validation', 'test', 'none')),
    gallery_member      INTEGER NOT NULL DEFAULT 1 CHECK (gallery_member IN (0, 1)),
    gold_member         INTEGER NOT NULL DEFAULT 0 CHECK (gold_member IN (0, 1)),
    pipeline_version    TEXT NOT NULL,
    processing_revision INTEGER NOT NULL DEFAULT 1 CHECK (processing_revision >= 1),
    label_version       TEXT NOT NULL,
    source_checksum     TEXT NOT NULL,
    line_art_checksum   TEXT NOT NULL,
    thumbnail_checksum  TEXT NOT NULL,
    faiss_row           INTEGER UNIQUE,           -- assigned when an index is built
    -- v3 derivative validity: 1 only when generation + bytes are verified.
    derivatives_current INTEGER NOT NULL DEFAULT 0 CHECK (derivatives_current IN (0, 1)),
    derivative_problems_json TEXT NOT NULL DEFAULT '[]',
    -- Derived serving flag: a cache of the canonical eligibility policy.
    -- Recomputed on gallery sync and label writes; never hand-edited.
    enabled             INTEGER NOT NULL DEFAULT 0 CHECK (enabled IN (0, 1)),
    -- Defence in depth for the frozen invariants (schema v3).
    CHECK (derivatives_current = 0 OR derivative_problems_json = '[]'),
    CHECK (enabled = 0 OR (gallery_member = 1 AND allowed_display = 1
                           AND permission_basis != 'unknown'
                           AND review_state = 'accepted'
                           AND COALESCE(sfw_human_safe, 0) = 1
                           AND blockers_json = '[]'
                           AND review_quality IN (2, 3)
                           AND derivatives_current = 1)),
    CHECK (allowed_trace = 0 OR allowed_display = 1),
    CHECK (allowed_display = 0 OR permission_basis != 'unknown'),
    CHECK (allowed_training = 0 OR permission_basis != 'unknown'),
    -- Gold has its own documented approval/quality conditions (schema v3).
    CHECK (gold_member = 0 OR (review_state = 'accepted'
                               AND review_quality IN (2, 3)
                               AND blockers_json = '[]'
                               AND primary_scope != 'unknown'
                               AND COALESCE(sfw_human_safe, 0) = 1
                               AND derivatives_current = 1)),
    CHECK (parent_asset_id IS NULL OR parent_asset_id != asset_id)
);

CREATE INDEX assets_enabled_style_idx ON assets (enabled, primary_style);
CREATE INDEX assets_source_work_idx ON assets (source_work_id);
CREATE INDEX assets_leakage_idx ON assets (leakage_group_id);
CREATE INDEX assets_learning_split_idx ON assets (learning_split);
CREATE INDEX assets_parent_idx ON assets (parent_asset_id);
CREATE INDEX assets_derivatives_idx ON assets (derivatives_current);

-- Scope membership (primary + secondary); the primary is disambiguated by
-- assets.primary_scope.
CREATE TABLE asset_scopes (
    asset_id    TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    scope       TEXT NOT NULL CHECK (scope IN
                    ('eye', 'eyebrow', 'mouth', 'face_head', 'hair', 'hand', 'foot',
                     'upper_body_clothing', 'full_body', 'multi_character')),
    PRIMARY KEY (asset_id, scope)
);
CREATE INDEX asset_scopes_scope_idx ON asset_scopes (scope);

-- Which manifest/index version is currently loaded (cache singleton), plus the
-- artifact contract it declares and how many rows were disabled by derivative
-- problems — the "report why" of the eligibility enforcement.
CREATE TABLE gallery_versions (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    dataset_version     TEXT NOT NULL,
    manifest_hash       TEXT NOT NULL,
    manifest_path       TEXT NOT NULL,
    asset_count         INTEGER NOT NULL,
    enabled_count       INTEGER NOT NULL,
    current_pipeline_version TEXT,
    current_label_version    TEXT,
    current_processing_revision INTEGER,
    derivative_problem_count INTEGER NOT NULL DEFAULT 0,
    loaded_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Human-readable audit of what each forward migration did (why legacy rows
-- are disabled). Never used to decide policy; it is a report.
CREATE TABLE IF NOT EXISTS migration_reports (
    version         INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    applied_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    notes_json      TEXT NOT NULL
);
INSERT INTO migration_reports (version, name, notes_json)
VALUES (3, '0003_eligibility_v3.sql',
 '{"summary":"assets cache rebuilt with derivative-currency and gold eligibility; no legacy row was promoted","detail":"enabled defaults to 0 and derivatives_current defaults to 0; derivative_problems_json records why per row; events, preferences, curation_labels, search_log, and snapshots are preserved verbatim; source bytes on disk are untouched"}');
