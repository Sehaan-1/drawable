-- 0002_contract_v2.sql
-- Schema v2 contract freeze: separate permission / SFW-human / learning fields.
--
-- Safety properties (see docs/contracts/migration-v1-to-v2.md):
--   * No old asset gains permission: the assets cache is dropped and rebuilt
--     from a v2 manifest whose allowed uses are explicit. The new columns all
--     default to deny (allowed_* = 0, permission_basis = 'unknown',
--     sfw_human_safe = NULL, gold_member = 0).
--   * No old asset gains human approval: curation labels are preserved
--     verbatim; automated-only SFW evidence lives in sfw_verdict, and
--     sfw_human_safe starts NULL for every row.
--   * `assets` / `asset_scopes` / `gallery_versions` are caches of the
--     manifest (the manifest is the source of truth), so they are rebuilt
--     rather than altered. Events, preferences, curation labels, and the
--     search log — user data — are preserved.

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
    -- Derived serving flag: a cache of the manifest's is_servable predicate.
    -- Recomputed on gallery sync and label writes; never hand-edited.
    enabled             INTEGER NOT NULL DEFAULT 0 CHECK (enabled IN (0, 1)),
    -- Defence in depth for the frozen invariants.
    CHECK (enabled = 0 OR (gallery_member = 1 AND allowed_display = 1
                           AND review_state = 'accepted'
                           AND COALESCE(sfw_human_safe, 0) = 1
                           AND blockers_json = '[]'
                           AND review_quality IN (2, 3))),
    CHECK (allowed_trace = 0 OR allowed_display = 1),
    CHECK (allowed_display = 0 OR permission_basis != 'unknown'),
    CHECK (allowed_training = 0 OR permission_basis != 'unknown'),
    CHECK (gold_member = 0 OR review_state = 'accepted'),
    CHECK (parent_asset_id IS NULL OR parent_asset_id != asset_id)
);

CREATE INDEX assets_enabled_style_idx ON assets (enabled, primary_style);
CREATE INDEX assets_source_work_idx ON assets (source_work_id);
CREATE INDEX assets_leakage_idx ON assets (leakage_group_id);
CREATE INDEX assets_learning_split_idx ON assets (learning_split);
CREATE INDEX assets_parent_idx ON assets (parent_asset_id);

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

-- Which manifest/index version is currently loaded (cache singleton).
CREATE TABLE gallery_versions (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    dataset_version     TEXT NOT NULL,
    manifest_hash       TEXT NOT NULL,
    manifest_path       TEXT NOT NULL,
    asset_count         INTEGER NOT NULL,
    enabled_count       INTEGER NOT NULL,
    loaded_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Immutable full-snapshot registry: one row per POST /curation/snapshots.
-- Lineage is the previous_snapshot_id chain; files are never edited.
CREATE TABLE snapshots (
    snapshot_id         TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    label_count         INTEGER NOT NULL,
    previous_snapshot_id TEXT REFERENCES snapshots(snapshot_id),
    path                TEXT NOT NULL
);

-- Interaction events gain the gallery namespace they were recorded under.
-- Legacy rows default to 'live' (they came from the user's real gallery) so
-- existing learned preferences survive the upgrade unchanged.
ALTER TABLE events ADD COLUMN gallery_kind TEXT NOT NULL DEFAULT 'live';
CREATE INDEX events_kind_idx ON events (gallery_kind, created_at);

-- Curation labels: named blockers replace the two booleans (kept verbatim on
-- historical rows), a label may record a human SFW decision, and scopes split
-- into primary + secondary. The legacy ``scopes_json`` column is retained
-- verbatim on historical rows and no longer written.
ALTER TABLE curation_labels ADD COLUMN blockers_json TEXT;
ALTER TABLE curation_labels ADD COLUMN sfw_safe INTEGER CHECK (sfw_safe IS NULL OR sfw_safe IN (0, 1));
ALTER TABLE curation_labels ADD COLUMN primary_scope TEXT;
ALTER TABLE curation_labels ADD COLUMN secondary_scopes_json TEXT;
UPDATE curation_labels
   SET blockers_json = CASE
        WHEN malformed_anatomy = 1 AND poor_extraction = 1 THEN '["anatomy", "extraction"]'
        WHEN malformed_anatomy = 1 THEN '["anatomy"]'
        WHEN poor_extraction = 1 THEN '["extraction"]'
        ELSE '[]'
       END;
-- Legacy label scopes were emitted in descending labeler score, so the first
-- entry was the primary — the same rule the manifest migration applies.
UPDATE curation_labels
   SET primary_scope = json_extract(scopes_json, '$[0]'),
       secondary_scopes_json = COALESCE(
           (SELECT json_group_array(value) FROM json_each(scopes_json) WHERE key > 0),
           '[]'
       )
 WHERE scopes_json IS NOT NULL;
