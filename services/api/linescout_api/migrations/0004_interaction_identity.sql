-- 0004_interaction_identity.sql
-- Idempotent interaction events + durable pin state.
--
-- Safety properties (see docs/contracts/api-contract.md):
--   * Every event row gains a client-generated identity (`event_uuid`) and a
--     hash of the authoritative payload (`payload_hash`). The UUID is UNIQUE,
--     so a retried POST /events can never write a second row, and a UUID
--     reused with a different payload is detectable (409 event_uuid_conflict).
--   * Legacy rows are backfilled with a random UUID and a NULL payload_hash:
--     they were written before clients sent an identity, so they can never be
--     "replayed" — a NULL hash is treated as "no recorded payload" and any
--     reuse of such a UUID is a conflict.
--   * `open` / `trace` contributions coalesce per (session, asset, revision,
--     gallery). That coalescing is now enforced by a partial UNIQUE index, so
--     a repeated click cannot append a row at all — it can neither inflate the
--     decayed weight nor refresh the contribution timestamp. Pre-existing
--     duplicates are collapsed onto the EARLIEST row (the first contribution
--     wins), which is exactly what the aggregator now reads.
--   * `pin` / `unpin` stay append-only: they are a toggle sequence, and an
--     unpin must be able to follow the pin it cancels.
--   * Pins themselves are durable state, not a learning signal: the new
--     `pins` table survives restarts and `reset_affinities`, and is namespaced
--     by gallery kind so fixture pins can never appear in a live session.
--   * No user data is dropped except exact duplicate open/trace rows, which
--     were already coalesced to a single contribution by the aggregator.

ALTER TABLE events ADD COLUMN event_uuid TEXT;
ALTER TABLE events ADD COLUMN payload_hash TEXT;

-- Backfill a random v4-shaped UUID per legacy row (randomblob is evaluated
-- per row, so the values are distinct).
UPDATE events
   SET event_uuid = lower(
        substr(hex(randomblob(4)), 1, 8) || '-' ||
        substr(hex(randomblob(2)), 1, 4) || '-4' ||
        substr(hex(randomblob(2)), 2, 3) || '-' ||
        substr('89ab', 1 + (abs(random()) % 4), 1) ||
        substr(hex(randomblob(2)), 2, 3) || '-' ||
        substr(hex(randomblob(6)), 1, 12))
 WHERE event_uuid IS NULL;

-- Collapse historic open/trace duplicates onto the earliest row so the new
-- coalescing index can be created and the retained timestamp is the first
-- contribution, never a refreshed one.
DELETE FROM events
 WHERE event IN ('open', 'trace')
   AND id NOT IN (
       SELECT MIN(id) FROM events
        WHERE event IN ('open', 'trace')
        GROUP BY session_id, asset_id, event, query_revision, gallery_kind
   );

CREATE UNIQUE INDEX events_event_uuid_idx ON events (event_uuid);

-- One contribution per (session, asset, event, revision, gallery) for the
-- coalescing event kinds. Pin/unpin are deliberately excluded.
CREATE UNIQUE INDEX events_contribution_idx
    ON events (session_id, asset_id, event, query_revision, gallery_kind)
 WHERE event IN ('open', 'trace');

-- Durable pinned references. Pins are application state, not training data:
-- they are kept when learning is disabled, kept across an affinity reset, and
-- namespaced by the gallery they were made against.
CREATE TABLE pins (
    gallery_kind    TEXT NOT NULL CHECK (gallery_kind IN ('fixture', 'live')),
    asset_id        TEXT NOT NULL,
    pinned_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (gallery_kind, asset_id)
);
CREATE INDEX pins_kind_idx ON pins (gallery_kind, pinned_at);

INSERT INTO migration_reports (version, name, notes_json)
VALUES (4, '0004_interaction_identity.sql',
 '{"summary":"event identity (client UUID + payload hash), storage-enforced open/trace coalescing, durable namespaced pins","detail":"legacy event rows keep their data and gain a random event_uuid with a NULL payload_hash (never replayable); duplicate open/trace rows are collapsed onto the earliest row, which is the timestamp the aggregator now uses; pin/unpin stay append-only; the new pins table is empty and survives affinity resets"}');
