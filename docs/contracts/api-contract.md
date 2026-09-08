# API contract (`/api/v1`) — schema v2

Source of truth: `services/api/linescout_api/` (FastAPI app; OpenAPI export is
generated into `packages/contracts/openapi.json`). Error envelope is identical
on every endpoint.

## Identifiers & version fields

| Field | Shape | Notes |
|---|---|---|
| `asset_id` | `ls_…` UUID-style string | stable asset identity |
| `session_id`, `request_id` | UUID4 | request ids echo `X-Request-Id` when valid |
| `revision` | int ≥ 1 | client search revision; echoed unchanged |
| `dataset_version` | `YYYY.MM.DD[-suffix]` | manifest calendar version; `null` when no gallery is loaded |
| `index_version` | 64-hex (clients may see 16-hex) | manifest content hash; changes whenever the dataset changes |
| `snapshot_id` | `curation_YYYYMMDD_HHMMSS_ffffff[_suffix]` | immutable export identity |

## Search

`POST /api/v1/search` (multipart): `session_id`, `revision` ≥ 1,
`canvas_width`/`canvas_height` (logical 2048), `stroke_count`, `point_count`,
`image` (PNG snapshot, ≤ 4 MiB), optional `strokes` (gzipped stroke sequence,
≤ 2 MiB), optional `text_hint` (≤ 120 chars), optional `selected_style`.

Request integers are bounded before any arithmetic or SQLite storage:
`revision` ≤ 2,000,000,000, `stroke_count` ≤ 1,000,000, `point_count`
≤ 10,000,000. Anything larger — or a non-2048 `canvas_width`/`canvas_height` —
is a structured `422` (`validation_error` / `canvas_dimensions`), never an
uncaught exception or an overflow.

Response (`schema_version: 2`):

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `2` | bump on any breaking response change |
| `request_id` | UUID | echoes the request identity (`X-Request-Id`); matches the error envelope |
| `api_version` | str | server build/release identifier for this response |
| `revision` | int | echo |
| `canvas_width` / `canvas_height` | int | logical canvas the query was validated against (always 2048) |
| `stroke_status` | `present` \| `absent` | whether a vector `strokes` payload accompanied this query |
| `counts_approximate` | bool | exact-verification flag, see below |
| `preprocessing_version` | str | snapshot-preprocessing pipeline identity |
| `mode` | `insufficient` \| `provisional` \| `confident` | too little input → `200` with `mode=insufficient`, never an error; the API has no `empty` mode, clients map `degradations: [blank_raster]` onto it |
| `scope_predictions` | `[{label, confidence}]` | early-scope reading; `unknown` label = no confident scope yet |
| `groups` | `[{kind, id?, title, style?, scope?, results}]` | `best_match`, `style`, or `provisional_scope` |
| `results[i]` | see below | per-result fields include `trace_allowed` (stored permission, never derived from origin) |
| `timing` | `{preprocessing_ms, embedding_ms, retrieval_ms, reranking_ms, total_ms}` | |
| `degradations` | `[{kind, detail}]` | **canonical** quality view; `[]` = full quality. kinds: `fixture_mode`, `cpu_fallback`, `branch_disabled`, `gallery_empty`, `blank_raster`, `vector_absent`, `vector_sparse` |
| `warning` | str \| null | `"; ".join(detail)` of `degradations`; kept for older clients, may be removed later |
| `dataset_version` / `index_version` | str \| null | provenance of the results |

**Sufficiency is a raster verdict; the vector branch is a separate verdict.**
`mode` comes from the ink on the uploaded snapshot, measured on the flattened
512² image (grayscale < 200 counts as ink, `preprocessing_version` 1.1.0):

| Verdict | Rule |
|---|---|
| `blank` | no ink pixels at all (`ink_pixels`/`coverage` are 0) — nothing was drawn, everything was erased, or the import is fully transparent/blank. Transparency is flattened onto white first, so an empty import measures as blank rather than as content. |
| `insufficient` | `blank`, or the ink bounding-box diagonal is below `min_ink_diagonal_ratio` of the snapshot diagonal (a mark too small to read a subject from) |

Stroke/point counts never produce either verdict. They describe the *vector
branch* only — `absent` (no strokes and no points), `sparse` (below
`min_points_for_search` sampled points), or `usable` — and a degraded branch is
disclosed (`vector_absent` / `vector_sparse`) without demoting a drawing that
has real ink. That distinction is the whole point: a PNG or flattened-SVG
import carries no vector geometry at all, and counting it as "too few points"
would make imported artwork unsearchable.

`blank` is reported structurally as `blank_raster` so a client can tell "this
canvas is empty" from "keep drawing" without re-deriving it from counts. On the
insufficient path the input degradations are the raster ones; on the ranked path
they are the vector ones (the query had enough ink to be searched at all).

**Count cross-checks are never silently trusted.** When a `strokes` payload
is present its `canvas_width`/`canvas_height` and stroke array are the ground
truth: a non-2048 vector canvas is a `422 canvas_dimensions` and a
`stroke_count` that does not equal the number of delivered strokes is a `422
stroke_count_mismatch`. `point_count` is *not* structural (the delivered
points may be an exact or resampled view of the ink), so a disagreement is
never an error — it is disclosed: `stroke_status` is `absent` for raster-only
queries, and `counts_approximate` is `true` whenever no vector payload was
delivered or the reported `point_count` differs from the delivered vector
point total. `counts_approximate` is `false` only when a `strokes` payload was
present and every reported count agreed exactly.

`dataset_version`/`index_version`/`preprocessing_version`/`api_version` are
*provenance identifiers*: each is whatever actually produced this response,
and an unavailable one (e.g. no gallery loaded) is an explicit `null` on the
wire, never a fabricated value.

Vector payloads: retrieval branches whose per-(asset, embedder) embedding
status is `missing`, `unsupported`, or `stale` simply do not contribute to
that branch's ranking — vectors are **never zero-filled** (a zero vector would
fake a semantic match). Approximate result counts are reported as exact
integers of what was ranked; there is no "about N" fuzziness on the wire.

## Errors

Every error is `{schema_version, request_id, retryable, error: {code, message, field?, details?}}`.
`details` is structured, machine-readable, and never echoes paths or secrets:

| Status | Code | `details` |
|---|---|---|
| 403 | `invalid_host` | — (loopback Host enforcement) |
| 403 | `cross_origin_mutation_forbidden` | — |
| 413 | `request_too_large` / `too_many_parts` | `request_too_large`: `{max_bytes, received_bytes}`; `too_many_parts`: `{max_parts}` |
| 422 | `validation_error` | `{errors: [{field, message, type}]}` |
| 503 | `not_ready` | `{setup_error, warmup, device_ready, fixture_mode, gallery_loaded, warnings}` — readiness warnings, incl. the v1-manifest case naming `linescout-manifest migrate-v1` |
| 404 | `queue_empty` / `gallery_unavailable` / `asset_not_found` | — |
| 409 | `event_uuid_conflict` | — (`field: event_uuid`; the id was already used for a *different* interaction) |
| 403 | `trace_not_permitted` | — (`field: asset_id`; the source does not permit tracing) |

`retryable` is true for 429/503. A v1 manifest loaded against a v2 API makes
the app permanently unready (503) with the migrate command in the message —
it is never partially interpreted.

## Events & preferences (gallery namespacing)

`POST /api/v1/events` records `open` / `trace` / `pin` / `unpin` interactions.

**Identity and idempotency.** The client generates an `event_uuid` (v4) per
interaction and reuses it verbatim on retries. The column is unique, and the
write is a single atomic statement:

- a replay with the **same** uuid and the same payload returns the original
  row (`id`, `created_at`) with `replayed: true` — nothing is inserted, no
  timestamp is refreshed, no weight is added;
- the same uuid with a **different** payload is a client bug, not a retry:
  `409 event_uuid_conflict` (`field: event_uuid`, `retryable: false`);
- `open` and `trace` additionally coalesce on
  `(session_id, asset_id, event, query_revision, gallery_kind)`, so a
  double-click or a second client-side attempt with a fresh uuid still
  resolves to the first row and its original timestamp. `pin` / `unpin` are
  append-only because their order is what carries meaning;
- rows written before schema version 4 have no payload hash, so their
  backfilled uuids can never match a replay — reusing one is a conflict.

**Server-derived fields.** `gallery_kind` is `"fixture"` iff the API runs in
fixture mode, else `"live"` — clients cannot assert it (legacy rows default to
`live`). `style` is likewise **not** a client input: it is read from the
enabled gallery row for `asset_id` (unknown/disabled asset → `404
asset_not_found`), so a client can neither invent a style nor train a profile
for an asset it cannot see. The payload hash covers session, asset, event,
revision, and gallery kind only — never the derived style.

**Permission.** `event: "trace"` on an asset whose source permissions forbid
tracing is rejected with `403 trace_not_permitted` (`field: asset_id`), after
the eligibility check. Origin (`native_line_art` / `extracted_line_art`) is
*not* consulted: permission is the recorded source metadata, so a native asset
may be untraceable and an extracted one traceable. A forged event cannot buy
the permission the gallery row does not grant.

**Learning toggle.** With `learning_enabled = false` no row is written at all
(the response is `{id: 0, recorded: false, replayed: false}`) and the uuid
stays usable for a later real write. Pin and unpin still change durable pin
state — pins work while learning is off.

- `GET/PUT /api/v1/preferences` computes style affinities from **live** events
  only: fixture-mode interactions never leak into the learned profile
  (Laplace smoothing, 30-day half-life; explicit row order wins, learned
  affinities reorder the rest). Weights are `open 1`, `pin 3`, `trace 4`,
  `unpin 0`. Resetting affinities ignores everything recorded before the
  reset; it does **not** touch pins.
- **Pins are durable state, not a learning signal.** `unpin` contributes zero
  weight — it is recorded so the pin history can be replayed, and it ends the
  active pin run so the asset's pin contribution stops (the earlier `pin`
  keeps its own weight and original timestamp; unpinning is never a negative
  vote).

### Pins

| Endpoint | Behaviour |
|---|---|
| `GET /api/v1/pins` | the whole pin set for the server's gallery kind |
| `PUT /api/v1/pins/{asset_id}` | idempotent; re-pinning keeps the original `pinned_at`; `404 asset_not_found` unless the asset is currently servable |
| `DELETE /api/v1/pins/{asset_id}` | idempotent, always `200` |

All three return the same `PinsResponse{schema_version, gallery_kind, pins,
revoked}`, so a client replaces its view atomically instead of guessing what
changed. Every read **revalidates**: a pin whose asset became ineligible
(disabled, display no longer permitted, missing derivative…) is dropped from
storage and reported in `revoked[]` with its blocker reasons, never returned
as a pin. `PinnedAsset.trace_allowed` carries the current permission; there is
no `trace_url` — a client derives it only when tracing is permitted.

Pins are namespaced by gallery kind and stored separately: **live** pins are
rows in the API database (they survive a page reload *and* an API restart),
**fixture** pins are offline browser state under
`drawable-pins:fixture:<version>` (currently version `1`). Neither namespace
ever reads the other's assets; the pre-v2 un-namespaced key
`drawable-fixture-pins` migrates into the **fixture** namespace exactly once,
dropping entries that fail validation or look like live gallery ids.

## Assets

`GET /api/v1/assets/{id}/thumbnail` and `/line-art` serve **enabled**
(`is_servable`) assets only. Serving re-verifies eligibility, file existence,
and the recorded sha256 at request time:

* a missing file → `404 error.code=asset_unavailable`, and the asset is
  dropped from the session's serving list;
* bytes that no longer match the manifest hash → same `asset_unavailable`
  (tampered content is never served);
* at gallery load, any missing/checksum-mismatched/stale derivative is
  disabled per row (`derivatives_current = 0`), reported in `derivative_problems_json`,
  and counted in the health warning — disable-and-report, never silent.

`GET /api/v1/assets/{id}/permissions` is the authoritative answer to "may this
be used?": `allowed_display`, `allowed_trace`, `permission_basis`,
`attribution`(`_required`), `origin`, and a `trace_url` that is populated
**only** when tracing is permitted. Clients call it when reopening a saved
document, so a trace layer written before a permission was revoked cannot be
resurrected from local storage. `/line-art` itself is not gated on
`allowed_trace` — display permission already governs full-size display, and
trace permission is enforced where tracing is actually claimed (this endpoint,
the `trace_allowed` projections, and `POST /events`).

## Curation (`curation_mode` only)

| Endpoint | Behaviour |
|---|---|
| `GET /curation/next?style=&scope=&session_id=` | one `unreviewed` candidate (v2 wire: primary/secondary scopes, `blockers`, `sfw_screening`, `sfw_human`, `permissions`, `allowed_uses`, `learning_split`, identity ids, `label_version`, `derivative_processing_state`); 404 `queue_empty` when drained. The queue applies **no SFW filter** — unsafe/unsure assets were quarantined at ingestion; human SFW approval flows through `/curation/sfw/{id}/adjudication`. `session_id` scopes the queue: the session cursor and its skip exclusions are applied, so Next after Skip (or after labelling with `session_id`) never re-serves the same asset in that session |
| `GET /curation/candidates/{asset_id}` | stable by-id retrieval of the live candidate row (404 `asset_not_found`) — used for Previous and 409 recovery, never a stale client copy |
| `GET /curation/progress` | reviewed/accepted/rejected/**quarantined**/remaining + per-style/per-scope breakdowns |
| `POST /curation/queue/skip` | body `{session_id, asset_id}`: advance the session cursor past an asset **without** a label. The asset is only excluded for that session (review state, version, eligibility untouched); response `{session_id, cursor_asset_id, excluded_asset_id, remaining}` |
| `POST /curation/labels` | see below; `201` returns `{blockers, sfw_human_approved, enabled, serving_blockers}` — `serving_blockers` is empty **iff** `enabled` |
| `GET /curation/quarantine` | SFW adjudication backlog as `QuarantineCandidate` rows — **metadata only** (quarantined/unsafe/unsure/human-flagged). `thumbnail_url`/`line_art_url` stay `null` until a reveal grant exists; the preview route enforces the grant independently |
| `POST /curation/quarantine/{asset_id}/reveal` | body `{reviewer}`: issue a deliberate, **expiring** reveal grant (recorded in `sfw_reveals`, TTL `curation_reveal_ttl_seconds`). Only the curation preview route honours it; public asset routes never do; the asset itself is unchanged. Reveal is idempotent (extends the grant) |
| `POST /curation/sfw/{asset_id}/adjudication` | body `{safe, expected_label_version, reviewer?, note?}`: explicit human SFW decision, atomic on `expected_label_version`. `safe` mirrors the decision and returns a quarantined record to `unreviewed` (it must still pass every serving gate); `unsafe` quarantines + human-flags it off every serving surface. Append-only audit in `sfw_adjudications`. Held or previously-decided records only (422 `asset_not_sfw_pending`) |
| `POST /curation/assets/{asset_id}/crops` | body `{crop, expected_label_version, reviewer?, note?}`: cut an **immutable child derivative** — own identity (deterministic from dataset+item+crop), parent identity links, bounded geometry, fresh files/hashes, own processing (`pending`) and review (`unreviewed`) state, `derivatives_current = 0`, `enabled = 0`. Nothing is inherited from the parent's review/SFW/gold state. 422s: `crop_out_of_bounds`, `crop_too_small` (min edge), `derivative_stale` (parent artifacts stale), `crop_source_restricted` (held parent — cropping must not launder held content into a fresh reviewable asset), `parent_artifact_invalid` (parent bytes fail checksum verification at cut time; `details.problems` lists each mismatched artifact) |
| `POST /curation/assets/{asset_id}/process` | run (or re-run) a derivative's required processing: verify the child's files against recorded checksums, rebuild every measurement from the child's **own** bytes, regenerate the thumbnail, then mark derivatives current and recompute index membership + `enabled` in one transaction. Failures are recorded (`processing_state = 'failed'`, `processing_error`) and are retryable. Returns measurements, attempts, `derivatives_current`, and the artifact stamp future embeddings must bind to (`embedding_status: "missing"`) |
| `POST /curation/snapshots` | see below |

`POST /curation/labels` body: `asset_id`, `expected_review_state` **and**
`expected_label_version` (optimistic concurrency on the per-asset curation
version; either mismatch → 409 `review_conflict` / `label_version_conflict`
with `details.current_label_version`, `details.current_review_state`,
`details.latest_decision`, `details.latest_reviewer`), `decision`
(`keep` \| `reject`), optional `primary_style`, `primary_scope`,
`secondary_scopes`, `sfw_safe` (tri-state human assertion), `quality`,
`note`, `reviewer`, `session_id` (advances that session's queue cursor);
`blockers: ["anatomy" | "extraction"]`. Crops are **no longer part of the
label** — a crop is an immutable derivative via
`POST /curation/assets/{asset_id}/crops`.

Validation (422s): duplicates in scopes/blockers; secondary scopes outside the
gallery set or equal to the primary; `keep` without `quality`; **`keep` with
blockers** (blockers force reject/quarantine); `keep` accepting an `unknown`
primary scope (`primary_scope_required`); labelling a screened-unsafe asset
(`asset_not_sfw` — quarantine happens at ingestion, not here; adjudicate
instead).

Side effects, in one transaction: append-only `curation_labels` audit row;
mirror decision + style/scope and human-SFW columns onto `assets`; bump
`curation_label_version`; recompute `enabled` and `serving_blockers`;
transactional `asset_scopes` upsert; advance the session cursor when
`session_id` is present.

## Snapshots

`POST /curation/snapshots` exports a **full, immutable snapshot of the latest
label per asset** — keeps *and* rejects (quality-model training needs both
classes). It is never an incremental export and never claims to be: the
payload is self-contained (`schema_version: 2`, every asset's current
decision).

Guarantees:

* **consistent view** — the label read, lineage read, file publication,
  registry insert, and the exact linking of exported `curation_labels` rows to
  the snapshot all run inside one `BEGIN IMMEDIATE` transaction, so a
  concurrent label write or a concurrent export can never interleave;
* **exclusive atomic publication** — the file is created with
  `O_CREAT | O_EXCL` under a lock; two concurrent exports serialize and chain
  via `previous_snapshot_id`, and neither file can overwrite the other;
* **failure recovery** — a failed publish never leaves a registered-but-missing
  or written-but-unregistered snapshot (the transaction rolls back together);
* **no absolute paths** — the registry stores a path **relative to the data
  root**, plus the published bytes' SHA-256, so the export stays verifiable
  after later edits;
* a `snapshots` table row records `snapshot_id`, label count, style
  breakdown, and `previous_snapshot_id` lineage; `curation_labels.snapshot_id`
  remains audit-only bookkeeping.

Relabelling an asset after an export leaves earlier snapshot files untouched.
