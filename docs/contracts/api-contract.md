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

Response (`schema_version: 2`):

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `2` | bump on any breaking response change |
| `revision` | int | echo |
| `mode` | `insufficient` \| `provisional` \| `confident` | blank/early input → `200` with `mode=insufficient`, never an error |
| `scope_predictions` | `[{label, confidence}]` | early-scope reading; `unknown` label = no confident scope yet |
| `groups` | `[{kind, id?, title, style?, scope?, results}]` | `best_match`, `style`, or `provisional_scope` |
| `results[i]` | see below | per-result fields include `trace_allowed` (stored permission, never derived from origin) |
| `timing` | `{preprocessing_ms, embedding_ms, retrieval_ms, reranking_ms, total_ms}` | |
| `degradations` | `[{kind, detail}]` | **canonical** quality view; `[]` = full quality. kinds: `fixture_mode`, `cpu_fallback`, `branch_disabled`, `gallery_empty` |
| `warning` | str \| null | `"; ".join(detail)` of `degradations`; kept for older clients, may be removed later |
| `dataset_version` / `index_version` | str \| null | provenance of the results |

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
| 413 | `payload_too_large` | `{max_bytes, received_bytes}` |
| 422 | `validation_error` | `{errors: [{field, message, type}]}` |
| 503 | `not_ready` | readiness warnings, incl. the v1-manifest case naming `linescout-manifest migrate-v1` |
| 404 | `queue_empty` / `gallery_unavailable` / `asset_not_found` | — |

`retryable` is true for 429/503. A v1 manifest loaded against a v2 API makes
the app permanently unready (503) with the migrate command in the message —
it is never partially interpreted.

## Events & preferences (gallery namespacing)

`POST /api/v1/events` records `pin` / `open` / `trace` interactions. The
server stamps every event with `gallery_kind`: `"fixture"` iff the API runs in
fixture mode, else `"live"` — clients cannot assert it. Legacy (v1-era) rows
default to `live`.

- `GET/PUT /api/v1/preferences` computes style affinities from **live** events
  only: fixture-mode interactions never leak into the learned profile
  (Laplace smoothing, 30-day half-life; explicit row order wins, learned
  affinities reorder the rest).
- **Pins are not events.** Pinning is durable *local application state*,
  stored client-side under `drawable-pins:<kind>:<version>`
  (`kind ∈ {fixture, live}`, currently version `1`). Switching galleries
  (`setGallery(kind)`) reloads the pin set for that namespace; the two never
  mix. The pre-v2 un-namespaced key migrates into the **fixture** namespace
  exactly once.

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

## Curation (`curation_mode` only)

| Endpoint | Behaviour |
|---|---|
| `GET /curation/next?style=&scope=` | one `unreviewed` candidate (v2 wire: primary/secondary scopes, `blockers`, `sfw_screening`, `sfw_human`, `permissions`, `allowed_uses`, `learning_split`, identity ids); 404 `queue_empty` when drained. The queue applies **no SFW filter** — unsafe/unsure assets were quarantined at ingestion; human SFW approval flows through this queue |
| `GET /curation/progress` | reviewed/accepted/rejected/remaining + per-style/per-scope breakdowns |
| `POST /curation/labels` | see below; `201` returns `{blockers, sfw_human_approved, enabled, serving_blockers}` — `serving_blockers` is empty **iff** `enabled` |
| `POST /curation/snapshots` | see below |

`POST /curation/labels` body: `asset_id`, `expected_review_state` (optimistic
concurrency; mismatch → 409 `review_conflict`), `decision`
(`keep` \| `reject`), optional `primary_style`, `primary_scope`,
`secondary_scopes`, `crop`, `sfw_safe` (tri-state human assertion), `quality`,
`note`, `reviewer`; `blockers: ["anatomy" | "extraction"]`.

Validation (422s): duplicates in scopes/blockers; secondary scopes outside the
gallery set or equal to the primary; `keep` without `quality`; **`keep` with
blockers** (blockers force reject/quarantine); `keep` accepting an `unknown`
primary scope (`primary_scope_required`); labelling a screened-unsafe asset
(`asset_not_sfw` — quarantine happens at ingestion, not here); crop outside
the image (`crop_out_of_bounds`).

Side effects, in one transaction: append-only `curation_labels` audit row;
mirror decision + `COALESCE`-style style/scope/crop and human-SFW columns onto
`assets`; recompute `enabled` and `serving_blockers`.

## Snapshots

`POST /curation/snapshots` exports a **full, immutable snapshot of the latest
label per asset** — keeps *and* rejects. It is never an incremental export and
never claims to be: the payload is self-contained
(`schema_version: 2`, every asset's current decision). A `snapshots` table row
records `snapshot_id`, label count, style breakdown, and
`previous_snapshot_id` lineage; `curation_labels.snapshot_id` remains
audit-only bookkeeping. Relabelling an asset after an export leaves earlier
snapshot files untouched.
