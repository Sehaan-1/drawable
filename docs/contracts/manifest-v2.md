# Manifest schema v2 — field/invariant matrix

Source of truth: `ml/linescout_ml/manifest.py` (schema
`packages/contracts/manifest.schema.json`). A manifest is
`{ schema_version: 2, dataset_version, records: [...] }`. Every record is
validated field-by-field; the matrix below lists type, default, and the
invariants enforced at load time.

Identity and location fields are **never invented**: unknown means `null`
(or the literal `unknown` label where the field is an enum), never a
synthesised id.

## Record fields

### Identity & provenance

| Field | Type | Default | Invariants |
|---|---|---|---|
| `asset_id` | `ls_*` string | required | unique across records; stable id of the line-art asset |
| `source_dataset` · `source_item_id` | str | required | where the item came from; item id unique within `source_dataset` |
| `source_work_id` | str | required | the artistic work the asset depicts; de-facto leakage group when `leakage_group_id` is null |
| `parent_asset_id` | id \| null | `null` | must reference another record in the same manifest; the parent chain must be acyclic; never self |
| `artist_id` | id \| null | `null` | unknown stays null; never fabricated |
| `leakage_group_id` | id \| null | `null` | unknown stays null; null → `source_work_id` is the de-facto group |
| `source_url` | url \| null | `null` | optional public URL of the source |

### Permissions & allowed uses (independent axes)

| Field | Type | Default | Invariants |
|---|---|---|---|
| `permissions.license_id` | str | required | carried verbatim; may coexist with `basis: unknown` |
| `permissions.basis` | `first_party` \| `public_domain` \| `license_terms` \| `explicit_consent` \| `unknown` | `unknown` | any granted use requires `basis != unknown` |
| `permissions.permission_url` | url \| null | `null` | evidence link (license page, consent record) |
| `permissions.attribution` | str \| null | `null` | credit line |
| `permissions.attribution_required` | bool | `false` | true → `attribution` must be non-null |
| `allowed_uses.display` | bool | **`false`** | show in the reference gallery |
| `allowed_uses.training` | bool | **`false`** | may enter the training split |
| `allowed_uses.trace` | bool | **`false`** | may be placed on the trace layer; `trace → display` |

Unknown permission grants nothing: `basis: unknown` with any `true` in
`allowed_uses` is a hard validation error, not a warning.

### SFW (screening ≠ approval)

| Field | Type | Default | Invariants |
|---|---|---|---|
| `sfw_screening` | `{verdict: safe\|unsafe\|unsure, confidence?: 0..1, method: none\|source_rating\|opennsfw2\|source_rating+opennsfw2}` \| null | `null` | automated only; `null` = never screened; `unsure` is *not* a pass |
| `sfw_human` | `{safe: bool, reviewer: str, decided_at?: timestamp}` \| null | `null` | the only thing that can gate display; `null` = no human decision yet |

Ingestion mapping: screening `safe` → `unreviewed` (awaiting a human);
`unsafe` or `unsure` → `quarantined`. Manifests produced by *manual* sources
(`source_rating` provenance recorded by a human) may carry `sfw_human` with
reviewer `ingestion:<pipeline_version>` and `sfw_screening: null` — the
screening-vs-approval distinction is never blurred.

### Art & extraction

| Field | Type | Default | Invariants |
|---|---|---|---|
| `original_path` · `line_art_path` · `thumbnail_path` | relative path | required | files must exist under the manifest directory |
| `origin` | `native_line_art` \| `extracted_line_art` | required | |
| `extraction_model` · `extraction_version` | str \| null | `null` | required when `origin = extracted_line_art` |
| `width` · `height` | int > 0 | required | pixel size of the line art |
| `crop` | `{x, y, width, height}` \| null | `null` | must lie inside `width × height` |
| `text_coverage` · `ink_coverage` · `quality_score` | 0..1 | required | coverage heuristics |
| `phash` | 16-hex | required | perceptual hash |
| `checksums` (`source`, `line_art`, `thumbnail`) | sha256 (64 hex) | required | content integrity; drive artifact invalidation |

### Taxonomy

| Field | Type | Default | Invariants |
|---|---|---|---|
| `primary_style` | style enum | required | exactly one |
| `primary_scope` | scope enum | required | **exactly one**; `unknown` marks the asset provisional |
| `secondary_scopes` | scope[] | `[]` | ⊆ gallery scopes (no `unknown`); no duplicates; ≠ `primary_scope` |
| `person_count` | int 0..50 \| null | `null` | null = not assessed; `multi_character` ∈ scopes → ≥ 2 |
| `person_count_approximate` | bool | `false` | `true` → `person_count` must be non-null |

### Review state

| Field | Type | Default | Invariants |
|---|---|---|---|
| `review.state` | `unreviewed` \| `accepted` \| `rejected` \| `quarantined` | `unreviewed` | quarantine = reversible hold; rejection = terminal |
| `review.quality` | 1..3 \| null | `null` | set → `state != unreviewed` |
| `review.blockers` | (`anatomy` \| `extraction`)[] | `[]` | named, use-blocking defects; unique |
| `review.note` | str ≤ 500 \| null | `null` | free text |

### Learning & gallery membership (independent axes)

| Field | Type | Default | Invariants |
|---|---|---|---|
| `learning_split` | `train` \| `validation` \| `test` \| `none` | required | *learning* assignment only; `none` = not for training |
| `gallery_member` | bool | `true` | *gallery* membership; independent of the split |
| `gold_member` | bool | `false` | requires `accepted` ∧ `quality != null` ∧ no blockers ∧ primary scope known |

Split policy: 70/15/15 train/validation/test over trainable assets, enforced
per source work **and** per leakage group (both may hold a single split only —
no work or group straddles train and test). `learning_split_report` reports
fractions with an `unassigned` bucket; a work that mixes splits is an error.

### Provenance & versioning

| Field | Type | Default | Invariants |
|---|---|---|---|
| `pipeline_version` | str | required | producing pipeline |
| `processing_revision` | int ≥ 1 | `1` | bump when the same pipeline re-processes the asset |
| `label_version` | str | `"1"` | curation label format the metadata reflects |
| `label_version`/`processing_revision` changes | | | **invalidate derived artifacts** (embeddings, thumbnails, index entries) — index keys include them; stale entries are re-embedded, not trusted |

## Derived predicates (frozen)

- **`is_servable`** = `gallery_member` ∧ `allowed_uses.display` ∧
  `review.state = accepted` ∧ `review.quality ≥ 2` ∧ no blockers ∧
  `sfw_human.safe` — the quality floor preserves the v1 rule that a keep with
  quality 1 does not serve.
- **`is_trainable`** = `allowed_uses.training` ∧ `learning_split = train` ∧
  accepted ∧ no blockers ∧ (no screening or `verdict != unsafe`).

The API mirrors `is_servable` in SQLite (`assets.enabled`, recomputed inside
the label-write transaction) and enforces it with a `CHECK` constraint.
