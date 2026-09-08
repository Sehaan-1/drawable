# Manifest schema v3 — field/invariant matrix

Source of truth: `ml/linescout_ml/manifest.py` (schema
`packages/contracts/manifest.schema.json`). A v3 manifest is
`{ schema_version: 3, dataset_version, artifact_contract, records: [...] }`.
Every record is validated field-by-field; the matrix below lists type,
default, and the invariants enforced at load time.

Identity and location fields are **never invented**: unknown means `null`
(or the literal `unknown` label where the field is an enum), never a
synthesised id.

## Manifest envelope

| Field | Type | Default | Invariants |
|---|---|---|---|
| `schema_version` | `3` | `3` | only v3 manifests load into the API |
| `dataset_version` | `YYYY.MM.DD[-suffix]` | required | calendar version |
| `artifact_contract` | `{pipeline_version, label_version, processing_revision}` \| `null` | required (nullable) | the *current* artifact generation. `null` = generation unknown → every record's derivatives are unverified → nothing serves or trains |
| `records` | record[] | required | unique `asset_id`; split integrity; parent integrity; gold membership must satisfy the v3 gold conditions against `artifact_contract` |

`artifact_contract` is the single declaration of what "current" means. A
record whose own generation fields (`pipeline_version`, `label_version`,
`processing_revision`) differ from it is **kept for audit but disabled** until
re-processed — it is never silently promoted, and it never leaks into search.

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
may carry `sfw_human` with reviewer `ingestion:<pipeline_version>` and
`sfw_screening: null` — the screening-vs-approval distinction is never blurred.

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
| `gold_member` | bool | `false` | requires accepted ∧ quality 2–3 ∧ no blockers ∧ known primary scope ∧ human SFW approval ∧ current derivatives (v3) |

### Generation & derivative currency

| Field | Type | Default | Invariants |
|---|---|---|---|
| `pipeline_version` | str | required | producing pipeline |
| `processing_revision` | int ≥ 1 | `1` | bump when the same pipeline re-processes the asset |
| `label_version` | str | `"1"` | curation label format the metadata reflects |
| mismatch vs `artifact_contract` | | | record is **stale**: kept for audit, disabled until re-processed |

## Derived predicates (canonical, frozen)

- **`is_servable(record, contract)`** =
  `gallery_member` ∧ `allowed_uses.display` ∧
  `permissions.basis != unknown` ∧ `review.state = accepted` ∧
  `review.quality ∈ {2, 3}` ∧ no blockers ∧ `sfw_human.safe` ∧
  derivatives current under `contract`.
  A `null` contract fails closed: nothing is verified current, so nothing serves.
- **`is_trainable(record, contract)`** =
  `allowed_uses.training` ∧ `permissions.basis != unknown` ∧
  `learning_split = train` ∧ accepted ∧ no blockers ∧
  screening not `unsafe` ∧ derivatives current.
- **`is_gold(record, contract)`** = accepted ∧ quality 2–3 ∧ no blockers ∧
  primary scope known ∧ `sfw_human.safe` ∧ derivatives current. (Stricter
  than v2: gold labels are evaluated against the current generation's bytes.)

`serving_reasons`, `training_reasons`, `gold_reasons`, and
`derivative_reasons` return stable machine-readable reasons for every
violation; the predicates are exactly "the reason list is empty".

## Split integrity (leakage checks)

The one-group-one-split rule is enforced over **every identity axis**;
two *different assigned splits* (train/validation/test) may never share:

1. `source_work_id` — the artistic work (the de-facto leakage group);
2. a non-null `leakage_group_id` — finest declared identity;
3. a non-null `artist_id` — one artist's works never straddle splits;
4. one parent chain — a derivative and its ancestors depict the same art.
   Every record is keyed by its chain's root (a root keys by its own id), so
   a derivative whose `source_work_id` disagrees with its ancestor is still
   caught.

`none` never conflicts, and the split targets remain 70/15/15
(`learning_split_report`). The validator refuses a manifest that violates any
axis, and the API refuses to load one.
