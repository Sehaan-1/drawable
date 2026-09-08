# Migration v1 → v2

Tool: `linescout-manifest migrate-v1 <v1-manifest> --out <v2-manifest>
[--report <path>]` (`ml/linescout_ml/migrate.py`). The SQL side
(`services/api/linescout_api/migrations/0002_contract_v2.sql`) mirrors the
same rules for an existing local database.

## Safety properties (asserted, not promised)

The migration report carries three safety counters. They are structurally
`0` — the mapping contains no code path that could set them — and they are
recorded on every run so an audit never has to trust the code, only the
report:

| Counter | Guarantee |
|---|---|
| `uses_granted` | no `allowed_uses` flag became `true` |
| `human_approvals_fabricated` | no `sfw_human` decision was invented |
| `gold_members_created` | no `gold_member` became `true` |

Consequence: **a migrated gallery is unservable until explicit permission and
an explicit human SFW decision are recorded per asset.** That is the point.
`serving_eligibility_lost` counts v1-`enabled` records that lost serving for
exactly those two reasons, so the operator knows the work queue size.

## Field mapping

| v1 field | v2 field | Rule |
|---|---|---|
| `schema_version: 1` | `schema_version: 2` | hard reject the other direction: a v1 manifest against a v2 API → unready/503 naming `linescout-manifest migrate-v1` |
| `asset_id`, `source_dataset`, `source_item_id`, `source_work_id` | same | carried verbatim |
| — | `parent_asset_id`, `artist_id`, `leakage_group_id` | **`null`** (unknown), never invented |
| — | `source_url` | `null` |
| `license_id` (top level) | `permissions.license_id` | verbatim |
| — | `permissions.basis` | **`unknown`** — a licence id alone proves nothing |
| — | `allowed_uses.{display,training,trace}` | **all `false`** — nothing carries over from v1 `enabled`, which mixed serving with review into one flag |
| `scopes: [s0, s1, …]` | `primary_scope` = `scopes[0]`, `secondary_scopes` = rest | ordering rule is frozen; empty v1 scopes → `primary_scope: unknown` |
| `person_count` | `person_count` + `person_count_approximate: false` | count carried; approximate flag is new and defaults exact |
| `sfw: {safe, confidence, method}` (automated method) | `sfw_screening: {verdict, confidence, method}` | `safe=false` → `verdict: unsafe` (quarantined), else `safe` (unreviewed) |
| `sfw: {safe, …, method: "manual"}` | `sfw_human: {safe, reviewer: "migrate-v1", decided_at: null}` **and `sfw_screening: null`** | manual *was* a human decision — the only approval that carries over |
| `review_state` | `review.state` | `accepted`/`rejected`/`unreviewed` verbatim; see quarantine row below |
| `quality` | `review.quality` | verbatim |
| `malformed_anatomy` / `poor_extraction` (booleans) | `review.blockers: ["anatomy" | "extraction"]` | both booleans map to named blockers; an accepted record that carries blockers keeps **both** facts (state and blockers) — the DB CHECK then forces `enabled=0` until it is resolved |
| — | `gold_member` | **`false`** for every migrated record, even accepted ones (gold requires the full v2 accept predicate) |
| `split: train\|validation\|test\|gallery_only` | `learning_split` | `gallery_only` → **`none`** + `gallery_member: true`; the other three verbatim |
| — | `gallery_member` | `true` (v1 membership notion) |
| `enabled` | *(derived)* `is_servable` | not migrated; recomputed from the new fields — and false by default |
| `pipeline_version` | `pipeline_version` | verbatim |
| — | `processing_revision` | `1` |
| — | `label_version` | `"1"` |
| checksums | same | verbatim |
| `origin`, `extraction_*`, paths, `width/height`, `crop`, coverage, `phash`, `quality_score`, `primary_style` | same | verbatim |

`split_mapping` and `blockers_created` in the report count what happened per
old value.

## SQL migration (existing local databases)

`0002_contract_v2.sql` rebuilds the serving cache under the same rules:

- **`assets` is dropped and reloaded** from the manifest. It is a cache, and
  every migrated v2 manifest is deny-by-default, so no old asset keeps (or
  gains) serving state.
- `events` survives verbatim; every legacy row is stamped `gallery_kind =
  'live'` (they came from the real gallery). New fixture-mode events are
  stamped server-side and excluded from affinity learning.
- `curation_labels` (human work) survives verbatim, plus new columns:
  `blockers_json` backfilled from the two v1 booleans, `primary_scope` /
  `secondary_scopes_json` split from legacy `scopes_json` by the same
  ordering rule. Legacy columns stay; they are no longer written.
- New `CHECK` constraints make the no-gain rules structural:
  `enabled = 1` requires the full servable predicate (including
  `COALESCE(sfw_human_safe, 0) = 1` — NULL never satisfies a CHECK);
  any granted use requires `permission_basis != 'unknown'`; `trace →
  display`; `gold_member = 1` only on accepted rows.
- `snapshots` table added for export lineage.

`tests/test_migration_safety.py` exercises exactly this: a populated v1
database migrates, loses its assets cache, keeps events and labels with the
new columns backfilled, and the CHECKs reject every no-gain violation.

## Compatibility failures (by design)

| Situation | Result |
|---|---|
| v1 manifest + v2 API | app unready; `/health` + 503 name `linescout-manifest migrate-v1`; the manifest is never partially interpreted |
| v1 `enabled: true`, unknown permission | migrated record **not servable**; counted in `serving_eligibility_lost` |
| v1 automated `sfw.safe: true` | screening only; **no human approval**; asset stays unreviewed, unservable until a human decides |
| v1 synthetic fixture gallery | regenerated first-party under the v2 generator rather than migrated (it can grant itself permission) |
| Old client expecting `warning` string | still present (join of `degradations` details); `degradations` is canonical |
| Old curation payload (`scopes`, `malformed_anatomy`, …) | 422 `validation_error` — the curation wire is versioned with the app, not negotiated |
| Pre-v2 pin storage (`drawable-fixture-pins`) | migrated once into the fixture namespace; never absorbed into the live namespace |

## Runbook

```bash
# 1. Migrate the manifest (report is written for the audit trail)
ml/.venv/bin/linescout-manifest migrate-v1 data/galleries/v1.json \
  --out data/galleries/v2.json --report data/galleries/v2.report.json

# 2. Point the API at it and restart; the SQL migration runs on boot
LINESCOUT_GALLERY_MANIFEST=data/galleries/v2.json npm run dev:api

# 3. Work the queue: record permission + human SFW per asset; assets
#    become servable only as both are recorded (watch /curation/progress).
```
