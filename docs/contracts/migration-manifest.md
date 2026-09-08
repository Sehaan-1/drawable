# Manifest migration v1/v2 → v3

Tool: `linescout-manifest convert <path> --from {1,2} --to 3 --out <v3-path>
[--report <path>] [--pipeline-version V --label-version V
[--processing-revision N]]` (`ml/linescout_ml/migrate.py`). The SQL side
(`services/api/linescout_api/migrations/0003_eligibility_v3.sql`) mirrors the
same rules for an existing local database.

`migrate-v1` remains as a compatibility alias for `convert --from 1 --to 3`.

## Safety properties (asserted, not promised)

The conversion report carries safety counters. They are structurally `0` —
the mapping contains no code path that could set them — and they are recorded
on every run so an audit never has to trust the code, only the report:

| Counter | Guarantee |
|---|---|
| `uses_granted` | no `allowed_uses` flag became `true` |
| `human_approvals_fabricated` | no `sfw_human` decision was invented |
| `gold_members_created` | no `gold_member` became `true` |

Consequence: **a migrated gallery is unservable until explicit permission and
an explicit human SFW decision are recorded per asset.** That is the point.
`serving_eligibility_lost` counts previously-`enabled` records that lost
serving for exactly those reasons, so the operator knows the work queue size.

Additional v3 properties:

| Property | Guarantee |
|---|---|
| no silent promotion | `gold_member` is carried from v2 only where the v3 gold conditions hold (accepted, quality 2–3, no blockers, known scope, human SFW approval); otherwise it is conservatively downgraded and counted in `gold_disabled` |
| ambiguous generations fail closed | when v2 records disagree about their `pipeline_version` / `label_version` / `processing_revision` (or no explicit contract is supplied), the v3 manifest is written with `artifact_contract: null`; under `null` every derivative is unverified, so **nothing** is servable or trainable — `records_disabled_stale` and `notes` say exactly why |
| audits and bytes preserved | curation labels, events, and source bytes on disk are never touched by conversion |

`contract_detected` / `contract_source` (explicit | record_consensus |
ambiguous) and `artifact_contract` in the report describe what the converter
decided and how.

## v1 → v3 mapping

v1 is first mapped to the frozen v2 shape under the v1→v2 rules, then carried
to v3:

- `permissions.basis` = **`unknown`**; `allowed_uses` all **`false`**; v1
  `enabled` becomes nothing.
- `sfw_human` is created only from a v1 `sfw.method == "manual"` decision
  (reviewer `migrate-v1`); automated methods become `sfw_screening` only.
- `gold_member` = **`false`** for every record.
- `parent_asset_id`, `artist_id`, `leakage_group_id` = **`null`**.
- `learning_split`: `gallery_only` → `none`; others verbatim.
- `processing_revision` = `1`; `label_version` = `"1"`; `pipeline_version`
  verbatim.
- `primary_scope` = v1 `scopes[0]` (frozen descending-score ordering rule).

Because every v1 record carries the same generation values, a v1 gallery
usually converts with a detected contract — but the permission and human-SFW
gates still keep every record disabled.

## v2 → v3 mapping

A pure materialisation plus conservative checks:

- `schema_version` → `3`; the record shape is carried field-by-field
  (generation fields verbatim).
- `artifact_contract` resolution order: explicit CLI flags > record consensus
  > `null` (ambiguous). Consensus means *every* record carries the exact same
  `(pipeline_version, label_version, processing_revision)`.
- Records from other generations are kept for audit and disabled.
- `gold_member`: kept only when the v3 gold conditions hold; otherwise
  downgraded to `false` and counted.
- Split/parent integrity and duplicate ids are validated in the v3 shape; a
  contradiction that conversion cannot conservatively resolve is a hard
  `ConversionError` (fix the source manifest by hand).

## SQL migration (existing local databases)

`0003_eligibility_v3.sql`:

- **`assets`, `asset_scopes`, `gallery_versions` are dropped and rebuilt**
  (caches of the manifest, never the source of truth). Every rebuilt row
  starts `enabled = 0` and `derivatives_current = 0`;
  `derivative_problems_json` records why per row. **No legacy row is
  promoted.**
- `events`, `preferences`, `curation_labels`, `search_log`, `snapshots` are
  preserved verbatim. Source bytes on disk are never touched by a migration.
- `assets` gains `derivatives_current`, `derivative_problems_json`, and the
  `enabled = 1` CHECK now requires the **full** canonical policy, including
  `permission_basis != 'unknown'` and `derivatives_current = 1`.
- The `gold_member = 1` CHECK now requires accepted ∧ quality 2–3 ∧ no
  blockers ∧ known primary scope ∧ human SFW approval ∧ current derivatives.
- `gallery_versions` records the loaded `artifact_contract` and
  `derivative_problem_count`; `migration_reports` records what the migration
  did (report, never policy).

`0004_interaction_identity.sql` (database schema version `4`):

- `events` gains `event_uuid` (unique) and `payload_hash`. Existing rows get a
  random uuid and a `NULL` hash, so a legacy uuid can never satisfy a replay —
  reusing one is reported as a conflict rather than silently accepted.
- Duplicate `open` / `trace` contributions for the same
  `(session_id, asset_id, event, query_revision, gallery_kind)` are collapsed
  to `MIN(id)` — the **earliest** row, so migrating cannot move a contribution
  forward in time — and a partial unique index keeps them collapsed. `pin` /
  `unpin` rows stay append-only.
- New `pins(gallery_kind, asset_id, pinned_at)` table, primary key
  `(gallery_kind, asset_id)`, `gallery_kind IN ('live','fixture')`. Pins are
  durable state: no migration, affinity reset, or learning toggle clears them.
- `preferences`, `curation_labels`, `search_log`, `snapshots` are untouched;
  `migration_reports` records the dedupe counts.

`tests/test_migration_safety.py` exercises this: a populated v1 database
migrates through every step, loses its assets cache, keeps user data, and the
CHECKs reject every no-gain violation.

## Compatibility failures (by design)

| Situation | Result |
|---|---|
| v1/v2 manifest + v3 API | app unready; the readiness error names `linescout-manifest convert --from N --to 3`; the manifest is never partially interpreted |
| v2 records disagree on generation, no explicit contract | `artifact_contract: null`; every record disabled; `notes` explain |
| v2 `gold_member: true` failing v3 gold conditions | downgraded to `false`; counted in `gold_disabled` |
| v1 `enabled: true`, unknown permission | migrated record **not servable**; counted in `serving_eligibility_lost` |
| v1 automated `sfw.safe: true` | screening only; **no human approval**; asset stays unreviewed, unservable until a human decides |
| stale/corrupt derivative bytes | record disabled and reported (`derivatives_current = 0`); never served, never silently ignored |

## Runbook

```bash
# 1. Convert the manifest (report is written for the audit trail)
ml/.venv/bin/linescout-manifest convert data/galleries/v2.json \
  --from 2 --to 3 --out data/galleries/v3.json \
  --report data/galleries/v3.report.json

# ...or migrate a v1 gallery straight to v3
ml/.venv/bin/linescout-manifest migrate-v1 data/galleries/v1.json \
  --out data/galleries/v3.json --report data/galleries/v3.report.json

# 2. Point the API at the v3 manifest and restart; the SQL migration runs on
#    boot (it is also exercised by `0003` on an existing local database).
LINESCOUT_GALLERY_MANIFEST=data/galleries/v3.json npm run dev:api

# 3. Work the queue: record permission + human SFW per asset; assets become
#    servable only as both are recorded (watch /curation/progress).
```
