# LineScout contract documentation (schema v2)

This directory is the **frozen, human-readable statement of the v2 dataset and
API contract**. The machine-readable sources of truth are:

| Artifact | Location | Role |
|---|---|---|
| Manifest schema + invariants | `ml/linescout_ml/manifest.py` | Pydantic model; validation *is* the spec |
| Taxonomy (styles, scopes, splits) | `ml/linescout_ml/taxonomy.py` | Enum definitions shared by ml + API |
| API wire schemas | `services/api/linescout_api/schemas.py`, `routers/curation.py` | Pydantic request/response models |
| Generated TypeScript contracts | `packages/contracts/` (regenerate: `npm run contracts`) | Frontend view of the wire |
| SQLite schema | `services/api/linescout_api/migrations/` | Local serving cache; CHECK constraints mirror the invariants |
| v1→v2 manifest migration | `ml/linescout_ml/migrate.py` (`linescout-manifest migrate-v1`) | Deny-by-default field mapping |

Documents:

- **[manifest-v2.md](manifest-v2.md)** — the manifest field/invariant matrix:
  every field, its type, default, and the invariants that make the dataset
  safe to serve and to learn from.
- **[api-contract.md](api-contract.md)** — the HTTP wire: identifiers, version
  fields, search request/response, degradations, structured errors, event and
  preference namespacing, the curation wire, snapshots, and client-side pins.
- **[migration-v1-to-v2.md](migration-v1-to-v2.md)** — the complete v1→v2
  mapping: every old field, every default, and the two safety properties that
  the migration structurally cannot violate.

## The two safety properties

Everything else in the v2 contract exists to make these auditable:

1. **No asset gains permission through migration or default.** `allowed_uses`
   (`display`, `training`, `trace`) all default to `false`; granting any use
   requires a permission basis that is not `unknown`; `trace` requires
   `display`. Nothing in the migration, the SQLite layer, or the API derives a
   use grant from anything but an explicit decision.
2. **No asset gains human approval through migration or default.** Serving
   requires `sfw_human.safe == true` — a *human* decision. Automated screening
   (`sfw_screening`) is never a substitute: it can only gate *out*
   (unsafe/unsure → quarantined at ingestion), never in.

Both properties are enforced three times over: in the Pydantic manifest
validators, in SQLite `CHECK` constraints (`migrations/0002_contract_v2.sql`),
and recorded by the migration report (`uses_granted`,
`human_approvals_fabricated`, `gold_members_created` — structurally `0` on
 every run).

## Versioning rules

- **Manifest `schema_version`** is `2`. A v1 manifest is rejected with a
  readiness error naming `linescout-manifest migrate-v1`; it is never
  interpreted hopefully.
- **Payload `schema_version`** (search, error envelopes) starts at `2` for
  search responses and stays `1` for the error envelope until it changes.
  Breaking response changes bump it.
- **`dataset_version`** is the manifest's calendar version
  (`YYYY.MM.DD[-suffix]`); **`index_version`** is the manifest content hash
  (64 hex chars; the retrieval index is keyed by it). The API may report the
  first 16 hex chars of `index_version` to clients.
- **Client pin storage** is versioned by key (`drawable-pins:<kind>:<version>`)
  so format changes are additive, never destructive rewrites.
