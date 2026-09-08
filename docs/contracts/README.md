# LineScout contract documentation (schema v3)

This directory is the **frozen, human-readable statement of the v3 dataset and
API contract**. The machine-readable sources of truth are:

| Artifact | Location | Role |
|---|---|---|
| Manifest schema + invariants | `ml/linescout_ml/manifest.py` | Pydantic model; validation *is* the spec |
| Taxonomy (styles, scopes, splits) | `ml/linescout_ml/taxonomy.py` | Enum definitions shared by ml + API |
| API wire schemas | `services/api/linescout_api/schemas.py`, `routers/curation.py` | Pydantic request/response models |
| Generated TypeScript contracts | `packages/contracts/` (regenerate: `npm run contracts`) | Frontend view of the wire |
| SQLite schema | `services/api/linescout_api/migrations/` | Local serving cache; CHECK constraints mirror the invariants |
| Manifest conversions | `ml/linescout_ml/migrate.py` (`linescout-manifest convert`, `migrate-v1` alias) | Deny-by-default, versioned field mapping |

Documents:

- **[manifest-v3.md](manifest-v3.md)** — the **current** manifest
  field/invariant matrix: every field, its type, default, and the invariants
  that make the dataset safe to serve and to learn from (including
  `artifact_contract`, derivative currency, gold conditions, and the four-axis
  split-integrity rule).
- **[api-contract.md](api-contract.md)** — the HTTP wire: identifiers, version
  fields, search request/response, degradations, structured errors, event and
  preference namespacing, the curation wire, snapshots, and client-side pins.
- **[migration-manifest.md](migration-manifest.md)** — the complete v1→v3 and
  v2→v3 mapping: every old field, every default, the artifact-contract
  decision procedure, and the safety properties the migration structurally
  cannot violate.
- **[manifest-v2.md](manifest-v2.md)** and **[migration-v1-to-v2.md](migration-v1-to-v2.md)** —
  the frozen v2 documents, kept for historical audit of the earlier freeze.

## The safety properties

Everything else in the v3 contract exists to make these auditable:

1. **No asset gains permission through migration or default.** `allowed_uses`
   (`display`, `training`, `trace`) all default to `false`; granting any use
   requires a permission basis that is not `unknown`; `trace` requires
   `display`. Nothing in the migration, the SQLite layer, or the API derives a
   use grant from anything but an explicit decision.
2. **No asset gains human approval through migration or default.** Serving
   requires `sfw_human.safe == true` — a *human* decision. Automated screening
   (`sfw_screening`) is never a substitute: it can only gate *out*
   (unsafe/unsure → quarantined at ingestion), never in.
3. **Derivatives must be current and intact.** Serving/search/curation all
   require the record's generation to match the manifest `artifact_contract`
   and its bytes to match the recorded checksums. A stale, missing, or tampered
   derivative is disabled and reported — never served, never silently ignored.
4. **Legacy data migrates conservatively.** `gold_member` is carried only
   where the v3 gold conditions hold; ambiguous generations produce
   `artifact_contract: null` (nothing servable) with the reason recorded.
5. **One-group-one-split, on every identity axis.** Train/validation/test must
   never mix within a work, leakage group, artist, or parent/derivative chain.

These properties are enforced at every layer: Pydantic manifest validators,
the colab pipeline build, the SQLite `CHECK` constraints
(`migrations/0002_contract_v2.sql`, `0003_eligibility_v3.sql`), the gallery
loader/serving routes, and the search ranker (defence in depth). The derived
`assets.enabled` column is a cache of the canonical predicate and is never
hand-edited.

## Versioning rules

- **Manifest `schema_version`** is `3`. A v1/v2 manifest is rejected with a
  readiness error naming `linescout-manifest convert --from N --to 3`; it is
  never interpreted hopefully.
- **Payload `schema_version`** (search, error envelopes) starts at `2` for
  search responses and stays `1` for the error envelope until it changes.
  Breaking response changes bump it.
- **`dataset_version`** is the manifest's calendar version
  (`YYYY.MM.DD[-suffix]`); **`index_version`** is the manifest content hash
  (64 hex chars; the retrieval index is keyed by it). The API may report the
  first 16 hex chars of `index_version` to clients.
- **Client pin storage** is versioned by key (`drawable-pins:<kind>:<version>`)
  so format changes are additive, never destructive rewrites.
