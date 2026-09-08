# drawable · LineScout

A local, line-art reference copilot for character artists: draw on a
pressure-sensitive canvas and matching references appear after every stroke.
Everything runs on your machine — no accounts, no cloud, no paid services.

> **Status: Milestone 1 (project foundation).** The canvas is fully usable and
> the API serves the complete v1 wire contract, but rankings come from a
> deterministic fixture over a synthetic gallery until the retrieval models
> land in Milestone 4. See [Roadmap](#roadmap).

## Quick start

Requirements: **Node.js 22+**, **Python 3.11**, and [`uv`](https://docs.astral.sh/uv/)
(`pip install uv` works). Linux and current Chrome/Edge are the primary targets.
No GPU is required for Milestone 1.

```bash
npm install          # frontend + generated contracts
npm run setup:py     # creates ml/.venv and services/api/.venv with uv
npm run dev:all      # API on :8000 and the web app on :5173, together
```

Open <http://127.0.0.1:5173/draw>. The reference panel badge shows which
backend you are on:

| Badge | Meaning |
|---|---|
| **API fixture** | Live FastAPI worker, deterministic ranker over the synthetic gallery (default) |
| **GPU** / **CPU fallback** | Live worker with real models (Milestone 4+) |
| **Fixture** | No API reachable; the frontend simulates results so drawing still works |

Run the two halves separately if you prefer:

```bash
npm run dev:api      # serves the synthetic gallery by default
npm run dev          # Vite; proxies /api → http://127.0.0.1:8000
```

Configuration is optional and documented in [`.env.example`](.env.example).
Nothing needs a secret, an absolute path, or a paid service.

Use **Export drawing → drawable project · Editable** to save a lossless `.drawable` project. **Import sketch** accepts `.drawable`, PNG, and self-contained SVG files and opens each imported sketch as an independent drawing in a new tab.

## Checks

```bash
npm run check        # TypeScript (contracts + web)
npm test             # Vitest
npm run build        # production bundle
npm run check:py     # ruff + mypy --strict + pytest for ml and services/api
npm run smoke        # end-to-end contract check against a running API
npm run test:e2e     # Playwright (needs `npx playwright install chromium`)
npm run contracts    # regenerate packages/contracts from the API (CI fails if stale)
```

CI runs all of the above, including Playwright, plus a synthetic-data smoke test
that boots the API against the committed fixture gallery. Python installs are
pinned by the committed ``uv.lock`` files under ``ml/`` and ``services/api/``.

## Repository layout

```
apps/web/            React 19 + Vite canvas, reference panel, curation & benchmark shells
services/api/        FastAPI worker: health, search, events, preferences, assets, curation (gated)
ml/                  Taxonomy, manifest schema + validator, fixture generator, ingestion pipeline
ml/colab/            Google Colab notebook that runs that pipeline on a free GPU runtime
packages/contracts/  TypeScript types generated from the API's OpenAPI document
scripts/             setup, checks, contract export, smoke test, dev runner
data/                Local datasets, indexes, models, SQLite — never committed
```

### Contracts (schema v2)

The dataset manifest, API wire, and their v1→v2 migration are frozen and
documented in [`docs/contracts/`](docs/contracts/):

- [manifest-v2.md](docs/contracts/manifest-v2.md) — field/invariant matrix (scopes, splits, permissions, SFW, review, identity)
- [api-contract.md](docs/contracts/api-contract.md) — search/events/curation wire, degradations, structured errors, pins
- [migration-v1-to-v2.md](docs/contracts/migration-v1-to-v2.md) — full field mapping; no old asset gains permission or human approval

### API surface (`/api/v1`)

| Endpoint | Milestone 1 |
|---|---|
| `GET /health` | readiness, CUDA/GPU, model + dataset/index versions, gallery size, warnings |
| `POST /search` | full multipart contract with structured `400/413/422`; unready → `503`; blank input → `200 mode=insufficient` |
| `POST /events` · `GET/PUT /preferences` | interaction logging, Laplace-smoothed 30-day-half-life style affinity |
| `GET /assets/{id}/thumbnail` · `/line-art` | enabled + SFW assets only; missing files auto-disable the asset |
| `/curation/*` | mounted only with `LINESCOUT_CURATION_MODE=1`; progress works, the rest is 501 until Milestone 2 |

Interactive docs: <http://127.0.0.1:8000/api/v1/docs>.

## Datasets

No third-party dataset is redistributed with this repository. `ml/fixtures/synthetic/`
is a generated, license-free stand-in used by tests and the default dev setup.
Real sources (Quick, Draw!, Amateur Drawings, Human-Art, Manga109, eBDtheque,
Safebooru, Smithsonian/Met Open Access) are downloaded into `data/` in
Milestone 2 under their own terms — several require an access application,
so start those early.

Ingestion runs in Google Colab on a free T4/A100 runtime, so no GPU is needed at
home: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sehaan-1/drawable/blob/main/ml/colab/linescout_gpu_pipeline.ipynb)
It reads a Drive folder of raw artwork and writes a validated gallery
(`originals/`, `line_art/`, `thumbnails/`, `manifest.json`) plus MobileCLIP2 and
DINOv2 feature shards, which you unzip into `data/` and point the API at. Every
stage is resumable, labels are written as provisional for the curation UI to
correct, and the notebook refuses to run with a placeholder dataset licence.
Details in [`ml/colab/README.md`](ml/colab/README.md).

## Roadmap

| # | Milestone | Status |
|---|---|---|
| 1 | Reproducible project foundation | **This branch** |
| 2 | Dataset and curation pipeline | next |
| 3 | Headless Clippy baseline | |
| 4 | LineScout retrieval models | |
| 5 | Polished canvas and live search | canvas ~70% done |
| 6 | Preference learning and calibrated ranking | event/preference plumbing done |
| 7 | Locked evaluation and demonstration release | |

## License

All rights reserved. See [`LICENSE`](LICENSE).
