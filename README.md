# drawable

A line-art reference engine for character artists.
<!--
The authoritative implementation and research roadmap is [docs/deep-draw.md](docs/deep-draw.md). It distinguishes verified frontend progress from remaining pilot, backend, data, ML, and validation work, and records which local systems are still review-gated.
-->

## Run the frontend

Requirements: Node.js 22 or newer.

```bash
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/draw`.

The frontend does not require a backend. Its fixture service simulates live reference results after each completed stroke. Type `slow`, `error`, or `empty` into the optional reference hint to exercise those states.

Use **Export drawing → drawable project · Editable** to save a lossless `.drawable` project. **Import sketch** accepts `.drawable`, PNG, and self-contained SVG files and opens each imported sketch as an independent drawing in a new tab.

## Checks

```bash
npm run check
npm test
npm run build
npm run test:e2e
```

The artist workspace is at `/draw`. `/benchmark` remains a fixture scaffold. `/setup` is the private source-terms and reference-review handoff; `/curate` is the opt-in local image-labeling workspace. Both require the FastAPI API and remain separate from real search and gallery activation.

To review the downloaded reference images for the local educational prototype, install the workspace dependencies if needed and run:

```bash
npm ci
npm run review
```

The command starts the local API and web app, then opens a pinned Chromium profile directly at `/curate?prototype=1`. It needs Node.js 22+, Python 3.11, and `uv`; the first launch may install locked Python dependencies and the pinned browser. Following the owner's explicit educational-prototype instruction, it reuses the existing `datasets/AnitaDataset` sketches, prepares a small batch across Hero, LMA test, and Pirate, and skips manual source/work approval forms. Review image quality, style, scope, and viewpoint, then keep/reject images or create detail crops. Use **Load another 24 images** to expand the batch. Reviews persist in `datasets/prototype/metadata.sqlite3`; file hashes and source/artist information are retained. Source terms remain recorded as unreviewed, and saved batches identify their educational-prototype basis. Close Chromium to stop both servers. No archive download is needed for this local path.

The optional `npm run review -- --guided` command opens `/setup` for the separate source/model terms workflow and uses the original review database. Its source archive preparation remains resumable. Prototype reviews and snapshots do not claim full research qualification.

To start the curation workspace manually during development, run the API in one terminal and Vite in another:

```bash
CURATION_MODE=1 DRAWABLE_API_MODE=fixture uv run --project services/api --locked drawable-api
npm run dev
```

Then open `http://127.0.0.1:5173/curate`. Keep the API bound to loopback. Curation is disabled by default; only enable it for local review. The current dataset files have not passed archive-provenance, terms, and per-asset local-display review, so the API will not show their image previews or permit crops until those gates are recorded. An empty queue or blocked preview is expected and is not permission to bypass the gate. `/curate?mode=fixture` explicitly opens a non-persistent UI demonstration; it never writes review data.

The review workspace stores local human decisions and manually selected detail crops. Crops inherit source-family split and applicable permissions but start without labels. Person count must be entered or explicitly marked uncertain; counts above eight remain valid, while pose matching separately caps its runtime input at eight. Asset viewpoint labels use a versioned qualitative rubric; pairwise asset-view match records can be stored by the API, but a query-prefix challenge UI and angle-aware retrieval do not exist yet. Saving a private curation snapshot does not activate search, create training bundles, upload data, or start training. The ML-side `drawable-ml curation validate-snapshot` command validates such a local snapshot only.

After reviewing a small batch, select suitable kept items under **Candidate state → Kept**, use **Add to reviewed batch**, then **Create review-only snapshot**. If the batch has already been saved, proceed directly to:

```bash
npm run search:prepare
npm run search:local
```

The first command installs the locked CPU dependencies, downloads the exact pinned MobileCLIP2-S2 and DINOv2-small files, verifies their hashes, and builds full/tight embeddings for the latest saved educational prototype snapshot (1–32 references). Repeat setup reuses an intact matching build. Set `--snapshot reports/private/curation-snapshots/<id>.json` to choose another batch. The CPU extra uses PyTorch's CPU wheel index, separately from the CUDA extra, following the [official uv configuration](https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies).

The second command opens `/draw?mode=real` with the actual reviewed images. Draw a head or figure, finish the stroke, and pause: a persistent, isolated CPU worker searches the saved batch. MobileCLIP is the initial serving branch; DINO vectors and forward timings are prepared for a separate comparison, without claiming that the extra branch improves retrieval. Results remain tentative, with no calibrated percentages or same-angle claim. Text assistance is disabled for this initial image-only path. Search runs offline after setup; the launcher blocks external browser requests. Close the browser window or press Ctrl+C to stop the app, API, and model worker. `npm run dev` still offers the clearly labeled fixture demonstration.

With the app closed, `npm run search:check` runs a headless drawing-to-real-reference smoke check, verifies server shutdown, and saves a private engineering report. It uses a synthetic drawing and does not measure retrieval quality or create human research judgments.

Prototype search uses the owner's educational-prototype instruction without recording unreviewed source/model terms as approved. It preserves image labels, provenance, contained paths, byte checksums, and the separate guided research requirements. This is a small local search experiment, not training or full pilot qualification.

The consent-aware M0 collection recorder is at `/record`. It is browser-local and separate from normal drawing; only explicitly accepted recordings are saved to its research IndexedDB store. Cloud-storage, cloud-processing, training, and public-demo permissions are separate opt-ins and do not cause uploads or training. Accepted saves verify event replay and the final rendered-pixel checksum. Older local records migrate to a review-required form with no inherited cloud permissions.

## Data and API preparation

Python backend and dataset tooling use Python 3.11 and `uv`:

```bash
uv sync --all-packages --frozen
uv run --package drawable-api drawable --help
uv run --package drawable-api drawable preflight --profile local-cpu
uv run --package drawable-api drawable sources audit
uv run --package drawable-api drawable data validate
uv run --package drawable-api drawable data sample --recipe pilot-v1
uv run --package drawable-api drawable data manifest --source anita-dataset --inspect-existing
uv run --package drawable-api drawable readiness report
uv run --package drawable-api drawable db migrate
uv run --package drawable-api pytest services/api/tests
npm run api:contracts:check
```
<!--
The source audit is read-only. On 2026-10-03 it found all downloaded content only as extracted trees: the three official archives and receipts are absent, so provenance remains unverified. Dataset paths, terms snapshots, manifests, receipts, recordings, and future model artifacts stay outside Git. Explicit acquisition commands download only from declared official URLs; manifest cataloging keeps assets quarantined, and the local curation API is disabled unless `CURATION_MODE=1` is set. Real search remains unavailable until approved artifacts, gallery and API integration exist.
-->

`drawable data sample --recipe pilot-v1` writes an immutable candidate-only source-yield sample under the private data root. It will currently report a blocked, empty sample because the official archives/receipts, reviewed rights, and authoritative book mapping are not yet verified. It never approves assets or activates a gallery.

`drawable readiness report` accepts a cumulative `--stage` of `m0`, `m1`, `m2`, or `m3` and does not probe hardware. API and ML tools share one hash-bound evidence ledger. For a prepared gate report, `drawable-ml readiness attest --gate <gate-id> --report <private-report.json> --reviewer-id owner-local --review-note "<review note>" --output reports/readiness-ledger-v1.json` records the explicit owner review; mere file presence never satisfies a gate. `--stage training_preparation` (also accepted as `training`) includes M0–M3 plus model, manifest, cloud-bundle and recipe review, but cannot authorize training. The M0 effort gate requires a measured five-hour checkpoint and enforces a ten-hour annotation stop.
<!-- 
Your local GPU is currently disabled. The default preflight does not probe it or infer a driver failure. The optional [Colab runtime preflight](notebooks/colab-runtime-preflight.ipynb) checks only the manually selected Colab VM and a tiny tensor operation; it does not load a retrieval model, upload data, or train. No training has been run, and there is no training-execution command. If cloud training is separately authorized in a future phase, model artifacts are intended to return to the local application for local inference; Colab is not the live application server. Local CPU inference must be measured until the local GPU is enabled and certified.
-->
<!--
## Training preparation (no training)

ML preparation lives in an independent Python 3.11 project and lockfile, separate from the API's dependencies. These commands validate local evidence and data, or create a manual-transfer bundle only after each rights and cloud-consent gate passes. They never upload anything or start training:

```bash
uv sync --project ml --locked --extra dev
uv run --project ml --locked drawable-ml readiness report --data-root datasets --stage training_preparation
uv run --project ml --locked drawable-ml models verify --data-root datasets
uv run --project ml --locked drawable-ml data validate \
  --data-root datasets --manifest manifests/training-dataset.json
uv run --project ml --locked drawable-ml data package \
  --data-root datasets --manifest manifests/training-dataset.json \
  --output transfers/training-bundle-run001.zip
uv run --project ml --locked drawable-ml train prepare \
  --data-root datasets --manifest manifests/training-dataset.json \
  --bundle transfers/training-bundle-run001.zip \
  --ledger reports/evidence-ledger.json \
  --output reports/training-readiness-run001.json
```

After a future, separately approved manual transfer, `drawable-ml data inspect-bundle --data-root /content --bundle training-bundle-run001.zip --expected-dataset-sha256 HASH_FROM_REPORT --expected-recipe-sha256 HASH_FROM_REPORT --expected-bundle-sha256 HASH_FROM_REPORT` checks that the received bytes match the local review report and that the archive contains no protected splits or unsafe/unreferenced members. Replace each placeholder with its corresponding hash from the local readiness/package report.

At the current repository state these checks are expected to report blockers: the official dataset archives and rights approvals are not verified, model artifacts/terms are pending, and M0/M1/pilot/full-data evidence is incomplete. The report is hash-bound and explicitly states that authorization is still false. A person must separately review any data transfer; no automatic Google Drive or Colab upload is provided. The free Colab runtime is optional and manually selected; it may disconnect or expire, so it is not a background training service. Do not leave a run unattended or assume the VM or its local files persist after disconnection.

The package has loss/model/checkpoint building blocks, but no optimizer, backward-pass, training, resume, calibration, or weight-update entry point. `drawable-ml train prepare` is a non-executing readiness report, not a training command. The separate educational prototype search path has loaded the exact frozen CPU weights and measured forward passes; no backward pass, training, or formal research qualification is claimed.
-->