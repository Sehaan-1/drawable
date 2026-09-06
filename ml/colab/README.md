# Colab GPU ingestion

The Milestone 2 entry point for turning raw source artwork into a validated
LineScout gallery, on a free Colab GPU — no account, no paid tier, nothing to
install locally.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sehaan-1/drawable/blob/main/ml/colab/linescout_gpu_pipeline.ipynb)

* **`linescout_gpu_pipeline.ipynb`** — the notebook. Configuration, progress,
  previews, and the numbers that come out.
* **`linescout_ml/colab/`** — the pipeline itself: typed, linted, and unit tested
  in CI, with every heavy import deferred until a stage actually needs it.

The notebook is a driver, not an implementation. Stage logic lives in the package
so that a bug found on a T4 is fixed in a module that CI can see, and so the same
code can be run headless on a workstation (`pip install -e ".[gpu]"`).

## Quick start

1. Open the notebook with the badge above, then **Runtime → Change runtime type →
   T4 GPU**.
2. Upload your sources to Drive, one folder per dataset:
   ```
   MyDrive/LineScout/sources/
       amateur_drawings/…
       manga109/<title>/page-001.png
       met_openaccess/…
   ```
3. In cell 1, set `DATASET_VERSION`, `DRIVE_ROOT`, and one `SOURCES` entry per
   dataset — each with a **`license_id` you verified yourself**. The pipeline
   refuses to write a provenance manifest with a placeholder in it. Start small:
   `LIMIT_PER_SOURCE = 20` for a first smoke run.
4. Run cell 2 (environment), then cell 3 (CPU sanity check on the committed
   fixture, ~10 s) to confirm the plumbing works in your runtime.
5. Run cells 4 → 12. Each stage prints what it did and persists before the next
   one starts.

Prefer one cell over eleven? The `run_all` alternative is documented at the end
of section 12. Use it after you have watched the stages once.

## What comes out

```
<DRIVE_ROOT>/gallery/<DATASET_VERSION>/
    originals/<asset_id>.png        source image, re-encoded as PNG
    line_art/<asset_id>.png         extracted (or native) line art
    thumbnails/<asset_id>.png       reference-panel tile
    manifest.json                   validated against linescout_ml/manifest.py
    _pipeline/candidates.jsonl      resumable per-image state
    _pipeline/run_report.json       config, GPU, model versions, counts
<DRIVE_ROOT>/indexes/<DATASET_VERSION>/
    mobileclip2_s2/{index.json, shard-*.npz}     512-d, text-aligned
    dinov2_vits14/{index.json, shard-*.npz}      384-d, self-supervised shape
```

Plus a zip of exactly the files the manifest vouches for, and an optional mirror
onto Drive. The zip is the reliable transport for a large gallery: Drive's FUSE
mount is slow and prone to partial writes across tens of thousands of small
files.

Landing it locally:

```bash
unzip linescout-<version>.zip -d data/gallery/<version>/
# services/api/.env
LINESCOUT_GALLERY_MANIFEST=data/gallery/<version>/manifest.json
LINESCOUT_CURATION_MODE=1
```

Then `npm run dev:all` and curate at `http://127.0.0.1:5173/curate`.

## Pipeline stages

| Stage | Module | Device | Notes |
|---|---|---|---|
| discover | `sources.py` | CPU | recursive walk, deterministic `asset_id`, work grouping, hash-derived splits |
| extract | `extract.py` | **GPU** | `anime2sketch` / `informative_drawings` via `controlnet_aux`, or native normalisation |
| measure | `measure.py` | CPU | ink, text, quality, 64-bit pHash, crop |
| dedupe | `measure.py` + `runner.py` | CPU | pHash groups at Hamming ≤ 6, recomputed each run |
| label | `label.py` | **GPU** | MobileCLIP2 zero-shot style/scope, provisional by design; SFW gate |
| embed | `embed.py` | **GPU** | MobileCLIP2-S2 + DINOv2 shards, resumable |
| build | `assets.py` | CPU | records, merge with any existing manifest, invariant checks |
| export | `export.py` | CPU | zip, Drive copy, run report |

Supporting modules: `config.py` (typed settings + dataset presets), `models.py`
(encoder wrappers and model cards), `runtime.py` (device resolution), `_optional.py`
(lazy imports with actionable errors).

### Resumability

Colab disconnects idle and heavy sessions, so nothing is a long transaction:

* `candidates.jsonl` holds per-image state; a restarted runtime reloads it and
  every stage skips work that is already done.
* Extraction checks for files on disk, embedding checks the shard index, and the
  manifest is **merged** rather than rewritten — so a second dataset added next
  week does not rebuild the first.
* A CUDA out-of-memory on one huge scan retries that image at half resolution
  instead of ending the run.
* De-duplication marks duplicates but keeps their files, so changing
  `dedupe_threshold` and re-running the cell can both drop and restore assets.

## Labels are provisional

Zero-shot CLIP labels are a starting point, not a verdict. Every record keeps
`labels.labelled_by` (`zero_shot` or `source_default`) and the raw probabilities,
and every freshly ingested asset arrives `review.state="unreviewed"` with
`enabled=true` so the curation UI can render and correct it. Assets that fail the
SFW gate are written `quarantined` + `enabled=false`, which the manifest enforces
as an invariant.

The SFW screen runs on the **original**, not the line art, because extraction
removes exactly the content a classifier needs to see. Sources whose terms
already guarantee SFW content are recorded as `source_rating` and pay nothing;
only scraped sources run `opennsfw2`.

## Licences

Model weights are downloaded at runtime and their terms are recorded in the
embedding index beside the vectors they produced.

| Component | Licence | Note |
|---|---|---|
| Anime2Sketch (`netG.pth`) | MIT | weights mirrored as `lllyasviel/Annotators` |
| Informative Drawings (`sk_model.pth`) | MIT | Chan *et al.*, CVPR 2022 |
| `controlnet_aux` | Apache-2.0 | loads both of the above |
| MobileCLIP2 | code MIT, **weights Apple ML Research Model License** | **research purposes only, no commercial use.** Derived features inherit that |
| DINOv2 | Apache-2.0 | the XRay/Cell variants are not Apache; never loaded here |
| `opennsfw2` | MIT | needs TensorFlow; Colab ships it |

**Dataset licences are yours to verify.** The presets in `config.py` describe what
to check for each source and deliberately ship no `license_id`: a provenance
manifest that guesses at licences is worse than no manifest at all.

## Running the stages without Colab

Everything except the GPU stages runs on a laptop with the CPU extras, which is
what CI does:

```bash
cd ml
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"

.venv/bin/pytest tests/test_colab_pipeline.py -q   # end-to-end on generated images
.venv/bin/pytest tests/test_colab_notebook.py -q   # notebook structure + dry-run cell
```

`tests/test_colab_notebook.py` compiles every notebook cell, checks that the
names it imports are really exported by the package, asserts that no cell tries
to reinstall torch, and **executes the notebook's dry-run cell** against the
committed fixture. A refactor that breaks the notebook fails CI rather than
failing someone's GPU session.

With the GPU extras installed (`uv pip install -e ".[gpu]"`) the same
`PipelineRunner` runs headless:

```python
from pathlib import Path
from linescout_ml.colab import PipelineConfig, PipelineRunner, preset_source

config = PipelineConfig(
    dataset_version="2026.09.06-local1",
    output_root=Path("data/gallery/2026.09.06-local1"),
    sources=[
        preset_source(
            "amateur_drawings", root=Path("data/sources/amateur_drawings"), license_id="cc-by-4.0"
        )
    ],
)
report = PipelineRunner(config).run_all(zip_path=Path("gallery.zip"))
```

Troubleshooting lives in section 14 of the notebook.
