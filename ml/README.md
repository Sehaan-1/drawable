# linescout-ml

Dataset manifest schema, taxonomy, and validation for LineScout, plus the
Milestone 2 ingestion pipeline that fills a gallery from raw source artwork.
Training, index construction, and evaluation arrive in later milestones.

Two halves live here:

* **`linescout_ml/`** — the manifest contract (`manifest.py`, `taxonomy.py`), the
  deterministic fixture generator, the `linescout-manifest` CLI, and
  **`linescout_ml/colab/`**: the ingestion pipeline (line-art extraction,
  measurement, pHash de-duplication, zero-shot labelling, feature embedding,
  manifest assembly, export).
* **`colab/`** — the Google Colab notebook that drives that pipeline on a free
  GPU runtime. See [`colab/README.md`](colab/README.md).

```bash
cd ml
uv sync --frozen --extra dev --python 3.11

# Validate a manifest (and check that enabled assets' files exist)
.venv/bin/linescout-manifest validate ../data/gallery/manifest.json --require-files

# Emit the JSON schema
.venv/bin/linescout-manifest schema --out ../packages/contracts/manifest.schema.json

# Regenerate the committed synthetic fixture (deterministic; safe to commit)
.venv/bin/linescout-manifest synth --out fixtures/synthetic --count 24 --seed 7

# Checks (what CI runs)
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy linescout_ml && .venv/bin/pytest -q

# Run the pipeline headless instead of in Colab: same lockfile, plus the GPU stack
uv sync --frozen --extra gpu --python 3.11

# Audit the Colab setup without a GPU: pins, lock coverage, the notebook's imports
.venv/bin/linescout-repro selfcheck

The `dev` extra is enough for everything CI does: the pipeline's CPU stages need
only `numpy` and `pillow`, and the GPU stages keep their imports lazy so a fresh
clone never has to install torch to validate a manifest. Nothing here installs a
GPU package implicitly: `linescout-repro selfcheck` asserts that, and it also checks
that `colab/requirements-colab.txt` agrees with `uv.lock`, that the checkpoint lock
covers every group the default pipeline wants, and that the notebook installs the plan
rather than a list of bare package names.

`fixtures/synthetic/` is the only dataset committed to Git. Real datasets live
under `data/` (ignored) and are never redistributed — the Colab notebook writes
its output to Drive, and section 13 of it explains how to land a gallery locally.
