# Colab GPU ingestion

The Milestone 2 entry point for turning raw source artwork into a validated
LineScout gallery, on a free Colab GPU — no account, no paid tier, nothing to
install locally.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/junosapollo/drawable/blob/main/ml/colab/linescout_gpu_pipeline.ipynb)

* **`linescout_gpu_pipeline.ipynb`** — the notebook. Configuration, progress,
  previews, and the numbers that come out.
* **`linescout_ml/colab/`** — the pipeline itself: typed, linted, and unit tested
  in CI, with every heavy import deferred until a stage actually needs it.

The notebook is a driver, not an implementation. Stage logic lives in the package
so that a bug found on a T4 is fixed in a module that CI can see, and so the same
code can be run headless on a workstation (`uv sync --frozen --extra gpu`).

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
   `LIMIT_PER_SOURCE = 20` for a first smoke run. `REPO_PIN` and
   `CHECKPOINT_POLICY` are the two reproducibility knobs, and both default to the
   values this repository was audited with.
4. Run cells 2 → 2d, in that order: the pinned checkout, then the sources resolved,
   then Colab's torch frozen into constraints, then the pinned installs. Cell 2 stops
   if HEAD is not the pin and cell 2b stops if a licence is a placeholder — both
   before anything installs, which is the whole ordering.
5. Run cell 3 (CPU sanity check on the committed fixture, ~10 s) to confirm the
   plumbing works in your runtime, then cells 4 → 12. Each stage prints what it did
   and persists before the next one starts.
6. Read the `reproducibility:` block cell 12b prints before you close the tab. It is
   what a future rebuild needs, and `NOT RECORDED` there means this run cannot be
   re-created no matter how good the numbers look.

Prefer one cell over eleven? The `run_all` alternative is documented at the end
of section 12. Use it after you have watched the stages once.

## Reproducibility

A GPU run is expensive enough that doing it twice by accident should be impossible,
so the pipeline pins everything that could silently change underneath it. Not one of
these is optional, and none of them are the author's machine being representative.

| What | Where | How it is enforced |
|---|---|---|
| The code you ran | `ml/colab/linescout_gpu_pipeline.ipynb` cell 2a | `git fetch --depth 1 origin <sha>` then `git checkout --force` |
| The environment | `ml/colab/requirements-colab.txt` | every install in the notebook is `pip(PLAN.pip_specs)`, never a bare name |
| The weights | `ml/linescout_ml/colab/models.lock.json` | size + SHA-256 verified before a stage opens the file |
| The runtime | `ml/colab/runtime-baseline.json` | actual versions recorded at run time, compared to the baseline |
| All of it, per run | `run_report.json`, `manifest.json` | recorded, then hashed with the manifest |

### The repository pin

The notebook defaults to **`junosapollo/drawable`** at commit
`81a8683a53ea9e1a9838925968bd6b3020a3d102`, fetched over
`https://github.com/junosapollo/drawable.git` — the unauthenticated transport, because
nothing that has to be reproducible should depend on a token you happen to have in a
runtime. `REPO_URL` is a form field for the private-repo case and for a fork; the pin
is what makes it safe to change. A tag would read better than 40 hex digits, but a tag
can be moved and a branch can be force-pushed; a commit cannot be either, and a
notebook that says "`main`" is a notebook that runs whatever `main` happens to be the
day someone presses the button.

The `Sehaan-1/drawable` mirror in `REPO_MIRROR_URL` is a fetch fallback only, tried
when the canonical URL is unreachable, and it cannot change what runs: whatever arrives
is checked out by SHA and the checkout is refused if HEAD is not the pin. Content
addressing is what makes a mirror safe; "a second URL I trust" would not be.

The pin names the commit that contains the finalized pipeline — the stage logic, the
SFW policy above, and the schema v3 contract the gallery is written against. **A contract
change counts as a pipeline change.** A pin predating one is not merely older, it is
*wrong*: the notebooks at such a pin emit records the current API refuses to read, so the
pin moved when v2 landed rather than staying put as a documentation courtesy — and moved
once more when running the notebook's own reporting cells turned up three v1 attribute
reads no test had been able to see, and again when schema v3 froze which records the API
loads at all. The churn is the mechanism working: every bump is a commit whose tree agrees
with itself, so the number cannot quietly drift out of date.

One lag is structural and worth knowing before you file it as a bug. A commit cannot
contain its own SHA, so the commit that *ships* a pin value is one past the commit the pin
names: check out the pin and its own `REPO_PIN`, `COLAB_PIN` and this paragraph will cite
the bump before it. The pinned *code* is what matters, and it is complete — cell 1's
procedure fails closed on any mismatch, so a reader following the stale number in the
pinned notebook simply re-checkouts the tree they are already in.

Each bump is verified the same way, against GitHub rather than a fixture:

```bash
.venv/bin/linescout-repro checkout --dir /tmp/ls-pin --rev <pin>   # canonical + mirror
# and prove the pinned tree is self-consistent, with *its* code on the path:
cd /tmp && PYTHONPATH=/tmp/ls-pin/ml <repo>/ml/.venv/bin/python -c \
  "from pathlib import Path; from linescout_ml.colab import run_selfcheck; \
print(run_selfcheck(Path('/tmp/ls-pin')))"
```

The first prints `HEAD : 81a8683a53ea  (matches the pin)` for both URLs; the second
prints `[]`, which is the stronger claim — the pinned commit's own notebook,
environment spec, checkpoint lock, and docs agree with *each other*, not merely with
whatever the working tree looks like now. Running it from the repository root instead
of `/tmp` silently audits the pinned files against the working tree's `COLAB_PIN` and
invents two spurious problems, which is the one way to get a false failure out of this
design. The reason the
canonical URL answers at all is that `Sehaan-1/drawable` is a *fork* of
`junosapollo/drawable` and GitHub shares object storage across a fork network — enough
to run today, a hazard tomorrow, because a pin reachable only through a fork breaks
the moment that fork is re-created or detached. Merging into the canonical repository
is what makes the pin permanent: a merge into `Sehaan-1/drawable` makes the mirror
permanent and leaves the primary URL relying on the fork network still.

One consequence of pinning a commit rather than a branch, worth naming because it
looks wrong at first glance: the notebook *inside* the pinned tree still defaults to
the pin that was current when that tree was written — a commit cannot contain its own
SHA. It does not matter what runs, because cell 2 verifies HEAD against whatever
`REPO_PIN` holds in the notebook you actually opened, and the code it puts on
`sys.path` is the pinned tree's. To see the pin mean something, check it from the
outside:

```bash
git -C /tmp/ls-pin rev-parse HEAD          # the pin
grep -c 'source_rating+opennsfw2' /tmp/ls-pin/ml/linescout_ml/colab/config.py
```

`0000000000000000000000000000000000000000` is what the pin held before that commit
existed: a format-valid SHA no repository contains, named out loud by
`COLAB_PIN_PLACEHOLDER` so that `linescout-repro pin --check` (wired into CI) fails
while it is in force and cell 2 says so. Bumping the pin is a small, boring,
procedural change:

```bash
# 1. land the pipeline, get its commit
git rev-parse HEAD                             # e.g. 3f9c1d2e...
.venv/bin/linescout-repro pin --pin <the 40 hex you just read>   # judge it first
# 2. COLAB_PIN in ml/linescout_ml/colab/repro.py, REPO_PIN in the notebook cell 1 form
git commit -am "docs: pin Colab to <sha>"
.venv/bin/linescout-repro pin --check && .venv/bin/linescout-repro selfcheck
# 3. prove a stranger's runtime can actually fetch it, from a scratch clone
.venv/bin/linescout-repro checkout --dir /tmp/ls-pin --rev <sha>
```

Nothing in the code derives the pin from the current checkout. If it did, `origin` on
your machine would decide what a stranger's GPU session clones.

### The environment

`ml/colab/requirements-colab.txt` is the only dependency list Colab installs from:
21 pins in six `# group:` sections — 19 installable, plus a `[runtime]` group of 2
(`torch`, `torchvision`) that exists to be *compared against* and never installed. The
distinction is the whole design:

* A **runtime** package's version is recorded, not forced. Letting pip replace the
  image's CUDA-enabled torch with the PyPI default build is how a free T4 quietly
  becomes a CPU run, and `pip` cannot see the difference. Cell 2c asserts the `+cu…`
  build tag survives and that torch can actually move a tensor onto the device.
* Every **other** pin is checked against `uv.lock` by `linescout-repro selfcheck` (CI
  runs it), so the file you run in Colab and the lockfile you run at home are the same
  version by construction, not by diligence.

Why the constraints file is not paranoia — resolving the 19 installable pins from a
clean Python 3.11 environment on linux, which is worth repeating whenever the file
changes:

```bash
cd ml && .venv/bin/python - <<'PY' > /tmp/colab-pins.txt
from linescout_ml.colab import repro

spec = repro.load_requirement_spec()
skip = set(spec.runtime_only)
print("\n".join(f"{name}=={version}" for name, version in spec.pins.items() if name not in skip))
PY
uv pip compile --python-version 3.11 /tmp/colab-pins.txt | grep -c '=='   # → 81
```

19 pins become **81 packages**, and 19 of those are CUDA-flavoured (`cuda-toolkit`,
fourteen `nvidia-*` wheels, `triton`, `cuda-bindings`) — every one of them arriving
`# via torch`, a package the file deliberately refuses to install. `torch==2.14.0`
enters the closure purely transitively, through `controlnet-aux`, `open-clip-torch`,
`timm`, and `torchvision`, which is exactly how an unpinned install ends up replacing
a `+cu` build with 2 GB of PyPI wheels. Nothing about that is visible in the file's own
lines; it is only visible one level down, which is why the notebook's `pip()` honours a
constraints file rather than trusting the pins to keep out what they never mentioned.

The closure has one more thing to say: `opencv-python` arrives `# via opennsfw2`, next
to our `opencv-python-headless` pin, so choosing the headless build is a preference and
not an exclusion — both are installed, and a container without `libGL.so.1` fails at
`import cv2` regardless of which one the pipeline meant to use.

The names are additionally a subset of the `[gpu]` extra in `ml/pyproject.toml`, which
is the one place they all get their versions resolved, and they must each have a lock
entry — a pin nobody can resolve is a pin nobody can install.

`pip` is told only the pinned `name==version` strings; it is never asked for a name.
The notebook installs nothing that is already present, because `pip install` of an
already-satisfied requirement still pays for the index round trip, and the whole point
of the preflight is that a wrong version is caught before weights download.

### The weights

`ml/linescout_ml/colab/models.lock.json` is a `ModelLock`: one entry per file a stage
needs, each with `repo` + `revision` + `filename` + `sha256` + `size_bytes`. A revision
is a 40-hex commit (`IMMUTABLE_REVISION`) for the same reason the repository pin is:
`main` is a moving pointer, and the whole point is that nobody has to trust it.

A `sha256` there is always the file's actual digest — for Hugging Face weights, the
`lfs.oid` returned by the API (which *is* the SHA-256), not the top-level `oid` (which
is a git blob SHA-1, and would fail every verification ever done against it). Files
with no published digest are honest about it: `sha256: null` plus a `note` saying who
has to look.

`CheckpointPolicy` then decides what to do about it:

| Policy | Behaviour |
|---|---|
| `record` (default) | download once, verify what arrives, record the digest into the run; a cached file whose digest disagrees with the lock still stops the run |
| `require` | a missing digest is an error, and every entry must carry a SHA-256 — the mode for anything that will be published |
| `off` | no verification; `run_report.json` says `"unverified"` in plain words |

Nothing is ever deleted over a digest disagreement; the error names the cache path so
you can look at what you got. Verification runs once per file per process, and an
already-verified cache directory is a directory whose digest matched a minute ago, so
`gated = already_verified or verifier.gated(...)` is how the label stage decides not to
re-open it.

### What a run leaves behind

Every manifest and every `run_report.json` carries a `PipelineProvenance`: repository +
commit + dirty flag, the environment pins file and its digest, the environment digest
itself, the runtime baseline and whether it matched, the source of truth, and each
model's resolved repo/revision/digest. `manifest.content_hash()` covers it, so a
provenance mismatch is a different gallery — you cannot quietly re-point an old
manifest at new weights.

`compare_runtime` compares the run's environment to `runtime-baseline.json` and records
the answer in `run_report.json` — with one gap, stated plainly there: an installed
package with **no** recorded version is an unknown, not a mismatch. `tensorflow` and
`opennsfw2` are exactly that, and the honest reason is that this environment had no way
to resolve them. The `unrecorded` list in the baseline names them so nobody mistakes
the baseline for a complete environment description.

### Auditing it

```bash
cd ml
uv sync --frozen --extra dev --python 3.11
.venv/bin/linescout-repro selfcheck     # also run in CI
.venv/bin/linescout-repro pin --check   # fails while the pin is still the placeholder
.venv/bin/linescout-repro environment plan   # what this run would install, from the pins
.venv/bin/linescout-repro environment report # what this runtime actually has
```

`selfcheck` reads files rather than importing them, so it checks the notebook you
committed and not the notebook you meant to commit. It catches a pin that drifted from
the lockfile, a GPU import hoisted to module level (which would make every CPU test
either fail or silently skip), an unpinned checkpoint group, a notebook cell reordered
to install before it resolves the source presets, and documentation that still tells you
to install the GPU extras by hand instead of through the lockfile.

### Running the GPU path

See [`SMOKE_TEST.md`](SMOKE_TEST.md): the runbook for the GPU path, with the exact
output each cell has to produce, and a plain statement of what has *not* been run —
no GPU and no route to the weight CDNs means nobody has done a real Colab run of this
pinning yet, and the file says so rather than implying a test report.


## What comes out

```
<DRIVE_ROOT>/gallery/<DATASET_VERSION>/
    originals/<asset_id>.<ext>      source image, copied byte-for-byte
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
* The weights cache is the one thing that does *not* survive: it defaults to
  `/root/.cache/linescout/checkpoints`, and a recycled runtime throws ~770 MB of
  pinned files away, verified again on download. Point `CHECKPOINT_CACHE_DIR` at
  Drive to keep them across disconnects, and accept that Drive's FUSE mount reads a
  400 MB file slower than a local disk — worth it on a long run, not on a smoke test.

## Labels are provisional

Zero-shot CLIP labels are a starting point, not a verdict. Every record keeps
`labels.labelled_by` (`zero_shot` or `source_default`) and the raw probabilities, and
every freshly ingested asset arrives `review.state="unreviewed"` with no `sfw_human`
approval and no quality grade — which is enough to keep it out of search, because
serving is the derived `is_servable` predicate in v2 and not a stored flag. The curation
UI renders such assets through dedicated preview routes. Assets that fail the SFW screen
(`unsafe`, or `unsure` under the confidence floor) are written `quarantined`, which the
manifest enforces as an invariant.

The SFW screen runs on the **original**, not the line art, because extraction
removes exactly the content a classifier needs to see. What a source may be
trusted on is a per-preset policy, and the rule is *who made the guarantee* — not
how innocuous the dataset looks:

| | Presets | Gate |
|---|---|---|
| A publisher's own programme, or a corpus gated behind an application | `synthetic`, `quickdraw`, `manga109`, `ebdtheque`, `met_openaccess`, `smithsonian_openaccess` | `source_rating`: no classifier runs, and every asset records that it did not |
| Community uploads, where the people posting enforce the rules | `amateur_drawings`, `safebooru` | `source_rating+opennsfw2`: both, stricter verdict wins |
| Artwork of people, with nothing to inherit | `human_art` | `opennsfw2` |

A booru's `rating` tags are the claim the gate exists to check, so they are not a
guarantee — honouring them would make the gallery's safety screen a copy of a
stranger's moderation queue. The only supported way to skip screening is
`sfw_method="manual"`, which records **no screen at all** — it writes your verdict as a
human `sfw_human` decision attributed to `ingestion:<pipeline_version>`, in every asset and
in the run report, because a screen is a claim about how a machine decided. Nothing else turns it off — not
`RUN_LABELS = False`, which skips the CLIP ranking and still loads the classifier for
the sources that need it — and an original the pipeline cannot read fails *closed*.
`SFW_POLICY` in `tests/test_colab_config.py` is the table, so changing a preset's
policy is a reviewed edit to a test rather than a default nobody read.

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
uv sync --frozen --extra dev --python 3.11

.venv/bin/pytest tests/test_colab_pipeline.py -q   # end-to-end on generated images + reporting cells
.venv/bin/pytest tests/test_colab_notebook.py -q   # notebook structure + dry-run cell
```

`tests/test_colab_notebook.py` compiles every notebook cell, checks that the
names it imports are really exported by the package, asserts that no cell tries
to reinstall torch, and **executes the notebook's dry-run cell** against the
committed fixture. A refactor that breaks the notebook fails CI rather than
failing someone's GPU session.

`tests/test_colab_pipeline.py` additionally **executes the notebook's reporting
cells** (extract, measure, de-duplicate, labels, manifest slice) against a completed
runner and asserts on what they print. This exercises model attribute access and
string lookups (like `getattr(item, field)`) against real outputs, preventing
stale attributes from surviving undetected in cells that only run at the end of
a session.

With the GPU extras installed (`uv sync --frozen --extra gpu`) the same
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
