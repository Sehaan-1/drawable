# GPU smoke test — the runbook, and what has not been run

**No real Colab or GPU run has been executed for this setup, and this file is where
that is written down instead of being hidden in a checkbox.** The sandbox that
produced the pinning machinery has no CUDA device (`nvidia-smi` is not installed) and
no route to the weight hosts: `huggingface.co:443` fails the TLS handshake here. PyPI
*is* reachable, which is how `requirements-colab.txt` and `models.lock.json` were
built from real wheel metadata and real `lfs.oid` values — but nothing here has ever
downloaded `netG.pth`, loaded a ControlNet, or pushed a tensor onto a T4.

So the GPU path is documented as a runbook with expected output, and the parts that can
be proven without a GPU are proven by tests. Do not read this file as a passing test
report, and do not merge a gallery produced by the first run of these cells without
reading the table at the bottom.

## What is verified without a GPU

| Claim | Evidence |
|---|---|
| Every notebook cell compiles; the names it imports exist in the package; the dry-run cell executes against the committed fixture | `pytest tests/test_colab_notebook.py` |
| A checkout is only accepted at the pinned commit; an existing one is verified and never reset under your feet; a dirty or wrong-HEAD tree is refused or recorded as dirty | `pytest tests/test_colab_repro.py` (runs against real throwaway `git` repos) |
| The notebook installs only pinned `name==version` specs from the plan, and only after presets resolve | same, plus `linescout-repro selfcheck` |
| `requirements-colab.txt` agrees with `ml/uv.lock`, its names are all in the `[gpu]` extra, and no runtime pin is installed by pip | `linescout-repro selfcheck` |
| Every checkpoint the default pipeline selects has a lock entry; nothing is fetched by tag; a digest or size mismatch stops the run | `pytest tests/test_colab_checkpoints.py` |
| No heavy import is reachable at module scope, so CPU tests need no torch | `linescout-repro selfcheck` |
| The docs still describe the setup CI runs | `linescout-repro selfcheck` |

Reproduce that block anywhere:

```bash
cd ml
uv sync --frozen --extra dev --python 3.11
.venv/bin/pytest -q
.venv/bin/linescout-repro selfcheck
```

## The runbook

Runtime: **T4**, default image, nothing pre-installed by hand. `git` and `curl` are
present; Colab's own credential helper authenticates the clone of a private repo after
the Drive/GitHub auth cell you already run.

| # | Cell | What must be true when it finishes |
|---|---|---|
| 1 | 0 · Before you start | Read it. If `REPO_PIN` still prints as a placeholder, that is a bug in this repository and not in your session — `linescout-repro pin --check` in CI should have caught it before you got here. |
| 2 | 1 · Run configuration | `LIMIT_PER_SOURCE = 20`, one `SOURCES` entry with a real `license_id`, `MOUNT_DRIVE = True`. |
| 3 | 2 · GPU, Drive, checkout | `GPU 0, T4, ...` appears, then the record: `repository : /content/drawable`, `pinned at : <40 hex>`, `HEAD : <12 hex>  (matches the pin)`, `working tree : clean (0 untracked)`. Anything else there — `(does NOT match the pin)`, `DIRTY` — has to be explained before you spend an hour on it. |
| 4 | 2b · Resolve sources | Each source prints `origin=`/`extractor=`/`sfw=`. Human-Art prints `sfw=opennsfw2` and the line `needs the opennsfw2 gate (its Keras backend comes from the runtime)` **even if you never wrote `sfw_method`**. That line is the reason this cell runs before the install. |
| 5 | 2c · Freeze torch | `constraints: torch==<the image's CUDA build>` — never `torch==2.x.0` alone (no `+cu`), and never "no torch importable here" on a GPU runtime. |
| 6 | 2d · Install and record | `install` lines for a handful of packages and no `Installing collected packages: torch`. Then `environment <12 hex> (requirements-colab.txt)` and `checkpoints <12 hex> (15 artifacts, 11 without a pinned digest, policy record)`. A `baseline differs` block is *information*, not failure — record it (below) when the runtime is a newer image, and treat a `matches` line on a fresh image as suspicious. |
| 7 | 3 · Sanity check | ~10 s, CPU, writes and validates a tiny manifest. This proves the plumbing without weights. |
| 8 | 4b · Runner | `code : <12 hex> from junosapollo/drawable`, and `weights: policy record, cache /root/.cache/linescout/checkpoints`. The `code` line is blank if cell 2 did not verify anything, which is the one output here that should make you stop. |
| 9 | 6 · Extract · *GPU* | The first image downloads weights: watch for `verified` notes. If `opennsfw2`'s or `controlnet_aux`'s file digest disagrees with `models.lock.json`, the run stops with a named cache path — that is the pin working, and the correct response is to look at what upstream did, not to delete the check. |
| 10 | 9 · Label · *GPU* | `opennsfw2` loads on the *originals*. A missing `keras` here is a Colab image change, not a lockfile bug: `opennsfw2` 0.18 declares no Keras version, and the environment spec pins `gdown` but deliberately does not fight the image over TensorFlow. |
| 11 | 10–12 · Embed, build, export | 20×N images, two shards, one manifest, one zip. |
| 12 | 12b · Run report | The `reproducibility:` block must print a `code ...@<12 hex> (clean tree, cloned)` line, the environment digest, the checkpoint digest count, and the runtime summary. `code NOT RECORDED — this run did not verify a checkout against a pin` means exactly that, and the run is not reproducible however clean the numbers look. |

## What to bring back

Paste this into the pull request that records a run — it is the only part of the table
that cannot be derived from the artifacts:

```
date:
runtime:              # e.g. colab T4, image dated ...
gpu:                  # nvidia-smi -L line
python / torch / cuda:            # from cell 2d's RUNTIME print
pin:                              # 40 hex, from cell 2
environment spec sha256:          # 12 hex is enough
baseline drift:                   # "none", or the lines printed
checkpoints verified:             # count, and "unverified" if policy was off
images: extracted / labelled / embedded / in manifest
failures:                         # count and one line each
```

Then, from that same runtime, refresh the baseline so the next person's diff is
smaller than yours:

```bash
.venv/bin/linescout-repro environment record     # writes ml/colab/runtime-baseline.json
.venv/bin/linescout-repro environment drift
```

Commit the regenerated `runtime-baseline.json` — and only that file, or a
`requirements-colab.txt` change that `uv lock` also agrees with. The baseline is a
*record* of one runtime, never a spec: the `unrecorded` list in it exists precisely so
that a package the sandbox could not resolve (TensorFlow, `opennsfw2`) is visible as a
gap rather than silently missing.

## Triage for the failures this design can produce

| Symptom | What it means | Do |
|---|---|---|
| `no commit ... in this repository` during the fetch | The pin names a commit the remote does not have (placeholder, unpushed branch, or a force-push) | Check `linescout-repro pin --check`; push the pinned commit or bump the pin. Never `git pull` inside the checkout to "fix" it. |
| `refusing a checkout with uncommitted changes` | `REPO_ALLOW_DIRTY = False` and the tree is dirty | Commit elsewhere and re-point the pin, or set it to `True` knowing the run is recorded `dirty` |
| `is N bytes, the lock says M` | A truncated or replaced download | Re-run; if it persists, the CDN served something else — record the URL and the digest in the issue |
| `SHA-256 mismatch ... Refusing to run on bytes that do not match the lock` | The cached file is not the pinned file | The message names the path; delete *that file*, not the directory, and re-run |
| `the checkpoint lock has no artifacts for group ...` | A model card names a group nobody pinned (e.g. `mobileclip2_s0`) | Intended fail-closed. Pin the group in `models.lock.json` or do not select the model |
| `MissingDependencyError` in the label stage after `AUTO_INSTALL = False` | Nothing installed, as asked | Run cell 2d |
| A stray `note:` about a mirror | `REPO_URL` was unreachable and the mirror served the fetch | Harmless *if* the printed HEAD equals `REPO_PIN`; it always is, or the run stopped |
