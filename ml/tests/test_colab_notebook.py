"""Structural and behavioural tests for the Colab notebook.

The notebook is a deliverable, and deliverables that are never executed rot. It
lives outside the import graph, so nothing else in CI would notice if a refactor
of ``linescout_ml.colab`` left it calling an API that no longer exists.

These tests therefore do three things:

* **compile** every code cell, so syntax errors and stale names are caught here
  rather than in a browser on a T4;
* **check the imports** it makes against the package's real public surface;
* **run the CPU sanity-check cell for real** against the committed fixture, which
  exercises discover → extract → measure → dedupe → build → export through the
  exact code path the notebook uses;
* **hold the environment cells to the reproducibility contract**: one immutable
  commit, installs only from the pinned spec, torch frozen rather than
  reinstalled, and the run stamped with what the runtime actually had.

Names a later cell reads are checked in ``test_names_flow_forward_across_cells``,
because a Colab notebook has no other way to notice that a cell defines nothing.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import json
import re
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from linescout_ml import colab
from linescout_ml.colab import read_manifest, repro

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "colab" / "linescout_gpu_pipeline.ipynb"
REPO_ROOT = Path(__file__).resolve().parents[2]
COLAB_BADGE_PREFIX = "https://colab.research.google.com/github/"

# The pipeline's canonical stage order, as method names on PipelineRunner.
STAGE_ORDER = (
    "discover",
    "run_extract",
    "run_measure",
    "run_dedupe",
    "run_label",
    "run_embed",
    "run_build",
    "run_export",
)


def load_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def cells(kind: str) -> list[dict[str, Any]]:
    return [cell for cell in load_notebook()["cells"] if cell["cell_type"] == kind]


def source(cell: dict[str, Any]) -> str:
    raw = cell["source"]
    return raw if isinstance(raw, str) else "".join(raw)


def code_sources() -> list[str]:
    return [source(cell) for cell in cells("code")]


def markdown_sources() -> list[str]:
    return [source(cell) for cell in cells("markdown")]


def strip_magics(text: str) -> str:
    """Drop IPython line magics and shell escapes so ``compile`` accepts a cell."""
    kept = [line for line in text.splitlines() if not line.lstrip().startswith(("%", "!"))]
    return "\n".join(kept)


def cell_titled(title: str) -> str:
    """The one code cell whose title starts with ``title``."""
    for text in code_sources():
        if f"@title {title}" in text:
            return text
    raise AssertionError(f"the notebook lost its {title!r} cell")


def notebook_index() -> int:
    """Which cell holds the CPU dry run (found by title, never by position)."""
    for position, text in enumerate(code_sources()):
        if "@title 3 ·" in text:
            return position
    raise AssertionError("the notebook lost its dry-run cell")


# --------------------------------------------------------------------- structure


def test_notebook_is_valid_nbformat_four() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert set(notebook) >= {"cells", "metadata", "nbformat", "nbformat_minor"}
    assert notebook["cells"], "an empty notebook is not a deliverable"

    for position, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] in {"markdown", "code"}, f"cell {position} has no type"
        assert "source" in cell, f"cell {position} has no source"
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], f"cell {position} committed its outputs"
            assert cell["execution_count"] is None, f"cell {position} committed a count"


def test_notebook_requests_a_gpu_runtime() -> None:
    metadata = load_notebook()["metadata"]
    assert metadata["accelerator"] == "GPU"
    assert metadata["colab"]["gpuType"] in {"T4", "L4", "A100"}
    assert metadata["colab"]["toc_visible"] is True
    assert metadata["kernelspec"]["name"] == "python3"


def test_notebook_is_stored_clean() -> None:
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "\r" not in text, "CRLF line endings break Colab diffs"
    assert text.endswith("\n")
    assert "REPLACE-ME" in text, "the placeholder licence guard lost its example"
    for forbidden in ("hf_token", "HF_TOKEN", "api_key", "password"):
        assert forbidden not in text, f"notebook mentions {forbidden}"


def test_badge_points_at_this_notebook_in_the_canonical_repo() -> None:
    """The badge opens *this file*, from the default branch of the canonical repo.

    That is a different URL from the one the run clones, on purpose: people need
    to find the notebook, while the code it executes has to come from one
    immutable commit (`REPO_PIN`, verified in cell 2). "Point the badge at the
    pin too" would make the notebook unfindable for anyone who is not already
    reading that commit.
    """
    first = markdown_sources()[0]
    match = re.search(r"\((https://colab\.research\.google\.com/github/[^)]+)\)", first)
    assert match, "the Open-in-Colab badge disappeared"
    url = match.group(1)
    relative = NOTEBOOK_PATH.relative_to(REPO_ROOT).as_posix()
    assert url.startswith(COLAB_BADGE_PREFIX)
    assert url.endswith(f"junosapollo/drawable/blob/main/{relative}"), url


PIN_LITERAL = re.compile(r'^REPO_PIN = "([0-9a-f]{40})"  # @param \{type:"string"\}', re.M)


def test_the_notebook_clones_one_immutable_commit() -> None:
    """Not a branch, not a tag, not "main": a full SHA, and HEAD is checked.

    This replaced a test that demanded `REPO_REF = "main"` — which was the right
    rule when the notebook was a moving target, and the wrong one once its output
    was a dataset. A branch can be re-pointed; a run recorded as "built by main"
    cannot be re-created.
    """
    config = cell_titled("1 ·")
    env = cell_titled("2 ·")
    match = PIN_LITERAL.search(config)
    assert match, "REPO_PIN lost its 40-hex SHA or its @param comment"
    assert match.group(1) == repro.COLAB_PIN, (
        "the notebook's pin and the package's default pin moved apart"
    )
    assert "junosapollo/drawable.git" in config
    assert "revision=REPO_PIN" in env
    assert "find_or_checkout" in env, "cell 2 stopped asking the package to verify HEAD"

    joined = "\n".join(code_sources())
    assert "REPO_REF" not in joined
    for forbidden in ("origin/", "git pull", "reset --hard", "clone -b ", "--branch"):
        assert forbidden not in joined, f"the notebook reached for {forbidden!r} again"


def test_an_existing_checkout_is_verified_and_its_state_recorded() -> None:
    """A reused checkout must be able to say what it is, dirty or not."""
    config = cell_titled("1 ·")
    env = cell_titled("2 ·")
    assert "REPO_ALLOW_DIRTY = True" in config
    assert "allow_dirty=REPO_ALLOW_DIRTY" in env
    assert "REPO_DIR" in config and "is_checkout(Path(REPO_DIR))" in env
    assert "CHECKOUT.describe()" in env, "the checkout facts are printed, not hidden"
    assert "preferred=" in env and "CLONE_TARGET" in env


# ------------------------------------------------------------------- cell hygiene


@pytest.mark.parametrize("position", range(len(cells("code"))))
def test_every_code_cell_compiles(position: int) -> None:
    text = strip_magics(code_sources()[position])
    compile(text, f"<notebook cell {position}>", "exec")


def test_notebook_imports_only_public_package_names() -> None:
    checked = 0
    for position, text in enumerate(code_sources()):
        tree = ast.parse(strip_magics(text))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if not node.module.startswith("linescout_ml"):
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                checked += 1
                if node.module == "linescout_ml.colab":
                    assert alias.name in colab.__all__, (
                        f"cell {position} imports {alias.name}, which the package does not export"
                    )
                assert hasattr(module, alias.name), (
                    f"cell {position} imports {alias.name} from {node.module}"
                )
    assert checked >= 12, f"only {checked} package imports found — is the test stale?"


def pip_specs(text: str) -> list[str]:
    """Every string literal handed to the notebook's ``pip`` helper."""
    specs: list[str] = []
    for node in ast.walk(ast.parse(strip_magics(text))):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", "")
        if name != "pip":
            continue
        specs.extend(
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        )
    return specs


def test_the_notebook_never_reinstalls_torch() -> None:
    """Colab's CUDA build must survive dependency installation.

    Cell 2c freezes whatever torch the runtime shipped into a constraints file and
    hands it to every install; cell 2d installs only the plan's pinned specs, which
    the environment spec keeps outside its installable groups. A torch mentioned by
    name anywhere in the notebook would undo that, so it is asserted, not commented.
    """
    freeze = cell_titled("2c ·")
    assert 'runtime_constraints(("torch", "torchvision")' in freeze
    assert "write_to=CONSTRAINTS" in freeze
    assert '-c", str(CONSTRAINTS)' in freeze, "the pip helper stopped honouring the pin"

    installed = [spec for text in code_sources() for spec in pip_specs(text)]
    packages = {re.split(r"[<>=!~[; ]", spec, maxsplit=1)[0] for spec in installed}
    assert not packages & {"torch", "torchvision"}, sorted(packages)

    joined = "\n".join(code_sources())
    for forbidden in ("pip install torch", "!pip", "pip install -r"):
        assert forbidden not in joined, f"the notebook reached for {forbidden!r}"


def test_installs_come_from_the_resolved_sources_and_the_pinned_spec() -> None:
    """Nothing is installed from a list typed into a cell.

    The plan is derived from the *resolved* sources (so a preset that implies the
    NSFW gate gets the gate) and every version comes from the environment spec,
    which is why `pip` is handed `PLAN.pip_specs` rather than package names.
    """
    install = cell_titled("2d ·")
    assert 'load_requirement_spec(REPO / "ml" / "colab" / "requirements-colab.txt")' in install
    assert "plan_environment(" in install and "pip(*PLAN.pip_specs)" in install
    for flag in ("extract_line_art=RUN_EXTRACTION", "label=RUN_LABELS", "embed=RUN_EMBEDDINGS"):
        assert flag in install, f"the plan stopped honouring {flag}"
    assert "SOURCE_SPECS" in install, "the plan is no longer driven by the resolved sources"
    assert "AUTO_INSTALL" in install, "the escape hatch for a CPU-only probe disappeared"


def test_the_runtime_is_recorded_and_diffed_against_the_baseline() -> None:
    """A preinstalled runtime is a fact to snapshot, not an assumption to make."""
    install = cell_titled("2d ·")
    for needed in (
        "runtime_report()",
        'load_baseline(REPO / "ml" / "colab" / "runtime-baseline.json")',
        "compare_runtime(RUNTIME, BASELINE)",
        "RUNTIME_SNAPSHOT.write_text",
    ):
        assert needed in install, f"cell 2d stopped recording the runtime ({needed})"
    config_cell = cell_titled("4 ·")
    for stamped in (
        "source_revision=CHECKOUT.revision",
        "source_dirty=CHECKOUT.dirty",
        "environment_sha256=SPEC.sha256",
        "model_lock_sha256=MODEL_LOCK_SHA",
        "checkpoint_policy=CHECKPOINT_POLICY",
    ):
        assert stamped in config_cell, f"the config lost {stamped}"


def test_weights_are_verified_through_the_lock_before_anything_loads() -> None:
    """The notebook configures the policy; the package enforces it."""
    config = cell_titled("1 ·")
    install = cell_titled("2d ·")
    assert 'CHECKPOINT_POLICY = "record"  # @param ["strict", "record", "off"]' in config
    assert "lock_digest(MODEL_LOCK)" in install
    assert "LOCK.unpinned()" in install
    assert "MODEL_LOCK_SHA256" in install, "the optional pin of the lock file itself vanished"
    config_cell = cell_titled("4 ·")
    assert "model_lock_path=MODEL_LOCK" in config_cell


def test_names_flow_forward_across_cells() -> None:
    """A notebook cell can read a name only if an earlier cell defined it.

    Python checks this at runtime, in the order the user clicks, which is how a
    renamed helper survives review and dies in a browser at 2 a.m. Here the
    definition order is the whole test.
    """
    defined: set[str] = set(dir(builtins))
    for text in code_sources():
        tree = ast.parse(strip_magics(text))
        local = {
            (alias.asname or alias.name).split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
                local.add(node.id)
            elif isinstance(node, ast.FunctionDef | ast.ClassDef):
                local.add(node.name)
            elif isinstance(node, ast.arg):
                local.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                local.add(node.name)
        loaded = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        missing = sorted(loaded - defined - local)
        assert not missing, f"a cell reads {missing} before any earlier cell defines it"
        defined |= local


def test_stage_cells_run_in_pipeline_order() -> None:
    calls: list[str] = []
    for text in code_sources():
        calls.extend(re.findall(r"RUNNER\.(\w+)\(", text))
    stages = [call for call in calls if call in STAGE_ORDER]
    assert stages == list(STAGE_ORDER), stages
    # run_all is offered as the hands-off alternative, in prose only.
    assert any("run_all" in text for text in markdown_sources())
    assert not any(re.search(r"RUNNER\.run_all\(", text) for text in code_sources())


def test_documentation_covers_licences_and_recovery() -> None:
    prose = "\n".join(markdown_sources())
    for expected in (
        "Apple ML Research Model License",  # the awkward truth about MobileCLIP2
        "Apache-2.0",
        "Troubleshooting",
        "LINESCOUT_GALLERY_MANIFEST",
        "T4 GPU",
        "candidates.jsonl",
        "dedupe_threshold",
    ):
        assert expected in prose, f"the notebook stopped documenting {expected}"


# ------------------------------------------------------------------- real dry run


def test_the_dry_run_cell_executes_end_to_end(tmp_path: Path) -> None:
    """Run the notebook's own sanity-check cell, unmodified except for its paths.

    This is the test that keeps the notebook honest: it drives the same
    ``PipelineRunner`` calls a Colab user would, on the committed fixture, with
    no GPU dependencies installed.
    """
    text = code_sources()[notebook_index()]
    assert "/content/linescout-dryrun" in text, "unexpected dry-run root"
    text = text.replace("/content/linescout-dryrun", (tmp_path / "dryrun").as_posix())

    namespace: dict[str, Any] = {
        "REPO": REPO_ROOT,
        "DATASET_VERSION": "2026.09.06-notebooktest",
        "THUMBNAIL_SIZE": 256,
        "Path": Path,
    }
    exec(compile(strip_magics(text), "<dry-run cell>", "exec"), namespace)

    report: dict[str, Any] = namespace["dry_report"]
    counts = report["summary"]["candidates"]
    assert counts["total"] == 24, "the committed fixture changed; update the notebook prose"
    # The fixture ships five byte-identical re-uploads, so dedupe must find at
    # least that many even if Pillow's resampling shifts a borderline distance.
    assert counts["duplicates"] >= 5, counts
    assert counts["active"] == counts["total"] - counts["duplicates"] - sum(
        counts["skipped"].values()
    )
    assert report["summary"]["total"] == counts["active"]
    assert [stage["name"] for stage in report["stages"]] == [
        "discover",
        "extract",
        "measure",
        "dedupe",
        "label",
        "embed",
        "build",
        "export",
    ]

    archive = Path(report["outputs"]["zip"]["path"])
    assert archive.is_file()
    assert report["outputs"]["zip"]["bytes"] == archive.stat().st_size
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert "manifest.json" in names
    assert "run_report.json" in names

    gallery = tmp_path / "dryrun" / "gallery"
    manifest = read_manifest(gallery / "manifest.json")
    assert manifest is not None
    assert len(manifest.records) == report["summary"]["total"]
    assert manifest.dataset_version == "2026.09.06-notebooktest"
    for record in manifest.records:
        for relative in (record.original_path, record.line_art_path, record.thumbnail_path):
            assert (gallery / relative).is_file(), relative
        for relative in (record.original_path, record.line_art_path, record.thumbnail_path):
            assert relative in names, f"{relative} is missing from the export zip"
        assert record.review.state.value == "unreviewed"
        assert record.enabled is False


def test_dry_run_cell_output_matches_the_documented_claims(tmp_path: Path) -> None:
    """The prose promises a gallery tree; check the dry run really produces it."""
    text = code_sources()[notebook_index()].replace(
        "/content/linescout-dryrun", (tmp_path / "dryrun").as_posix()
    )
    namespace: dict[str, Any] = {
        "REPO": REPO_ROOT,
        "DATASET_VERSION": "2026.09.06-notebooktest",
        "THUMBNAIL_SIZE": 256,
        "Path": Path,
    }
    exec(compile(strip_magics(text), "<dry-run cell>", "exec"), namespace)

    gallery = tmp_path / "dryrun" / "gallery"
    for directory in ("originals", "line_art", "thumbnails"):
        assert (gallery / directory).is_dir(), directory
        assert list((gallery / directory).glob("*.png")), f"{directory} is empty"
    assert (gallery / "_pipeline" / "candidates.jsonl").is_file()
    assert (gallery / "_pipeline" / "run_report.json").is_file()

    prose = "\n".join(markdown_sources())
    for documented in (
        "originals/<asset_id>.<ext>",
        "line_art/<asset_id>.png",
        "thumbnails/<asset_id>.png",
        "_pipeline/candidates.jsonl",
        "_pipeline/run_report.json",
    ):
        assert documented in prose, f"the overview no longer documents {documented}"


# ------------------------------------------------------ resolve cell (notebook 2b)


def sources_namespace(tmp_path: Path, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """What the resolve cell expects to find already defined by cell 1."""
    return {
        "Path": Path,
        "SOURCES": sources,
        "SOURCES_ROOT": tmp_path / "sources",
    }


def run_resolve_cell(tmp_path: Path, sources: list[dict[str, Any]]) -> dict[str, Any]:
    namespace = sources_namespace(tmp_path, sources)
    exec(compile(strip_magics(cell_titled("2b ·")), "<resolve cell>", "exec"), namespace)
    return namespace


def test_the_resolve_cell_expands_presets_and_overrides(tmp_path: Path, capsys: Any) -> None:
    """Presets win where they are silent; the form wins where it speaks."""
    namespace = run_resolve_cell(
        tmp_path,
        [
            {
                "preset": "amateur_drawings",
                "folder": "amateur",
                "license_id": "cc-by-4.0",
                "extractor": "",
                "sfw_method": "",
                "work_grouping": "parent_dir",
            },
            {
                "preset": "manga109",
                "root": str(tmp_path / "manga109"),
                "license_id": "m109-research",
                "extractor": "informative_drawings",
            },
        ],
    )
    specs = namespace["SOURCE_SPECS"]
    assert [spec.name for spec in specs] == ["amateur_drawings", "manga109"]
    assert specs[0].root == tmp_path / "sources" / "amateur"
    assert specs[0].license_id == "cc-by-4.0"
    assert specs[0].work_grouping == "parent_dir"  # per-source override won
    assert specs[0].extractor == "none"  # the preset's own choice survived
    assert specs[1].root == tmp_path / "manga109"  # absolute "root" beats "folder"
    assert specs[1].extractor == "informative_drawings"

    printed = capsys.readouterr().out
    for preset in ("synthetic", "quickdraw", "amateur_drawings", "manga109", "met_openaccess"):
        assert preset in printed, f"cell 2b stopped advertising the {preset} preset"


@pytest.mark.parametrize("license_id", ["REPLACE-ME", "", "tbd", "  "])
def test_the_resolve_cell_blocks_placeholder_licences(tmp_path: Path, license_id: str) -> None:
    """Provenance is the point of the manifest, so guesses are refused up front —
    and up front means *before the install*, in a cell that imports no weights."""
    with pytest.raises(SystemExit, match="license_id") as raised:
        run_resolve_cell(
            tmp_path, [{"preset": "safebooru", "folder": "safebooru", "license_id": license_id}]
        )
    assert "nothing was installed yet" in str(raised.value)


def test_the_resolve_cell_rejects_an_unknown_preset(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="unknown source preset"):
        run_resolve_cell(tmp_path, [{"preset": "not_a_dataset", "license_id": "cc0-1.0"}])


def test_the_resolve_cell_explains_an_impossible_combination(tmp_path: Path) -> None:
    """A native-line-art source cannot also name an extractor; say so in English."""
    with pytest.raises(SystemExit, match="not a valid combination"):
        run_resolve_cell(
            tmp_path,
            [
                {
                    "preset": "quickdraw",  # native line art: extractor must stay "none"
                    "license_id": "google-terms",
                    "extractor": "anime2sketch",
                }
            ],
        )


def test_the_resolve_cell_rejects_an_empty_source_list(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="SOURCES is empty"):
        run_resolve_cell(tmp_path, [])


def test_a_preset_that_needs_a_classifier_says_so_before_anything_installs(
    tmp_path: Path, capsys: Any
) -> None:
    """The Human-Art case, and the reason resolution happens two cells early.

    Nothing in `SOURCES` mentions NSFW here: the *preset* rates its own images, and
    the cell still announces the opennsfw2 gate, which is what cell 2d installs.
    """
    namespace = run_resolve_cell(
        tmp_path, [{"preset": "human_art", "folder": "human_art", "license_id": "human-art-terms"}]
    )
    spec = namespace["SOURCE_SPECS"][0]
    assert spec.requires_nsfw and spec.uses_extractor
    printed = capsys.readouterr().out
    assert "opennsfw2 gate" in printed
    assert "GPU extractor stage" in printed


# --------------------------------------------------------- config cell (notebook 4)

REVISION = "5" * 40


def config_namespace(tmp_path: Path, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything cell 4 reads: cell 1's form, cell 2's checkout, 2b's specs, 2d's record."""
    spec = repro.load_requirement_spec()
    assert spec is not None, "the committed environment spec disappeared"
    lock = repro.repo_root() / "ml" / "linescout_ml" / "colab" / "models.lock.json"
    return {
        **sources_namespace(tmp_path, sources),
        "SOURCE_SPECS": colab.resolve_sources(sources, sources_root=tmp_path / "sources"),
        "CHECKOUT": repro.CheckoutRecord(
            path=str(tmp_path / "checkout"),
            url="https://github.com/junosapollo/drawable.git",
            requested_revision=REVISION,
            revision=REVISION,
            short_revision=REVISION[:12],
            branch=None,
            detached=True,
            dirty=False,
            untracked=0,
            action="cloned",
            verified=True,
        ),
        "SPEC": spec,
        "MODEL_LOCK": lock,
        "MODEL_LOCK_SHA": colab.lock_digest(lock)["sha256"],
        "CHECKPOINT_POLICY": "record",
        "CHECKPOINT_CACHE_DIR": str(tmp_path / "checkpoints"),
        "MODEL_LOCK_SHA256": "",
        "DATASET_VERSION": "2026.09.06-notebooktest",
        "GALLERY_ROOT": tmp_path / "gallery" / "2026.09.06-notebooktest",
        "INDEX_ROOT": tmp_path / "indexes" / "2026.09.06-notebooktest",
        "DEVICE": "cpu",
        "BATCH_SIZE": 4,
        "LIMIT_PER_SOURCE": 0,
        "THUMBNAIL_SIZE": 256,
        "LINE_ART_RESOLUTION": 512,
        "RUN_EXTRACTION": True,
        "RUN_DEDUPE": True,
        "RUN_LABELS": False,
        "RUN_EMBEDDINGS": False,
        "EMBEDDERS": "dinov2_vits14",
        "LABELER_MODEL": "MobileCLIP2-S2",
    }


def test_the_config_cell_carries_the_run_attribution(tmp_path: Path, capsys: Any) -> None:
    """What a run *did* and what a run *recorded* come from the same objects.

    Cell 4 does not re-read `SOURCES` and does not guess at the environment: it
    stamps the checkout the previous cells verified and the spec it installed, so
    the config cannot describe a run that did not happen.
    """
    namespace = config_namespace(
        tmp_path,
        [
            {
                "preset": "amateur_drawings",
                "folder": "amateur",
                "license_id": "cc-by-4.0",
                "extractor": "",
                "sfw_method": "",
                "work_grouping": "parent_dir",
            },
            {
                "preset": "manga109",
                "root": str(tmp_path / "manga109"),
                "license_id": "m109-research",
                "extractor": "informative_drawings",
            },
        ],
    )
    exec(compile(strip_magics(cell_titled("4 ·")), "<config cell>", "exec"), namespace)
    config = namespace["CONFIG"]

    assert [spec.name for spec in config.sources] == ["amateur_drawings", "manga109"]
    assert config.sources[0].work_grouping == "parent_dir"
    assert config.sources[1].extractor == "informative_drawings"
    assert config.dataset_version == "2026.09.06-notebooktest"
    assert config.limit_per_source is None  # 0 means "no limit"
    assert config.embedders == ["dinov2_vits14"]
    assert config.label is False and config.embed is False
    assert config.source_revision == REVISION
    assert config.source_dirty is False
    assert config.source_action == "cloned"
    assert config.source_repo == "junosapollo/drawable"
    assert config.environment_sha256 == namespace["SPEC"].sha256
    assert config.model_lock_sha256 == namespace["MODEL_LOCK_SHA"]
    assert config.checkpoint_policy == "record"
    assert config.checkpoint_cache_dir == tmp_path / "checkpoints"
    assert config.model_lock_path == namespace["MODEL_LOCK"]
    assert config.source_provenance() is not None

    printed = capsys.readouterr().out
    assert REVISION[:12] in printed
    assert "clean" in printed

    # The config survives a round trip, which is what "recorded" has to mean.
    again = type(config).model_validate_json(config.model_dump_json())
    assert again == config


def test_the_config_cell_refuses_a_revision_that_is_not_a_sha(tmp_path: Path) -> None:
    """`source_revision` is typed as a SHA, so "main" cannot be recorded."""
    namespace = config_namespace(tmp_path, [{"preset": "quickdraw", "license_id": "cc0-1.0"}])
    namespace["CHECKOUT"] = replace(namespace["CHECKOUT"], revision="main", verified=False)
    with pytest.raises(ValidationError, match="source_revision"):
        exec(compile(strip_magics(cell_titled("4 ·")), "<config cell>", "exec"), namespace)


# ------------------------------------------------------ the one-cell alternative
#
# `run_all` is offered inside a markdown cell, which means nothing imports it, nothing
# compiles it, and no ordinary test would notice it drifting from the stages it claims
# to replace. These assertions exist so that a shortcut written in prose stays a
# shortcut through the same pipeline.


def test_the_one_cell_alternative_runs_every_stage_the_cells_run() -> None:
    import inspect

    from linescout_ml.colab import PipelineRunner

    body = inspect.getsource(PipelineRunner.run_all)
    joined = "\n".join(code_sources())
    solo = {match.group(1) for match in re.finditer(r"RUNNER\.(run_\w+)\(", joined)}
    assert solo, "the notebook stopped calling stages by name"
    missing = sorted(name for name in solo if f"self.{name}(" not in body)
    assert not missing, f"the documented one-cell run no longer covers {missing}"


def test_the_one_cell_alternative_leaves_the_same_files_behind() -> None:
    """Provenance a reader cannot see is worse than no shortcut at all.

    The stage calls are equivalent by the test above; the file copy cell 12 does is
    not, so the snippet in the markdown has to carry it. A gallery that is missing
    `runtime.json` looks exactly like a gallery that never had one.
    """
    alternative = next(text for text in markdown_sources() if "run_all(" in text)
    assert "runtime.json" in alternative and "RUNTIME_SNAPSHOT" in alternative, (
        "cell 12 copies the runtime record next to the manifest; the one-cell path "
        "must copy it too or the two documented paths leave different galleries"
    )
    assert "CONFIG" in alternative, "the alternative must build on the verified cells"

    block = re.search(r"```python\n(.*?)```", alternative, re.S)
    assert block, "the alternative lost its python fence"
    compile(block.group(1), "<run_all alternative>", "exec")


# --------------------------------------------------------------- form hint hygiene


def test_the_configuration_hints_only_name_values_the_model_accepts() -> None:
    """Cell 1 is a form: its comments are the only documentation most runs read.

    A hint that names a value the schema rejects sends someone to a traceback they
    blame on the pipeline, and a hint that omits a legal value makes a feature
    undiscoverable. Both are invisible to every other test here, because the comment
    is never executed.
    """
    from typing import get_args

    from linescout_ml.colab.config import SourceSpec

    legal = {
        name: {str(arg) for arg in get_args(SourceSpec.model_fields[name].annotation)}
        for name in ("sfw_method", "extractor", "work_grouping", "origin")
    }
    config = cell_titled("1 ·")
    checked = 0
    for match in re.finditer(
        r'"(\w+)": ".*?#.*?([a-z_][a-z0-9_+]*(?:\s*\|\s*[a-z_][a-z0-9_+]*)+)', config
    ):
        key, listing = match.group(1), match.group(2)
        if key not in legal:
            continue
        for token in (part.strip() for part in listing.split("|")):
            assert token in legal[key], f"cell 1 suggests {key}={token!r}, which is not a value"
            checked += 1
    assert checked >= 6, f"the hints stopped naming values at all ({checked} found)"

    # `manual` is the operator-asserted verdict and the only way to skip screening, so
    # it must stay discoverable in the same comment the gate's other options live in.
    assert "manual" in legal["sfw_method"]
    assert re.search(r'"sfw_method":.*#.*manual', config, re.S), (
        "the manual option went undocumented"
    )
