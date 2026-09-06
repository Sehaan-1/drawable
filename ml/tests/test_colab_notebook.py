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
  exact code path the notebook uses.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pytest

from linescout_ml import colab
from linescout_ml.colab import read_manifest

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


def test_badge_points_at_this_notebook_in_this_repo() -> None:
    first = markdown_sources()[0]
    match = re.search(r"\((https://colab\.research\.google\.com/github/[^)]+)\)", first)
    assert match, "the Open-in-Colab badge disappeared"
    url = match.group(1)
    relative = NOTEBOOK_PATH.relative_to(REPO_ROOT).as_posix()
    assert url.startswith(COLAB_BADGE_PREFIX)
    assert url.endswith(f"Sehaan-1/drawable/blob/main/{relative}"), url


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


def test_notebook_never_reinstalls_torch() -> None:
    """Colab's CUDA build must survive dependency installation.

    Cell 2 pins the installed torch/torchvision versions into a constraints file
    instead of letting pip resolve them again. Installing torch by name would
    undo that, so the decision is encoded here rather than left in a comment.
    """
    installed = [spec for text in code_sources() for spec in pip_specs(text)]
    assert installed, "the notebook installs nothing — did the helper get renamed?"
    packages = {re.split(r"[<>=!~[; ]", spec, maxsplit=1)[0] for spec in installed}
    assert not packages & {"torch", "torchvision"}, sorted(packages)

    shell = [line for text in code_sources() for line in text.splitlines() if "pip install" in line]
    assert not [
        line for line in shell if re.search(r"(?:^|\s)torch(?:vision)?(?:[<>=!\s]|$)", line)
    ]
    assert "pin_torch" in code_sources()[1], "the torch-pinning helper disappeared"


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
    text = text.replace("/content/linescout-dryrun", str(tmp_path / "dryrun"))

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
        assert record.enabled is True


def test_dry_run_cell_output_matches_the_documented_claims(tmp_path: Path) -> None:
    """The prose promises a gallery tree; check the dry run really produces it."""
    text = code_sources()[notebook_index()].replace(
        "/content/linescout-dryrun", str(tmp_path / "dryrun")
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
        "originals/<asset_id>.png",
        "line_art/<asset_id>.png",
        "thumbnails/<asset_id>.png",
        "_pipeline/candidates.jsonl",
        "_pipeline/run_report.json",
    ):
        assert documented in prose, f"the overview no longer documents {documented}"


# --------------------------------------------------------------- config cell


def cell_titled(title: str) -> str:
    for text in code_sources():
        if f"@title {title}" in text:
            return text
    raise AssertionError(f"the notebook lost its {title!r} cell")


def config_namespace(tmp_path: Path, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything cell 4 expects to find already defined by cells 1 and 2."""
    return {
        "Path": Path,
        "SOURCES": sources,
        "SOURCES_ROOT": tmp_path / "sources",
        "GALLERY_ROOT": tmp_path / "gallery" / "2026.09.06-notebooktest",
        "INDEX_ROOT": tmp_path / "indexes" / "2026.09.06-notebooktest",
        "DATASET_VERSION": "2026.09.06-notebooktest",
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


def test_the_sources_cell_expands_presets_and_overrides(tmp_path: Path, capsys: Any) -> None:
    sources = [
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
    ]
    namespace = config_namespace(tmp_path, sources)
    exec(compile(strip_magics(cell_titled("4 ·")), "<sources cell>", "exec"), namespace)

    specs = namespace["source_specs"]
    assert [spec.name for spec in specs] == ["amateur_drawings", "manga109"]
    assert specs[0].root == tmp_path / "sources" / "amateur"
    assert specs[0].license_id == "cc-by-4.0"
    assert specs[0].work_grouping == "parent_dir"  # per-source override won
    assert specs[0].extractor == "none"  # the preset's own choice survived
    assert specs[1].root == tmp_path / "manga109"  # absolute "root" beats "folder"
    assert specs[1].extractor == "informative_drawings"

    config = namespace["CONFIG"]
    assert config.dataset_version == "2026.09.06-notebooktest"
    assert config.limit_per_source is None  # 0 means "no limit"
    assert config.embedders == ["dinov2_vits14"]
    assert config.label is False and config.embed is False

    printed = capsys.readouterr().out
    for preset in ("synthetic", "quickdraw", "amateur_drawings", "manga109", "met_openaccess"):
        assert preset in printed, f"cell 4 stopped advertising the {preset} preset"


@pytest.mark.parametrize("license_id", ["REPLACE-ME", "", "tbd", "  "])
def test_the_sources_cell_blocks_placeholder_licences(tmp_path: Path, license_id: str) -> None:
    """Provenance is the point of the manifest, so guesses are refused up front."""
    namespace = config_namespace(
        tmp_path,
        [{"preset": "safebooru", "folder": "safebooru", "license_id": license_id}],
    )
    with pytest.raises(SystemExit, match="license_id"):
        exec(compile(strip_magics(cell_titled("4 ·")), "<sources cell>", "exec"), namespace)


def test_the_sources_cell_rejects_an_unknown_preset(tmp_path: Path) -> None:
    namespace = config_namespace(tmp_path, [{"preset": "not_a_dataset", "license_id": "cc0-1.0"}])
    with pytest.raises(SystemExit, match="unknown source preset"):
        exec(compile(strip_magics(cell_titled("4 ·")), "<sources cell>", "exec"), namespace)


def test_the_sources_cell_explains_an_impossible_combination(tmp_path: Path) -> None:
    """A native-line-art source cannot also name an extractor; say so in English."""
    namespace = config_namespace(
        tmp_path,
        [
            {
                "preset": "quickdraw",  # native line art: extractor must stay "none"
                "license_id": "google-terms",
                "extractor": "anime2sketch",
            }
        ],
    )
    with pytest.raises(SystemExit, match="not a valid combination"):
        exec(compile(strip_magics(cell_titled("4 ·")), "<sources cell>", "exec"), namespace)


def test_the_sources_cell_rejects_an_empty_source_list(tmp_path: Path) -> None:
    namespace = config_namespace(tmp_path, [])
    with pytest.raises(SystemExit, match="SOURCES is empty"):
        exec(compile(strip_magics(cell_titled("4 ·")), "<sources cell>", "exec"), namespace)
