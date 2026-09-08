"""Tests for the notebook ↔ package ↔ lockfile ↔ docs audit.

``linescout-repro selfcheck`` is the check that keeps the Colab notebook honest
without a GPU: it reads the notebook, the environment spec, the checkpoint lock,
the package's public surface, and the docs, and reports what no longer agrees.

A snapshot of the files it reads is copied into ``tmp_path``, one file is broken
at a time, and the expected problem has to appear. That is also what makes the
clean-tree assertion below meaningful: it proves the audit is not simply vacuous.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from linescout_ml import colab
from linescout_ml.colab import repro, selfcheck
from linescout_ml.repro_cli import main as repro_main

#: Everything the audit reads, so a copy of these is a whole tree for its purposes.
SNAPSHOT = (
    selfcheck.NOTEBOOK_RELATIVE,
    selfcheck.ENVIRONMENT_RELATIVE,
    selfcheck.BASELINE_RELATIVE,
    selfcheck.LOCK_RELATIVE,
    selfcheck.PACKAGE_INIT_RELATIVE,
    selfcheck.PYPROJECT_RELATIVE,
    selfcheck.UV_LOCK_RELATIVE,
    *selfcheck.DOCUMENTED,
)


def clean_tree(tmp_path: Path) -> Path:
    """A repository-shaped copy of the files the audit reads."""
    root = tmp_path / "repo"
    real = repro.repo_root()
    for relative in SNAPSHOT:
        source = real / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copytree(
        real / "ml" / "linescout_ml",
        root / "ml" / "linescout_ml",
        ignore=shutil.ignore_patterns("__pycache__"),
        dirs_exist_ok=True,
    )
    return root


def patch_all(root: Path, relative: Path, old: str, new: str) -> None:
    """Like `patch`, for files where the assertion is "mentions it at all"."""
    path = root / relative
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{relative} no longer contains the text this test mutates"
    path.write_text(text.replace(old, new), encoding="utf-8")


def patch(root: Path, relative: Path, old: str, new: str) -> None:
    path = root / relative
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{relative} no longer contains the text this test mutates"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_notebook_cell(root: Path, match_in_cell: str, mutate: Any) -> None:
    path = root / selfcheck.NOTEBOOK_RELATIVE
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if match_in_cell in text:
            lines = mutate(text).split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
            break
    else:  # pragma: no cover - a missing cell is the failure this test wants
        raise AssertionError(f"no cell contains {match_in_cell!r}")
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------- the real tree


def test_the_repository_passes_its_own_audit() -> None:
    """The one place where "the notebook, the pins, and the docs agree" is decided."""
    assert colab.run_selfcheck() == []


def test_the_cli_reports_the_same_verdict_as_the_function(tmp_path: Path, capsys: Any) -> None:
    code = repro_main(["selfcheck", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload == {"problems": []}

    root = clean_tree(tmp_path)
    patch(root, selfcheck.ENVIRONMENT_RELATIVE, "numpy==2.4.6", "numpy==1.24.0")
    code = repro_main(["selfcheck", "--root", str(root)])
    printed = capsys.readouterr().out
    assert code == 1
    assert "consistency problem" in printed


# ------------------------------------------------------------------- what it catches


@pytest.mark.parametrize(
    ("pin", "expected"),
    [
        ("main", "not a full 40-hex commit"),
        ("abc123", "not a full 40-hex commit"),
        ("f" * 40, "!= repro.COLAB_PIN"),
    ],
)
def test_it_notices_the_notebook_drifting_from_the_package_pin(
    tmp_path: Path, pin: str, expected: str
) -> None:
    root = clean_tree(tmp_path)
    patch_notebook_cell(
        root,
        "# @title 1 · Run configuration",
        lambda text: text.replace('REPO_PIN = "' + "0" * 40 + '"', f'REPO_PIN = "{pin}"'),
    )
    problems = colab.run_selfcheck(root)
    assert any(expected in problem and "REPO_PIN" in problem for problem in problems), problems


def test_it_notices_a_pinned_package_that_the_lockfile_does_not_agree_with(
    tmp_path: Path,
) -> None:
    root = clean_tree(tmp_path)
    patch(root, selfcheck.ENVIRONMENT_RELATIVE, "numpy==2.4.6", "numpy==1.24.0")
    assert any("numpy==1.24.0" in problem for problem in colab.run_selfcheck(root))

    patch(root, selfcheck.ENVIRONMENT_RELATIVE, "numpy==1.24.0", "numpy")
    problems = colab.run_selfcheck(root)
    assert any("unpinned lines" in problem for problem in problems), problems


def test_it_keeps_torch_out_of_the_installable_groups(tmp_path: Path) -> None:
    """The one rule whose violation silently turns a GPU run into a CPU run."""
    root = clean_tree(tmp_path)
    spec = root / selfcheck.ENVIRONMENT_RELATIVE
    text = spec.read_text(encoding="utf-8")
    moved = text.replace("# group: base\npydantic", "# group: base\npydantic\ntorch==2.5.1")
    moved = moved.replace("torch==2.14.0\n", "")
    spec.write_text(moved, encoding="utf-8")

    problems = colab.run_selfcheck(root)
    assert any("torch must not be installable" in problem for problem in problems), problems


def test_it_notices_a_stage_the_checkpoint_lock_does_not_cover(tmp_path: Path) -> None:
    root = clean_tree(tmp_path)
    lock = root / selfcheck.LOCK_RELATIVE
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["artifacts"] = [item for item in payload["artifacts"] if item["group"] != "opennsfw2"]
    lock.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    problems = colab.run_selfcheck(root)
    assert any("SFW gate has no checkpoint group" in problem for problem in problems), problems

    # A truncated digest is a different failure: the lock stops being loadable,
    # because `Sha256` is a type and not a hope. Either way the audit says so.
    payload["artifacts"][0]["sha256"] = "deadbeef"
    lock.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    problems = colab.run_selfcheck(root)
    assert any("models.lock.json" in problem for problem in problems), problems


def test_it_notices_a_heavy_import_at_module_level(tmp_path: Path) -> None:
    """The rule that keeps `pytest` from needing a GPU.

    A module-level `import torch` in the package would make every CPU test either
    fail or silently skip, which is how a "lightweight" test suite stops being one.
    """
    root = clean_tree(tmp_path)
    module = root / "ml" / "linescout_ml" / "colab" / "measure.py"
    module.write_text("import torch\n" + module.read_text(encoding="utf-8"))
    problems = colab.run_selfcheck(root)
    assert any("measure.py" in problem and "at module level" in problem for problem in problems), (
        problems
    )


def test_it_notices_installing_before_the_sources_are_resolved(tmp_path: Path) -> None:
    """Order matters: an unresolved preset cannot know it needs the NSFW gate."""
    root = clean_tree(tmp_path)
    path = root / selfcheck.NOTEBOOK_RELATIVE
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    resolver = next(
        index for index, cell in enumerate(cells) if "resolve_sources(" in "".join(cell["source"])
    )
    installer = next(
        index
        for index, cell in enumerate(cells)
        if "pip(*PLAN.pip_specs)" in "".join(cell["source"])
    )
    assert resolver < installer, "the clean tree already has the wrong order"
    cells[installer], cells[resolver] = cells[resolver], cells[installer]
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    problems = colab.run_selfcheck(root)
    assert any("before resolving presets" in problem for problem in problems), problems


def test_it_notices_docs_that_describe_a_different_setup(tmp_path: Path) -> None:
    """Docs are checked because stale docs are how people end up with two setups.

    The sentence-level check is "does this file still mention the command CI runs",
    so the mutation removes all three mentions rather than the first.
    """
    root = clean_tree(tmp_path)
    patch_all(root, Path("ml/colab/README.md"), "uv sync --frozen", "pip install -e")
    problems = colab.run_selfcheck(root)
    assert any(
        "ml/colab/README.md" in problem and "not `uv sync --frozen`" in problem
        for problem in problems
    ), problems

    readme = root / "ml" / "colab" / "README.md"
    text = "".join(
        line
        for line in readme.read_text(encoding="utf-8").splitlines(keepends=True)
        if "requirements-colab.txt" not in line
    )
    readme.write_text(text, encoding="utf-8")
    problems = colab.run_selfcheck(root)
    assert any("pinned environment spec" in problem for problem in problems), problems


def test_it_notices_a_runtime_baseline_that_stopped_saying_what_it_measured(
    tmp_path: Path,
) -> None:
    root = clean_tree(tmp_path)
    baseline = root / selfcheck.BASELINE_RELATIVE
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload.pop("recorded_on")
    baseline.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    problems = colab.run_selfcheck(root)
    assert any("does not say what runtime" in problem for problem in problems), problems

    payload["recorded_on"] = "colab"
    payload.pop("packages")
    baseline.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    problems = colab.run_selfcheck(root)
    assert any("no 'packages' block" in problem for problem in problems), problems


def test_a_tree_that_is_not_the_repository_is_reported_as_such(tmp_path: Path) -> None:
    problems = colab.run_selfcheck(tmp_path)
    assert any("linescout_gpu_pipeline.ipynb is missing" in problem for problem in problems)
    assert any("ml/README.md is missing" in problem for problem in problems)
