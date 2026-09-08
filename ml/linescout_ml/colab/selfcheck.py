"""Consistency audit: notebook, package, environment spec, lockfiles, and docs.

The notebook is a deliverable that CI cannot execute on a GPU, so every promise
between its cells and the rest of the repository is checked here as plain text:
the pinned commit the cells clone equals the pinned commit the package declares,
the packages the cells install are the packages the environment spec pins, the
environment spec's pins are the ones ``uv.lock`` resolved, the checkpoint lock
covers every model the default config loads, and the heavy modules stay lazily
imported so ``pytest`` never needs torch.

Every check returns strings rather than raising, so one run reports everything
that drifted instead of only the first thing. It is exposed as
``linescout-repro selfcheck`` (and therefore to ``npm run check:colab`` and CI)
and asserted to be empty by ``ml/tests/test_colab_selfcheck.py``.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path
from typing import Any

#: Module-level imports that must never appear outside a lazily loaded stage.
HEAVY_MODULES: tuple[str, ...] = ("torch", "torchvision", "open_clip", "controlnet_aux")
GPU_ONLY_MODULES: tuple[str, ...] = ("tensorflow", "opennsfw2")
NOTEBOOK_RELATIVE = Path("ml/colab/linescout_gpu_pipeline.ipynb")
ENVIRONMENT_RELATIVE = Path("ml/colab/requirements-colab.txt")
BASELINE_RELATIVE = Path("ml/colab/runtime-baseline.json")
LOCK_RELATIVE = Path("ml/linescout_ml/colab/models.lock.json")
PACKAGE_INIT_RELATIVE = Path("ml/linescout_ml/colab/__init__.py")
PYPROJECT_RELATIVE = Path("ml/pyproject.toml")
UV_LOCK_RELATIVE = Path("ml/uv.lock")
DOCUMENTED = (Path("README.md"), Path("ml/README.md"), Path("ml/colab/README.md"))
#: Groups whose pins must appear in the package's `[gpu]` extra.
GPU_GROUPS: tuple[str, ...] = ("extraction", "clip", "nsfw")
PIN_ASSIGNMENT = re.compile(r"^([A-Z_]+)\s*=\s*[\"']([0-9a-f]{40})[\"']", re.MULTILINE)
REPO_URL_ASSIGNMENT = re.compile(r"^REPO_URL\s*=\s*[\"']([^\"']+)[\"']", re.MULTILINE)
BADGE_URL = re.compile(
    r"https://colab\.research\.google\.com/github/([^/]+)/([^/]+)/blob/[^/]+/(\S+)"
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _strip_magics(text: str) -> str:
    kept = [line for line in text.splitlines() if not line.lstrip().startswith(("%", "!"))]
    return "\n".join(kept)


def notebook_cells(path: Path) -> list[dict[str, Any]]:
    """Parse the notebook, tolerating nothing: a broken file is itself a problem."""
    raw = _read_text(path)
    if not raw:
        return []
    try:
        notebook = json.loads(raw)
    except json.JSONDecodeError:
        return []
    cells = notebook.get("cells", [])
    return [cell for cell in cells if isinstance(cell, dict)] if isinstance(cells, list) else []


def cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def code_sources(path: Path) -> list[str]:
    return [cell_source(cell) for cell in notebook_cells(path) if cell.get("cell_type") == "code"]


def markdown_sources(path: Path) -> list[str]:
    return [
        cell_source(cell) for cell in notebook_cells(path) if cell.get("cell_type") == "markdown"
    ]


def _assign(text: str, name: str) -> str | None:
    match = re.search(rf"^{name}\s*=\s*[\"']([^\"']*)[\"']", text, re.MULTILINE)
    return match.group(1) if match else None


# --------------------------------------------------------------------- individual checks


#: Anything that reaches pip, whether a helper call or a shell line.
INSTALL_CALL = re.compile(r"\bpip\(|!pip install|pip install")


def check_notebook_parity(root: Path, problems: list[str]) -> None:
    """The notebook and the package must declare the same reproducibility contract."""
    from linescout_ml.colab import repro

    notebook = root / NOTEBOOK_RELATIVE
    if not notebook.is_file():
        problems.append(f"{NOTEBOOK_RELATIVE} is missing")
        return
    cells = code_sources(notebook)
    if not cells:
        problems.append(f"{NOTEBOOK_RELATIVE} has no readable cells")
        return
    joined = "\n".join(cells)

    config = cells[0] if "@title 1 ·" in cells[0] else next(iter(cells), "")
    pinned = _assign(config, "REPO_PIN") or _assign(joined, "REPO_PIN")
    if pinned is None:
        problems.append("the notebook no longer declares REPO_PIN")
    elif not repro.is_immutable_revision(pinned):
        problems.append(
            f"REPO_PIN {pinned!r} is not a full 40-hex commit; branches and tags move, "
            "so a dataset built from one cannot be re-created"
        )
    elif pinned != repro.COLAB_PIN:
        problems.append(f"REPO_PIN {pinned[:12]} != repro.COLAB_PIN {repro.COLAB_PIN[:12]}")

    url = _assign(config, "REPO_URL") or _assign(joined, "REPO_URL")
    if url != repro.COLAB_REPO_URL:
        problems.append(
            f"notebook REPO_URL {url!r} != repro.COLAB_REPO_URL {repro.COLAB_REPO_URL!r}"
        )

    for position, text in enumerate(cells):
        try:
            compile(_strip_magics(text), f"<cell {position}>", "exec")
        except SyntaxError as error:
            problems.append(f"notebook cell {position} does not compile: {error.msg}")

    # The preflight must come after resolution: an unresolved preset cannot know
    # it needs the NSFW classifier.
    resolver = [index for index, text in enumerate(cells) if "resolve_sources(" in text]
    installer = [
        index
        for index, text in enumerate(cells)
        if "install_dependencies(" in text or INSTALL_CALL.search(text)
    ]
    if not resolver:
        problems.append(
            "no notebook cell calls resolve_sources(); presets are being expanded inline"
        )
    elif installer and min(installer) < min(resolver):
        problems.append(
            "the notebook installs dependencies before resolving presets into SourceSpecs"
        )

    for name in sorted(_imported_names(cells, "linescout_ml.colab")):
        if name not in repro_public_names(root):
            problems.append(f"notebook imports linescout_ml.colab.{name}, which is not exported")

    badges = {match.group(1) for match in BADGE_URL.finditer(_read_text(notebook))}
    if badges and badges != {repro.COLAB_REPO.split("/")[0]}:
        problems.append(f"the notebook's Colab badge(s) point at {sorted(badges)}")


def _imported_names(cells: list[str], module: str) -> set[str]:
    found: set[str] = set()
    for text in cells:
        try:
            tree = ast.parse(_strip_magics(text))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == module:
                found.update(alias.name for alias in node.names)
    return found


def repro_public_names(root: Path) -> set[str]:
    """Names ``ml/linescout_ml/colab/__init__.py`` exports, parsed without importing."""
    text = _read_text(root / PACKAGE_INIT_RELATIVE)
    match = re.search(r"__all__ = \[(.*?)\]", text, re.DOTALL)
    if not match:
        return set()
    return {name.strip().strip('",') for name in match.group(1).splitlines() if name.strip()}


def check_environment_spec(root: Path, problems: list[str]) -> None:
    """Pinned, complete, and identical to what ``uv.lock`` resolved.

    A separate requirements file rots the moment someone adds a dependency to the
    notebook. Requiring it to agree with the lockfile and with the package's own
    ``[gpu]`` extra means an unnoticed edit fails CI instead of quietly making the
    Colab runtime and the developer runtime differ.
    """
    from linescout_ml.colab import repro

    spec_path = root / ENVIRONMENT_RELATIVE
    if not spec_path.is_file():
        problems.append(f"{ENVIRONMENT_RELATIVE} is missing")
        return
    spec = repro.load_requirement_spec(spec_path)
    if spec is None:  # pragma: no cover - guarded by is_file()
        problems.append(f"{ENVIRONMENT_RELATIVE} could not be parsed")
        return
    if not spec.pins:
        problems.append(f"{ENVIRONMENT_RELATIVE} pins nothing")
        return

    unpinned = [
        line.strip()
        for line in spec_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "==" not in line
    ]
    if unpinned:
        problems.append(f"{ENVIRONMENT_RELATIVE} has unpinned lines: {unpinned[:3]}")

    locked = repro.lockfile_versions(root / UV_LOCK_RELATIVE)
    for name, version in sorted(spec.pins.items()):
        versions = locked.get(name)
        if versions is None:
            problems.append(f"{name}=={version} is not in {UV_LOCK_RELATIVE}")
        elif version not in versions:
            problems.append(
                f"{name}=={version} disagrees with {UV_LOCK_RELATIVE} ({', '.join(versions)})"
            )

    gpu_names = _gpu_extra_names(root / PYPROJECT_RELATIVE)
    for group in GPU_GROUPS:
        for name in spec.groups.get(group, ()):
            if name not in gpu_names:
                problems.append(
                    f"{name} is pinned for Colab ({group}) but absent from the [gpu] extra"
                )

    installable = set(spec.groups_for(group for group in spec.groups if group != "runtime"))
    for name in ("torch", "torchvision"):
        if name in installable:
            problems.append(f"{name} must not be installable from {ENVIRONMENT_RELATIVE}")

    notebook_install = "\n".join(code_sources(root / NOTEBOOK_RELATIVE))
    if "load_requirement_spec" not in notebook_install and "requirements-colab.txt" not in (
        notebook_install
    ):
        problems.append("the notebook no longer installs from the pinned environment spec")


def _gpu_extra_names(pyproject: Path) -> set[str]:
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    requirements = data.get("project", {}).get("optional-dependencies", {}).get("gpu", [])
    names: set[str] = set()
    for item in requirements:
        match = re.match(r"^([A-Za-z0-9_.-]+)", str(item).strip())
        if match:
            names.add(match.group(1).lower())
    return names


def check_checkpoint_lock(root: Path, problems: list[str]) -> None:
    """The lock must cover what the default config loads, with immutable refs."""
    from linescout_ml.colab.checkpoints import CheckpointError, load_model_lock
    from linescout_ml.colab.config import PipelineConfig, SourceSpec
    from linescout_ml.colab.extract import EXTRACTOR_SPECS
    from linescout_ml.colab.models import MODEL_CARDS

    try:
        lock = load_model_lock(root / LOCK_RELATIVE)
    except (CheckpointError, OSError) as error:
        problems.append(f"{LOCK_RELATIVE}: {error}")
        return

    seen: set[str] = set()
    for artifact in lock.artifacts:
        if artifact.id in seen:
            problems.append(f"{LOCK_RELATIVE}: duplicate artifact id {artifact.id}")
        seen.add(artifact.id)
        too_short = artifact.revision is not None and not re.fullmatch(
            r"[0-9a-f]{40}", artifact.revision
        )
        if too_short and artifact.source != "url":
            shown = artifact.revision
            problems.append(f"{artifact.id}: revision {shown!r} is not a 40-hex commit")
        if artifact.sha256 is not None and not re.fullmatch(r"[a-f0-9]{64}", artifact.sha256):
            problems.append(f"{artifact.id}: sha256 is not 64 hex characters")
        if (
            artifact.source == "huggingface"
            and artifact.sha256 is None
            and artifact.size_bytes is None
        ):
            problems.append(f"{artifact.id}: neither a digest nor a size — nothing to verify")

    config = PipelineConfig(
        dataset_version="2026.09.06-selfcheck",
        output_root=Path("/tmp/gallery"),
        sources=[SourceSpec.model_validate(_minimal_source())],
    )
    for key in config.embedders:
        card = MODEL_CARDS.get(key)
        if card is None:
            problems.append(f"default embedder {key!r} has no model card")
        elif not lock.for_group(card.artifacts):
            problems.append(f"default embedder {key!r} has no checkpoint group in {LOCK_RELATIVE}")
    for extractor_key, spec in EXTRACTOR_SPECS.items():
        if not lock.for_group(spec.checkpoint_group):
            problems.append(
                f"extractor {extractor_key!r} has no checkpoint group in {LOCK_RELATIVE}"
            )
    if not lock.for_group("opennsfw2"):
        problems.append(f"the SFW gate has no checkpoint group in {LOCK_RELATIVE}")
    pinned = {artifact.id for artifact in lock.artifacts if artifact.pinned}
    for required in ("annotators/netG.pth", "annotators/sk_model.pth", "annotators/sk_model2.pth"):
        if required not in pinned:
            problems.append(f"{required} must carry a pinned sha256")


def _minimal_source() -> dict[str, Any]:
    return {
        "name": "selfcheck",
        "root": "/tmp/sources",
        "license_id": "CC0-1.0",
        "origin": "native_line_art",
        "extractor": "none",
    }


def check_lazy_imports(root: Path, problems: list[str]) -> None:
    """Keep the GPU stack behind a function call, so tests stay lightweight."""
    package = root / "ml" / "linescout_ml"
    for module in sorted(package.rglob("*.py")):
        if not module.is_file():
            continue
        try:
            tree = ast.parse(module.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:  # pragma: no cover - broken file, other check fires
            problems.append(f"{module}: unreadable ({error})")
            continue
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                head = name.split(".")[0]
                if head in HEAVY_MODULES or head in GPU_ONLY_MODULES:
                    problems.append(
                        f"{module.relative_to(root)} imports {name} at module level; "
                        "resolve it through linescout_ml.colab._optional instead"
                    )


def check_documentation(root: Path, problems: list[str]) -> None:
    """The docs must describe the setup CI actually runs."""
    from linescout_ml.colab import repro

    for relative in DOCUMENTED:
        text = _read_text(root / relative)
        if not text:
            problems.append(f"{relative} is missing")
            continue
        if "uv sync --frozen" not in text and "setup:py" not in text:
            problems.append(f"{relative} documents a Python setup that is not `uv sync --frozen`")
    colab_readme = _read_text(root / "ml/colab/README.md")
    for expected, where in (
        (str(ENVIRONMENT_RELATIVE), "the pinned environment spec"),
        (str(LOCK_RELATIVE), "the checkpoint lock"),
        (repro.COLAB_REPO, "the canonical repository"),
        (repro.COLAB_PIN[:12], "the pinned commit"),
    ):
        if expected not in colab_readme:
            problems.append(f"ml/colab/README.md no longer mentions {where} ({expected})")
    if "uv pip install -e" in colab_readme or "uv pip install -e" in _read_text(
        root / "ml/README.md"
    ):
        problems.append("the docs still tell people to `uv pip install -e`, bypassing the lockfile")
    for relative in DOCUMENTED:
        text = _read_text(root / relative)
        for match in BADGE_URL.finditer(text):
            if match.group(1) != repro.COLAB_REPO.split("/")[0]:
                problems.append(
                    f"{relative} links the Colab badge at {match.group(1)}, "
                    f"not {repro.COLAB_REPO.split('/')[0]}"
                )


def check_runtime_baseline(root: Path, problems: list[str]) -> None:
    """A recorded baseline must be honest about what it recorded."""
    path = root / BASELINE_RELATIVE
    if not path.is_file():
        problems.append(f"{BASELINE_RELATIVE} is missing")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        problems.append(f"{BASELINE_RELATIVE} is not valid JSON: {error}")
        return
    if not isinstance(payload, dict) or "packages" not in payload:
        problems.append(f"{BASELINE_RELATIVE} has no 'packages' block")
        return
    if not payload.get("recorded_on"):
        problems.append(f"{BASELINE_RELATIVE} does not say what runtime it was recorded on")


def run_selfcheck(root: Path | None = None) -> list[str]:
    """Every consistency check, as a list of human-readable problems."""
    base = Path(root) if root is not None else _default_root()
    problems: list[str] = []
    check_notebook_parity(base, problems)
    check_environment_spec(base, problems)
    check_checkpoint_lock(base, problems)
    check_lazy_imports(base, problems)
    check_documentation(base, problems)
    check_runtime_baseline(base, problems)
    return problems


def _default_root() -> Path:
    from linescout_ml.colab.repro import repo_root

    return repo_root()


__all__ = ["run_selfcheck", "HEAVY_MODULES", "ENVIRONMENT_RELATIVE", "LOCK_RELATIVE"]
