"""Reproducibility plumbing: which code ran, which packages it ran on, what to install.

Three jobs, all of them answerable without a GPU:

* **Pinned checkout** — :func:`ensure_checkout` clones (or reuses) the repository
  at *one immutable commit*, verifies HEAD, and refuses to quietly move a dirty
  checkout. The result is recorded in the run report and the manifest, so a
  dataset says which source tree produced it rather than "whatever was on main
  that day".
* **Pinned environment** — :func:`load_requirement_spec` reads
  ``ml/colab/requirements-colab.txt``, the single list of package versions the
  pipeline runs on, and :func:`plan_environment` turns the *resolved* sources and
  stage toggles into the subset of it a run actually needs.
* **Recorded runtime** — :func:`runtime_report` reads the versions that are
  actually importable — including Colab's preinstalled torch and CUDA — and
  :func:`compare_runtime` reports drift against a recorded baseline. Nothing here
  assumes the runtime matches the spec; the point is to notice when it does not.

torch is never imported by this module (only *reported* on when something else
already imported it), so importing it stays cheap and CI never needs a GPU stack.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from linescout_ml.colab.config import SourceSpec
from linescout_ml.taxonomy import LineArtOrigin

#: The canonical repository the notebook clones.
COLAB_REPO = "junosapollo/drawable"
COLAB_REPO_URL = "https://github.com/junosapollo/drawable.git"
#: Read-only fallbacks, tried in order. Mirrors are safe *only* because every
#: attempt is checked against :data:`COLAB_PIN`: a mirror cannot smuggle in
#: different code, since a different commit is a different SHA.
COLAB_MIRROR_URLS: tuple[str, ...] = ("https://github.com/Sehaan-1/drawable.git",)
#: The finalized pipeline commit. Bumping this is a reviewed change: run
#: ``linescout-repro selfcheck`` and the notebook's tests to see what moved, and
#: re-run the documented Colab smoke test before landing a new pin.
#:
#: The commit must be *reachable from a ref in the canonical repository* before a
#: Colab runtime can fetch it; a pin that only exists on an unpushed local branch
#: fails in a notebook, not in CI. A pinned commit that resolves only because a fork
#: shares the parent's object storage is one `git remote remove` away from being
#: unreachable, which is the difference between this pin and a permanent one.
#: Verified with: ``linescout-repro checkout --dir /tmp/x --rev <COLAB_PIN>``.
COLAB_PIN = "81a8683a53ea9e1a9838925968bd6b3020a3d102"

#: A revision must be a full commit hash. Abbreviated hashes and branch names are
#: exactly the mutable refs reproducibility is about avoiding.
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
#: What identifies a usable checkout of this project.
CHECKOUT_MARKER = ("ml", "linescout_ml", "colab", "repro.py")
#: Environment spec, relative to the repository root.
ENVIRONMENT_SPEC_PATH = ("ml", "colab", "requirements-colab.txt")
RUNTIME_BASELINE_PATH = ("ml", "colab", "runtime-baseline.json")

#: Packages whose import name differs from their distribution name.
MODULE_NAMES: dict[str, str] = {
    "pillow": "PIL",
    "opencv-python-headless": "cv2",
    "scikit-image": "skimage",
    "huggingface-hub": "huggingface_hub",
    "open-clip-torch": "open_clip",
    "importlib-metadata": "importlib.metadata",
}
#: Installed from the spec, never by name at runtime: Colab's own build must win.
RUNTIME_ONLY_GROUP = "runtime"
#: Progress bars and previews: useful in a notebook, irrelevant to a headless run.
DISPLAY_GROUP = "display"
BASE_GROUP = "base"

#: What :func:`runtime_report` records. TensorFlow is here because opennsfw2
#: rides on it and Colab's copy is part of the runtime's identity.
RUNTIME_PACKAGES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "numpy",
    "pillow",
    "pydantic",
    "controlnet-aux",
    "open-clip-torch",
    "timm",
    "opennsfw2",
    "tensorflow",
    "tf-keras",
    "keras",
    "gdown",
    "huggingface-hub",
    "safetensors",
    "einops",
    "opencv-python-headless",
    "scikit-image",
    "scipy",
    "matplotlib",
    "tqdm",
)

GIT_TIMEOUT_SECONDS = 900


class CheckoutError(RuntimeError):
    """The repository could not be cloned, verified, or reused."""


class RevisionMismatchError(CheckoutError):
    """HEAD is not the pinned revision — the run would not be reproducible."""


# --------------------------------------------------------------------- git plumbing


def _git(args: Sequence[str], cwd: Path | None = None, *, check: bool = True) -> str:
    """Run git, returning trimmed stdout. ``check=False`` swallows a failure."""
    if shutil.which("git") is None:  # pragma: no cover - git is on every Colab image
        msg = "git is not installed; the pinned checkout cannot be verified without it"
        raise CheckoutError(msg)
    command = ["git", *(str(arg) for arg in args)]
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0 and check:
        msg = f"`{' '.join(command)}` failed ({result.returncode}): {result.stderr.strip()}"
        raise CheckoutError(msg)
    return result.stdout.strip()


#: The value :data:`COLAB_PIN` holds until the finalized pipeline is committed.
#: A placeholder is a *format*-valid SHA that no repository contains, so the only
#: honest handling is to name it out loud wherever the pin is printed.
COLAB_PIN_PLACEHOLDER = "0" * 40


def pin_is_finalised(pin: object) -> bool:
    """Whether ``pin`` is a real commit rather than the placeholder."""
    return (
        isinstance(pin, str)
        and is_immutable_revision(pin.strip())
        and pin.strip() != COLAB_PIN_PLACEHOLDER
    )


def is_immutable_revision(ref: object) -> bool:
    """Whether ``ref`` is a full 40-hex commit hash (not a branch, tag, or short sha)."""
    return isinstance(ref, str) and IMMUTABLE_REVISION.match(ref) is not None


def require_immutable_revision(ref: object, *, field: str = "REPO_PIN") -> str:
    """Validate a pin, with the reason it is not allowed to be a branch."""
    if ref is None or (isinstance(ref, str) and not ref.strip()):
        msg = (
            f"{field} is empty. Point it at an immutable commit — the full 40-hex SHA of a "
            f"revision that contains the pipeline (the default is {COLAB_PIN})."
        )
        raise CheckoutError(msg)
    text = str(ref).strip()
    if not IMMUTABLE_REVISION.match(text):
        msg = (
            f"{field} must be a full 40-hex commit SHA, got {text!r}. Branches and tags move; "
            "a dataset recorded as produced by `main` cannot be re-created later."
        )
        raise CheckoutError(msg)
    return text


def repo_is_here(path: Path) -> bool:
    """Whether ``path`` holds a checkout of this project."""
    marker = Path(path)
    for part in CHECKOUT_MARKER:
        marker = marker / part
    return marker.is_file()


def candidate_repo_dirs(*, drive_root: Path | str | None = None) -> list[Path]:
    """Where a Colab runtime usually already has a copy of the repository."""
    candidates: list[Path] = []
    override = __import__("os").environ.get("LINESCOUT_REPO")
    if override:
        candidates.append(Path(override))
    if drive_root:
        candidates.append(Path(drive_root).parent / "drawable")
    candidates.append(Path("/content/drawable"))
    return candidates


def untracked_count(path: Path) -> int:
    """How many untracked files the checkout carries (not counted as dirty)."""
    out = _git(["status", "--porcelain", "--untracked-files=all"], cwd=path, check=False)
    return sum(1 for line in out.splitlines() if line.startswith("??"))


@dataclass(frozen=True)
class CheckoutRecord:
    """What a checkout actually is — recorded next to every run report."""

    path: str
    url: str | None
    requested_revision: str
    revision: str | None
    short_revision: str
    branch: str | None
    detached: bool
    #: Tracked files differ from HEAD. Recorded, and never silently reset.
    dirty: bool
    untracked: int
    action: Literal["cloned", "reused", "fetched", "failed"]
    #: HEAD equals the requested revision after this call.
    verified: bool
    #: Which URL supplied the bytes (the canonical repo or a verified mirror).
    source_url: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def repo(self) -> str | None:
        """``owner/name`` of the repository these bytes came from.

        Derived from the remote that actually served the fetch, not from a
        constant: a run against a fork should say so in the report. Anything that
        is not a GitHub URL — a local path in a test, a server on a LAN — yields
        ``None`` rather than a claim the report cannot support.
        """
        for candidate in (self.source_url, self.url):
            if not candidate or "github.com" not in candidate:
                continue
            tail = candidate.split("github.com/")[-1]
            repo = tail.removesuffix(".git").strip("/").removeprefix("/")
            if repo.count("/") == 1:
                return repo
        return None

    @property
    def matches_pin(self) -> bool:
        return self.revision is not None and self.revision == self.requested_revision

    @property
    def usable(self) -> bool:
        return self.verified and self.revision is not None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        payload["repo"] = self.repo
        return payload

    def describe(self) -> list[str]:
        """Lines for the notebook to print; the dirty state must not be subtle."""
        lines = [
            f"repository   : {self.path}",
            f"pinned at    : {self.requested_revision}",
            f"HEAD         : {self.short_revision or 'unknown'}"
            f"{'  (matches the pin)' if self.matches_pin else '  (does NOT match the pin)'}",
            f"branch       : {self.branch or 'detached'}{'  [detached]' if self.detached else ''}",
            f"source       : {self.source_url or self.url or 'existing checkout'}  [{self.action}]",
            f"working tree : {'DIRTY' if self.dirty else 'clean'} ({self.untracked} untracked)",
        ]
        lines.extend(f"  note: {note}" for note in self.notes)
        return lines


def read_checkout(path: Path, *, requested_revision: str, action: str = "reused") -> CheckoutRecord:
    """Collect the facts about an existing checkout, without changing it."""
    path = Path(path)
    revision = _git(["rev-parse", "HEAD"], cwd=path, check=False) or None
    branch = _git(["symbolic-ref", "--quiet", "--short", "HEAD"], cwd=path, check=False) or None
    status = _git(["status", "--porcelain"], cwd=path, check=False)
    url = _git(["remote", "get-url", "origin"], cwd=path, check=False) or None
    dirty = bool(status.strip())
    return CheckoutRecord(
        path=str(path),
        url=url,
        requested_revision=requested_revision,
        revision=revision or None,
        short_revision=(revision or "")[:12],
        branch=branch,
        detached=branch is None,
        dirty=dirty,
        untracked=sum(1 for line in status.splitlines() if line.startswith("??")),
        action=action,  # type: ignore[arg-type]
        verified=bool(revision) and revision == requested_revision,
    )


def _clean_target(target: Path) -> None:
    """Remove a partially cloned directory. Never touches a directory with content."""
    entries = [entry for entry in target.iterdir()] if target.is_dir() else []
    if entries and not (entries[0].name == ".git" and len(entries) == 1):
        return
    shutil.rmtree(target, ignore_errors=True)


def ensure_checkout(
    target: Path,
    *,
    revision: str,
    url: str = COLAB_REPO_URL,
    mirrors: Sequence[str] = COLAB_MIRROR_URLS,
    fetch: bool = True,
    allow_dirty: bool = True,
    on_note: Callable[[str], None] | None = None,
) -> CheckoutRecord:
    """Get ``target`` to *exactly* ``revision``, or refuse.

    * No checkout there → clone from ``url``, then each mirror, fetching only the
      pinned commit (``--depth 1``), and verify HEAD equals the pin.
    * A checkout on the right revision → reuse it; if the working tree is dirty,
      that is recorded, because the report must describe what ran. Pass
      ``allow_dirty=False`` to refuse it instead of merely marking it.
    * A checkout on the wrong revision → fetch the pin and detach onto it, but
      **only if the tree is clean**. Moving a dirty checkout would destroy edits
      the pipeline cannot judge, so that case fails with instructions instead.
    """
    note = on_note or (lambda message: None)
    requested = require_immutable_revision(revision)
    target = Path(target)
    attempts: list[str] = []

    if (target / ".git").is_dir():
        record = read_checkout(target, requested_revision=requested)
        if record.verified:
            if record.dirty and not allow_dirty:
                msg = (
                    f"{target} is at the pinned revision but has uncommitted changes, and this "
                    "run is configured to refuse a dirty tree. Commit or stash them, or set "
                    "REPO_ALLOW_DIRTY back to True to keep the run and record it as dirty."
                )
                raise CheckoutError(msg)
            if record.dirty:
                message = (
                    "reusing an existing checkout with local edits; the run report records "
                    f"revision {record.short_revision} as dirty — push these changes and bump "
                    "REPO_PIN for a reproducible dataset"
                )
                note(message)
                return replace(record, notes=(*record.notes, message))
            return record
        if record.dirty:
            msg = (
                f"{target} is at {record.short_revision} (dirty) but the run is pinned to "
                f"{requested}. Refusing to move a checkout with uncommitted edits.\n"
                "  · commit or stash your changes and re-run, or\n"
                "  · point REPO_DIR at a clean checkout, or\n"
                "  · delete this checkout to let the notebook clone the pinned revision."
            )
            raise RevisionMismatchError(msg)
        if not fetch:
            msg = (
                f"{target} is at {record.short_revision}, the run is pinned to {requested}, and "
                "auto-fetch is off; re-run with fetch enabled or check the revision out yourself."
            )
            raise RevisionMismatchError(msg)
        note(f"existing checkout is at {record.short_revision}; fetching {requested[:12]}")
        for candidate in [url, *mirrors]:
            attempts.append(candidate)
            if (
                _git(
                    ["fetch", "--quiet", "--depth", "1", candidate, requested],
                    cwd=target,
                    check=False,
                )
                or _git(["rev-parse", "--verify", "FETCH_HEAD^{commit}"], cwd=target, check=False)
                != requested
            ):
                continue
            _git(["checkout", "--quiet", "--detach", requested], cwd=target)
            fetched = read_checkout(target, requested_revision=requested, action="fetched")
            if not fetched.verified:  # pragma: no cover - belt and braces
                msg = (
                    f"checkout drifted after fetching {requested}: HEAD is {fetched.short_revision}"
                )
                raise RevisionMismatchError(msg)
            return fetched
        hints = [
            f"could not fetch {requested} from any of: {', '.join(attempts)}. The revision may "
            "not be published yet — set REPO_URL to the repository that carries it."
        ]
        if not pin_is_finalised(requested):
            hints.append(
                f"Note: {requested} is the placeholder pin. ml/colab/README.md explains which "
                "commit to bump REPO_PIN and COLAB_PIN to."
            )
        raise RevisionMismatchError("\n".join(hints))

    if target.exists() and any(target.iterdir()):
        msg = (
            f"{target} exists, is not a git checkout, and is not empty — refusing to clone "
            "into it. Point REPO_DIR at a different path."
        )
        raise CheckoutError(msg)

    for candidate in [url, *mirrors]:
        attempts.append(candidate)
        _clean_target(target)
        target.mkdir(parents=True, exist_ok=True)
        try:
            _git(["init", "--quiet", str(target)])
            _git(["remote", "add", "origin", candidate], cwd=target)
            _git(["fetch", "--quiet", "--depth", "1", "origin", requested], cwd=target)
            _git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=target)
        except CheckoutError as error:
            note(f"{candidate}: {error}")
            _clean_target(target)
            continue
        record = read_checkout(target, requested_revision=requested, action="cloned")
        if not record.verified:
            _clean_target(target)
            msg = (
                f"{candidate} answered with {record.short_revision}, not the pinned "
                f"{requested}; refusing to run code that is not the reviewed revision"
            )
            raise RevisionMismatchError(msg)
        notes: list[str] = []
        if candidate != url:
            message = f"pinned commit came from the mirror {candidate} (content verified by SHA)"
            note(message)
            notes.append(message)
        return replace(record, source_url=candidate, notes=tuple(notes))

    problem = (
        f"could not obtain {requested[:12]} from {', '.join(attempts)}. Check that the revision "
        "has been pushed and that REPO_URL points at a repository that carries it."
    )
    if not pin_is_finalised(requested):
        problem += (
            f"\nNote: {requested} is this repository's placeholder pin — ml/colab/README.md "
            "explains which commit to bump REPO_PIN and COLAB_PIN to."
        )
    raise CheckoutError(problem)


def find_or_checkout(
    *,
    revision: str,
    url: str = COLAB_REPO_URL,
    mirrors: Sequence[str] = COLAB_MIRROR_URLS,
    preferred: Iterable[Path | str] = (),
    target: Path | str = Path("/content/drawable"),
    allow_dirty: bool = True,
    on_note: Callable[[str], None] | None = None,
) -> CheckoutRecord:
    """Reuse a checkout that already has this project, else clone into ``target``.

    ``preferred`` is where a Colab runtime keeps copies that survive a restart —
    a Drive folder, typically. An existing copy is *verified*, not trusted: a
    wrong or dirty checkout is reported or refused by :func:`ensure_checkout`.
    """
    requested = require_immutable_revision(revision)
    for candidate in preferred:
        path = Path(candidate)
        if (path / ".git").is_dir():
            return ensure_checkout(
                path,
                revision=requested,
                url=url,
                mirrors=mirrors,
                allow_dirty=allow_dirty,
                on_note=on_note,
            )
    return ensure_checkout(
        Path(target),
        revision=requested,
        url=url,
        mirrors=mirrors,
        allow_dirty=allow_dirty,
        on_note=on_note,
    )


# ------------------------------------------------------------------- environment spec


@dataclass(frozen=True)
class RequirementSpec:
    """The parsed ``requirements-colab.txt``."""

    path: str
    #: ``package -> pinned version`` for every line in the file.
    pins: Mapping[str, str]
    #: ``group -> packages`` in file order.
    groups: Mapping[str, tuple[str, ...]]
    #: Packages that must never be installed from here (Colab owns them).
    runtime_only: tuple[str, ...] = ()
    #: SHA-256 of the spec file's bytes: name this in a run report and the exact
    #: set of pins is re-derivable, which is the whole point of having a spec.
    sha256: str | None = None

    def specs(self, packages: Iterable[str]) -> list[str]:
        return [f"{name}=={self.pins[name]}" for name in sorted(set(packages)) if name in self.pins]

    def groups_for(self, groups: Iterable[str]) -> list[str]:
        selected: list[str] = []
        for group in groups:
            selected.extend(self.groups.get(group, ()))
        return sorted(dict.fromkeys(selected))

    def summary(self) -> list[dict[str, str]]:
        return [
            {"group": group, "package": package, "version": self.pins[package]}
            for group, packages in self.groups.items()
            for package in packages
        ]


_REQUIREMENT_LINE = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^;\s]+)")
_GROUP_LINE = re.compile(r"^#\s*group:\s*(?P<group>[a-z0-9_-]+)\s*$")


def parse_requirements(text: str, *, path: str = "<inline>") -> RequirementSpec:
    """Parse ``# group:`` sections of ``name==version`` pins.

    A pin in a requirements file is meaningless if it can silently disappear, so
    unparsed content is *not* an error but an unpinned package is: only exact
    ``==`` lines are accepted, and anything else is skipped by design (comments).
    """
    pins: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    current = BASE_GROUP
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        group = _GROUP_LINE.match(stripped)
        if group:
            current = str(group.group("group"))
            groups.setdefault(current, [])
            continue
        match = _REQUIREMENT_LINE.match(stripped)
        if match is None:
            continue
        name = match.group("name").lower()
        version = match.group("version")
        pins[name] = version
        groups.setdefault(current, []).append(name)
    return RequirementSpec(
        path=path,
        pins=pins,
        groups={key: tuple(value) for key, value in groups.items()},
        runtime_only=tuple(groups.get(RUNTIME_ONLY_GROUP, ())),
    )


def repo_root(start: Path | None = None) -> Path:
    """The repository root, found by walking up from ``start``.

    Installed from a wheel there is no repository to find, so the caller's path
    is returned unchanged and every reader treats a missing file as "no spec".
    """
    current = Path(start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "ml" / "colab" / "requirements-colab.txt").is_file():
            return candidate
        if (candidate / "ml" / "pyproject.toml").is_file():
            return candidate
    return current


def environment_spec_path(root: Path | None = None) -> Path:
    return Path(root or repo_root()) / Path(*ENVIRONMENT_SPEC_PATH)


def runtime_baseline_path(root: Path | None = None) -> Path:
    return Path(root or repo_root()) / Path(*RUNTIME_BASELINE_PATH)


def load_requirement_spec(path: Path | None = None) -> RequirementSpec | None:
    """The pinned environment, or ``None`` when the file is genuinely absent."""
    candidate = Path(path) if path is not None else environment_spec_path()
    if not candidate.is_file():
        return None
    spec = parse_requirements(candidate.read_text(encoding="utf-8"), path=str(candidate))
    return replace(spec, sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())


def environment_digest(path: Path | None = None) -> dict[str, Any]:
    """SHA-256 of the environment spec, so a run names the pins it used."""
    candidate = Path(path) if path is not None else environment_spec_path()
    if not candidate.is_file():
        return {"path": str(candidate), "sha256": None}
    payload = candidate.read_bytes()
    return {"path": str(candidate), "sha256": hashlib.sha256(payload).hexdigest()}


def lockfile_versions(path: Path) -> dict[str, list[str]]:
    """``package -> [versions]`` from a ``uv.lock``.

    Used by the consistency checks to prove the Colab pins and the committed
    lockfile still agree, which is what makes "reinstall it later" mean something.
    """
    versions: dict[str, list[str]] = {}
    if not Path(path).is_file():
        return versions
    name: str | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("name = "):
            name = stripped.split("=", 1)[1].strip().strip('"').lower()
        elif stripped.startswith("version = ") and name:
            versions.setdefault(name, []).append(stripped.split("=", 1)[1].strip().strip('"'))
            name = None
    return versions


# ------------------------------------------------------------------ the install plan


@dataclass(frozen=True)
class EnvironmentPlan:
    """What one run needs, derived from resolved sources and stage toggles."""

    groups: tuple[str, ...]
    packages: tuple[str, ...]
    specs: tuple[str, ...]
    reasons: Mapping[str, str]
    #: Packages that are importable already and therefore not installed.
    already_present: tuple[str, ...]
    #: Packages to hand to pip.
    to_install: tuple[str, ...]
    #: The spec file the pins came from, so a printout can name its source.
    spec: RequirementSpec | None = None

    def pip_args(self) -> list[str]:
        """Bare package names, for a caller that supplies its own versions."""
        return list(self.to_install)

    @property
    def pip_specs(self) -> tuple[str, ...]:
        """Pinned ``name==version`` specs for exactly the packages to install.

        Never a bare name: installing unpinned is how a runtime silently drifts
        away from the environment this pipeline was validated on.
        """
        wanted = set(self.to_install)
        return tuple(spec for spec in self.specs if spec.split("==", 1)[0] in wanted)

    def describe(self) -> list[str]:
        lines = [
            f"environment: {self.spec.path if self.spec else 'MISSING'}",
            f"groups: {', '.join(self.groups) or 'base only'}",
        ]
        for package in self.to_install:
            lines.append(
                f"  install {package} — {self.reasons.get(package, 'required by a stage')}"
            )
        for package in self.already_present:
            lines.append(f"  present {package}")
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "groups": list(self.groups),
            "packages": list(self.packages),
            "reasons": dict(self.reasons),
            "to_install": list(self.to_install),
            "already_present": list(self.already_present),
            "pip_specs": list(self.pip_specs),
            "environment_sha256": self.spec.sha256 if self.spec else None,
        }


def needs_detector(source: SourceSpec) -> bool:
    """Whether a *resolved* source will run a line-art extractor.

    Native line-art sources set ``extractor="none"`` and never load a detector,
    which is why this reads the spec rather than a form field: the preset's own
    choice is only visible after resolution.
    """
    return source.origin is LineArtOrigin.EXTRACTED and source.extractor != "none"


def plan_environment(
    sources: Sequence[SourceSpec],
    *,
    spec: RequirementSpec | None,
    extract_line_art: bool = True,
    label: bool = True,
    embed: bool = True,
    embedders: Sequence[str] = (),
) -> EnvironmentPlan:
    """Derive the pinned packages a run needs — from **resolved** sources.

    Resolving presets first is the whole trick: the Human-Art preset screens SFW
    with opennsfw2 *by default*, and a raw form entry that leaves ``sfw_method``
    blank would never ask for it.
    """
    reasons: dict[str, str] = {}
    groups: list[str] = [BASE_GROUP]
    why: dict[str, str] = {BASE_GROUP: "every stage imports numpy, Pillow, and pydantic"}
    if spec is not None and DISPLAY_GROUP in spec.groups:
        groups.append(DISPLAY_GROUP)
        why[DISPLAY_GROUP] = "the notebook's progress bars and previews"

    detector_sources = [source for source in sources if extract_line_art and needs_detector(source)]
    if detector_sources and spec is not None and "extraction" in spec.groups:
        groups.append("extraction")
        listed = ", ".join(sorted(f"{s.extractor}:{s.name}" for s in detector_sources))
        why["extraction"] = f"line-art extractors {listed}"

    clip_needed = bool(label) or (
        embed and any(str(key).startswith("mobileclip") for key in embedders)
    )
    if clip_needed and spec is not None and "clip" in spec.groups:
        groups.append("clip")
        why["clip"] = (
            "zero-shot labels" if label else "MobileCLIP2 embeddings"
        ) + f" ({', '.join(embedders) or 'labeler only'})"

    nsfw_sources = [source for source in sources if source.requires_nsfw]
    if nsfw_sources and spec is not None and "nsfw" in spec.groups:
        groups.append("nsfw")
        names = ", ".join(sorted(source.name for source in nsfw_sources))
        why["nsfw"] = f"SFW gate for source(s) {names}"

    packages: list[str] = []
    if spec is not None:
        packages = [name for name in spec.groups_for(groups) if name not in spec.runtime_only]
    else:  # No spec: fall back to the names the notebook has always understood.
        packages = sorted(reasons)

    for group in groups:
        for name in spec.groups.get(group, ()) if spec is not None else ():
            reasons.setdefault(name, why.get(group, f"{group} group"))
    present = [name for name in packages if has_package(name)]
    to_install = [name for name in packages if name not in present]
    return EnvironmentPlan(
        groups=tuple(dict.fromkeys(groups)),
        packages=tuple(packages),
        specs=tuple(spec.specs(packages)) if spec is not None else (),
        reasons=reasons,
        already_present=tuple(present),
        to_install=tuple(sorted(to_install)),
        spec=spec,
    )


def has_package(name: str) -> bool:
    """Whether a distribution's import name is importable right now."""
    module = MODULE_NAMES.get(name.lower(), name.lower().replace("-", "_"))
    return importlib.util.find_spec(module) is not None


def runtime_constraints(
    packages: Sequence[str] = ("torch", "torchvision"),
    *,
    write_to: Path | None = None,
) -> list[str]:
    """pip ``--constraint`` lines freezing the builds the runtime already shipped.

    Colab's torch is a CUDA build matched to the host driver; a transitive
    dependency must not be able to swap it. Pinning the *full local version*
    (``2.6.0+cu124``, not ``2.6.0``) is what makes that stick: a CPU wheel of the
    same base version does not satisfy the constraint.
    """
    lines: list[str] = []
    for name in packages:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        lines.append(f"{name}=={version}")
    if write_to is not None:
        write_to.parent.mkdir(parents=True, exist_ok=True)
        write_to.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return lines


def installed_versions(packages: Iterable[str]) -> dict[str, str | None]:
    """Versions from package metadata, without importing anything."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def pip_snapshot(*, python: str | None = None) -> dict[str, str]:
    """``pip list --format=json``: the exact set the runtime ended up with.

    Recorded beside a run so the environment can be rebuilt later without
    guessing which transitive versions pip happened to pick.
    """
    executable = python or sys.executable
    result = subprocess.run(
        [executable, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        return {}
    try:
        rows = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return {}
    if not isinstance(rows, list):  # pragma: no cover - defensive against odd pip builds
        return {}
    return {
        str(row["name"]).lower(): str(row["version"])
        for row in rows
        if isinstance(row, dict) and "name" in row and "version" in row
    }


# ------------------------------------------------------------------- runtime report


def runtime_report(*, snapshot: bool = False, python: str | None = None) -> dict[str, Any]:
    """What is actually installed — including what the runtime shipped preinstalled.

    torch is read from package metadata rather than imported, so building this
    report costs microseconds and never drags a CUDA context into a CPU test.
    ``torch_imported``/``cuda`` say whether a CUDA device is visible *to the
    process that already loaded torch*, and report ``"not queried"`` instead of
    guessing when torch has not been imported.
    """
    import os  # local: only needed to spot a Colab/Kaggle kernel

    torch = sys.modules.get("torch")
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "packages": {
            name: version
            for name, version in installed_versions(RUNTIME_PACKAGES).items()
            if version is not None
        },
        "torch_imported": torch is not None,
        "cuda": "not queried (torch not imported in this process)",
        "runtime": "unknown",
    }
    if torch is not None:  # pragma: no cover - exercised on a runtime that has it
        cudnn = torch.backends.cudnn.version()
        report["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "version": str(getattr(torch.version, "cuda", "") or ""),
            "cudnn": str(cudnn) if cudnn else None,
            "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
        }
    if os.environ.get("COLAB_GPU") or Path("/content/colab-ssh.log").exists():
        report["runtime"] = "google-colab"
    elif shutil.which("kaggle") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):  # pragma: no cover
        report["runtime"] = "kaggle"
    if snapshot:
        report["pip_freeze"] = pip_snapshot(python=python)
    return report


def load_baseline(path: Path | None = None) -> dict[str, Any] | None:
    """The recorded reference runtime, or ``None`` if it has never been recorded."""
    candidate = Path(path) if path is not None else runtime_baseline_path()
    if not candidate.is_file():
        return None
    try:
        loaded = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def compare_runtime(
    report: Mapping[str, Any], baseline: Mapping[str, Any] | None
) -> dict[str, dict[str, str]]:
    """Where the live runtime differs from the recorded one.

    Empty means "matches the baseline", never "was never checked": a missing
    baseline is reported as one ``baseline`` row, so a run can never mistake the
    absence of data for agreement.
    """
    if baseline is None or not baseline.get("packages"):
        return {
            "baseline": {
                "expected": "unrecorded",
                "actual": "unrecorded",
                "note": "no runtime-baseline.json; run `linescout-repro environment record`",
            }
        }
    drift: dict[str, dict[str, str]] = {}
    expected_packages = dict(baseline.get("packages") or {})
    actual_packages = dict(report.get("packages") or {})
    for name, expected in sorted(expected_packages.items()):
        actual = actual_packages.get(name)
        if actual is None:
            drift[name] = {"expected": str(expected), "actual": "absent"}
        elif str(expected) != str(actual):
            drift[name] = {"expected": str(expected), "actual": str(actual)}
    for name, actual in sorted(actual_packages.items()):
        if name not in expected_packages:
            drift[name] = {"expected": "unpinned", "actual": str(actual)}
    for name in ("torch", "cuda", "python"):
        expected = (baseline.get("runtime") or {}).get(name)
        actual = report.get(name)
        if isinstance(actual, dict):
            actual = actual.get("version")
        if (
            expected
            and str(expected) not in {"", "null", "None"}
            and str(expected) != str(actual or "")
        ):
            drift[f"runtime:{name}"] = {
                "expected": str(expected),
                "actual": str(actual or "absent"),
            }
    return drift


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON atomically: a crash mid-write must not eat a previous record."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def checkout_provenance(record: CheckoutRecord | Mapping[str, Any]) -> dict[str, Any]:
    """The subset of checkout facts that belongs on a config/report."""
    payload = record.as_dict() if isinstance(record, CheckoutRecord) else dict(record)
    return {
        "repo": payload.get("repo") or COLAB_REPO,
        "url": payload.get("url"),
        "path": payload.get("path"),
        "revision": payload.get("revision"),
        "requested_revision": payload.get("requested_revision"),
        "dirty": bool(payload.get("dirty")),
        "untracked": int(payload.get("untracked") or 0),
        "action": payload.get("action"),
        "verified": bool(payload.get("verified")),
        "source_url": payload.get("source_url"),
    }


__all__ = [
    "CHECKOUT_MARKER",
    "COLAB_MIRROR_URLS",
    "COLAB_PIN",
    "COLAB_REPO",
    "COLAB_REPO_URL",
    "CheckoutError",
    "CheckoutRecord",
    "EnvironmentPlan",
    "MODULE_NAMES",
    "RequirementSpec",
    "RevisionMismatchError",
    "RUNTIME_PACKAGES",
    "checkout_provenance",
    "compare_runtime",
    "ensure_checkout",
    "environment_digest",
    "environment_spec_path",
    "find_or_checkout",
    "has_package",
    "installed_versions",
    "needs_detector",
    "is_immutable_revision",
    "load_baseline",
    "load_requirement_spec",
    "lockfile_versions",
    "parse_requirements",
    "plan_environment",
    "read_checkout",
    "repo_is_here",
    "repo_root",
    "require_immutable_revision",
    "runtime_baseline_path",
    "runtime_constraints",
    "runtime_report",
    "untracked_count",
    "write_json",
]
