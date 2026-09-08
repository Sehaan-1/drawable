"""Pinned model code and weights, verified before a stage uses them.

A pipeline that downloads weights at runtime is only reproducible if it says
*which* bytes it used. Two things are therefore pinned in
``models.lock.json`` beside this module:

* the **repository revision** — a 40-hex commit, never a branch or a tag, so a
  Hugging Face repo or a torch.hub clone cannot silently move under a run;
* the **SHA-256 of each checkpoint**, verified after download and *before* the
  file is loaded. A mismatch raises :class:`CheckpointMismatchError` and the run
  stops; nothing is silently repaired.

Publishers that expose no digest (``dl.fbaipublicfiles.com``, GitHub release
assets) get ``"sha256": null`` and the ``"record"`` policy: the size is still
checked, the digest the run actually saw is written to
``<cache>/recorded.json``, and every later run verifies against it. ``"strict"``
refuses to load anything without a pinned digest, which is what a release run
should use.

Nothing here imports torch, numpy, or Pillow, so the checks run in a plain
``uv sync --frozen --extra dev`` environment and in CI without a GPU stack.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from linescout_ml.colab._optional import has_module

IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
ArtifactId = Annotated[str, StringConstraints(min_length=3, max_length=128)]

CHECKPOINT_SCHEMA_VERSION: Literal[1] = 1
#: ``strict`` refuses unpinned bytes, ``record`` pins what it can and records the
#: rest, ``off`` skips verification (debug only — the run report says so).
CheckpointPolicy = Literal["strict", "record", "off"]
POLICIES: tuple[str, ...] = ("strict", "record", "off")

HINT_FILES = "pip install huggingface-hub   # shipped with controlnet-aux and open-clip-torch"
RECORDED_FILENAME = "recorded.json"
LOCK_FILENAME = "models.lock.json"
#: Where a Colab session keeps verified weights. Drive is deliberately not the
#: default: the FUSE mount is slow and its partial writes look like corruption.
DEFAULT_CACHE_DIRNAME = "linescout/checkpoints"

DOWNLOAD_TIMEOUT_SECONDS = 120
CHUNK_BYTES = 1 << 20


class CheckpointError(RuntimeError):
    """The checkpoint lock could not be read, or an artifact could not be resolved."""


class CheckpointMismatchError(CheckpointError):
    """A downloaded file is not the bytes the lock says the model needs."""


class CheckpointUnpinnedError(CheckpointError):
    """Policy is ``strict`` and the artifact has no digest to check against."""


def default_cache_dir() -> Path:
    """``$LINESCOUT_CACHE_DIR/checkpoints`` or ``~/.cache/linescout/checkpoints``."""
    override = os.environ.get("LINESCOUT_CACHE_DIR")
    base = Path(override).expanduser() if override else Path.home() / ".cache" / "linescout"
    return base / "checkpoints"


def package_lock_path() -> Path:
    """The lock shipped inside the package (also the repo copy in a checkout)."""
    return Path(__file__).with_name(LOCK_FILENAME)


def sha256_file(path: Path) -> str:
    """Streaming SHA-256, so a 400 MB checkpoint does not have to fit in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


class CheckpointArtifact(BaseModel):
    """One pinned file: the weights a loader must be given, and nothing else."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    id: ArtifactId
    #: Which loader asks for it: an extractor key, an encoder key, or ``opennsfw2``.
    group: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    source: Literal["huggingface", "url", "git"]
    repo: Annotated[str, StringConstraints(max_length=128)] | None = None
    #: The commit the bytes came from. Hugging Face and git artifacts must name a
    #: 40-hex commit — a branch there moves, and a moving revision is a silent
    #: change to the model. A ``url`` artifact may instead name the release tag its
    #: publisher served it under, because that tag is part of the URL.
    revision: Revision | str | None = None
    path: Annotated[str, StringConstraints(max_length=256)] | None = None
    url: Annotated[str, StringConstraints(max_length=512)] | None = None
    sha256: Sha256 | None = None
    size_bytes: Annotated[int, Field(ge=1)] | None = None
    license: Annotated[str, StringConstraints(max_length=256)] = ""
    used_by: Annotated[str, StringConstraints(max_length=64)] = ""
    note: str = ""

    @model_validator(mode="after")
    def _source_shape(self) -> Self:
        if self.source == "huggingface" and not (self.repo and self.path):
            msg = f"{self.id}: a huggingface artifact needs repo and path"
            raise ValueError(msg)
        if self.source == "url" and not (self.url and self.path):
            msg = f"{self.id}: a url artifact needs url and path"
            raise ValueError(msg)
        if self.source == "git" and not self.repo:
            msg = f"{self.id}: a git artifact needs repo"
            raise ValueError(msg)
        if (
            self.source != "git"
            and self.sha256 is None
            and self.size_bytes is None
            and not self.note.strip()
        ):
            msg = (
                f"{self.id}: neither a digest nor a size, and no note saying why — "
                "an entry nothing can check is not a pin. Publish a size at minimum, or "
                "write in `note` why this file cannot be verified yet."
            )
            raise ValueError(msg)
        if (
            self.source in {"huggingface", "git"}
            and self.revision is not None
            and not IMMUTABLE_REVISION.fullmatch(str(self.revision))
        ):
            msg = (
                f"{self.id}: revision {self.revision!r} is not a 40-hex commit. Pin the commit you "
                "verified, not the branch it was on: a branch moves, and the model changes under "
                "the dataset without anything failing."
            )
            raise ValueError(msg)
        return self

    @property
    def pinned(self) -> bool:
        """Whether this entry can fail closed on a digest mismatch."""
        return self.sha256 is not None

    @property
    def filename(self) -> str:
        return Path(str(self.path or self.url or self.id)).name

    def locator(self) -> str:
        if self.source == "huggingface":
            return f"{self.repo}@{self.short_revision}/{self.path}"
        if self.source == "git":
            return f"{self.repo}@{self.short_revision}"
        return str(self.url)

    @property
    def short_revision(self) -> str:
        revision = self.revision
        if revision is None:
            return "unpinned"
        return revision[:12] if len(revision) >= 12 else revision


class ModelLock(BaseModel):
    """The parsed ``models.lock.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = CHECKPOINT_SCHEMA_VERSION
    note: list[str] = Field(default_factory=list)
    artifacts: list[CheckpointArtifact] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [artifact.id for artifact in self.artifacts]
        if len(set(ids)) != len(ids):
            msg = f"duplicate artifact ids in the checkpoint lock: {sorted(ids)}"
            raise ValueError(msg)
        return self

    def by_id(self) -> dict[str, CheckpointArtifact]:
        return {artifact.id: artifact for artifact in self.artifacts}

    def groups(self) -> dict[str, list[CheckpointArtifact]]:
        grouped: dict[str, list[CheckpointArtifact]] = {}
        for artifact in self.artifacts:
            grouped.setdefault(artifact.group, []).append(artifact)
        for entries in grouped.values():
            entries.sort(key=lambda artifact: artifact.id)
        return grouped

    def for_group(self, group: str) -> list[CheckpointArtifact]:
        return self.groups().get(group, [])

    def unpinned(self) -> list[CheckpointArtifact]:
        return [artifact for artifact in self.artifacts if not artifact.pinned]


def load_model_lock(path: Path | None = None) -> ModelLock:
    """Read and validate the lock. Unreadable or malformed is an error, not a default."""
    candidate = Path(path) if path is not None else package_lock_path()
    if not candidate.is_file():
        msg = f"checkpoint lock not found: {candidate}"
        raise CheckpointError(msg)
    try:
        raw = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"checkpoint lock {candidate} is not readable JSON: {error}"
        raise CheckpointError(msg) from error
    try:
        return ModelLock.model_validate(raw)
    except ValueError as error:
        msg = f"checkpoint lock {candidate} failed validation: {error}"
        raise CheckpointError(msg) from error


def lock_digest(path: Path | None = None) -> dict[str, Any]:
    """Identity of the lock file itself, so a run names the bytes it checked against."""
    candidate = Path(path) if path is not None else package_lock_path()
    if not candidate.is_file():
        return {"path": str(candidate), "sha256": None}
    return {"path": str(candidate), "sha256": sha256_file(candidate)}


def fetch_huggingface(artifact: CheckpointArtifact, destination: Path) -> None:
    """Download one file from a Hugging Face repo at the pinned revision."""
    if artifact.revision is None:
        msg = f"{artifact.id}: no pinned revision to download from"
        raise CheckpointUnpinnedError(msg)
    if not has_module("huggingface_hub"):  # pragma: no cover - GPU stack only
        msg = f"{artifact.id}: downloading needs huggingface_hub. Try: {HINT_FILES}"
        raise CheckpointError(msg)
    from huggingface_hub import hf_hub_download  # lazy: CPU-only tests never need it

    destination.parent.mkdir(parents=True, exist_ok=True)
    fetched = hf_hub_download(
        repo_id=str(artifact.repo),
        filename=str(artifact.path),
        revision=str(artifact.revision),
        cache_dir=str(destination.parent / "hf"),
        local_files_only=False,
    )
    shutil.copyfile(fetched, destination)


def fetch_url(artifact: CheckpointArtifact, destination: Path) -> None:
    """Download a plain file over HTTPS (Meta's and GitHub's own URLs)."""
    if not artifact.url:  # pragma: no cover - guarded by the model validator
        msg = f"{artifact.id}: artifact has no url"
        raise CheckpointError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(str(artifact.url), headers={"User-Agent": "linescout-ml"})
    try:
        with (
            urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            temporary.open("wb") as handle,
        ):
            for block in iter(lambda: response.read(CHUNK_BYTES), b""):
                handle.write(block)
    except (urllib.error.URLError, OSError) as error:
        temporary.unlink(missing_ok=True)
        msg = f"{artifact.id}: could not download {artifact.url}: {error}"
        raise CheckpointError(msg) from error
    shutil.move(str(temporary), destination)


FETCHERS: dict[str, Callable[[CheckpointArtifact, Path], None]] = {
    "huggingface": fetch_huggingface,
    "url": fetch_url,
}


def fetch_artifact(artifact: CheckpointArtifact, destination: Path) -> None:
    """Materialise an artifact's bytes into ``destination`` (no verification yet)."""
    fetcher = FETCHERS.get(artifact.source)
    if fetcher is None:
        msg = (
            f"{artifact.id}: source {artifact.source!r} is not downloadable (revision-pinned code)"
        )
        raise CheckpointError(msg)
    if destination.is_file() and matches_digest(destination, artifact):
        return
    fetcher(artifact, destination)


def matches_digest(
    path: Path, artifact: CheckpointArtifact, *, expected: str | None = None
) -> bool:
    """Whether ``path`` is the file the lock (or the ledger) describes."""
    if not path.is_file():
        return False
    if artifact.size_bytes is not None and path.stat().st_size != artifact.size_bytes:
        return False
    digest = expected if expected is not None else artifact.sha256
    if digest is None:
        return True
    return sha256_file(path) == digest


class Verification(BaseModel):
    """What the checker concluded about one artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: ArtifactId
    group: str
    locator: str
    path: str | None = None
    sha256: Sha256 | None = None
    size_bytes: int | None = None
    #: ``verified`` (matched a pin), ``recorded`` (digest stored for next time),
    #: ``revision`` (code pinned by commit only), ``unchecked`` (policy=off).
    status: Literal["verified", "recorded", "revision", "unchecked"]
    pinned: bool

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


#: Alias kept so callers written against either name still work.
VerificationError = CheckpointError


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class CheckpointLedger:
    """Digests this machine recorded for artifacts the lock cannot pin.

    A first-use record, not a trust-on-first-use shortcut: once an entry exists,
    every run verifies against it and fails closed on a mismatch. Deleting the
    file is therefore the only way to accept new bytes, which is the point.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict[str, Any]] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
            artifacts = loaded.get("artifacts") if isinstance(loaded, dict) else None
            if isinstance(artifacts, dict):
                self._entries = {
                    str(key): dict(value)
                    for key, value in artifacts.items()
                    if isinstance(value, dict)
                }

    def expected(self, artifact_id: str) -> str | None:
        entry = self._entries.get(artifact_id)
        digest = entry.get("sha256") if entry else None
        return str(digest) if isinstance(digest, str) and len(digest) == 64 else None

    def record(self, artifact: CheckpointArtifact, digest: str, size: int) -> None:
        self._entries[artifact.id] = {
            "sha256": digest,
            "size_bytes": size,
            "locator": artifact.locator(),
            "recorded_at": _utc_now(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "artifacts": self._entries}
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def as_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "artifacts": dict(self._entries)}


class CheckpointVerifier:
    """Resolves, downloads, and verifies the artifacts a stage asked for.

    The loaders in :mod:`~linescout_ml.colab.extract`, :mod:`~linescout_ml.colab.
    label`, and :mod:`~linescout_ml.colab.models` receive one of these and call
    :meth:`directory` or :meth:`file` instead of naming a repo id, which is what
    turns "download whatever is current" into "load exactly these bytes".
    """

    def __init__(
        self,
        lock: ModelLock,
        *,
        cache_dir: Path | None = None,
        policy: CheckpointPolicy = "record",
        fetcher: Callable[[CheckpointArtifact, Path], None] | None = None,
        lock_path: Path | None = None,
    ) -> None:
        if policy not in POLICIES:
            msg = f"unknown checkpoint policy {policy!r}; expected one of {', '.join(POLICIES)}"
            raise CheckpointError(msg)
        self.lock = lock
        self.policy: CheckpointPolicy = policy
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self._fetcher = fetcher
        self._lock_path = Path(lock_path) if lock_path is not None else package_lock_path()
        self.ledger = CheckpointLedger(self.cache_dir / RECORDED_FILENAME)
        self._verifications: dict[str, Verification] = {}

    @classmethod
    def for_config(cls, config: Any) -> CheckpointVerifier:
        """Build from a :class:`~linescout_ml.colab.config.PipelineConfig`.

        ``config`` is duck-typed so the checkpoint machinery can be exercised
        without the pydantic config module (and so a test can pass a stub).
        """
        policy: CheckpointPolicy = getattr(config, "checkpoint_policy", "record")
        lock_path = getattr(config, "model_lock_path", None)
        cache = getattr(config, "checkpoint_cache_dir", None)
        lock = load_model_lock(Path(lock_path) if lock_path is not None else None)
        return cls(
            lock,
            cache_dir=Path(cache) if cache is not None else None,
            policy=policy,
            lock_path=Path(lock_path) if lock_path is not None else None,
        )

    @classmethod
    def disabled(cls) -> CheckpointVerifier:
        """A verifier that records nothing and downloads nothing (policy ``off``)."""
        return cls(load_model_lock(), policy="off")

    # ------------------------------------------------------------------ lookups

    def artifacts(self, group: str) -> list[CheckpointArtifact]:
        return self.lock.for_group(group)

    def artifact(self, artifact_id: str) -> CheckpointArtifact:
        try:
            return self.lock.by_id()[artifact_id]
        except KeyError:
            known = ", ".join(sorted(self.lock.by_id()))
            msg = f"unknown checkpoint artifact {artifact_id!r}; the lock names: {known}"
            raise CheckpointError(msg) from None

    def knows(self, group: str) -> bool:
        return bool(self.lock.for_group(group))

    def revisions(self, group: str) -> dict[str, str]:
        """``repo -> revision`` for a group, used by hub loaders that clone code."""
        return {
            str(artifact.repo): str(artifact.revision)
            for artifact in self.lock.for_group(group)
            if artifact.repo and artifact.revision
        }

    def weights_file(self, group: str) -> Path | None:
        """The verified checkpoint of a group, or ``None`` when the lock has none.

        A group may also carry code (``git``) and tokenizer files; this returns
        the weights a loader should hand to ``load_state_dict`` directly.
        """
        for artifact in self.artifacts(group):
            if artifact.source != "git":
                return self.file(artifact.id)
        return None

    def hub_repo_ref(self, group: str) -> str | None:
        """``owner/repo:<commit>`` for torch.hub, or ``None`` when unpinned."""
        for artifact in self.lock.for_group(group):
            if artifact.source == "git" and artifact.repo and artifact.revision:
                return f"{artifact.repo}:{artifact.revision}"
        return None

    # ------------------------------------------------------------------ verification

    def verify(self, artifact: CheckpointArtifact, path: Path) -> Verification:
        """Check ``path`` against the lock (and the ledger), fail closed on mismatch."""
        if self.policy == "off":
            verification = self._build(artifact, path, "unchecked", pinned=artifact.pinned)
            self._verifications[artifact.id] = verification
            return verification
        if not path.is_file():
            msg = f"{artifact.id}: expected a file at {path}, found nothing"
            raise CheckpointError(msg)
        size = path.stat().st_size
        if artifact.size_bytes is not None and size != artifact.size_bytes:
            msg = (
                f"{artifact.id}: {path.name} is {size} bytes, the lock says "
                f"{artifact.size_bytes} — a truncated or replaced download"
            )
            raise CheckpointMismatchError(msg)
        digest = sha256_file(path)
        expected = artifact.sha256 or self.ledger.expected(artifact.id)
        if expected is not None and digest != expected:
            msg = (
                f"{artifact.id}: SHA-256 mismatch for {path}\n"
                f"  expected {expected}\n  found    {digest}\n"
                f"  source   {artifact.locator()}\n"
                "Refusing to run on bytes that do not match the lock. If the upstream "
                "file genuinely changed, update ml/linescout_ml/colab/models.lock.json "
                "in a reviewed commit (and delete the stale file from the cache dir)."
            )
            raise CheckpointMismatchError(msg)
        if expected is not None:
            status: Literal["verified", "recorded", "revision", "unchecked"] = "verified"
        elif self.policy == "strict":
            msg = (
                f"{artifact.id}: policy 'strict' requires a pinned sha256 but the lock has "
                f"none for {artifact.locator()}"
            )
            raise CheckpointUnpinnedError(msg)
        else:
            self.ledger.record(artifact, digest, size)
            status = "recorded"
        verification = self._build(artifact, path, status, pinned=artifact.pinned, digest=digest)
        self._verifications[artifact.id] = verification
        return verification

    def _build(
        self,
        artifact: CheckpointArtifact,
        path: Path | None,
        status: Literal["verified", "recorded", "revision", "unchecked"],
        *,
        pinned: bool,
        digest: str | None = None,
    ) -> Verification:
        size = path.stat().st_size if path is not None and path.is_file() else None
        return Verification(
            id=artifact.id,
            group=artifact.group,
            locator=artifact.locator(),
            path=str(path) if path is not None else None,
            sha256=digest if digest is not None else artifact.sha256,
            size_bytes=size,
            status=status,
            pinned=pinned,
        )

    def _fetch(self, artifact: CheckpointArtifact, destination: Path) -> None:
        if destination.is_file() and matches_digest(destination, artifact):
            return
        expected = artifact.sha256 or self.ledger.expected(artifact.id)
        if (
            expected is not None
            and destination.is_file()
            and not matches_digest(destination, artifact, expected=expected)
        ):
            msg = (
                f"{artifact.id}: the cached file at {destination} does not match the pin; "
                "delete it and re-run rather than overwriting silently"
            )
            raise CheckpointMismatchError(msg)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self._fetcher is not None:
            self._fetcher(artifact, destination)
        else:
            fetch_artifact(artifact, destination)

    # ------------------------------------------------------------------ resolutions

    def file(self, artifact_id: str) -> Path:
        """A verified local copy of one artifact, downloaded into the cache dir if needed."""
        artifact = self.artifact(artifact_id)
        if artifact.source == "git":
            self._verifications[artifact.id] = self._build(artifact, None, "revision", pinned=False)
            return self.cache_dir / "git" / artifact.id
        destination = self._path_for(artifact)
        if self.policy == "off":
            return destination
        self._fetch(artifact, destination)
        self.verify(artifact, destination)
        return destination

    def group_files(self, group: str) -> dict[str, Path]:
        """Every file of a group, verified, keyed by the filename the loader expects."""
        return {artifact.filename: self.file(artifact.id) for artifact in self.artifacts(group)}

    def directory(self, group: str) -> Path:
        """A directory holding a group's verified files, named the way loaders want.

        ``controlnet_aux`` accepts *a folder* and joins its own filenames, and
        open_clip's ``local-dir:`` schema reads config, weights, and tokenizer
        from one folder — so handing it a folder of bytes we verified is enough
        to pin the whole load, with no fork of either library.
        """
        entries = self.artifacts(group)
        if not entries:
            msg = (
                f"the checkpoint lock has no artifacts for group {group!r}, so nothing in it "
                "can be verified. Either pin the files that group needs in "
                f"{self._lock_path.name} (repo, 40-hex revision, filename, digest) or do not "
                "select it: fetching weights by tag because no pin existed is how a pinned "
                "pipeline quietly stops being one."
            )
            raise CheckpointError(msg)
        target = self.cache_dir / "dirs" / group
        if self.policy == "off":
            return target
        target.mkdir(parents=True, exist_ok=True)
        for artifact in entries:
            if artifact.source == "git":
                # Clone-time code, pinned by commit: nothing to hash here.
                self._verifications[artifact.id] = self._build(
                    artifact, None, "revision", pinned=False
                )
                continue
            self._stage(artifact, target / artifact.filename)
        return target

    def _stage(self, artifact: CheckpointArtifact, staged: Path) -> None:
        """Verify the staged copy when we have one, else fetch, verify, then link it in."""
        if staged.is_file():
            self.verify(artifact, staged)
            return
        source = self.file(artifact.id)
        self.verify(artifact, source)
        staged.parent.mkdir(parents=True, exist_ok=True)
        try:
            staged.hardlink_to(source)
        except OSError:  # exotic filesystems (Drive) have no hardlinks
            shutil.copyfile(source, staged)

    def _path_for(self, artifact: CheckpointArtifact) -> Path:
        return self.cache_dir / "files" / artifact.id

    def ensure_at(self, artifact_id: str, destination: Path) -> Verification:
        """Verify (and if necessary fetch) a file at a path the *library* chose.

        opennsfw2 insists on loading ``~/.opennsfw2/weights/open_nsfw_weights.h5``
        and downloads it itself on first use; this gives the pipeline the same
        file with the pin checked before anything is loaded.
        """
        artifact = self.artifact(artifact_id)
        destination = Path(destination)
        if self.policy == "off":
            seen = destination if destination.is_file() else None
            return self._build(artifact, seen, "unchecked", pinned=artifact.pinned)
        if not destination.is_file():
            self._fetch(artifact, destination)
        return self.verify(artifact, destination)

    def verify_existing(self, artifact_id: str, path: Path) -> Verification:
        """Check a file a loader already produced, without moving anything."""
        return self.verify(self.artifact(artifact_id), Path(path))

    # ------------------------------------------------------------------ reporting

    @property
    def verifications(self) -> list[Verification]:
        return [self._verifications[key] for key in sorted(self._verifications)]

    def requirement_notes(self) -> list[str]:  # pragma: no cover - kept for API symmetry
        return self.notes()

    def notes(self) -> list[str]:
        """One-line summaries for the stage notes and the notebook."""
        notes: list[str] = []
        for verification in self.verifications:
            notes.append(
                f"{verification.id}: {verification.status}{' @' + (verification.sha256 or '')[:12]}"
            )
        unpinned = [artifact.id for artifact in self.lock.unpinned()]
        if unpinned and self.policy != "off":
            listed = ", ".join(sorted(unpinned)[:6])
            notes.append(f"{len(unpinned)} artifact(s) in the lock carry no digest: {listed}")
        return notes

    def as_dict(self) -> dict[str, Any]:
        """The block written into the run report and the manifest's provenance."""
        return {
            "policy": self.policy,
            "lock": lock_digest(self._lock_path),
            "cache_dir": str(self.cache_dir),
            "verified_at_record": bool(self.ledger.as_dict()["artifacts"]),
            "artifacts": [verification.as_dict() for verification in self.verifications],
        }

    def ledger_path(self) -> Path:
        return self.ledger.path


def summarize_lock(lock: ModelLock) -> list[dict[str, str]]:
    """Table rows for the notebook: which weights run which stage, and how pinned they are."""
    rows: list[dict[str, str]] = []
    for artifact in sorted(lock.artifacts, key=lambda item: (item.group, item.id)):
        rows.append(
            {
                "group": artifact.group,
                "artifact": artifact.filename,
                "revision": artifact.short_revision,
                "sha256": (artifact.sha256 or "unpinned")[:12],
                "license": artifact.license,
            }
        )
    return rows


def iter_groups(
    lock: ModelLock, groups: Iterable[str]
) -> list[tuple[str, list[CheckpointArtifact]]]:
    """``(group, artifacts)`` pairs, preserving the caller's order."""
    return [(group, lock.for_group(group)) for group in groups]


def missing_groups(lock: ModelLock, groups: Sequence[str]) -> list[str]:
    """Groups a run needs that the lock does not pin — reported, never guessed at."""
    return [group for group in groups if not lock.for_group(group)]


__all__ = [
    "CheckpointArtifact",
    "CheckpointError",
    "CheckpointLedger",
    "CheckpointMismatchError",
    "CheckpointUnpinnedError",
    "CheckpointVerifier",
    "ModelLock",
    "Verification",
    "VerificationError",
    "default_cache_dir",
    "fetch_artifact",
    "iter_groups",
    "load_model_lock",
    "lock_digest",
    "matches_digest",
    "missing_groups",
    "package_lock_path",
    "sha256_file",
    "summarize_lock",
]
