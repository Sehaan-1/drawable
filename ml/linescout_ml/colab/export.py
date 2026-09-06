"""Run report, zip export, and Drive export.

A Colab session is ephemeral, so the run has to leave behind enough to be
trusted later: what config produced it, on what GPU, with which model versions,
how many assets survived each stage, and where the bytes went. That is
``_pipeline/run_report.json``.

Exports are zip-first. Drive copies of a gallery mean tens of thousands of small
file writes through the FUSE mount, which is slow and prone to partial failures;
a single zip next to it costs one write and can be verified by checksum. Both
are offered, and the report records whichever ran.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from linescout_ml.colab._optional import optional_module
from linescout_ml.colab.measure import sha256_file
from linescout_ml.manifest import Manifest

RUN_REPORT_FILENAME = "run_report.json"
TEMPORARY_SUFFIXES: tuple[str, ...] = (".tmp", ".tmp.npz", ".jsonl.tmp")


def utc_now() -> str:
    """UTC timestamp used across reports and indexes."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


_now = utc_now


def _is_temporary(path: Path) -> bool:
    return path.name.endswith(TEMPORARY_SUFFIXES)


@dataclass
class StageResult:
    """What one stage did. ``notes`` carries per-run detail worth keeping."""

    name: str
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seconds"] = round(self.seconds, 3)
        return payload


def build_run_report(
    *,
    config_dump: Mapping[str, Any],
    stages: Iterable[StageResult],
    summary: Mapping[str, Any],
    gpu: Mapping[str, Any],
    embedders: Iterable[Mapping[str, Any]],
    outputs: Mapping[str, Any] | None = None,
    started_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the report payload. Pure, so it is testable without a GPU."""
    return {
        "schema_version": 1,
        "created_at": _now(),
        "started_at": started_at or _now(),
        "config": json.loads(json.dumps(config_dump, default=str)),
        "gpu": dict(gpu),
        "stages": [stage.as_dict() for stage in stages],
        "summary": dict(summary),
        "embedders": [dict(embedder) for embedder in embedders],
        "outputs": dict(outputs or {}),
    }


def write_run_report(root: Path, report: Mapping[str, Any]) -> Path:
    path = root / RUN_REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def zip_tree(
    root: Path,
    destination: Path,
    *,
    exclude_dirs: Iterable[str] = (),
    arcname_prefix: str = "",
) -> Path:
    """Zip ``root`` deterministically: sorted entries, temporaries skipped.

    ``exclude_dirs`` matches directory *names* anywhere in the tree, which is
    how ``__pycache__`` and scratch folders stay out of a dataset archive.
    """
    if not root.is_dir():
        msg = f"nothing to zip: {root} is not a directory"
        raise FileNotFoundError(msg)
    excluded = set(exclude_dirs)
    entries: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_temporary(path):
            continue
        if excluded.intersection(part for part in path.relative_to(root).parts[:-1]):
            continue
        entries.append(path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in entries:
            relative = path.relative_to(root).as_posix()
            archive.write(path, arcname=f"{arcname_prefix}{relative}")
    os.replace(temporary, destination)
    return destination


def zip_gallery(
    root: Path,
    manifest: Manifest,
    destination: Path,
    *,
    manifest_name: str = "manifest.json",
    extra: Iterable[Path] = (),
) -> Path:
    """Zip exactly the files a manifest references — nothing else.

    ``zip_tree`` would also sweep up candidates that dedupe dropped, half-written
    temporaries, and any leftovers from an earlier run. A dataset archive should
    contain the artefacts the manifest vouches for, so the manifest drives the
    file list here.
    """
    entries: list[tuple[Path, str]] = []
    for record in manifest.records:
        for relative in (record.original_path, record.line_art_path, record.thumbnail_path):
            path = root / relative
            if path.is_file():
                entries.append((path, relative))
    manifest_path = root / manifest_name
    if manifest_path.is_file():
        entries.append((manifest_path, manifest_name))
    for path in extra:
        if path.is_file():
            entries.append((path, path.name))

    seen: set[str] = set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in sorted(entries, key=lambda entry: entry[1]):
            if arcname in seen:
                continue
            seen.add(arcname)
            archive.write(path, arcname=arcname)
    os.replace(temporary, destination)
    return destination


def copy_tree(source: Path, destination: Path, *, exclude_dirs: Iterable[str] = ()) -> Path:
    """Mirror a tree (used for the Google Drive export)."""
    if not source.is_dir():
        msg = f"nothing to copy: {source} is not a directory"
        raise FileNotFoundError(msg)
    excluded = set(exclude_dirs)
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        if _is_temporary(path):
            continue
        relative = path.relative_to(source)
        if excluded.intersection(relative.parts[:-1]):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    if copied == 0:
        msg = f"copy produced no files: {source} -> {destination}"
        raise RuntimeError(msg)
    return destination


def colab_download(path: Path) -> bool:
    """Trigger the browser download in Colab. Returns False off-Colab."""
    try:
        files = optional_module("google.colab.files", "run this cell inside Google Colab")
    except RuntimeError:
        return False
    files.download(str(path))
    return True


def describe_file(path: Path) -> dict[str, Any]:
    """Size and checksum of an artefact, for the report's ``outputs`` block."""
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "sha256": sha256_file(path),
        "written_at": _now(),
    }
