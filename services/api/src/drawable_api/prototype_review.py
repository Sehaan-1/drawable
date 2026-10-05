from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .database import connect
from .intake import IntakeError, _anita_work_records, ensure_inside, load_registry, sha256_file
from .settings import Settings

PROTOTYPE_AUTHORIZATION = "owner-local-educational-prototype"
PROTOTYPE_PERMISSIONS = {
    key: key in {"local_display", "evaluation", "derive"}
    for key in (
        "local_display", "training", "evaluation", "derive", "trace", "export",
        "public_demo", "redistribute", "cloud_storage", "cloud_processing",
    )
}
DEFAULT_WORKS = ("hero", "lma_test", "pirate")


def is_prototype_asset(settings: Settings, metadata: dict[str, Any]) -> bool:
    return settings.local_prototype_mode and metadata.get("authorization_basis") == PROTOTYPE_AUTHORIZATION


def _sketch_paths(tree: Path, works: tuple[str, ...]) -> list[Path]:
    work_queues = []
    for work in works:
        folder = tree / work / "sketch"
        if folder.is_symlink() or (tree / work).is_symlink():
            raise IntakeError("Prototype sketch folders may not be symbolic links")
        clips: dict[str, list[Path]] = defaultdict(list)
        for current, directories, filenames in os.walk(folder, followlinks=False):
            directories.sort()
            if any((Path(current) / name).is_symlink() for name in directories):
                raise IntakeError("Prototype sketch folders may not contain symbolic links")
            for name in sorted(filenames):
                candidate = Path(current) / name
                if candidate.is_symlink():
                    raise IntakeError("Prototype sketches may not be symbolic links")
                if candidate.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}:
                    clips[candidate.parent.relative_to(folder).as_posix()].append(candidate)
        work_queues.append([
            candidate
            for row in zip_longest(*(clips[key] for key in sorted(clips)))
            for candidate in row if candidate is not None
        ])
    return [candidate for row in zip_longest(*work_queues) for candidate in row if candidate is not None]


def prepare_prototype_batch(settings: Settings, *, limit: int = 24, reuse_pending: bool = False) -> dict[str, int]:
    """Catalog local sketches for the owner's educational prototype, without recording terms approval."""
    if not settings.curation_mode or not settings.local_prototype_mode:
        raise PermissionError("Local educational prototype review is not enabled")
    if not 1 <= limit <= 32:
        raise ValueError("Prototype batches must contain 1–32 images")
    root = settings.data_root.resolve(strict=True)
    source = next(row for row in load_registry(settings.source_registry_path)["sources"] if row["id"] == "anita-dataset")
    tree = ensure_inside(root, source["extracted_path"])
    if tree.is_symlink() or not tree.is_dir():
        raise IntakeError("No local AnitaDataset sketches were found. Place the extracted dataset in datasets/AnitaDataset.")
    work_records = _anita_work_records(settings.anita_work_terms_path)
    registry_hash = sha256_file(settings.source_registry_path)
    work_terms_hash = sha256_file(settings.anita_work_terms_path)
    db = connect(settings.metadata_db)
    try:
        pending = db.execute(
            "SELECT COUNT(*) FROM assets WHERE state IN ('quarantined', 'candidate_pending_rights') AND json_extract(metadata_json, '$.authorization_basis') = ?",
            (PROTOTYPE_AUTHORIZATION,),
        ).fetchone()[0]
        if reuse_pending and pending:
            return {"imported": 0, "pending": pending, "skipped": 0}
        known = {row["relative_path"] for row in db.execute("SELECT relative_path FROM assets WHERE source_id = 'anita-dataset'")}
        imported = skipped = 0
        prepared = []
        for candidate in _sketch_paths(tree, DEFAULT_WORKS):
            relative = candidate.relative_to(tree).as_posix()
            if relative in known:
                continue
            if not candidate.is_file() or not 0 < candidate.stat().st_size <= 50 * 1024 * 1024:
                skipped += 1
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(candidate) as image:
                        width, height = image.size
                        if image.format not in {"PNG", "JPEG", "WEBP"} or getattr(image, "n_frames", 1) != 1 or width * height > 40_000_000:
                            skipped += 1
                            continue
                        media_type = Image.MIME[image.format]
                        image.verify()
                    with Image.open(candidate) as image:
                        image.load()
            except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                skipped += 1
                continue
            work_id = candidate.relative_to(tree).parts[0]
            work = work_records[work_id]
            group_id = f"anita:work:{work_id}"
            digest = sha256_file(candidate)
            asset_id = f"prototype-anita-{hashlib.sha256(relative.encode()).hexdigest()[:24]}"
            metadata = {
                "relative_root": tree.relative_to(root).as_posix(),
                "source_registry_sha256": registry_hash,
                "anita_work_terms_sha256": work_terms_hash,
                "authorization_basis": PROTOTYPE_AUTHORIZATION,
                "archive_provenance": "existing-local-tree",
                "work_id": work_id,
                "work_title": work.get("title", work_id),
                "artist": work.get("artist"),
                "group_id": group_id,
                "candidate_type": "sketch",
                "decoder_status": "decoded",
                "license_id": work.get("license"),
                "attribution_text": work.get("artist"),
                "rights_status": "review_required",
                "use_permissions": dict(PROTOTYPE_PERMISSIONS),
                "learning_split": "pilot_development",
                "review_status": "pending",
                "gallery_member": False,
                "gallery_eligible_candidate": False,
            }
            prepared.append((asset_id, "anita-dataset", relative, digest, candidate.stat().st_size, media_type, width, height, group_id, "review_required", "quarantined", 0, json.dumps(metadata, ensure_ascii=False)))
            imported += 1
            if imported >= limit:
                break
        if not prepared and not known:
            raise IntakeError("No readable native sketches were found in the selected local works.")
        db.execute("BEGIN IMMEDIATE")
        try:
            db.executemany(
                "INSERT OR IGNORE INTO assets(asset_id, source_id, relative_path, sha256, byte_length, media_type, width, height, group_id, rights_status, state, revision, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                prepared,
            )
            # Existing family assignments must also include newly added frames.
            from .curation import _refresh_active_family_assignment
            for group_id in {row[8] for row in prepared}:
                _refresh_active_family_assignment(db, "anita-dataset", group_id)
            db.execute("COMMIT")
        except Exception:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        os.chmod(settings.metadata_db.parent, 0o700)
        os.chmod(settings.metadata_db, 0o600)
        return {"imported": imported, "pending": pending + imported, "skipped": skipped}
    finally:
        db.close()
