from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from drawable_research_contracts import (
    PrototypeGalleryCatalogV2,
    PrototypeGalleryItemV2,
    ReviewProvenanceV1,
    SnapshotLabelsV1,
    SnapshotPermissionsV1,
)
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError

from .database import connect, migrate
from .intake import IntakeError, _anita_work_records, ensure_inside, load_registry, sha256_file
from .local_reviewer import (
    CropBox,
    LlamaReviewer,
    LocalPairReview,
    LocalVisualReview,
    ReviewerError,
    _gpu_probe,
    review_image,
    review_pair,
    reviewer_config,
    setup_reviewer,
)
from .settings import REPOSITORY_ROOT

MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_PIXELS = 40_000_000
SUPPORTED_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
RUBRIC_VERSION = "drawable-reference-review-rubric-v1"
CROP_RECIPE = "orientation-normalized-box-v1"
SOFT_SCOPE_TARGETS = {
    "face_head": 200,
    "upper_body_clothing": 150,
    "full_body": 75,
    "eye": 25,
    "hand": 25,
    "multi_character": 25,
}
PHASH_COSINE = tuple(
    tuple(math.cos(((2 * coordinate + 1) * frequency * math.pi) / 64) for frequency in range(8))
    for coordinate in range(32)
)


def _db_path(root: Path) -> Path:
    return root.resolve() / "prototype" / "metadata.sqlite3"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:28]
    return f"{prefix}-{digest}"


def _safe_image_path(root: Path, relative: str) -> Path:
    path = ensure_inside(root, relative)
    cursor = root.resolve()
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise IntakeError("Reference candidate path traverses a symbolic link")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise IntakeError("Reference candidate is missing or outside the private data root")
    return path


def _white_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def _normalized_annotation_box(
    box: dict[str, float],
    raw_width: int,
    raw_height: int,
    orientation: int,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    corners = [
        (box["xmin"], box["ymin"]),
        (box["xmax"], box["ymin"]),
        (box["xmin"], box["ymax"]),
        (box["xmax"], box["ymax"]),
    ]
    def transform(x: float, y: float) -> tuple[float, float]:
        if orientation == 2:
            return raw_width - x, y
        if orientation == 3:
            return raw_width - x, raw_height - y
        if orientation == 4:
            return x, raw_height - y
        if orientation == 5:
            return y, x
        if orientation == 6:
            return raw_height - y, x
        if orientation == 7:
            return raw_height - y, raw_width - x
        if orientation == 8:
            return y, raw_width - x
        return x, y

    points = [transform(x, y) for x, y in corners]
    x0 = max(0.0, min(float(image_width), min(point[0] for point in points)))
    y0 = max(0.0, min(float(image_height), min(point[1] for point in points)))
    x1 = max(0.0, min(float(image_width), max(point[0] for point in points)))
    y1 = max(0.0, min(float(image_height), max(point[1] for point in points)))
    if x1 - x0 < 96 or y1 - y0 < 96:
        return None
    return x0 / image_width, y0 / image_height, x1 / image_width, y1 / image_height


def _pixel_hash(image: Image.Image) -> str:
    normalized = _white_rgb(image)
    digest = hashlib.sha256()
    digest.update(f"{normalized.width}x{normalized.height}:RGB\0".encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def _perceptual_hash(image: Image.Image) -> str:
    pixels = list(_white_rgb(image).convert("L").resize((32, 32), Image.Resampling.LANCZOS).getdata())
    horizontal = [
        [sum(pixels[y * 32 + x] * PHASH_COSINE[x][u] for x in range(32)) for u in range(8)]
        for y in range(32)
    ]
    values = [
        sum(horizontal[y][u] * PHASH_COSINE[y][v] for y in range(32))
        for v in range(8)
        for u in range(8)
    ]
    median = sorted(values[1:])[len(values[1:]) // 2]
    bits = sum((1 << index) for index, value in enumerate(values) if value > median)
    return f"{bits:016x}"


def _screen_metrics(image: Image.Image) -> dict[str, float]:
    gray = _white_rgb(image).convert("L")
    sample = gray.resize((min(256, gray.width), min(256, gray.height)), Image.Resampling.BOX)
    values = list(sample.getdata())
    ink = sum(value < 248 for value in values) / max(1, len(values))
    return {"sample_ink_ratio": round(ink, 6), "sample_contrast": float(max(values) - min(values)) / 255}


def _anita_inventory(root: Path, source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    source_root = ensure_inside(root, source["extracted_path"])
    if not source_root.is_dir() or source_root.is_symlink():
        return []
    terms = _anita_work_records(REPOSITORY_ROOT / "config" / "anita-work-terms.json")
    results: list[dict[str, Any]] = []
    for work_dir in sorted(source_root.iterdir(), key=lambda path: path.name.casefold()):
        folder = work_dir / "sketch"
        if not work_dir.is_dir() or work_dir.is_symlink() or not folder.is_dir() or folder.is_symlink():
            continue
        for current, directories, filenames in os.walk(folder, followlinks=False):
            directories.sort(key=str.casefold)
            current_path = Path(current)
            if any((current_path / name).is_symlink() for name in directories):
                continue
            for filename in sorted(filenames, key=str.casefold):
                candidate = current_path / filename
                if candidate.is_symlink() or candidate.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                relative = candidate.relative_to(root).as_posix()
                clip = candidate.parent.relative_to(folder).as_posix()
                work_id = work_dir.name
                results.append({
                    "source_id": source["id"],
                    "relative_path": relative,
                    "work_id": work_id,
                    "clip_id": f"anita:{work_id}:{clip}",
                    "source_family": f"anita:work:{work_id}",
                    "upstream_partition": "not_applicable",
                    "source_identity_verified": True,
                    "artist": terms.get(work_id.casefold(), {}).get("artist"),
                    "license_id": terms.get(work_id.casefold(), {}).get("license") or "unspecified-local-prototype",
                    "work_title": terms.get(work_id.casefold(), {}).get("title", work_id),
                })
    return results


def _emoreccom_partition_allowlist(root: Path) -> tuple[set[str], set[str], dict[str, dict[str, Any]]]:
    """Collect filename membership before any panel is decoded; test membership wins."""
    directory = ensure_inside(root, "visemocomic_csvs")
    partitions: dict[str, set[str]] = {"source_train": set(), "source_test_locked": set()}
    annotations: dict[str, dict[str, Any]] = defaultdict(lambda: {"annotations": [], "partition": None})
    for partition, filename in (("source_train", "emoreccom_train.csv"), ("source_test_locked", "emoreccom_test.csv")):
        csv_path = directory / filename
        if not csv_path.is_file() or csv_path.is_symlink():
            continue
        csv_hash = sha256_file(csv_path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "filename" not in reader.fieldnames:
                raise IntakeError(f"EmoRecCom {filename} has no filename allowlist column")
            for row_number, row in enumerate(reader, start=2):
                name = (row.get("filename") or "").strip()
                if not name or Path(name).name != name or ".." in name or "\\" in name:
                    raise IntakeError(f"EmoRecCom {filename} contains an unsafe panel filename")
                folded = name.casefold()
                partitions[partition].add(folded)
                # Parse annotation values only after partition identity is known. Image bytes
                # remain untouched here; test files are only remembered for exclusion.
                if partition == "source_test_locked":
                    continue
                panel = annotations[folded]
                panel["partition"] = partition
                panel["filename"] = name
                panel["annotation_source_file"] = filename
                panel["annotation_source_sha256"] = csv_hash
                box_data: dict[str, dict[str, float]] = {}
                for kind in ("face", "body"):
                    try:
                        box = {
                            axis: float(row[f"{kind}_{axis}"])
                            for axis in ("xmin", "ymin", "xmax", "ymax")
                        }
                    except (KeyError, TypeError, ValueError):
                        continue
                    if all(math.isfinite(value) for value in box.values()) and box["xmin"] < box["xmax"] and box["ymin"] < box["ymax"]:
                        box_data[kind] = box
                if box_data:
                    panel["annotations"].append({"row_number": row_number, "boxes": box_data})
    train = partitions["source_train"] - partitions["source_test_locked"]
    return train, partitions["source_test_locked"], dict(annotations)


def _emoreccom_inventory(root: Path, source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    train_names, _test_names, annotations = _emoreccom_partition_allowlist(root)
    source_root = ensure_inside(root, source["extracted_path"])
    image_root = source_root / "emoreccom"
    if not image_root.is_dir() or image_root.is_symlink():
        return []
    result: list[dict[str, Any]] = []
    # Only exact official training filenames are opened/decoded. Test and unknown panels
    # never cross this allowlist boundary.
    for name in sorted(train_names):
        path = image_root / name
        if path.is_symlink() or not path.is_file():
            continue
        info = annotations.get(name, {})
        prefix = name.split("_", 1)[0]
        result.append({
            "source_id": source["id"],
            "relative_path": path.relative_to(root).as_posix(),
            "work_id": None,
            "clip_id": f"emoreccom:train:{prefix}",
            "source_family": None,
            "upstream_partition": "source_train",
            "source_identity_verified": False,
            "provisional_book_prefix": prefix,
            "license_id": "unspecified-local-prototype-source",
            "artist": None,
            "work_title": source.get("title", "EmoRecCom panel"),
            "annotation": info,
        })
    return result


def _inventory(root: Path) -> list[dict[str, Any]]:
    sources = load_registry(REPOSITORY_ROOT / "config" / "sources.json")["sources"]
    source_map = {row["id"]: row for row in sources}
    entries: list[dict[str, Any]] = []
    anita = source_map.get("anita-dataset")
    comic = source_map.get("emoreccom-panels")
    if anita:
        entries.extend(_anita_inventory(root, anita))
    if comic:
        entries.extend(_emoreccom_inventory(root, comic))
    return entries


def _inventory_digest(entries: list[dict[str, Any]], root: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(entries, key=lambda row: (row["source_id"], row["relative_path"])):
        path = _safe_image_path(root, item["relative_path"])
        stat = path.stat()
        digest.update(f"{item['source_id']}\0{item['relative_path']}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def create_run(root: Path, *, target: int, seed: int = 20261005) -> dict[str, Any]:
    if not 1 <= target <= 10_000:
        raise ValueError("Reference target must be between 1 and 10,000")
    root = root.resolve(strict=True)
    database = _db_path(root)
    migrate(database)
    entries = _inventory(root)
    inventory_sha = _inventory_digest(entries, root)
    run_id = "refs-" + uuid4().hex[:24]
    recipe = {
        "schema_version": 1,
        "seed": seed,
        "target_count": target,
        "maximum_candidates": 20_000,
        "sources": ["anita-dataset", "emoreccom-panels"],
        "maximum_variants_per_root": 3,
        "maximum_work_share": 0.20,
        "pixel_hash_version": "exif-white-rgb-exact-v1",
        "perceptual_hash_version": "dct-32x32-8x8-v1",
        "near_duplicate_proposal": {
            "mobileclip_model_id": "mobileclip2-s2-dfndr2b",
            "cosine_threshold": 0.94,
            "perceptual_hash_max_hamming": 12,
            "pair_review_limit_per_candidate": 3,
        },
        "crop_recipe_version": CROP_RECIPE,
        "local_model_profile_required": True,
        "model_review_prompt_sha256": sha256_file(REPOSITORY_ROOT / "config" / "references-review-prompt-v1.txt"),
    }
    recipe_json = _json(recipe)
    db = connect(database)
    inserted = 0
    inventory_excluded = 0
    try:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO reference_runs(run_id,target_count,recipe_json,recipe_sha256,inventory_sha256,status,cursor_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (run_id, target, recipe_json, hashlib.sha256(recipe_json.encode()).hexdigest(), inventory_sha, "inventory", _json({"inventory_count": len(entries), "screened": 0, "reviewed": 0}), _now(), _now()),
        )
        for item in entries:
            path = _safe_image_path(root, item["relative_path"])
            size = path.stat().st_size
            if size <= 0 or size > MAX_IMAGE_BYTES:
                _record_error(
                    db,
                    run_id,
                    None,
                    "inventory",
                    "file_size_outside_bound",
                    f"Excluded {item['relative_path']} at {size} bytes; allowed range is 1..{MAX_IMAGE_BYTES}.",
                    False,
                )
                inventory_excluded += 1
                continue
            candidate_id = _stable_id("refc", run_id, item["source_id"], item["relative_path"])
            metadata = {
                **item,
                "inventory_size": size,
                "asset_id": _stable_id("reference", item["source_id"], item["relative_path"]),
            }
            cursor = db.execute(
                "INSERT OR IGNORE INTO reference_candidates(candidate_id,run_id,source_id,relative_path,source_sha256,byte_length,media_type,width,height,work_id,clip_id,status,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (candidate_id, run_id, item["source_id"], item["relative_path"], "0" * 64, size, "application/octet-stream", 1, 1, item.get("work_id"), item.get("clip_id"), "discovered", _json(metadata)),
            )
            inserted += cursor.rowcount
        if inventory_excluded:
            cursor = json.loads(
                db.execute("SELECT cursor_json FROM reference_runs WHERE run_id=?", (run_id,)).fetchone()[0]
            )
            cursor["inventory_exclusions"] = inventory_excluded
            db.execute(
                "UPDATE reference_runs SET error_count=error_count+?,cursor_json=?,updated_at=? WHERE run_id=?",
                (inventory_excluded, _json(cursor), _now(), run_id),
            )
        db.execute("COMMIT")
    except Exception:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise
    finally:
        db.close()
    return {"run_id": run_id, "status": "inventory", "inventory_count": len(entries), "candidate_count": inserted, "inventory_excluded": inventory_excluded, "inventory_sha256": inventory_sha}


def _record_error(db: sqlite3.Connection, run_id: str, candidate_id: str | None, stage: str, code: str, message: str, retryable: bool) -> None:
    db.execute(
        "INSERT INTO reference_errors(error_id,run_id,candidate_id,stage,error_code,message,retryable,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (str(uuid4()), run_id, candidate_id, stage, code[:120], message[:1000], int(retryable), _now()),
    )


def screen_run(root: Path, run_id: str, *, limit: int = 20_000) -> dict[str, int]:
    root = root.resolve(strict=True)
    db = connect(_db_path(root))
    try:
        run = db.execute("SELECT status FROM reference_runs WHERE run_id=?", (run_id,)).fetchone()
        if not run:
            raise LookupError("Reference run was not found")
        rows = db.execute(
            "SELECT * FROM reference_candidates WHERE run_id=? AND status='discovered' ORDER BY source_id,relative_path LIMIT ?",
            (run_id, limit),
        ).fetchall()
        screened = skipped = failed = 0
        for row in rows:
            try:
                path = _safe_image_path(root, row["relative_path"])
                byte_count = path.stat().st_size
                if not 0 < byte_count <= MAX_IMAGE_BYTES:
                    raise ValueError("unsupported_byte_length")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        if image.format not in SUPPORTED_FORMATS or getattr(image, "n_frames", 1) != 1 or image.width * image.height > MAX_PIXELS:
                            raise ValueError("unsupported_image_dimensions_or_format")
                        media_type = SUPPORTED_FORMATS[image.format]
                        image.verify()
                    with Image.open(path) as image:
                        image.load()
                        normalized = _white_rgb(image)
                        pixel_hash = _pixel_hash(normalized)
                        perceptual_hash = _perceptual_hash(normalized)
                        metrics = _screen_metrics(normalized)
                        if metrics["sample_ink_ratio"] == 0:
                            raise ValueError("blank_image")
                        width, height = normalized.size
                digest = sha256_file(path)
                metadata = json.loads(row["metadata_json"])
                metadata.update({"screen_metrics": metrics, "screened_at": _now(), "source_mtime_ns": path.stat().st_mtime_ns})
                db.execute(
                    "UPDATE reference_candidates SET source_sha256=?,pixel_sha256=?,perceptual_hash=?,byte_length=?,media_type=?,width=?,height=?,status='screened',metadata_json=? WHERE candidate_id=? AND status='discovered'",
                    (digest, pixel_hash, perceptual_hash, byte_count, media_type, width, height, _json(metadata), row["candidate_id"]),
                )
                screened += 1
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning, IntakeError) as error:
                code = str(error)[:120] or "decode_failed"
                db.execute("UPDATE reference_candidates SET status='error',revision=revision+1 WHERE candidate_id=?", (row["candidate_id"],))
                _record_error(db, run_id, row["candidate_id"], "screening", code, str(error), False)
                failed += 1
        # Group byte-identical and orientation-normalized decoded pixels. Keep one stable
        # representative, preserving every source alias in the candidate inventory.
        screeneds = db.execute(
            "SELECT candidate_id,source_sha256,pixel_sha256 FROM reference_candidates WHERE run_id=? AND status='screened' ORDER BY source_id,relative_path",
            (run_id,),
        ).fetchall()
        representatives: dict[str, tuple[str, bool]] = {}
        # Existing build hashes are already trusted, content-addressed review records;
        # seed exact-byte deduplication without rehashing every gallery image from disk.
        current_items, _catalog_id = _current_catalog_items(root)
        current_by_hash = {item.sha256: item for item in current_items}
        for item in current_items:
            representatives.setdefault("bytes:" + item.sha256, (item.asset_id, True))
        known_pixel_hashes: set[str] = set()
        prior_rows = db.execute(
            "SELECT source_sha256,pixel_sha256 FROM reference_candidates WHERE status='accepted' AND pixel_sha256 IS NOT NULL"
        ).fetchall()
        for prior in prior_rows:
            if prior["source_sha256"] in current_by_hash:
                representatives.setdefault(
                    "pixels:" + prior["pixel_sha256"],
                    (current_by_hash[prior["source_sha256"]].asset_id, True),
                )
                known_pixel_hashes.add(prior["source_sha256"])
        for item in current_items:
            if item.sha256 in known_pixel_hashes:
                continue
            try:
                path = _safe_image_path(root, item.relative_path)
                if path.stat().st_size != item.byte_length or sha256_file(path) != item.sha256:
                    continue
                with Image.open(path) as image:
                    image.load()
                    pixels = _pixel_hash(_white_rgb(image))
                representatives.setdefault("pixels:" + pixels, (item.asset_id, True))
            except (OSError, ValueError, IntakeError, UnidentifiedImageError):
                continue
        for item in screeneds:
            byte_key = "bytes:" + item["source_sha256"]
            pixel_key = "pixels:" + item["pixel_sha256"]
            representative = representatives.get(byte_key) or representatives.get(pixel_key)
            if representative is None:
                representatives[byte_key] = (item["candidate_id"], False)
                representatives[pixel_key] = (item["candidate_id"], False)
            else:
                representative_id, is_existing_asset = representative
                representatives.setdefault(byte_key, representative)
                representatives.setdefault(pixel_key, representative)
                if is_existing_asset:
                    db.execute(
                        "UPDATE reference_candidates SET status='exact_duplicate',metadata_json=json_set(metadata_json,'$.exact_representative_asset_id',?) WHERE candidate_id=?",
                        (representative_id, item["candidate_id"]),
                    )
                else:
                    db.execute(
                        "UPDATE reference_candidates SET status='exact_duplicate',parent_candidate_id=?,metadata_json=json_set(metadata_json,'$.exact_representative',?) WHERE candidate_id=?",
                        (representative_id, representative_id, item["candidate_id"]),
                    )
                skipped += 1
        run_row = db.execute("SELECT cursor_json FROM reference_runs WHERE run_id=?", (run_id,)).fetchone()
        cursor = json.loads(run_row["cursor_json"])
        cursor["screened"] = db.execute("SELECT COUNT(*) FROM reference_candidates WHERE run_id=? AND status='screened'", (run_id,)).fetchone()[0]
        cursor["errors"] = db.execute("SELECT COUNT(*) FROM reference_errors WHERE run_id=?", (run_id,)).fetchone()[0]
        db.execute(
            "UPDATE reference_runs SET status='screening',cursor_json=?,error_count=error_count+?,updated_at=? WHERE run_id=?",
            (_json(cursor), failed, _now(), run_id),
        )
        return {"screened": screened, "exact_duplicates": skipped, "errors": failed}
    finally:
        db.close()


def _selected_order(rows: list[sqlite3.Row], seed: int, budget: int) -> list[sqlite3.Row]:
    groups: dict[tuple[str, str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        key = (row["source_id"], row["work_id"] or "", row["clip_id"] or "")
        groups[key].append(row)
    for key, items in groups.items():
        items.sort(key=lambda item: hashlib.sha256(f"{seed}:{key}:{item['candidate_id']}".encode()).hexdigest())
        # Spread review effort across visually different frames in each source/work/clip
        # before the expensive vision pass. The qualified pair reviewer still adjudicates
        # every near-duplicate proposal after a candidate is accepted.
        remaining = list(items)
        diversified: list[sqlite3.Row] = []
        chosen_hashes: list[int] = []
        while remaining:
            if not chosen_hashes:
                chosen = remaining[0]
            else:
                def diversity_key(
                    item: sqlite3.Row,
                    prior_hashes: list[int] = chosen_hashes,
                    group_key: tuple[str, str, str] = key,
                ) -> tuple[int, str]:
                    value = item["perceptual_hash"]
                    distance = -1 if not value else min(
                        (int(value, 16) ^ prior).bit_count() for prior in prior_hashes
                    )
                    tie_breaker = hashlib.sha256(
                        f"{seed}:{group_key}:{item['candidate_id']}".encode()
                    ).hexdigest()
                    return distance, tie_breaker

                chosen = max(remaining, key=diversity_key)
            remaining.remove(chosen)
            diversified.append(chosen)
            if chosen["perceptual_hash"]:
                chosen_hashes.append(int(chosen["perceptual_hash"], 16))
        groups[key] = diversified
    keys = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    output: list[sqlite3.Row] = []
    offset = 0
    while len(output) < min(budget, len(rows)):
        progressed = False
        for key in keys:
            bucket = groups[key]
            if offset < len(bucket):
                output.append(bucket[offset])
                progressed = True
                if len(output) >= budget:
                    break
        if not progressed:
            break
        offset += 1
    return output


def select_candidates(root: Path, run_id: str, *, budget: int) -> dict[str, int]:
    db = connect(_db_path(root.resolve()))
    try:
        run = db.execute("SELECT recipe_json FROM reference_runs WHERE run_id=?", (run_id,)).fetchone()
        if not run:
            raise LookupError("Reference run was not found")
        recipe = json.loads(run["recipe_json"])
        rows = db.execute(
            "SELECT * FROM reference_candidates WHERE run_id=? AND status='screened' ORDER BY source_id,relative_path",
            (run_id,),
        ).fetchall()
        selected = _selected_order(rows, int(recipe["seed"]), budget)
        for row in selected:
            db.execute("UPDATE reference_candidates SET status='selected' WHERE candidate_id=? AND status='screened'", (row["candidate_id"],))
        return {"available": len(rows), "selected": len(selected), "remaining": len(rows) - len(selected)}
    finally:
        db.close()


def _reference_embedding_neighbors(
    root: Path, run_id: str, rows: list[sqlite3.Row]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    candidates = [
        {
            "candidate_id": row["candidate_id"],
            "relative_path": row["relative_path"],
            "image_sha256": row["source_sha256"],
            "byte_length": row["byte_length"],
        }
        for row in rows
    ]
    if not candidates:
        return {}, {"status": "skipped_empty_selection", "candidate_count": 0}
    candidate_hash = hashlib.sha256(_json(candidates).encode()).hexdigest()[:24]
    directory = root / "prototype/reviewer/embedding-proposals"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    stem = f"{run_id}-{candidate_hash}"
    input_relative = f"prototype/reviewer/embedding-proposals/candidates-{stem}.json"
    output_relative = f"prototype/reviewer/embedding-proposals/neighbors-{stem}.json"
    manifest = {
        "schema_version": 1,
        "kind": "drawable-reference-embedding-candidates",
        "run_id": run_id,
        "similarity_threshold": 0.94,
        "items": candidates,
    }
    input_path = ensure_inside(root, input_relative)
    if input_path.exists():
        if json.loads(input_path.read_text(encoding="utf-8")) != manifest:
            raise IntakeError("Immutable reference embedding candidate manifest changed")
    else:
        input_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(input_path, 0o600)
    output_path = ensure_inside(root, output_relative)
    if not output_path.is_file():
        executable = REPOSITORY_ROOT / "ml/.venv/bin/python"
        command = [
            str(executable),
            "-u",
            "-m",
            "drawable_ml.prototype_worker",
            "reference-neighbors",
            "--data-root",
            str(root),
            "--candidate-manifest",
            input_relative,
            "--output",
            output_relative,
        ]
        environment = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "TOKENIZERS_PARALLELISM": "false",
        }
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=6 * 60 * 60,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr[-3000:] or "Frozen embedding proposal failed")
    result = json.loads(output_path.read_text(encoding="utf-8"))
    if (
        result.get("kind") != "drawable-reference-embedding-neighbors"
        or result.get("candidate_manifest_sha256") != sha256_file(input_path)
        or result.get("candidate_count") != len(candidates)
        or not isinstance(result.get("neighbors"), dict)
    ):
        raise IntakeError("Frozen embedding proposal artifact is stale or malformed")
    return result["neighbors"], {
        key: result.get(key)
        for key in (
            "encoder_model_id",
            "encoder_sha256",
            "preprocessing_version",
            "implementation_sha256",
            "similarity_threshold",
            "candidate_count",
            "active_gallery_count",
            "cache_hits",
            "cache_misses",
        )
    } | {"status": "complete", "output_path": output_relative}


def _review_is_acceptable(review: LocalVisualReview) -> bool:
    return (
        review.useful_character_drawing
        and review.safe_content
        and review.style is not None
        and review.primary_scope is not None
        and review.quality >= 2
        and review.text_coverage <= 0.10
        and not review.text_crosses_important_anatomy
        and not review.essential_anatomy_clipped
        and not review.damaged_or_unreadable
        and not review.uncertain
    )


def _review_labels(review: LocalVisualReview, elapsed_ms: float) -> dict[str, Any]:
    return {
        "style": review.style or "gesture_sketch",
        "primary_scope": review.primary_scope or "full_body",
        "secondary_scopes": review.secondary_scopes,
        "person_count": review.person_count,
        "quality": review.quality,
        "sfw": "safe" if review.safe_content else "unsafe",
        "character_useful": review.useful_character_drawing,
        "text_coverage": review.text_coverage,
        "clipped": review.essential_anatomy_clipped,
        "damaged": review.damaged_or_unreadable,
        "duplicate_of": None,
        "review_seconds": 0,
        "keep_reject_reason": review.reason,
        "inference_ms": round(elapsed_ms, 3),
        "text_crosses_important_anatomy": review.text_crosses_important_anatomy,
        "uncertain": review.uncertain,
    }


def _review_candidate(root: Path, db: sqlite3.Connection, run_id: str, row: sqlite3.Row, reviewer: LlamaReviewer, profile: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    path = _safe_image_path(root, row["relative_path"])
    if path.stat().st_size != row["byte_length"] or sha256_file(path) != row["source_sha256"]:
        raise IntakeError("Candidate bytes changed after inventory screening")
    with Image.open(path) as source:
        source.load()
        raw_width, raw_height = source.size
        try:
            orientation = int(source.getexif().get(274, 1))
        except (TypeError, ValueError):
            orientation = 1
        image = _white_rgb(source)
    review, elapsed_ms = review_image(reviewer, image)
    accepted = _review_is_acceptable(review)
    review_id = str(uuid4())
    provenance = ReviewProvenanceV1(
        origin="local_model",
        reviewer_id=f"model-{reviewer.model['id']}",
        profile_id=profile["profile_id"],
        model_id=reviewer.model["id"],
        model_sha256=reviewer.model_sha256,
        runtime_id="llama.cpp-b11425",
        runtime_sha256=reviewer.runtime_sha256,
        prompt_version=reviewer.prompt_version,
        rubric_version=RUBRIC_VERSION,
        run_id=run_id,
        image_sha256=row["source_sha256"],
        reviewed_revision=int(row["revision"]),
        evidence={"reason": review.reason, "quality": review.quality, "accepted_by_frozen_rule": accepted},
        inference_ms=elapsed_ms,
    )
    decision = _review_labels(review, elapsed_ms)
    decision.update({"accepted_by_frozen_rule": accepted, "review_provenance": provenance.model_dump(mode="json")})
    db.execute(
        "INSERT INTO reference_reviews(review_id,candidate_id,reviewed_revision,origin,reviewer_id,profile_id,model_sha256,runtime_sha256,prompt_version,rubric_version,run_id,image_sha256,evidence_json,decision_json,inference_ms,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (review_id, row["candidate_id"], row["revision"], "local_model", provenance.reviewer_id, profile["profile_id"], reviewer.model_sha256, reviewer.runtime_sha256, reviewer.prompt_version, RUBRIC_VERSION, run_id, row["source_sha256"], _json(provenance.model_dump(mode="json")), _json(decision), elapsed_ms, _now()),
    )
    new_status = "accepted" if accepted else "rejected" if not review.uncertain else "deferred"
    db.execute("UPDATE reference_candidates SET status=?,revision=revision+1,metadata_json=json_set(metadata_json,'$.review',json(?)) WHERE candidate_id=?", (new_status, _json(decision), row["candidate_id"]))
    crops: list[dict[str, Any]] = []
    if accepted and json.loads(row["metadata_json"]).get("candidate_type") != "crop":
        parent_metadata = json.loads(row["metadata_json"])
        proposed_boxes: list[CropBox] = []
        annotation = parent_metadata.get("annotation", {})
        annotation_rows = annotation.get("annotations", []) if isinstance(annotation, dict) else []
        for kind in ("face", "body"):
            for annotation_row in annotation_rows:
                boxes = annotation_row.get("boxes", {}) if isinstance(annotation_row, dict) else {}
                raw_box = boxes.get(kind) if isinstance(boxes, dict) else None
                if not isinstance(raw_box, dict):
                    continue
                try:
                    normalized = _normalized_annotation_box(
                        raw_box, raw_width, raw_height, orientation, image.width, image.height
                    )
                except (KeyError, TypeError, ValueError):
                    normalized = None
                if normalized is not None:
                    x0, y0, x1, y1 = normalized
                    proposed_boxes.append(
                        CropBox(
                            x=x0,
                            y=y0,
                            width=x1 - x0,
                            height=y1 - y0,
                            purpose=f"Official EmoRecCom {kind} annotation crop",
                        )
                    )
                    break
        def overlaps(left: CropBox, right: CropBox) -> bool:
            x_overlap = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
            y_overlap = max(0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y))
            intersection = x_overlap * y_overlap
            union = left.width * left.height + right.width * right.height - intersection
            return union > 0 and intersection / union >= 0.75

        for box in review.crop_boxes:
            if not any(overlaps(box, prior) for prior in proposed_boxes):
                proposed_boxes.append(box)
        for box in proposed_boxes[:3]:
            if len(crops) >= 3:
                break
            x0 = max(0, min(image.width, round(box.x * image.width)))
            y0 = max(0, min(image.height, round(box.y * image.height)))
            x1 = max(x0, min(image.width, round((box.x + box.width) * image.width)))
            y1 = max(y0, min(image.height, round((box.y + box.height) * image.height)))
            if x1 - x0 < 96 or y1 - y0 < 96:
                continue
            # Existing roots can have at most two accepted derivatives, for three variants total.
            siblings = db.execute("SELECT COUNT(*) FROM reference_candidates WHERE run_id=? AND parent_candidate_id=? AND status='accepted'", (run_id, row["candidate_id"])).fetchone()[0]
            if siblings >= 2:
                break
            cropped = image.crop((x0, y0, x1, y1))
            root_asset_id = parent_metadata.get("root_asset_id") or parent_metadata.get("asset_id") or row["candidate_id"]
            crop_asset_id = _stable_id("reference-crop", str(root_asset_id), row["source_sha256"], CROP_RECIPE, f"{x0}:{y0}:{x1}:{y1}")
            crop_id = _stable_id("cropc", row["candidate_id"], crop_asset_id)
            rel = f"prototype/reference-derivatives/{crop_id}.png"
            crop_path = _safe_image_path(root, rel) if (root / rel).exists() else ensure_inside(root, rel)
            crop_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not crop_path.exists():
                cropped.save(crop_path, format="PNG", optimize=True)
                os.chmod(crop_path, 0o600)
            crop_hash = sha256_file(crop_path)
            crop_meta = json.loads(row["metadata_json"])
            crop_meta.update({
                "candidate_type": "crop",
                "asset_id": crop_asset_id,
                "root_asset_id": str(root_asset_id),
                "parent_asset_id": parent_metadata.get("asset_id") or row["candidate_id"],
                "parent_hash": row["source_sha256"],
                "crop_recipe": CROP_RECIPE,
                "crop_coordinates_px": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "parent_width": image.width,
                "parent_height": image.height,
                "crop_purpose": box.purpose,
            })
            db.execute(
                "INSERT OR IGNORE INTO reference_candidates(candidate_id,run_id,source_id,relative_path,source_sha256,pixel_sha256,perceptual_hash,byte_length,media_type,width,height,work_id,clip_id,parent_candidate_id,status,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (crop_id, run_id, row["source_id"], rel, crop_hash, _pixel_hash(cropped), _perceptual_hash(cropped), crop_path.stat().st_size, "image/png", cropped.width, cropped.height, row["work_id"], row["clip_id"], row["candidate_id"], "selected", _json(crop_meta)),
            )
            # Parent review never approves the derivative. Review it independently.
            crop_row = db.execute("SELECT * FROM reference_candidates WHERE candidate_id=?", (crop_id,)).fetchone()
            if crop_row and crop_row["status"] == "selected":
                crop_ok, _ = _review_candidate(root, db, run_id, crop_row, reviewer, profile)
                crops.append({"candidate_id": crop_id, "accepted": crop_ok, "purpose": box.purpose, "size": [cropped.width, cropped.height]})
    return accepted, crops


def _resolve_near_duplicates(
    root: Path,
    db: sqlite3.Connection,
    run_id: str,
    candidate_id: str,
    reviewer: LlamaReviewer,
    profile: dict[str, Any],
    active_image_hashes: set[str],
    embedding_neighbors: dict[str, list[dict[str, Any]]],
) -> int:
    """Directly compare pHash proposals to accepted representatives; uncertain items stay out."""
    candidate = db.execute(
        "SELECT * FROM reference_candidates WHERE candidate_id=? AND run_id=? AND status='accepted'",
        (candidate_id, run_id),
    ).fetchone()
    if candidate is None or not candidate["perceptual_hash"]:
        return 0
    metadata = json.loads(candidate["metadata_json"])
    root_id = str(metadata.get("root_asset_id") or metadata.get("asset_id") or candidate_id)
    similarity_by_sha = {
        row.get("reference_sha256"): float(row.get("cosine", 0.0))
        for row in embedding_neighbors.get(candidate_id, [])
        if isinstance(row, dict) and isinstance(row.get("reference_sha256"), str)
    }
    neighbors: list[tuple[int, float, sqlite3.Row]] = []
    candidates = db.execute(
        "SELECT * FROM reference_candidates WHERE status='accepted' AND candidate_id<>? AND perceptual_hash IS NOT NULL",
        (candidate_id,),
    ).fetchall()
    for other in candidates:
        if other["run_id"] != run_id and other["source_sha256"] not in active_image_hashes:
            continue
        other_metadata = json.loads(other["metadata_json"])
        other_root = str(other_metadata.get("root_asset_id") or other_metadata.get("asset_id") or other["candidate_id"])
        if other_root == root_id:
            continue
        distance = (int(candidate["perceptual_hash"], 16) ^ int(other["perceptual_hash"], 16)).bit_count()
        embedding_cosine = similarity_by_sha.get(other["source_sha256"], 0.0)
        if distance <= 12 or embedding_cosine >= 0.94:
            neighbors.append((distance, embedding_cosine, other))
    neighbors.sort(key=lambda pair: (pair[0] if pair[0] <= 12 else 64, -pair[1], pair[2]["candidate_id"]))
    removed = 0
    for distance, embedding_cosine, other in neighbors[:3]:
        left, right = sorted((candidate, other), key=lambda row: row["candidate_id"])
        pair_id = _stable_id("refpair", run_id, left["candidate_id"], right["candidate_id"])
        db.execute(
            "INSERT OR IGNORE INTO reference_pairs(pair_id,run_id,left_candidate_id,right_candidate_id,left_image_sha256,right_image_sha256,state,evidence_json,reviewer_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (pair_id, run_id, left["candidate_id"], right["candidate_id"], left["source_sha256"], right["source_sha256"], "proposed", "{}", None, _now()),
        )
        try:
            loaded: list[Image.Image] = []
            for row in (left, right):
                path = _safe_image_path(root, row["relative_path"])
                if path.stat().st_size != row["byte_length"] or sha256_file(path) != row["source_sha256"]:
                    raise IntakeError("Pair proposal bytes changed after screening")
                with Image.open(path) as source:
                    source.load()
                    loaded.append(_white_rgb(source))
            review, elapsed_ms = review_pair(reviewer, loaded[0], loaded[1])
            evidence = {
                "relation": review.relation,
                "reason": review.reason,
                "pose_difference": review.pose_difference,
                "expression_difference": review.expression_difference,
                "facing_or_view_difference": review.facing_or_view_difference,
                "perspective_or_silhouette_difference": review.perspective_or_silhouette_difference,
                "useful_detail_difference": review.useful_detail_difference,
                "perceptual_distance": distance,
                "embedding_cosine": embedding_cosine,
                "profile_id": profile["profile_id"],
                "model_sha256": reviewer.model_sha256,
                "runtime_sha256": reviewer.runtime_sha256,
                "pair_prompt_sha256": profile["pair_review_prompt_sha256"],
                "inference_ms": round(elapsed_ms, 3),
            }
            if review.relation in {"same_image", "adjacent_frame_duplicate"}:
                state = "same"
            elif review.relation in {"distinct", "related_distinct_pose"}:
                state = "distinct"
            else:
                state = "uncertain"
        except (OSError, IntakeError, ReviewerError, ValidationError, ValueError, RuntimeError) as error:
            state = "uncertain"
            evidence = {
                "relation": "uncertain",
                "reason": f"Pair reviewer failed: {type(error).__name__}",
                "perceptual_distance": distance,
                "profile_id": profile["profile_id"],
            }
            _record_error(db, run_id, candidate_id, "pair_review", type(error).__name__, str(error), True)
            db.execute(
                "UPDATE reference_runs SET error_count=error_count+1,updated_at=? WHERE run_id=?",
                (_now(), run_id),
            )
        db.execute(
            "UPDATE reference_pairs SET state=?,evidence_json=?,reviewer_id=? WHERE pair_id=?",
            (state, _json(evidence), reviewer.model["id"], pair_id),
        )
        if state in {"same", "uncertain"}:
            metadata["near_duplicate_review"] = {"pair_id": pair_id, **evidence}
            db.execute(
                "UPDATE reference_candidates SET status='near_duplicate',revision=revision+1,metadata_json=? WHERE candidate_id=? AND status='accepted'",
                (_json(metadata), candidate_id),
            )
            children = db.execute(
                "SELECT candidate_id,metadata_json FROM reference_candidates WHERE parent_candidate_id=? AND status='accepted'",
                (candidate_id,),
            ).fetchall()
            for child in children:
                child_metadata = json.loads(child["metadata_json"])
                child_metadata["near_duplicate_parent"] = candidate_id
                db.execute(
                    "UPDATE reference_candidates SET status='near_duplicate',revision=revision+1,metadata_json=? WHERE candidate_id=?",
                    (_json(child_metadata), child["candidate_id"]),
                )
            removed += 1 + len(children)
            break
    return removed


def _qualified_profile(root: Path) -> dict[str, Any]:
    path = ensure_inside(root, "prototype/reviewer/qualified-profile.json")
    if path.is_symlink() or not path.is_file():
        raise PermissionError("A passing held-out local reviewer qualification is required before automatic acceptance")
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewerError("Qualified reviewer profile is malformed") from error
    if profile.get("schema_version") != 1 or profile.get("status") != "qualified" or profile.get("profile_id") is None:
        raise PermissionError("The local reviewer profile has not passed held-out qualification")
    if profile.get("prompt_sha256") != sha256_file(REPOSITORY_ROOT / "config" / "references-review-prompt-v1.txt"):
        raise PermissionError("The qualified prompt changed; reviewer requalification is required")
    if (
        profile.get("pair_reviewer_status") != "qualified"
        or profile.get("pair_review_prompt_sha256")
        != sha256_file(REPOSITORY_ROOT / "config" / "references-pair-review-prompt-v1.txt")
    ):
        raise PermissionError("A qualified, unchanged pair reviewer is required before automatic acceptance")
    report_path = ensure_inside(root, str(profile.get("qualification_report", "")))
    if sha256_file(report_path) != profile.get("qualification_report_sha256"):
        raise PermissionError("The reviewer qualification report changed; requalification is required")
    return profile


def expand_run(root: Path, *, target: int, run_id: str | None = None, review_budget: int | None = None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    profile = _qualified_profile(root)
    if run_id is None:
        result = create_run(root, target=target)
        run_id = result["run_id"]
    db = connect(_db_path(root))
    try:
        run = db.execute("SELECT target_count,recipe_json,status FROM reference_runs WHERE run_id=?", (run_id,)).fetchone()
        if not run:
            raise LookupError("Reference run was not found")
        if int(run["target_count"]) != target:
            raise ValueError("Resume target differs from the immutable run recipe")
    finally:
        db.close()
    screen = screen_run(root, run_id)
    current = status_run(root, run_id)
    already_accepted = current["accepted_this_run"]
    remaining = max(0, target - current["gallery_size_including_existing"] - already_accepted)
    budget = review_budget or max(remaining * 3, remaining + 50)
    selection = select_candidates(root, run_id, budget=budget)
    db = connect(_db_path(root))
    try:
        active_items, _active_catalog_id = _current_catalog_items(root)
        active_image_hashes = {item.sha256 for item in active_items}
        rows = db.execute(
            "SELECT * FROM reference_candidates WHERE run_id=? AND status='selected' ORDER BY source_id,work_id,clip_id,candidate_id",
            (run_id,),
        ).fetchall()
        embedding_neighbors: dict[str, list[dict[str, Any]]] = {}
        try:
            embedding_neighbors, embedding_metrics = _reference_embedding_neighbors(root, run_id, rows)
            print(
                "Frozen embedding proposals: "
                f"{embedding_metrics.get('candidate_count', 0)} candidates, "
                f"{embedding_metrics.get('cache_hits', 0)} cache hits, "
                f"{embedding_metrics.get('cache_misses', 0)} cache misses.",
                flush=True,
            )
        except (OSError, IntakeError, RuntimeError, ValueError, KeyError, TypeError) as error:
            embedding_metrics = {"status": "failed", "error": str(error)[:1000]}
            _record_error(
                db,
                run_id,
                None,
                "embedding_proposal",
                type(error).__name__,
                str(error),
                True,
            )
            db.execute(
                "UPDATE reference_runs SET error_count=error_count+1,updated_at=? WHERE run_id=?",
                (_now(), run_id),
            )
            print(f"Frozen embedding proposals failed; pHash proposals remain active: {error}", flush=True)
        reviewed = accepted = rejected = deferred = near_duplicate = 0
        crop_reviews = 0
        errors: list[str] = []
        model_id = str(profile["model_id"])
        with LlamaReviewer(root, model_id, q8=profile.get("quantization") == "Q8_0") as reviewer:
            if (
                profile.get("model_sha256") != reviewer.model_sha256
                or profile.get("runtime_sha256") != reviewer.runtime_sha256
                or profile.get("gpu_layers") != reviewer.gpu_layers
                or profile.get("image_max_edge") != reviewer.config["review_defaults"]["image_max_edge"]
            ):
                raise PermissionError("Qualified reviewer artifacts or runtime settings changed")
            for row in rows:
                if accepted >= remaining:
                    break
                existing_work = db.execute(
                    "SELECT COUNT(*) FROM reference_candidates WHERE run_id=? AND status='accepted' AND work_id=?",
                    (run_id, row["work_id"]),
                ).fetchone()[0]
                max_work_count = max(1, math.floor(target * 0.20))
                if row["work_id"] and existing_work >= max_work_count:
                    db.execute("UPDATE reference_candidates SET status='deferred' WHERE candidate_id=?", (row["candidate_id"],))
                    deferred += 1
                    continue
                try:
                    accepted_item, crops = _review_candidate(root, db, run_id, row, reviewer, profile)
                    reviewed += 1
                    newly_accepted = int(accepted_item) + sum(int(crop["accepted"]) for crop in crops)
                    removed = 0
                    if accepted_item:
                        removed += _resolve_near_duplicates(
                            root,
                            db,
                            run_id,
                            row["candidate_id"],
                            reviewer,
                            profile,
                            active_image_hashes,
                            embedding_neighbors,
                        )
                    parent_still_accepted = db.execute(
                        "SELECT status FROM reference_candidates WHERE candidate_id=?", (row["candidate_id"],)
                    ).fetchone()["status"] == "accepted"
                    if parent_still_accepted:
                        for crop in crops:
                            if crop["accepted"]:
                                child = db.execute(
                                    "SELECT status FROM reference_candidates WHERE candidate_id=?", (crop["candidate_id"],)
                                ).fetchone()
                                if child and child["status"] == "accepted":
                                    removed += _resolve_near_duplicates(
                                        root,
                                        db,
                                        run_id,
                                        crop["candidate_id"],
                                        reviewer,
                                        profile,
                                        active_image_hashes,
                                        embedding_neighbors,
                                    )
                    accepted += newly_accepted - removed
                    near_duplicate += removed
                    rejected += int(not accepted_item) + sum(int(not crop["accepted"]) for crop in crops)
                    crop_reviews += len(crops)
                except (OSError, IntakeError, ReviewerError, ValidationError, ValueError, RuntimeError) as error:
                    db.execute("UPDATE reference_candidates SET status='error' WHERE candidate_id=?", (row["candidate_id"],))
                    _record_error(db, run_id, row["candidate_id"], "review", type(error).__name__, str(error), True)
                    errors.append(f"{row['candidate_id']}: {error}")
        totals = db.execute("SELECT status,COUNT(*) AS n FROM reference_candidates WHERE run_id=? GROUP BY status", (run_id,)).fetchall()
        previous_cursor = json.loads(
            db.execute("SELECT cursor_json FROM reference_runs WHERE run_id=?", (run_id,)).fetchone()[0]
        )
        cursor = {
            **previous_cursor,
            "screened": screen["screened"],
            "reviewed": reviewed,
            "accepted": sum(row["n"] for row in totals if row["status"] == "accepted"),
            "stage_counts": {row["status"]: row["n"] for row in totals},
            "embedding_proposals": embedding_metrics,
        }
        db.execute(
            "UPDATE reference_runs SET status=?,cursor_json=?,error_count=error_count+?,updated_at=? WHERE run_id=?",
            ("building" if accepted >= remaining else "reviewing", _json(cursor), len(errors), _now(), run_id),
        )
        return {
            "run_id": run_id,
            "status": "awaiting_finalize" if accepted >= remaining else "screened_candidates_exhausted_or_resume_needed",
            "target": target,
            "accepted_this_pass": accepted,
            "rejected_this_pass": rejected,
            "near_duplicate_this_pass": near_duplicate,
            "deferred_this_pass": deferred,
            "crop_reviews": crop_reviews,
            "embedding_proposals": embedding_metrics,
            "selected": selection["selected"],
            "errors": errors[:20],
            "stage_counts": cursor["stage_counts"],
        }
    finally:
        db.close()


def _active_catalog_path(root: Path) -> Path | None:
    pointer_path = root / "prototype/search/active.json"
    if not pointer_path.is_file() or pointer_path.is_symlink():
        return None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        manifest_path = root / "prototype/search/builds" / pointer["build_id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256_file(manifest_path) != pointer["manifest_sha256"]:
            return None
        snapshot = root / manifest["snapshot_path"]
        if sha256_file(snapshot) != manifest["snapshot_sha256"]:
            return None
        return snapshot
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _active_available_ids(root: Path, catalog_id: str | None, fallback: set[str]) -> set[str]:
    if not catalog_id:
        return fallback
    try:
        pointer = json.loads((root / "prototype/search/active.json").read_text(encoding="utf-8"))
        if not pointer.get("build_id") or not pointer.get("availability_path"):
            return fallback
        mask_path = ensure_inside(root, pointer["availability_path"])
        if sha256_file(mask_path) != pointer.get("availability_sha256"):
            return fallback
        mask = json.loads(mask_path.read_text(encoding="utf-8"))
        if mask.get("catalog_id") != catalog_id or not isinstance(mask.get("items"), dict):
            return fallback
        active_path = _active_catalog_path(root)
        if active_path is None:
            return fallback
        document = json.loads(active_path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 2 or document.get("catalog_id") != catalog_id:
            return fallback
        items = {item["asset_id"]: item for item in document.get("items", []) if isinstance(item, dict)}
        return {
            asset_id
            for asset_id, row in mask["items"].items()
            if asset_id in items
            and isinstance(row, dict)
            and row.get("sha256") == items[asset_id].get("sha256")
            and row.get("reviewed_revision") == items[asset_id].get("asset_revision")
        }
    except (OSError, ValueError, TypeError, KeyError):
        return fallback


def withdraw_availability(root: Path, asset_ids: list[str]) -> dict[str, Any] | None:
    """Withdraw bytes from every retained V2 mask so rollback cannot resurrect them."""
    pointer_path = root / "prototype/search/active.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        current_path = ensure_inside(root, pointer["availability_path"])
        if sha256_file(current_path) != pointer.get("availability_sha256"):
            return None
        current_mask = json.loads(current_path.read_text(encoding="utf-8"))
        directory = root / "prototype/search/availability"
        if directory.is_symlink() or not directory.is_dir():
            return None
        grouped: dict[str, tuple[int, Path, dict[str, Any], dict[str, Any]]] = {}
        paths = [path for path in directory.glob("gallery-*.json") if path.is_file() and not path.is_symlink()]
        if current_path not in paths:
            paths.append(current_path)
        for path in paths:
            try:
                mask = json.loads(path.read_text(encoding="utf-8"))
                catalog_id = mask.get("catalog_id")
                revision = int(mask.get("revision", 0))
                rows = mask.get("items")
                if (
                    not isinstance(catalog_id, str)
                    or not re.fullmatch(r"gallery-[0-9a-f]{24}", catalog_id)
                    or revision < 0
                    or mask.get("schema_version") != 1
                    or not isinstance(rows, dict)
                ):
                    continue
                catalog_path = ensure_inside(root, f"prototype/catalogs/{catalog_id}.json")
                catalog = PrototypeGalleryCatalogV2.model_validate_json(catalog_path.read_bytes())
                by_id = {item.asset_id: item for item in catalog.items}
                if catalog.catalog_id != catalog_id or not set(rows).issubset(by_id):
                    continue
                if any(
                    not isinstance(row, dict)
                    or row.get("sha256") != by_id[asset_id].sha256
                    or row.get("reviewed_revision") != by_id[asset_id].asset_revision
                    for asset_id, row in rows.items()
                ):
                    continue
                existing = grouped.get(catalog_id)
                if existing is None or revision > existing[0]:
                    grouped[catalog_id] = (revision, path, mask, by_id)
            except (OSError, ValueError, TypeError, KeyError):
                continue
        changed: dict[str, tuple[int, str, str]] = {}
        for catalog_id, (revision, _path, mask, _by_id) in grouped.items():
            rows = mask["items"]
            actually_withdrawn = [asset_id for asset_id in asset_ids if asset_id in rows]
            if not actually_withdrawn:
                continue
            updated = dict(mask)
            updated["items"] = {asset_id: row for asset_id, row in rows.items() if asset_id not in actually_withdrawn}
            next_revision = revision + 1
            relative = f"prototype/search/availability/{catalog_id}-r{next_revision}.json"
            destination = ensure_inside(root, relative)
            while destination.exists() or destination.is_symlink():
                next_revision += 1
                relative = f"prototype/search/availability/{catalog_id}-r{next_revision}.json"
                destination = ensure_inside(root, relative)
            updated["revision"] = next_revision
            temporary = destination.with_name(destination.name + ".part")
            temporary.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
            changed[catalog_id] = (next_revision, relative, sha256_file(destination))
        current_catalog_id = current_mask.get("catalog_id")
        active_change = changed.get(current_catalog_id)
        if active_change:
            _revision, relative, checksum = active_change
            pointer["availability_path"] = relative
            pointer["availability_sha256"] = checksum
            pointer_tmp = pointer_path.with_name("active.json.part")
            pointer_tmp.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.chmod(pointer_tmp, 0o600)
            os.replace(pointer_tmp, pointer_path)
        return {
            "withdrawn": sorted({asset_id for _revision, _path, mask, _by_id in grouped.values() for asset_id in asset_ids if asset_id in mask["items"]}),
            "updated_catalogs": {catalog_id: {"revision": revision, "path": path} for catalog_id, (revision, path, _hash) in changed.items()},
            "availability_path": pointer.get("availability_path"),
        }
    except (OSError, ValueError, TypeError, KeyError):
        return None


def status_run(root: Path, run_id: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    db = connect(_db_path(root))
    try:
        run = db.execute("SELECT * FROM reference_runs WHERE run_id=?", (run_id,)).fetchone()
        if not run:
            raise LookupError("Reference run was not found")
        counts = {row["status"]: row["n"] for row in db.execute(
            "SELECT status,COUNT(*) AS n FROM reference_candidates WHERE run_id=? GROUP BY status", (run_id,)
        )}
        by_source = {
            row["source_id"]: row["n"]
            for row in db.execute(
                "SELECT source_id,COUNT(*) AS n FROM reference_candidates WHERE run_id=? AND status='accepted' GROUP BY source_id", (run_id,)
            )
        }
        scopes = Counter()
        styles = Counter()
        for row in db.execute(
            "SELECT decision_json FROM reference_reviews WHERE run_id=? AND origin IN ('human','assistant','local_model')",
            (run_id,),
        ):
            decision = json.loads(row["decision_json"])
            if decision.get("accepted_by_frozen_rule") or decision.get("accepted_for_gallery"):
                labels = decision
                scopes[str(labels.get("primary_scope"))] += 1
                styles[str(labels.get("style"))] += 1
        cursor = json.loads(run["cursor_json"])
    finally:
        db.close()
    catalog_size = 0
    active_path = _active_catalog_path(root)
    if active_path:
        try:
            active = json.loads(active_path.read_text(encoding="utf-8"))
            catalog_size = len(active.get("items", []))
            if active.get("schema_version") == 2:
                catalog_size = len(_active_available_ids(root, active.get("catalog_id"), set()))
        except (OSError, ValueError, TypeError):
            catalog_size = 0
    return {
        "run_id": run_id,
        "status": run["status"],
        "target": run["target_count"],
        "accepted_this_run": counts.get("accepted", 0),
        "gallery_size_including_existing": catalog_size,
        "remaining_to_target": max(0, int(run["target_count"]) - catalog_size - counts.get("accepted", 0)),
        "inventory_sha256": run["inventory_sha256"],
        "stage_counts": counts,
        "accepted_by_source": by_source,
        "accepted_by_scope": dict(scopes),
        "accepted_by_style": dict(styles),
        "errors": run["error_count"],
        "cursor": cursor,
    }


def latest_run_id(root: Path) -> str:
    db = connect(_db_path(root.resolve()))
    try:
        row = db.execute("SELECT run_id FROM reference_runs ORDER BY created_at DESC LIMIT 1").fetchone()
        if not row:
            raise LookupError("No reference curation run exists yet")
        return str(row["run_id"])
    finally:
        db.close()


def _benchmark_category(row: sqlite3.Row) -> str:
    metadata = json.loads(row["metadata_json"])
    if metadata.get("candidate_type") == "crop":
        return "crop"
    if row["source_id"] == "emoreccom-panels":
        annotation = metadata.get("annotation", {})
        if annotation.get("annotations"):
            return "comic-panel-with-character-boxes"
        return "comic-panel"
    name = Path(row["relative_path"]).stem
    if name.isdigit():
        return "native-sketch-frame"
    return "native-sketch"


def _write_contact_sheets(root: Path, items: list[dict[str, Any]], output_dir: Path) -> list[str]:
    from PIL import ImageDraw, ImageFont

    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    sheet_paths: list[str] = []
    font = ImageFont.load_default()
    for batch_start in range(0, len(items), 12):
        batch = items[batch_start : batch_start + 12]
        sheet = Image.new("RGB", (1024, 840), "#f5f5f5")
        draw = ImageDraw.Draw(sheet)
        for offset, item in enumerate(batch):
            x = (offset % 4) * 256
            y = (offset // 4) * 280
            try:
                path = _safe_image_path(root, item["relative_path"])
                with Image.open(path) as source:
                    preview = _white_rgb(source)
                preview.thumbnail((240, 228), Image.Resampling.LANCZOS)
                sheet.paste(preview, (x + (240 - preview.width) // 2 + 8, y + 4))
            except (OSError, ValueError, IntakeError):
                draw.rectangle((x + 8, y + 4, x + 248, y + 232), fill="#dddddd")
                draw.text((x + 24, y + 100), "unavailable", font=font, fill="black")
            label = f"{item['candidate_id'][-8:]} {item['source_id']}"
            detail = f"{item.get('category', '')} · {Path(item['relative_path']).name}"
            draw.text((x + 6, y + 238), label[:40], font=font, fill="black")
            draw.text((x + 6, y + 254), detail[:40], font=font, fill="#333333")
        path = output_dir / f"contact-{batch_start // 12 + 1:02d}.jpg"
        sheet.save(path, format="JPEG", quality=90, optimize=True)
        os.chmod(path, 0o600)
        sheet_paths.append(path.relative_to(root).as_posix())
    return sheet_paths


def _write_pair_contact_sheets(root: Path, pairs: list[dict[str, Any]], output_dir: Path) -> list[str]:
    from PIL import ImageDraw, ImageFont

    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    font = ImageFont.load_default()
    sheets: list[str] = []
    for batch_start in range(0, len(pairs), 4):
        batch = pairs[batch_start : batch_start + 4]
        sheet = Image.new("RGB", (1024, 680), "#f5f5f5")
        draw = ImageDraw.Draw(sheet)
        for offset, pair in enumerate(batch):
            x = (offset % 2) * 512
            y = (offset // 2) * 340
            for side, tag in enumerate(("left", "right")):
                item = pair[tag]
                try:
                    with Image.open(_safe_image_path(root, item["relative_path"])) as source:
                        preview = _white_rgb(source)
                    preview.thumbnail((244, 286), Image.Resampling.LANCZOS)
                    px = x + side * 256 + (256 - preview.width) // 2
                    sheet.paste(preview, (px, y + 4))
                except (OSError, ValueError, IntakeError):
                    draw.rectangle((x + side * 256 + 6, y + 4, x + side * 256 + 250, y + 290), fill="#ddd")
                    draw.text((x + side * 256 + 70, y + 140), "unavailable", font=font, fill="black")
            draw.text((x + 6, y + 294), f"{pair['pair_id']} · {pair['split']}", font=font, fill="black")
            draw.text((x + 6, y + 311), f"A {Path(pair['left']['relative_path']).name}", font=font, fill="#333")
            draw.text((x + 258, y + 311), f"B {Path(pair['right']['relative_path']).name}", font=font, fill="#333")
        path = output_dir / f"pair-contact-{batch_start // 4 + 1:02d}.jpg"
        sheet.save(path, format="JPEG", quality=90, optimize=True)
        os.chmod(path, 0o600)
        sheets.append(path.relative_to(root).as_posix())
    return sheets


def _benchmark_pairs(rows: list[sqlite3.Row], run_id: str, seed: int) -> list[dict[str, Any]]:
    pool = sorted(rows, key=lambda row: hashlib.sha256(f"pair-pool:{seed}:{row['candidate_id']}".encode()).hexdigest())[:800]
    close: list[tuple[int, sqlite3.Row, sqlite3.Row]] = []
    distinct: list[tuple[int, sqlite3.Row, sqlite3.Row]] = []
    for left_index, left in enumerate(pool):
        if not left["perceptual_hash"]:
            continue
        for right in pool[left_index + 1 :]:
            if not right["perceptual_hash"] or left["candidate_id"] == right["candidate_id"]:
                continue
            distance = (int(left["perceptual_hash"], 16) ^ int(right["perceptual_hash"], 16)).bit_count()
            same_known_work = left["work_id"] is not None and left["work_id"] == right["work_id"]
            same_clip = left["clip_id"] is not None and left["clip_id"] == right["clip_id"]
            if left["source_id"] == right["source_id"] and (same_known_work or same_clip):
                if 1 <= distance <= 15:
                    close.append((distance, left, right))
                elif distance >= 23:
                    distinct.append((distance, left, right))
    close.sort(key=lambda pair: (pair[0], hashlib.sha256(f"{pair[1]['candidate_id']}:{pair[2]['candidate_id']}".encode()).hexdigest()))
    distinct.sort(key=lambda pair: (-pair[0], hashlib.sha256(f"{pair[1]['candidate_id']}:{pair[2]['candidate_id']}".encode()).hexdigest()))
    selected = close[:30] + distinct[:30]
    if len(selected) < 40:
        used = {(pair[1]["candidate_id"], pair[2]["candidate_id"]) for pair in selected}
        for left_index, left in enumerate(pool):
            if not left["perceptual_hash"]:
                continue
            for right in pool[left_index + 1 :]:
                if not right["perceptual_hash"] or left["source_id"] != right["source_id"]:
                    continue
                pair_key = (left["candidate_id"], right["candidate_id"])
                if pair_key in used:
                    continue
                distance = (int(left["perceptual_hash"], 16) ^ int(right["perceptual_hash"], 16)).bit_count()
                if distance < 5 or distance > 22:
                    continue
                selected.append((distance, left, right))
                used.add(pair_key)
                if len(selected) >= 40:
                    break
            if len(selected) >= 40:
                break
    selected = selected[:60]
    if len(selected) < 20:
        raise IntakeError(f"Pair-review qualification needs at least 20 candidate pairs; found {len(selected)}")
    output: list[dict[str, Any]] = []
    for index, (distance, first, second) in enumerate(selected):
        left, right = sorted((first, second), key=lambda row: row["candidate_id"])
        output.append({
            "pair_id": _stable_id("pairbench", run_id, left["candidate_id"], right["candidate_id"]),
            "run_id": run_id,
            "left": {
                "candidate_id": left["candidate_id"],
                "source_id": left["source_id"],
                "relative_path": left["relative_path"],
                "image_sha256": left["source_sha256"],
            },
            "right": {
                "candidate_id": right["candidate_id"],
                "source_id": right["source_id"],
                "relative_path": right["relative_path"],
                "image_sha256": right["source_sha256"],
            },
            "perceptual_distance_for_sampling_only": distance,
            "split": "development" if index % 2 == 0 else "held_out",
            "assistant_review": None,
        })
    return output


def prepare_benchmark(root: Path, *, run_id: str | None = None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if run_id is None:
        created = create_run(root, target=500)
        run_id = created["run_id"]
    screen_run(root, run_id)
    db = connect(_db_path(root))
    try:
        rows = db.execute(
            "SELECT * FROM reference_candidates WHERE run_id=? AND status='screened' ORDER BY source_id,relative_path",
            (run_id,),
        ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            grouped[row["source_id"]].append(row)
        selected: list[sqlite3.Row] = []
        seed = 20261005
        for source_id, source_rows in sorted(grouped.items()):
            source_rows.sort(key=lambda row: hashlib.sha256(f"benchmark:{seed}:{row['candidate_id']}".encode()).hexdigest())
            selected.extend(source_rows[:100])
        if len(selected) < 200:
            used = {row["candidate_id"] for row in selected}
            extra = [row for row in rows if row["candidate_id"] not in used]
            extra.sort(key=lambda row: hashlib.sha256(f"benchmark-extra:{seed}:{row['candidate_id']}".encode()).hexdigest())
            selected.extend(extra[: 200 - len(selected)])
        if len(selected) < 200:
            raise IntakeError(f"Qualification benchmark needs 200 readable images; found {len(selected)}")
        pairs = _benchmark_pairs(rows, run_id, seed)
        # Keep each source adapter represented in both development and held-out halves.
        items: list[dict[str, Any]] = []
        source_counts = Counter()
        split_counts = Counter()
        for row in selected[:200]:
            category = _benchmark_category(row)
            source_count = source_counts[row["source_id"]]
            split = "development" if source_count % 2 == 0 else "held_out"
            source_counts[row["source_id"]] += 1
            split_counts[(row["source_id"], split)] += 1
            items.append({
                "candidate_id": row["candidate_id"],
                "run_id": run_id,
                "source_id": row["source_id"],
                "relative_path": row["relative_path"],
                "image_sha256": row["source_sha256"],
                "source_category": category,
                "work_id": row["work_id"],
                "clip_id": row["clip_id"],
                "split": split,
                "category": category,
                "assistant_review": None,
            })
    finally:
        db.close()
    directory = root / "prototype/reviewer/benchmark"
    benchmark_id = _stable_id("benchmark", run_id, str(len(items)))
    output_dir = directory / benchmark_id
    crop_count = 0
    crop_candidates = [
        row
        for row in selected
        if row["source_id"] == "emoreccom-panels"
        and json.loads(row["metadata_json"]).get("annotation", {}).get("annotations")
    ]
    crop_candidates.sort(
        key=lambda row: hashlib.sha256(
            f"benchmark-crop:{seed}:{row['candidate_id']}".encode()
        ).hexdigest()
    )
    for row in crop_candidates:
        if crop_count >= 20:
            break
        annotation = json.loads(row["metadata_json"]).get("annotation", {})
        with Image.open(_safe_image_path(root, row["relative_path"])) as source:
            source.load()
            raw_width, raw_height = source.size
            try:
                orientation = int(source.getexif().get(274, 1))
            except (TypeError, ValueError):
                orientation = 1
            normalized_image = _white_rgb(source)
        chosen_box: tuple[float, float, float, float] | None = None
        chosen_kind: str | None = None
        annotation_rows = annotation.get("annotations", [])
        for kind in ("face", "body"):
            for annotation_row in annotation_rows:
                boxes = annotation_row.get("boxes", {}) if isinstance(annotation_row, dict) else {}
                raw_box = boxes.get(kind) if isinstance(boxes, dict) else None
                if not isinstance(raw_box, dict):
                    continue
                try:
                    chosen_box = _normalized_annotation_box(
                        raw_box,
                        raw_width,
                        raw_height,
                        orientation,
                        normalized_image.width,
                        normalized_image.height,
                    )
                except (KeyError, TypeError, ValueError):
                    chosen_box = None
                if chosen_box is not None:
                    chosen_kind = kind
                    break
            if chosen_box is not None:
                break
        if chosen_box is None or chosen_kind is None:
            continue
        x0 = max(0, min(normalized_image.width, round(chosen_box[0] * normalized_image.width)))
        y0 = max(0, min(normalized_image.height, round(chosen_box[1] * normalized_image.height)))
        x1 = max(x0, min(normalized_image.width, round(chosen_box[2] * normalized_image.width)))
        y1 = max(y0, min(normalized_image.height, round(chosen_box[3] * normalized_image.height)))
        if x1 - x0 < 96 or y1 - y0 < 96:
            continue
        crop = normalized_image.crop((x0, y0, x1, y1))
        crop_id = _stable_id("benchcrop", row["candidate_id"], row["source_sha256"], f"{x0}:{y0}:{x1}:{y1}")
        relative = f"prototype/reviewer/benchmark/{benchmark_id}/benchmark-crops/{crop_id}.png"
        crop_path = ensure_inside(root, relative)
        crop_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        crop.save(crop_path, format="PNG", optimize=True)
        os.chmod(crop_path, 0o600)
        parent_item = next(item for item in items if item["candidate_id"] == row["candidate_id"])
        crop_item = {
            **parent_item,
            "candidate_id": crop_id,
            "relative_path": relative,
            "image_sha256": sha256_file(crop_path),
            "source_category": f"comic-{chosen_kind}-annotation-crop",
            "category": f"comic-{chosen_kind}-annotation-crop",
            "parent_candidate_id": row["candidate_id"],
            "parent_image_sha256": row["source_sha256"],
            "crop_recipe_version": CROP_RECIPE,
            "crop_coordinates_px": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "assistant_review": None,
        }
        items[items.index(parent_item)] = crop_item
        crop_count += 1
    sheets = _write_contact_sheets(root, items, output_dir / "contact-sheets")
    pair_sheets = _write_pair_contact_sheets(root, pairs, output_dir / "pair-contact-sheets")
    benchmark = {
        "schema_version": 1,
        "kind": "drawable-local-review-qualification-benchmark",
        "benchmark_id": benchmark_id,
        "created_at": _now(),
        "run_id": run_id,
        "source_inventory_sha256": status_run(root, run_id)["inventory_sha256"],
        "required_example_count": 200,
        "crop_example_count": crop_count,
        "required_split_counts": {"development": 100, "held_out": 100},
        "required_accepted_held_out_per_adapter": 30,
        "contact_sheets": sheets,
        "required_pair_count": len(pairs),
        "required_pair_split_counts": {"development": (len(pairs) + 1) // 2, "held_out": len(pairs) // 2},
        "pair_contact_sheets": pair_sheets,
        "items": items,
        "pair_items": pairs,
        "assistant_judgments_complete": False,
    }
    benchmark_path = output_dir / "benchmark.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    benchmark_path.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(benchmark_path, 0o600)
    return {
        "status": "awaiting_assistant_review",
        "benchmark_id": benchmark_id,
        "benchmark_path": benchmark_path.relative_to(root).as_posix(),
        "example_count": len(items),
        "split_counts": dict(Counter(item["split"] for item in items)),
        "source_counts": dict(source_counts),
        "source_split_counts": {f"{source}:{split}": count for (source, split), count in split_counts.items()},
        "contact_sheets": sheets,
        "pair_contact_sheets": pair_sheets,
        "pair_count": len(pairs),
    }


def extend_benchmark_heldout(
    root: Path,
    benchmark_relative: str,
    extensions: dict[str, int],
) -> dict[str, Any]:
    """Add an immutable, diverse held-out sample when an adapter lacks 30 gold accepts."""
    root = root.resolve(strict=True)
    benchmark_path, benchmark = _load_benchmark(root, benchmark_relative)
    if not extensions or set(extensions) - {"anita-dataset", "emoreccom-panels"}:
        raise ValueError("Held-out extensions must name a supported source adapter")
    if any(not 1 <= count <= 500 for count in extensions.values()):
        raise ValueError("Each held-out extension must contain 1–500 images")
    used_candidate_ids = {item["candidate_id"] for item in benchmark["items"]}
    used_groups = {
        (item["source_id"], item.get("clip_id") or item.get("work_id") or item["candidate_id"])
        for item in benchmark["items"]
    }
    db = connect(_db_path(root))
    additions: list[dict[str, Any]] = []
    try:
        for source_id, count in sorted(extensions.items()):
            rows = db.execute(
                "SELECT * FROM reference_candidates WHERE run_id=? AND source_id=? AND status='screened'",
                (benchmark["run_id"], source_id),
            ).fetchall()
            grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
            for row in rows:
                candidate_id = str(row["candidate_id"])
                group_id = str(row["clip_id"] or row["work_id"] or candidate_id)
                if candidate_id in used_candidate_ids or (source_id, group_id) in used_groups:
                    continue
                grouped[group_id].append(row)
            for group_id, group_rows in grouped.items():
                group_rows.sort(
                    key=lambda row: hashlib.sha256(
                        f"qualification-heldout-v1:{benchmark['source_inventory_sha256']}:{source_id}:{group_id}:{row['candidate_id']}".encode()
                    ).hexdigest()
                )
            selected: list[sqlite3.Row] = []
            group_ids = sorted(
                grouped,
                key=lambda group_id: hashlib.sha256(
                    f"qualification-heldout-v1:{benchmark['source_inventory_sha256']}:{source_id}:{group_id}".encode()
                ).hexdigest(),
            )
            # One item per source clip/work keeps the extension broad and avoids near-frame leakage.
            for group_id in group_ids:
                if grouped[group_id]:
                    selected.append(grouped[group_id][0])
                if len(selected) == count:
                    break
            if len(selected) < count:
                raise IntakeError(
                    f"Held-out extension for {source_id} requested {count} images; found {len(selected)} unused work/clip groups"
                )
            for row in selected:
                category = _benchmark_category(row)
                additions.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "run_id": benchmark["run_id"],
                        "source_id": source_id,
                        "relative_path": row["relative_path"],
                        "image_sha256": row["source_sha256"],
                        "source_category": category,
                        "work_id": row["work_id"],
                        "clip_id": row["clip_id"],
                        "split": "held_out",
                        "category": category,
                        "assistant_review": None,
                    }
                )
                used_candidate_ids.add(str(row["candidate_id"]))
                used_groups.add((source_id, str(row["clip_id"] or row["work_id"] or row["candidate_id"])))
    finally:
        db.close()

    source_identity = {
        "parent_benchmark_sha256": sha256_file(benchmark_path),
        "source_counts": extensions,
        "candidate_ids": [item["candidate_id"] for item in additions],
    }
    extension_id = hashlib.sha256(_json(source_identity).encode()).hexdigest()[:24]
    output_dir = benchmark_path.parent.parent / f"benchmark-{extension_id}"
    output_path = output_dir / "benchmark.json"
    if output_path.is_file() and not output_path.is_symlink():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("heldout_extensions", [{}])[-1].get("extension_id") != extension_id:
            raise IntakeError("An existing held-out extension does not match its content identity")
        return {
            "status": "awaiting_assistant_review",
            "benchmark_id": existing["benchmark_id"],
            "benchmark_path": output_path.relative_to(root).as_posix(),
            "added_item_count": len(additions),
            "added_by_source": extensions,
            "split_counts": dict(Counter(item["split"] for item in existing["items"])),
            "contact_sheets": [
                path for path in existing["contact_sheets"] if path not in benchmark["contact_sheets"]
            ],
            "reused": True,
        }
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    sheet_paths: list[str] = []
    for source_id in sorted(extensions):
        subset = [item for item in additions if item["source_id"] == source_id]
        sheet_paths.extend(
            _write_contact_sheets(root, subset, output_dir / "heldout-contact-sheets" / source_id)
        )
    extended = dict(benchmark)
    extended["benchmark_id"] = f"benchmark-{extension_id}"
    extended["items"] = [*benchmark["items"], *additions]
    extended["contact_sheets"] = [*benchmark["contact_sheets"], *sheet_paths]
    extended["required_example_count"] = len(extended["items"])
    extended["required_split_counts"] = dict(Counter(item["split"] for item in extended["items"]))
    extended["heldout_extensions"] = [
        *benchmark.get("heldout_extensions", []),
        {
            "extension_id": extension_id,
            "source_counts": extensions,
            "item_count": len(additions),
            "parent_benchmark_sha256": source_identity["parent_benchmark_sha256"],
            "candidate_ids_sha256": hashlib.sha256(
                _json(source_identity["candidate_ids"]).encode()
            ).hexdigest(),
        },
    ]
    extended["assistant_image_judgments_complete"] = False
    extended["assistant_judgments_complete"] = False
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(extended, indent=2, sort_keys=True) + "\n")
    os.chmod(output_path, 0o600)
    return {
        "status": "awaiting_assistant_review",
        "benchmark_id": extended["benchmark_id"],
        "benchmark_path": output_path.relative_to(root).as_posix(),
        "added_item_count": len(additions),
        "added_by_source": extensions,
        "split_counts": dict(Counter(item["split"] for item in extended["items"])),
        "contact_sheets": sheet_paths,
    }


def _load_benchmark(root: Path, relative: str) -> tuple[Path, dict[str, Any]]:
    path = ensure_inside(root, relative)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise IntakeError("Qualification benchmark must be a regular bounded file in the private data root")
    try:
        benchmark = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError("Qualification benchmark is malformed") from error
    if benchmark.get("kind") != "drawable-local-review-qualification-benchmark" or benchmark.get("schema_version") != 1:
        raise IntakeError("Qualification benchmark schema is unsupported")
    items = benchmark.get("items")
    required_count = benchmark.get("required_example_count")
    if (
        not isinstance(items, list)
        or not 200 <= len(items) <= 1000
        or required_count != len(items)
        or len({item.get("candidate_id") for item in items if isinstance(item, dict)}) != len(items)
    ):
        raise IntakeError("Qualification benchmark must contain 200–1,000 unique images")
    split_counts = Counter(item.get("split") for item in items if isinstance(item, dict))
    if (
        split_counts.get("development", 0) < 100
        or split_counts.get("held_out", 0) < 100
        or dict(split_counts) != benchmark.get("required_split_counts")
    ):
        raise IntakeError("Qualification benchmark split counts differ from its recorded recipe")
    for item in items:
        image = _safe_image_path(root, item["relative_path"])
        if sha256_file(image) != item["image_sha256"]:
            raise IntakeError(f"Benchmark image bytes changed: {item['candidate_id']}")
    pairs = benchmark.get("pair_items")
    if not isinstance(pairs, list) or len(pairs) != benchmark.get("required_pair_count") or len(pairs) < 20:
        raise IntakeError("Qualification benchmark must contain a bounded near-duplicate pair sample")
    for pair in pairs:
        if not isinstance(pair, dict) or pair.get("assistant_review") is None and benchmark.get("assistant_pair_judgments_complete"):
            raise IntakeError("Pair benchmark labels are incomplete")
        for side in ("left", "right"):
            record = pair.get(side)
            if not isinstance(record, dict):
                raise IntakeError("Pair benchmark entry is malformed")
            image = _safe_image_path(root, record["relative_path"])
            if sha256_file(image) != record["image_sha256"]:
                raise IntakeError(f"Pair benchmark image bytes changed: {pair.get('pair_id')}")
    return path, benchmark


def _judgment_accepts(review: LocalVisualReview) -> bool:
    return _review_is_acceptable(review)


def _model_metrics(
    items: list[dict[str, Any]],
    predictions: dict[str, LocalVisualReview],
    source_ids: list[str],
    *,
    split: str,
) -> dict[str, Any]:
    sample = [item for item in items if item["split"] == split]
    model_accepted = [item for item in sample if _review_is_acceptable(predictions[item["candidate_id"]])]
    agreement = 0
    style_total = style_correct = scope_total = scope_correct = 0
    critical: list[str] = []
    per_adapter: dict[str, dict[str, int]] = {}
    for item in sample:
        assistant = LocalVisualReview.model_validate(item["assistant_review"])
        model = predictions[item["candidate_id"]]
        assistant_ok = _judgment_accepts(assistant)
        model_ok = _review_is_acceptable(model)
        if model_ok and assistant_ok:
            agreement += 1
            style_total += 1
            scope_total += 1
            style_correct += model.style == assistant.style
            scope_correct += model.primary_scope == assistant.primary_scope
        elif model_ok:
            is_critical = (
                not assistant.safe_content
                or not assistant.useful_character_drawing
                or assistant.essential_anatomy_clipped
                or assistant.text_crosses_important_anatomy
                or assistant.damaged_or_unreadable
            )
            if is_critical:
                critical.append(item["candidate_id"])
        if model_ok:
            row = per_adapter.setdefault(item["source_id"], {"held_out_items": 0, "model_accepted": 0, "agreement": 0})
            row["held_out_items"] += 1
            row["model_accepted"] += 1
            row["agreement"] += int(assistant_ok)
        elif item["source_id"] in source_ids:
            per_adapter.setdefault(item["source_id"], {"held_out_items": 0, "model_accepted": 0, "agreement": 0})["held_out_items"] += 1
    accepted_agreement = agreement / len(model_accepted) if model_accepted else 0.0
    style_agreement = style_correct / style_total if style_total else 0.0
    scope_agreement = scope_correct / scope_total if scope_total else 0.0
    adapter_counts = {
        source: int(per_adapter.get(source, {}).get("model_accepted", 0))
        for source in source_ids
    }
    passed = (
        accepted_agreement >= 0.95
        and style_agreement >= 0.90
        and scope_agreement >= 0.90
        and not critical
        and (split != "held_out" or all(count >= 30 for count in adapter_counts.values()))
    )
    return {
        "passed": passed,
        "sample_count": len(sample),
        "model_accepted_count": len(model_accepted),
        "useful_acceptance_yield": len(model_accepted) / max(1, len(sample)),
        "accepted_usability_agreement": accepted_agreement,
        "accepted_style_agreement": style_agreement,
        "accepted_primary_scope_agreement": scope_agreement,
        "critical_acceptance_failures": critical,
        "accepted_held_out_by_adapter": adapter_counts,
        "per_adapter_counts": per_adapter,
    }


def _pair_model_metrics(
    items: list[dict[str, Any]], predictions: dict[str, LocalPairReview], *, split: str
) -> dict[str, Any]:
    sample = [item for item in items if item["split"] == split]
    exact = predicted_duplicates = true_duplicates = 0
    false_duplicate_merges: list[str] = []
    duplicate_relations = {"same_image", "adjacent_frame_duplicate"}
    for item in sample:
        assistant = LocalPairReview.model_validate(item["assistant_review"])
        model = predictions[item["pair_id"]]
        exact += int(model.relation == assistant.relation)
        model_duplicate = model.relation in duplicate_relations
        assistant_duplicate = assistant.relation in duplicate_relations
        predicted_duplicates += int(model_duplicate)
        true_duplicates += int(model_duplicate and assistant_duplicate)
        if model_duplicate and not assistant_duplicate:
            false_duplicate_merges.append(item["pair_id"])
    accuracy = exact / max(1, len(sample))
    precision = true_duplicates / max(1, predicted_duplicates)
    passed = len(sample) >= 10 and accuracy >= 0.90 and precision >= 0.95 and not false_duplicate_merges
    return {
        "passed": passed,
        "sample_count": len(sample),
        "exact_relation_accuracy": accuracy,
        "predicted_duplicate_count": predicted_duplicates,
        "duplicate_precision": precision,
        "false_duplicate_merges": false_duplicate_merges,
    }


def _run_benchmark_model(root: Path, benchmark: dict[str, Any], model_id: str, *, q8: bool = False) -> dict[str, Any]:
    predictions: dict[str, LocalVisualReview] = {}
    timing: list[float] = []
    pair_predictions: dict[str, LocalPairReview] = {}
    pair_timing: list[float] = []
    invalid_image_reviews: list[dict[str, str]] = []
    invalid_pair_reviews: list[dict[str, str]] = []
    memory_before: dict[str, Any] = {}
    memory_after: dict[str, Any] = {}
    memory_peak: dict[str, Any] = {}
    with LlamaReviewer(root, model_id, q8=q8, preferred_gpu_layers=32 if q8 else None) as reviewer:
        memory_before = reviewer.startup_memory
        for index, item in enumerate(benchmark["items"], start=1):
            path = _safe_image_path(root, item["relative_path"])
            with Image.open(path) as source:
                source.load()
                normalized = _white_rgb(source)
            try:
                output, elapsed = review_image(reviewer, normalized)
                timing.append(elapsed)
            except ReviewerError as error:
                invalid_image_reviews.append({"candidate_id": item["candidate_id"], "error": str(error)[:500]})
                output = LocalVisualReview(
                    useful_character_drawing=False,
                    safe_content=False,
                    style=None,
                    primary_scope=None,
                    secondary_scopes=[],
                    person_count=None,
                    quality=1,
                    text_coverage=0,
                    text_crosses_important_anatomy=False,
                    essential_anatomy_clipped=False,
                    damaged_or_unreadable=False,
                    uncertain=True,
                    reason="The model returned invalid output after one retry; this image was excluded.",
                    crop_boxes=[],
                )
            predictions[item["candidate_id"]] = output
            if index % 10 == 0:
                print(f"{model_id}{' Q8' if q8 else ' Q4_K_M'}: {index}/{len(benchmark['items'])}", flush=True)
        for index, pair in enumerate(benchmark["pair_items"], start=1):
            images: list[Image.Image] = []
            for side in ("left", "right"):
                with Image.open(_safe_image_path(root, pair[side]["relative_path"])) as source:
                    source.load()
                    images.append(_white_rgb(source))
            try:
                output, elapsed = review_pair(reviewer, images[0], images[1])
                pair_timing.append(elapsed)
            except ReviewerError as error:
                invalid_pair_reviews.append({"pair_id": pair["pair_id"], "error": str(error)[:500]})
                output = LocalPairReview(
                    relation="uncertain",
                    reason="The model returned invalid output after one retry; the pair remains unresolved.",
                    pose_difference=False,
                    expression_difference=False,
                    facing_or_view_difference=False,
                    perspective_or_silhouette_difference=False,
                    useful_detail_difference=False,
                )
            pair_predictions[pair["pair_id"]] = output
            if index % 10 == 0:
                print(f"{model_id}{' Q8' if q8 else ' Q4_K_M'} pairs: {index}/{len(benchmark['pair_items'])}", flush=True)
        memory_after = reviewer.config and _gpu_probe()
        memory_peak = reviewer.peak_memory
        model_hash = reviewer.model_sha256
        runtime_hash = reviewer.runtime_sha256
        gpu_layers = reviewer.gpu_layers
        prompt_version = reviewer.prompt_version
    return {
        "model_id": model_id,
        "quantization": "Q8_0" if q8 else "Q4_K_M",
        "model_sha256": model_hash,
        "runtime_sha256": runtime_hash,
        "gpu_layers": gpu_layers,
        "prompt_version": prompt_version,
        "image_max_edge": int(reviewer.config["review_defaults"]["image_max_edge"]),
        "inference_ms": {"mean": sum(timing) / max(1, len(timing)), "median": sorted(timing)[len(timing) // 2] if timing else 0, "max": max(timing, default=0), "count": len(timing), "invalid_count": len(invalid_image_reviews)},
        "pair_inference_ms": {"mean": sum(pair_timing) / max(1, len(pair_timing)), "median": sorted(pair_timing)[len(pair_timing) // 2] if pair_timing else 0, "max": max(pair_timing, default=0), "count": len(pair_timing), "invalid_count": len(invalid_pair_reviews)},
        "gpu_memory_before": memory_before,
        "gpu_memory_after": memory_after,
        "gpu_memory_peak": memory_peak,
        "invalid_image_reviews": invalid_image_reviews,
        "predictions": {candidate_id: result.model_dump(mode="json") for candidate_id, result in predictions.items()},
        "metrics": _model_metrics(
            benchmark["items"], predictions,
            sorted({item["source_id"] for item in benchmark["items"]}),
            split="held_out",
        ),
        "development_metrics": _model_metrics(
            benchmark["items"], predictions,
            sorted({item["source_id"] for item in benchmark["items"]}),
            split="development",
        ),
        "pair_predictions": {pair_id: result.model_dump(mode="json") for pair_id, result in pair_predictions.items()},
        "invalid_pair_reviews": invalid_pair_reviews,
        "pair_metrics": _pair_model_metrics(benchmark["pair_items"], pair_predictions, split="held_out"),
        "development_pair_metrics": _pair_model_metrics(
            benchmark["pair_items"], pair_predictions, split="development"
        ),
    }


def qualify_reviewer(root: Path, benchmark_relative: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    benchmark_path, benchmark = _load_benchmark(root, benchmark_relative)
    unreviewed_images = sum(item.get("assistant_review") is None for item in benchmark["items"])
    unreviewed_pairs = sum(item.get("assistant_review") is None for item in benchmark["pair_items"])
    if unreviewed_images or unreviewed_pairs:
        return {
            "status": "awaiting_assistant_review",
            "benchmark_path": benchmark_relative,
            "unreviewed_image_count": unreviewed_images,
            "unreviewed_pair_count": unreviewed_pairs,
        }
    for item in benchmark["items"]:
        LocalVisualReview.model_validate(item["assistant_review"])
    for item in benchmark["pair_items"]:
        LocalPairReview.model_validate(item["assistant_review"])
    counts = Counter(item["split"] for item in benchmark["items"])
    if counts.get("development", 0) < 100 or counts.get("held_out", 0) < 100:
        raise IntakeError("Qualification benchmark requires at least 100 development and 100 held-out items")
    source_ids = sorted({item["source_id"] for item in benchmark["items"]})
    accepted_by_source = Counter(
        item["source_id"]
        for item in benchmark["items"]
        if item["split"] == "held_out"
        and _judgment_accepts(LocalVisualReview.model_validate(item["assistant_review"]))
    )
    missing_accepts = {source: 30 - accepted_by_source[source] for source in source_ids if accepted_by_source[source] < 30}
    if missing_accepts:
        return {
            "status": "held_out_sample_needs_extension",
            "benchmark_path": benchmark_relative,
            "accepted_held_out_by_adapter": dict(accepted_by_source),
            "additional_accepted_examples_needed": missing_accepts,
            "message": "Extend and assistant-review the held-out sample before model qualification.",
        }
    pair_counts = Counter(item["split"] for item in benchmark["pair_items"])
    if pair_counts != Counter(benchmark["required_pair_split_counts"]):
        raise IntakeError("Pair-review benchmark split counts differ from its recorded recipe")
    reviewer_cfg = reviewer_config()
    model_ids = [row["id"] for row in reviewer_cfg["models"]]
    results = [_run_benchmark_model(root, benchmark, model_id) for model_id in model_ids]
    # Select Q8 for the stronger Q4 only with development results. Hold-out metrics remain
    # untouched until final qualification and selection.
    q4_rank = sorted(
        results,
        key=lambda row: (
            min(row["development_metrics"]["accepted_usability_agreement"], row["development_metrics"]["accepted_style_agreement"], row["development_metrics"]["accepted_primary_scope_agreement"], row["development_pair_metrics"]["exact_relation_accuracy"], row["development_pair_metrics"]["duplicate_precision"]),
            row["development_metrics"]["useful_acceptance_yield"],
            -row["inference_ms"]["mean"],
        ),
        reverse=True,
    )
    stronger_model_id = q4_rank[0]["model_id"]
    setup_reviewer(root, q8_model_id=stronger_model_id)
    q8_result = _run_benchmark_model(root, benchmark, stronger_model_id, q8=True)
    results.append(q8_result)
    eligible = [row for row in results if row["metrics"]["passed"] and row["pair_metrics"]["passed"]]
    chosen = None
    if eligible:
        chosen = max(
            eligible,
            key=lambda row: (
                row["development_metrics"]["accepted_usability_agreement"],
                min(
                    row["development_metrics"]["accepted_style_agreement"],
                    row["development_metrics"]["accepted_primary_scope_agreement"],
                    row["development_pair_metrics"]["exact_relation_accuracy"],
                    row["development_pair_metrics"]["duplicate_precision"],
                ),
                row["development_metrics"]["useful_acceptance_yield"],
                -row["inference_ms"]["mean"],
            ),
        )
    report = {
        "schema_version": 1,
        "kind": "drawable-local-reviewer-qualification",
        "created_at": _now(),
        "benchmark_path": benchmark_relative,
        "benchmark_sha256": sha256_file(benchmark_path),
        "runtime_release": "b11425",
        "prompt_sha256": sha256_file(REPOSITORY_ROOT / "config/references-review-prompt-v1.txt"),
        "pair_review_prompt_sha256": sha256_file(REPOSITORY_ROOT / "config/references-pair-review-prompt-v1.txt"),
        "rubric_version": RUBRIC_VERSION,
        "split_counts": dict(counts),
        "pair_split_counts": dict(pair_counts),
        "acceptance_thresholds": {"usability_agreement": 0.95, "style_agreement": 0.90, "primary_scope_agreement": 0.90, "critical_acceptance_failures": 0, "accepted_held_out_per_adapter": 30},
        "pair_acceptance_thresholds": {"exact_relation_accuracy": 0.90, "duplicate_precision": 0.95, "false_duplicate_merges": 0},
        "selection_rule": "Gate each model on held-out metrics; among passing profiles, rank by development correctness, useful acceptance yield, then speed.",
        "models": results,
        "selected": None if chosen is None else {key: chosen[key] for key in ("model_id", "quantization", "model_sha256", "runtime_sha256", "gpu_layers", "prompt_version", "image_max_edge", "metrics", "pair_metrics", "inference_ms", "pair_inference_ms")},
        "status": "qualified" if chosen else "not_qualified",
        "pair_reviewer_status": "qualified" if chosen else "not_qualified",
        "assistant_review_is_ground_truth": False,
        "training_started": False,
        "data_uploaded": False,
    }
    report_dir = root / "prototype/reviewer/qualifications"
    report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    report_path = report_dir / ("qualification-" + sha256_file(benchmark_path)[:24] + ".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(report_path, 0o600)
    if chosen:
        profile_id = "reviewer-" + hashlib.sha256(_json(report["selected"]).encode()).hexdigest()[:24]
        profile = {
            "schema_version": 1,
            "profile_id": profile_id,
            "status": "qualified",
            "model_id": chosen["model_id"],
            "quantization": chosen["quantization"],
            "model_sha256": chosen["model_sha256"],
            "runtime_id": "llama.cpp-b11425",
            "runtime_sha256": chosen["runtime_sha256"],
            "gpu_layers": chosen["gpu_layers"],
            "image_max_edge": chosen["image_max_edge"],
            "context_tokens": 4096,
            "max_response_tokens": 512,
            "parallel_slots": 1,
            "temperature": 0,
            "prompt_version": chosen["prompt_version"],
            "prompt_sha256": report["prompt_sha256"],
            "pair_review_prompt_sha256": report["pair_review_prompt_sha256"],
            "pair_reviewer_status": report["pair_reviewer_status"],
            "rubric_version": RUBRIC_VERSION,
            "qualification_report": report_path.relative_to(root).as_posix(),
            "qualification_report_sha256": sha256_file(report_path),
            "qualification_metrics": chosen["metrics"],
        }
        target = root / "prototype/reviewer/qualified-profile.json"
        target.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(target, 0o600)
        report["profile_id"] = profile_id
    return {"status": report["status"], "report_path": report_path.relative_to(root).as_posix(), "selected_model": chosen["model_id"] if chosen else None, "models": [{"model_id": row["model_id"], "quantization": row["quantization"], "metrics": row["metrics"], "pair_metrics": row["pair_metrics"], "inference_ms": row["inference_ms"], "pair_inference_ms": row["pair_inference_ms"]} for row in results]}


def _assistant_import_records(
    root: Path, benchmark: dict[str, Any]
) -> list[dict[str, Any]]:
    """Load only accepted benchmark judgments, binding every image to current bytes."""
    db = connect(_db_path(root))
    try:
        train_names, test_names, _annotations = _emoreccom_partition_allowlist(root)
        records: list[dict[str, Any]] = []
        for item in benchmark["items"]:
            review = LocalVisualReview.model_validate(item.get("assistant_review"))
            if not _judgment_accepts(review):
                continue
            source_candidate_id = item["candidate_id"]
            crop = source_candidate_id.startswith("benchcrop-")
            if crop:
                parent_id = item.get("parent_candidate_id")
                parent = db.execute(
                    "SELECT * FROM reference_candidates WHERE candidate_id=? AND run_id=?",
                    (parent_id, benchmark["run_id"]),
                ).fetchone()
                if parent is None:
                    continue
                source = dict(parent)
                parent_metadata = json.loads(source["metadata_json"])
                if (
                    item.get("source_id") != source["source_id"]
                    or item.get("work_id") != source["work_id"]
                    or item.get("clip_id") != source["clip_id"]
                    or item.get("parent_image_sha256") != source["source_sha256"]
                ):
                    continue
                if source["source_id"] == "emoreccom-panels":
                    filename = Path(source["relative_path"]).name.casefold()
                    if (
                        parent_metadata.get("upstream_partition") != "source_train"
                        or filename not in train_names
                        or filename in test_names
                    ):
                        continue
                parent_path = _safe_image_path(root, source["relative_path"])
                if sha256_file(parent_path) != source["source_sha256"]:
                    continue
                coordinates = item.get("crop_coordinates_px")
                if not isinstance(coordinates, dict):
                    continue
                try:
                    x0, y0, x1, y1 = (int(coordinates[key]) for key in ("x0", "y0", "x1", "y1"))
                except (KeyError, TypeError, ValueError):
                    continue
                if item.get("crop_recipe_version") != CROP_RECIPE:
                    continue
                crop_path = _safe_image_path(root, item["relative_path"])
                if crop_path.stat().st_size <= 0 or crop_path.stat().st_size > MAX_IMAGE_BYTES:
                    continue
                if sha256_file(crop_path) != item["image_sha256"]:
                    continue
                with Image.open(parent_path) as parent_image:
                    parent_image.load()
                    normalized_parent = _white_rgb(parent_image)
                if (
                    x0 < 0 or y0 < 0 or x1 > normalized_parent.width or y1 > normalized_parent.height
                    or x1 - x0 < 96 or y1 - y0 < 96
                ):
                    continue
                reconstructed = normalized_parent.crop((x0, y0, x1, y1))
                with Image.open(crop_path) as stored_crop:
                    stored_crop.load()
                    normalized_crop = _white_rgb(stored_crop)
                if (
                    normalized_crop.size != reconstructed.size
                    or _pixel_hash(normalized_crop) != _pixel_hash(reconstructed)
                ):
                    continue
                metadata = dict(parent_metadata)
                parent_asset_id = str(parent_metadata.get("asset_id") or parent["candidate_id"])
                crop_asset_id = _stable_id(
                    "reference-crop",
                    parent_asset_id,
                    source["source_sha256"],
                    CROP_RECIPE,
                    f"{x0}:{y0}:{x1}:{y1}",
                )
                metadata.update({
                    "candidate_type": "crop",
                    "asset_id": crop_asset_id,
                    "root_asset_id": str(parent_metadata.get("root_asset_id") or parent_asset_id),
                    "parent_asset_id": parent_asset_id,
                    "parent_candidate_id": parent["candidate_id"],
                    "parent_hash": source["source_sha256"],
                    "crop_recipe": CROP_RECIPE,
                    "crop_coordinates_px": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "parent_width": normalized_parent.width,
                    "parent_height": normalized_parent.height,
                    "crop_purpose": item.get("category", "assistant-reviewed benchmark crop"),
                })
                relative_path = item["relative_path"]
                source_sha256 = item["image_sha256"]
                source_row_id = parent["candidate_id"]
                byte_length = crop_path.stat().st_size
                width, height = normalized_crop.size
                media_type = "image/png"
                pixel_sha256 = _pixel_hash(normalized_crop)
                perceptual_hash = _perceptual_hash(normalized_crop)
            else:
                original = db.execute(
                    "SELECT * FROM reference_candidates WHERE candidate_id=? AND run_id=?",
                    (source_candidate_id, benchmark["run_id"]),
                ).fetchone()
                if original is None:
                    continue
                source = dict(original)
                metadata = json.loads(source["metadata_json"])
                relative_path = source["relative_path"]
                source_sha256 = source["source_sha256"]
                source_row_id = source["candidate_id"]
                if source["source_id"] == "emoreccom-panels":
                    filename = Path(relative_path).name.casefold()
                    if (
                        metadata.get("upstream_partition") != "source_train"
                        or filename not in train_names
                        or filename in test_names
                    ):
                        continue
                image_path = _safe_image_path(root, relative_path)
                if source_sha256 != item["image_sha256"] or sha256_file(image_path) != source_sha256:
                    continue
                with Image.open(image_path) as source_image:
                    source_image.load()
                    normalized_image = _white_rgb(source_image)
                byte_length = image_path.stat().st_size
                width, height = normalized_image.size
                with Image.open(image_path) as source_image:
                    media_type = SUPPORTED_FORMATS.get(source_image.format or "", "application/octet-stream")
                pixel_sha256 = _pixel_hash(normalized_image)
                perceptual_hash = _perceptual_hash(normalized_image)

            if source["source_id"] != item["source_id"]:
                continue
            if source["source_id"] == "emoreccom-panels":
                filename = Path(source["relative_path"]).name.casefold()
                if (
                    metadata.get("upstream_partition") != "source_train"
                    or filename not in train_names
                    or filename in test_names
                ):
                    continue
            metadata["relative_path"] = relative_path
            records.append({
                "benchmark_candidate_id": source_candidate_id,
                "source_candidate_id": source_row_id,
                "source_id": source["source_id"],
                "relative_path": relative_path,
                "source_sha256": source_sha256,
                "pixel_sha256": pixel_sha256,
                "perceptual_hash": perceptual_hash,
                "byte_length": byte_length,
                "media_type": media_type,
                "width": width,
                "height": height,
                "work_id": source["work_id"],
                "clip_id": source["clip_id"],
                "parent_candidate_id": metadata.get("parent_candidate_id"),
                "metadata": metadata,
                "review": review,
                "benchmark_split": item["split"],
            })
    finally:
        db.close()
    return records


def _assistant_pair_review_artifact(
    root: Path,
    benchmark_path: Path,
    benchmark: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any], list[str]]:
    accepted_hash = sha256_file(benchmark_path)
    import_id = "assistant-import-" + accepted_hash[:24]
    directory = root / "prototype/reviewer/assistant-imports" / import_id
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    endpoints: list[dict[str, Any]] = []
    for record in records:
        endpoints.append({
            "candidate_id": record["benchmark_candidate_id"],
            "kind": "candidate",
            "source_id": record["source_id"],
            "relative_path": record["relative_path"],
            "image_sha256": record["source_sha256"],
            "pixel_sha256": record["pixel_sha256"],
            "perceptual_hash": record["perceptual_hash"],
        })
    active_items, _catalog_id = _current_catalog_items(root)
    for item in active_items:
        path = _safe_image_path(root, item.relative_path)
        if sha256_file(path) != item.sha256:
            continue
        with Image.open(path) as image:
            image.load()
            normalized = _white_rgb(image)
        endpoints.append({
            "candidate_id": item.asset_id,
            "kind": "active",
            "source_id": item.source_id,
            "relative_path": item.relative_path,
            "image_sha256": item.sha256,
            "pixel_sha256": _pixel_hash(normalized),
            "perceptual_hash": _perceptual_hash(normalized),
        })
    pair_items: list[dict[str, Any]] = []
    for index, left in enumerate(endpoints):
        for right in endpoints[index + 1 :]:
            if left["kind"] == right["kind"] == "active":
                continue
            same_pixels = left["pixel_sha256"] == right["pixel_sha256"]
            if not same_pixels and (
                left["source_id"] != right["source_id"]
                or (int(left["perceptual_hash"], 16) ^ int(right["perceptual_hash"], 16)).bit_count() > 12
            ):
                continue
            first, second = sorted((left, right), key=lambda row: (row["kind"], row["candidate_id"]))
            distance = (int(first["perceptual_hash"], 16) ^ int(second["perceptual_hash"], 16)).bit_count()
            pair_id = _stable_id(
                "refpair", benchmark["benchmark_id"], first["candidate_id"], second["candidate_id"]
            )
            review = None
            if same_pixels:
                review = LocalPairReview(
                    relation="same_image",
                    reason="The EXIF-normalized RGB pixels are identical.",
                    pose_difference=False,
                    expression_difference=False,
                    facing_or_view_difference=False,
                    perspective_or_silhouette_difference=False,
                    useful_detail_difference=False,
                ).model_dump(mode="json")
            pair_items.append({
                "pair_id": pair_id,
                "split": "curation",
                "left": {key: first[key] for key in ("candidate_id", "kind", "source_id", "relative_path", "image_sha256")},
                "right": {key: second[key] for key in ("candidate_id", "kind", "source_id", "relative_path", "image_sha256")},
                "perceptual_distance_for_sampling_only": distance,
                "assistant_review": review,
            })
    pair_items.sort(key=lambda item: item["pair_id"])
    pair_signature = hashlib.sha256(_json([
        {key: item[key] for key in ("pair_id", "left", "right", "perceptual_distance_for_sampling_only")}
        for item in pair_items
    ]).encode()).hexdigest()
    path = directory / "pair-review.json"
    expected = {
        "schema_version": 1,
        "kind": "drawable-reference-assistant-import-pairs",
        "import_id": import_id,
        "benchmark_path": benchmark_path.relative_to(root).as_posix(),
        "benchmark_sha256": accepted_hash,
        "pair_signature": pair_signature,
        "required_pair_count": len(pair_items),
        "pair_items": pair_items,
        "assistant_judgments_complete": all(item["assistant_review"] is not None for item in pair_items),
    }
    if path.exists():
        if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
            raise IntakeError("Assistant import pair review must be a regular bounded private file")
        existing = json.loads(path.read_text(encoding="utf-8"))
        expected_identity = {key: value for key, value in expected.items() if key not in {"pair_items", "assistant_judgments_complete"}}
        existing_identity = {key: value for key, value in existing.items() if key not in {"pair_items", "assistant_judgments_complete"}}
        if existing_identity != expected_identity:
            raise IntakeError("Assistant import pair queue no longer matches the current benchmark images")
        by_id = {item.get("pair_id"): item for item in existing.get("pair_items", []) if isinstance(item, dict)}
        if set(by_id) != {item["pair_id"] for item in pair_items}:
            raise IntakeError("Assistant import pair queue was edited or is incomplete")
        for item in pair_items:
            prior = by_id[item["pair_id"]]
            if any(prior.get(key) != item[key] for key in ("left", "right", "perceptual_distance_for_sampling_only")):
                raise IntakeError("Assistant import pair endpoints or image hashes changed")
            if prior.get("assistant_review") is not None:
                item["assistant_review"] = LocalPairReview.model_validate(prior["assistant_review"]).model_dump(mode="json")
        expected["pair_items"] = pair_items
        expected["assistant_judgments_complete"] = all(item["assistant_review"] is not None for item in pair_items)
    else:
        path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(path, 0o600)
    incomplete = [item for item in pair_items if item["assistant_review"] is None]
    sheets = _write_pair_contact_sheets(root, incomplete, directory / "pair-contact-sheets") if incomplete else []
    return path, expected, sheets


def _assistant_pair_states(
    root: Path, expected: dict[str, Any], review_relative: str | None
) -> tuple[dict[str, LocalPairReview], str | None]:
    required = [item for item in expected["pair_items"] if item["assistant_review"] is None]
    if not required:
        return {
            item["pair_id"]: LocalPairReview.model_validate(item["assistant_review"])
            for item in expected["pair_items"]
        }, None
    if not review_relative:
        return {}, None
    path = ensure_inside(root, review_relative)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise IntakeError("Assistant pair review must be a regular bounded file inside the private data root")
    document = json.loads(path.read_text(encoding="utf-8"))
    for key in ("kind", "import_id", "benchmark_sha256", "pair_signature", "required_pair_count"):
        if document.get(key) != expected.get(key):
            raise IntakeError("Assistant pair review is stale or belongs to a different import")
    supplied = document.get("pair_items")
    if (
        document.get("assistant_judgments_complete") is not True
        or not isinstance(supplied, list)
        or len(supplied) != len(expected["pair_items"])
    ):
        raise IntakeError("Assistant pair review has the wrong pair count")
    expected_by_id = {item["pair_id"]: item for item in expected["pair_items"]}
    reviews: dict[str, LocalPairReview] = {}
    for item in supplied:
        expected_item = expected_by_id.get(item.get("pair_id")) if isinstance(item, dict) else None
        if expected_item is None or any(
            item.get(key) != expected_item[key]
            for key in ("left", "right", "perceptual_distance_for_sampling_only")
        ):
            raise IntakeError("Assistant pair review changes or invents a pair endpoint")
        value = item.get("assistant_review")
        if value is None:
            raise IntakeError("Every proposed near-duplicate pair needs an assistant judgment")
        reviews[item["pair_id"]] = LocalPairReview.model_validate(value)
    return reviews, path.relative_to(root).as_posix()


def import_assistant_reviewed(
    root: Path,
    benchmark_relative: str,
    *,
    pair_review_relative: str | None = None,
) -> dict[str, Any]:
    """Import frozen assistant judgments without enabling automatic model acceptance."""
    root = root.resolve(strict=True)
    benchmark_path, benchmark = _load_benchmark(root, benchmark_relative)
    records = _assistant_import_records(root, benchmark)
    if not records:
        raise IntakeError("No complete, byte-verified assistant-approved candidates can be imported")
    import_id = "assistant-import-" + sha256_file(benchmark_path)[:24]
    database = _db_path(root)
    migrate(database)
    db = connect(database)
    try:
        prior = db.execute("SELECT run_id,target_count,recipe_json FROM reference_runs").fetchall()
    finally:
        db.close()
    for row in prior:
        prior_recipe = json.loads(row["recipe_json"])
        if prior_recipe.get("assistant_import_id") != import_id:
            continue
        pair_artifact = root / "prototype/reviewer/assistant-imports" / import_id / "pair-review.json"
        if (
            not pair_artifact.is_file()
            or pair_artifact.is_symlink()
            or sha256_file(pair_artifact) != prior_recipe.get("pair_review_sha256")
        ):
            raise IntakeError("This assistant import's committed pair-review artifact changed")
        return {
            "status": "already_imported",
            "run_id": row["run_id"],
            "target": row["target_count"],
            "gallery_size": int(row["target_count"]),
            "accepted_new": int(prior_recipe.get("accepted_new", 0)),
        }
    pair_path, pair_document, contact_sheets = _assistant_pair_review_artifact(
        root, benchmark_path, benchmark, records
    )
    pair_reviews, _review_path = _assistant_pair_states(
        root, pair_document,
        pair_review_relative or (pair_path.relative_to(root).as_posix() if pair_document["assistant_judgments_complete"] else None),
    )
    if not pair_reviews and any(item["assistant_review"] is None for item in pair_document["pair_items"]):
        return {
            "status": "awaiting_assistant_pair_review",
            "import_id": pair_document["import_id"],
            "approved_candidate_count": len(records),
            "pair_count": len(pair_document["pair_items"]),
            "unreviewed_pair_count": sum(item["assistant_review"] is None for item in pair_document["pair_items"]),
            "pair_review_path": pair_path.relative_to(root).as_posix(),
            "contact_sheets": contact_sheets,
        }
    if len(pair_reviews) != len(pair_document["pair_items"]):
        raise IntakeError("Assistant pair review is incomplete")

    current_items, current_catalog_id = _current_catalog_items(root)
    active_asset_ids = {item.asset_id for item in current_items}
    active_by_pixel: dict[str, list[PrototypeGalleryItemV2]] = defaultdict(list)
    for item in current_items:
        path = _safe_image_path(root, item.relative_path)
        if sha256_file(path) != item.sha256:
            continue
        with Image.open(path) as image:
            image.load()
            normalized = _white_rgb(image)
        active_by_pixel[_pixel_hash(normalized)].append(item)

    by_benchmark_id = {record["benchmark_candidate_id"]: record for record in records}
    dispositions: dict[str, tuple[str, str | None]] = {
        record["benchmark_candidate_id"]: ("accepted", None) for record in records
    }
    for record in records:
        asset_id = str(record["metadata"].get("asset_id") or record["benchmark_candidate_id"])
        if asset_id in active_asset_ids:
            dispositions[record["benchmark_candidate_id"]] = ("exact_duplicate", "asset already exists in the active gallery")
        elif record["pixel_sha256"] in active_by_pixel:
            dispositions[record["benchmark_candidate_id"]] = ("exact_duplicate", "pixel-identical to an active gallery asset")
    for pair in pair_document["pair_items"]:
        review = pair_reviews[pair["pair_id"]]
        left_id, right_id = pair["left"]["candidate_id"], pair["right"]["candidate_id"]
        if pair["left"]["kind"] == "active":
            candidate_id = right_id
            if review.relation in {"same_image", "adjacent_frame_duplicate"}:
                dispositions[candidate_id] = ("near_duplicate", f"duplicate of active gallery asset {left_id}")
            elif review.relation == "uncertain":
                dispositions[candidate_id] = ("deferred", f"unresolved pair with active gallery asset {left_id}")
            continue
        if pair["right"]["kind"] == "active":
            candidate_id = left_id
            if review.relation in {"same_image", "adjacent_frame_duplicate"}:
                dispositions[candidate_id] = ("near_duplicate", f"duplicate of active gallery asset {right_id}")
            elif review.relation == "uncertain":
                dispositions[candidate_id] = ("deferred", f"unresolved pair with active gallery asset {right_id}")
            continue
        if review.relation == "uncertain":
            dispositions[left_id] = ("deferred", f"unresolved near-duplicate pair {pair['pair_id']}")
            dispositions[right_id] = ("deferred", f"unresolved near-duplicate pair {pair['pair_id']}")
        elif review.relation in {"same_image", "adjacent_frame_duplicate"}:
            left, right = by_benchmark_id[left_id], by_benchmark_id[right_id]
            left_rank = (-left["review"].quality, 1 if left["metadata"].get("candidate_type") == "crop" else 0, left_id)
            right_rank = (-right["review"].quality, 1 if right["metadata"].get("candidate_type") == "crop" else 0, right_id)
            loser, winner = (left_id, right_id) if left_rank <= right_rank else (right_id, left_id)
            dispositions[loser] = ("near_duplicate", f"duplicate of assistant-reviewed representative {winner}")

    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if dispositions[record["benchmark_candidate_id"]][0] == "accepted":
            exact_groups[record["pixel_sha256"]].append(record)
    for group in exact_groups.values():
        if len(group) <= 1:
            continue
        ordered = sorted(group, key=lambda row: (-row["review"].quality, row["metadata"].get("candidate_type") == "crop", row["benchmark_candidate_id"]))
        for duplicate in ordered[1:]:
            dispositions[duplicate["benchmark_candidate_id"]] = (
                "exact_duplicate", f"pixel-identical to {ordered[0]['benchmark_candidate_id']}"
            )

    active_roots = Counter(item.root_asset_id for item in current_items)
    eligible = [record for record in records if dispositions[record["benchmark_candidate_id"]][0] == "accepted"]
    for record in sorted(eligible, key=lambda row: (-row["review"].quality, row["benchmark_candidate_id"])):
        root_id = str(record["metadata"].get("root_asset_id") or record["metadata"].get("asset_id") or record["benchmark_candidate_id"])
        if active_roots[root_id] >= 3:
            dispositions[record["benchmark_candidate_id"]] = ("deferred", "maximum three variants per root image")
        else:
            active_roots[root_id] += 1
    eligible = [record for record in records if dispositions[record["benchmark_candidate_id"]][0] == "accepted"]
    while True:
        work_counts = Counter(record["work_id"] for record in eligible if record["source_id"] == "anita-dataset" and record["work_id"])
        limit = max(1, math.floor(len(eligible) * 0.20))
        excessive = {work for work, count in work_counts.items() if count > limit}
        if not excessive:
            break
        remove = sorted(
            (record for record in eligible if record["source_id"] == "anita-dataset" and record["work_id"] in excessive),
            key=lambda row: (row["review"].quality, row["benchmark_candidate_id"]),
        )
        for record in remove:
            if work_counts[record["work_id"]] <= limit:
                continue
            dispositions[record["benchmark_candidate_id"]] = ("deferred", "soft source cap: one Anita work may supply at most 20% of new references")
            work_counts[record["work_id"]] -= 1
        eligible = [record for record in records if dispositions[record["benchmark_candidate_id"]][0] == "accepted"]

    accepted_count = len(eligible)
    target_count = len(current_items) + accepted_count
    if not target_count or target_count > 10_000:
        raise IntakeError("Assistant import would produce an invalid V2 gallery size")
    pair_digest = sha256_file(pair_path)
    import_id = pair_document["import_id"]
    database = _db_path(root)
    migrate(database)
    db = connect(database)
    try:
        prior_runs = db.execute("SELECT run_id,target_count,recipe_json FROM reference_runs").fetchall()
    finally:
        db.close()
    for prior in prior_runs:
        prior_recipe = json.loads(prior["recipe_json"])
        if prior_recipe.get("assistant_import_id") == import_id:
            if prior_recipe.get("pair_review_sha256") != pair_digest:
                raise IntakeError("This assistant import was already committed with different pair judgments")
            return {
                "status": "already_imported",
                "run_id": prior["run_id"],
                "target": prior["target_count"],
                "gallery_size": int(prior["target_count"]),
                "accepted_new": int(prior_recipe.get("accepted_new", 0)),
            }
    run_id = "refs-" + uuid4().hex[:24]
    recipe = {
        "schema_version": 1,
        "kind": "assistant-reviewed-subset-import",
        "assistant_import_id": import_id,
        "target_count": target_count,
        "accepted_new": accepted_count,
        "benchmark_path": benchmark_relative,
        "benchmark_sha256": sha256_file(benchmark_path),
        "pair_review_sha256": pair_digest,
        "source_run_id": benchmark["run_id"],
        "review_origin": "assistant",
        "automatic_model_acceptance": False,
        "maximum_variants_per_root": 3,
        "maximum_work_share": 0.20,
    }
    recipe_json = _json(recipe)
    recipe_sha = hashlib.sha256(recipe_json.encode()).hexdigest()
    database = _db_path(root)
    db = connect(database)
    try:
        prior = db.execute("SELECT run_id,target_count FROM reference_runs WHERE recipe_json=?", (recipe_json,)).fetchone()
        if prior:
            return {"status": "already_imported", "run_id": prior["run_id"], "target": prior["target_count"], "gallery_size": len(current_items) + accepted_count, "accepted_new": accepted_count}
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO reference_runs(run_id,target_count,recipe_json,recipe_sha256,inventory_sha256,status,cursor_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (run_id, target_count, recipe_json, recipe_sha, recipe["benchmark_sha256"], "building", _json({"inventory_count": len(records), "screened": len(records), "reviewed": len(records), "accepted": accepted_count, "assistant_import_id": import_id}), _now(), _now()),
        )
        new_ids: dict[str, str] = {}
        for record in records:
            new_id = _stable_id("refimport", run_id, record["benchmark_candidate_id"])
            new_ids[record["benchmark_candidate_id"]] = new_id
            review = record["review"]
            status, exclusion_reason = dispositions[record["benchmark_candidate_id"]]
            provenance = ReviewProvenanceV1(
                origin="assistant",
                reviewer_id="assistant-codex",
                prompt_version="assistant-reviewed-benchmark-v1",
                rubric_version=RUBRIC_VERSION,
                run_id=run_id,
                image_sha256=record["source_sha256"],
                reviewed_revision=0,
                evidence={
                    "benchmark_id": benchmark["benchmark_id"],
                    "benchmark_candidate_id": record["benchmark_candidate_id"],
                    "benchmark_split": record["benchmark_split"],
                    "reason": review.reason,
                    "assistant_review_is_ground_truth": False,
                },
            )
            decision = _review_labels(review, 0.0)
            decision.update({
                "accepted_by_assistant_rule": _judgment_accepts(review),
                "accepted_for_gallery": status == "accepted",
                "exclusion_reason": exclusion_reason,
                "review_provenance": provenance.model_dump(mode="json"),
            })
            metadata = dict(record["metadata"])
            if status != "accepted":
                metadata["assistant_import_exclusion"] = {"status": status, "reason": exclusion_reason}
            metadata["review"] = decision
            db.execute(
                "INSERT INTO reference_candidates(candidate_id,run_id,source_id,relative_path,source_sha256,pixel_sha256,perceptual_hash,byte_length,media_type,width,height,work_id,clip_id,parent_candidate_id,status,revision,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (new_id, run_id, record["source_id"], record["relative_path"], record["source_sha256"], record["pixel_sha256"], record["perceptual_hash"], record["byte_length"], record["media_type"], record["width"], record["height"], record["work_id"], record["clip_id"], record["source_candidate_id"] if record["metadata"].get("candidate_type") == "crop" else None, status, 1, _json(metadata)),
            )
            db.execute(
                "INSERT INTO reference_reviews(review_id,candidate_id,reviewed_revision,origin,reviewer_id,profile_id,model_sha256,runtime_sha256,prompt_version,rubric_version,run_id,image_sha256,evidence_json,decision_json,inference_ms,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(uuid4()), new_id, 0, "assistant", provenance.reviewer_id, None, None, None, provenance.prompt_version, RUBRIC_VERSION, run_id, record["source_sha256"], _json(provenance.model_dump(mode="json")), _json(decision), None, _now()),
            )
        pair_count = 0
        for pair in pair_document["pair_items"]:
            left_source, right_source = pair["left"]["candidate_id"], pair["right"]["candidate_id"]
            if left_source not in new_ids or right_source not in new_ids:
                continue
            left_id, right_id = sorted((new_ids[left_source], new_ids[right_source]))
            review = pair_reviews[pair["pair_id"]]
            if review.relation in {"same_image", "adjacent_frame_duplicate"}:
                pair_state = "same"
            elif review.relation == "uncertain":
                pair_state = "uncertain"
            else:
                pair_state = "distinct"
            evidence = {
                "relation": review.relation,
                "reason": review.reason,
                "review_origin": "assistant",
                "assistant_review_is_ground_truth": False,
                "perceptual_distance": pair["perceptual_distance_for_sampling_only"],
                "pair_review_id": pair["pair_id"],
            }
            left_hash = next(record["source_sha256"] for record in records if new_ids[record["benchmark_candidate_id"]] == left_id)
            right_hash = next(record["source_sha256"] for record in records if new_ids[record["benchmark_candidate_id"]] == right_id)
            pair_id = _stable_id("refpair", run_id, left_id, right_id)
            db.execute(
                "INSERT INTO reference_pairs(pair_id,run_id,left_candidate_id,right_candidate_id,left_image_sha256,right_image_sha256,state,evidence_json,reviewer_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (pair_id, run_id, left_id, right_id, left_hash, right_hash, pair_state, _json(evidence), "assistant-codex", _now()),
            )
            pair_count += 1
        counts = Counter(status for status, _reason in dispositions.values())
        cursor = {
            "inventory_count": len(records),
            "screened": len(records),
            "reviewed": len(records),
            "accepted": accepted_count,
            "assistant_import_id": import_id,
            "assistant_import_exclusions": dict(counts),
            "assistant_pair_review_count": pair_count,
            "assistant_pair_review_sha256": pair_digest,
            "stage_counts": dict(counts),
            "active_catalog_id_at_import": current_catalog_id,
        }
        db.execute(
            "UPDATE reference_runs SET status='building',cursor_json=?,updated_at=? WHERE run_id=?",
            (_json(cursor), _now(), run_id),
        )
        db.execute("COMMIT")
    except Exception:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise
    finally:
        db.close()
    summary = Counter(record["source_id"] for record in eligible)
    return {
        "status": "ready_to_audit",
        "run_id": run_id,
        "target": target_count,
        "existing_gallery_count": len(current_items),
        "accepted_new": accepted_count,
        "sources": dict(summary),
        "excluded": dict(Counter(status for status, _reason in dispositions.values() if status != "accepted")),
        "pair_reviews": pair_count,
        "assistant_import_id": import_id,
    }


def _current_catalog_items(root: Path) -> tuple[list[PrototypeGalleryItemV2], str | None]:
    active_path = _active_catalog_path(root)
    if active_path is None:
        return [], None
    document = json.loads(active_path.read_text(encoding="utf-8"))
    if document.get("schema_version") == 2:
        catalog = PrototypeGalleryCatalogV2.model_validate_json(active_path.read_bytes())
        available = _active_available_ids(root, catalog.catalog_id, set())
        return [item for item in catalog.items if item.asset_id in available], catalog.catalog_id
    # Preserve the active V1 batch exactly while recording the historical origin as human
    # because the existing snapshot records its reviewer as owner-local.
    from drawable_research_contracts import CurationSnapshotV1

    snapshot = CurationSnapshotV1.model_validate_json(active_path.read_bytes())
    build_id = "legacy-v1"
    try:
        pointer = json.loads((root / "prototype/search/active.json").read_text(encoding="utf-8"))
        build_id = str(pointer.get("build_id", build_id))
    except (OSError, ValueError, TypeError):
        pass
    output: list[PrototypeGalleryItemV2] = []
    for item in snapshot.items:
        provenance = item.review_provenance or ReviewProvenanceV1(
            origin="human",
            reviewer_id=item.reviewer_id,
            evidence={"legacy_record": True, "active_build": build_id, "legacy_snapshot": True},
            prompt_version="preserved-v1-review",
            rubric_version="preserved-v1-curation-rubric",
            run_id=f"legacy-{build_id}",
            image_sha256=item.sha256,
            reviewed_revision=item.asset_revision,
        )
        output.append(PrototypeGalleryItemV2(
            asset_id=item.asset_id,
            source_id=item.source_id,
            source_work_id=item.source_work_id,
            relative_path=item.relative_path,
            sha256=item.sha256,
            byte_length=item.byte_length,
            media_type=item.media_type,
            width=item.width,
            height=item.height,
            source_family=item.source_family,
            source_identity_verified=bool(item.source_work_id),
            upstream_partition="source_train" if item.source_id == "emoreccom-panels" else "not_applicable",
            research_family_id=None,
            authorization_basis="owner-local-educational-prototype",
            review_status="approved",
            gallery_eligible_candidate=True,
            license_id=item.license_id or "unspecified-local-prototype-source",
            attribution=item.attribution or item.source_id,
            permissions=item.permissions,
            labels=item.labels,
            parent_asset_id=item.parent_asset_id,
            root_asset_id=item.parent_asset_id or item.asset_id,
            clip_id=f"{item.source_id}:{item.source_work_id or 'unknown'}",
            crop_rectangle=item.crop_rectangle if isinstance(item.crop_rectangle, dict) else None,
            asset_revision=item.asset_revision,
            review_id=item.review_id,
            reviewer_id=item.reviewer_id,
            review_provenance=provenance,
        ))
    return output, None


def _thumbnail(root: Path, item: PrototypeGalleryItemV2) -> PrototypeGalleryItemV2:
    path = _safe_image_path(root, item.relative_path)
    with Image.open(path) as source:
        source.load()
        image = _white_rgb(source)
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    relative = f"prototype/thumbnails/{item.sha256}-768.webp"
    destination = ensure_inside(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        if destination.is_symlink() or destination.stat().st_size > 4 * 1024 * 1024:
            raise IntakeError("Existing reference thumbnail is unsafe")
        digest = sha256_file(destination)
    else:
        temporary = destination.with_name(destination.name + f".{uuid4().hex}.part")
        image.save(temporary, format="WEBP", quality=86, method=6)
        os.chmod(temporary, 0o600)
        with Image.open(temporary) as check:
            check.verify()
        digest = sha256_file(temporary)
        os.replace(temporary, destination)
    return item.model_copy(update={"thumbnail_path": relative, "thumbnail_sha256": digest})


def _accepted_run_items(root: Path, run_id: str) -> list[PrototypeGalleryItemV2]:
    db = connect(_db_path(root))
    try:
        rows = db.execute(
            "SELECT c.*,r.review_id,r.reviewer_id,r.profile_id,r.model_sha256,r.runtime_sha256,r.prompt_version,r.rubric_version,r.evidence_json,r.decision_json,r.inference_ms,r.created_at AS reviewed_at FROM reference_candidates c JOIN reference_reviews r ON r.candidate_id=c.candidate_id AND r.reviewed_revision=c.revision-1 WHERE c.run_id=? AND c.status='accepted' AND r.origin IN ('human','assistant','local_model') ORDER BY c.source_id,c.relative_path,c.candidate_id",
            (run_id,),
        ).fetchall()
    finally:
        db.close()
    registry = load_registry(REPOSITORY_ROOT / "config/sources.json")
    source_map = {row["id"]: row for row in registry["sources"]}
    result: list[PrototypeGalleryItemV2] = []
    for row in rows:
        metadata = json.loads(row["metadata_json"])
        decision = json.loads(row["decision_json"])
        if not (
            decision.get("accepted_by_frozen_rule")
            or decision.get("accepted_for_gallery")
        ) or not isinstance(decision.get("style"), str) or not isinstance(decision.get("primary_scope"), str):
            raise IntakeError(f"Candidate {row['candidate_id']} lacks a complete frozen acceptance decision")
        if metadata.get("upstream_partition") == "source_test_locked":
            raise PermissionError("Protected EmoRecCom test data cannot enter a prototype catalog")
        if row["source_id"] == "emoreccom-panels" and metadata.get("upstream_partition") != "source_train":
            raise PermissionError("Only allowlisted EmoRecCom training panels can enter a prototype catalog")
        source = source_map.get(row["source_id"], {})
        source_title = str(source.get("title") or row["source_id"])
        artist = metadata.get("artist")
        attribution = f"{source_title} · {artist}" if artist else source_title
        root_id = metadata.get("root_asset_id") or metadata.get("asset_id") or row["candidate_id"]
        box = metadata.get("crop_coordinates_px")
        image_width = int(row["width"])
        image_height = int(row["height"])
        crop_rectangle = None
        if isinstance(box, dict):
            crop_rectangle = {
                "x": float(box["x0"]) / max(1, int(metadata.get("parent_width", image_width))),
                "y": float(box["y0"]) / max(1, int(metadata.get("parent_height", image_height))),
                "width": max(0.0, float(box["x1"] - box["x0"])) / max(1, int(metadata.get("parent_width", image_width))),
                "height": max(0.0, float(box["y1"] - box["y0"])) / max(1, int(metadata.get("parent_height", image_height))),
            }
        provenance = ReviewProvenanceV1.model_validate_json(row["evidence_json"])
        labels = SnapshotLabelsV1.model_validate({
            "style": decision["style"],
            "primary_scope": decision["primary_scope"],
            "secondary_scopes": decision.get("secondary_scopes", []),
            "person_count": decision.get("person_count"),
            "quality": decision["quality"],
            "sfw": decision["sfw"],
            "character_useful": decision["character_useful"],
            "text_coverage": decision["text_coverage"],
            "clipped": decision["clipped"],
            "damaged": decision["damaged"],
            "duplicate_of": decision.get("duplicate_of"),
            "review_seconds": 0,
            "keep_reject_reason": decision["keep_reject_reason"],
        })
        source_family = metadata.get("source_family")
        item = PrototypeGalleryItemV2(
            asset_id=str(metadata.get("asset_id") or row["candidate_id"]),
            source_id=row["source_id"],
            source_work_id=row["work_id"],
            relative_path=row["relative_path"],
            sha256=row["source_sha256"],
            byte_length=row["byte_length"],
            media_type=row["media_type"],
            width=image_width,
            height=image_height,
            source_family=source_family,
            source_identity_verified=bool(metadata.get("source_identity_verified", False)),
            upstream_partition=metadata.get("upstream_partition", "not_applicable"),
            research_family_id=None,
            authorization_basis="owner-local-educational-prototype",
            review_status="approved",
            gallery_eligible_candidate=True,
            license_id=str(metadata.get("license_id") or "unspecified-local-prototype-source"),
            attribution=attribution,
            permissions=SnapshotPermissionsV1(local_display=True, evaluation=True, derive=True),
            labels=labels,
            parent_asset_id=metadata.get("parent_asset_id"),
            root_asset_id=str(root_id),
            clip_id=row["clip_id"],
            crop_rectangle=crop_rectangle,
            asset_revision=int(row["revision"] - 1),
            review_id=row["review_id"],
            reviewer_id=row["reviewer_id"],
            review_provenance=provenance,
        )
        result.append(_thumbnail(root, item))
    return result


def _collection_report_metrics(
    root: Path, run_id: str, items: list[PrototypeGalleryItemV2]
) -> dict[str, Any]:
    db = connect(_db_path(root))
    try:
        run = db.execute(
            "SELECT created_at,updated_at,inventory_sha256,error_count,cursor_json FROM reference_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise LookupError("Reference curation run was not found while building its report")
        stage_counts = {
            row["status"]: int(row["n"])
            for row in db.execute(
                "SELECT status,COUNT(*) AS n FROM reference_candidates WHERE run_id=? GROUP BY status",
                (run_id,),
            )
        }
        duplicate_counts = {
            "exact_duplicate_candidates": int(
                db.execute(
                    "SELECT COUNT(*) FROM reference_candidates WHERE run_id=? AND status='exact_duplicate'",
                    (run_id,),
                ).fetchone()[0]
            ),
            "near_duplicate_candidates": int(
                db.execute(
                    "SELECT COUNT(*) FROM reference_candidates WHERE run_id=? AND status='near_duplicate'",
                    (run_id,),
                ).fetchone()[0]
            ),
        }
        pair_outcomes = {
            row["state"]: int(row["n"])
            for row in db.execute(
                "SELECT state,COUNT(*) AS n FROM reference_pairs WHERE run_id=? GROUP BY state",
                (run_id,),
            )
        }
        review_times = [
            float(row["inference_ms"])
            for row in db.execute(
                "SELECT inference_ms FROM reference_reviews WHERE run_id=? AND origin='local_model' AND inference_ms IS NOT NULL",
                (run_id,),
            )
        ]
        local_model_review_count = int(
            db.execute(
                "SELECT COUNT(*) FROM reference_reviews WHERE run_id=? AND origin='local_model'",
                (run_id,),
            ).fetchone()[0]
        )
    finally:
        db.close()
    scope_counts = Counter(item.labels.primary_scope for item in items)
    if local_model_review_count:
        profile = _qualified_profile(root)
        qualification = {
            key: profile.get(key)
            for key in (
                "profile_id",
                "model_id",
                "quantization",
                "model_sha256",
                "runtime_sha256",
                "gpu_layers",
                "image_max_edge",
                "qualification_report",
                "qualification_report_sha256",
                "qualification_metrics",
                "pair_reviewer_status",
            )
        }
        qualification_report_path = ensure_inside(root, str(profile["qualification_report"]))
        qualification_report = json.loads(qualification_report_path.read_text(encoding="utf-8"))
        qualification["qualification_results"] = {
            key: qualification_report.get(key)
            for key in (
                "status",
                "split_counts",
                "pair_split_counts",
                "acceptance_thresholds",
                "pair_acceptance_thresholds",
                "assistant_review_is_ground_truth",
            )
        }
        qualification["model_comparisons"] = [
            {
                key: result.get(key)
                for key in (
                    "model_id",
                    "quantization",
                    "gpu_layers",
                    "metrics",
                    "pair_metrics",
                    "inference_ms",
                    "pair_inference_ms",
                    "gpu_memory_before",
                    "gpu_memory_after",
                    "gpu_memory_peak",
                )
            }
            for result in qualification_report.get("models", [])
        ]
    else:
        qualification = {
            "status": "not_used_for_assistant_reviewed_subset",
            "automatic_model_acceptance": False,
            "assistant_review_is_ground_truth": False,
        }
    try:
        elapsed = max(
            0.0,
            (
                datetime.fromisoformat(run["updated_at"])
                - datetime.fromisoformat(run["created_at"])
            ).total_seconds(),
        )
    except (TypeError, ValueError):
        elapsed = None
    roots = {item.root_asset_id for item in items}
    source_counts = Counter(item.source_id for item in items)
    train_names, test_names, _annotations = _emoreccom_partition_allowlist(root)
    comic_source = next(
        (row for row in load_registry(REPOSITORY_ROOT / "config/sources.json")["sources"] if row["id"] == "emoreccom-panels"),
        None,
    )
    comic_root = ensure_inside(root, comic_source["extracted_path"]) / "emoreccom" if comic_source else None
    present_comics = (
        {
            path.name.casefold()
            for path in comic_root.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        if comic_root is not None and comic_root.is_dir() and not comic_root.is_symlink()
        else set()
    )
    sorted_review_times = sorted(review_times)
    run_cursor = json.loads(run["cursor_json"])
    return {
        "run_id": run_id,
        "source_inventory_sha256": run["inventory_sha256"],
        "inventory_count": run_cursor.get("inventory_count"),
        "inventory_exclusions": run_cursor.get("inventory_exclusions", 0),
        "embedding_proposals": run_cursor.get("embedding_proposals", {"status": "not_run"}),
        "processing_elapsed_seconds": elapsed,
        "candidate_stage_counts": stage_counts,
        "candidate_error_count": int(run["error_count"]),
        "distinct_root_count": len(roots),
        "distinct_clip_count": len({item.clip_id for item in items if item.clip_id}),
        "distinct_work_count": len({item.source_work_id for item in items if item.source_work_id}),
        "source_counts": dict(source_counts),
        "upstream_partition_exclusions": {
            "eligible_training_filenames": len(train_names),
            "locked_test_filenames": len(test_names),
            "unknown_partition_files_not_decoded": len(present_comics - train_names - test_names),
            "published_comic_items": int(source_counts.get("emoreccom-panels", 0)),
        },
        "scope_coverage": {
            "targets": SOFT_SCOPE_TARGETS,
            "actual": dict(scope_counts),
            "gaps": {
                scope: max(0, target_count - int(scope_counts.get(scope, 0)))
                for scope, target_count in SOFT_SCOPE_TARGETS.items()
            },
        },
        "near_duplicate_pair_outcomes": pair_outcomes,
        "duplicate_candidate_counts": duplicate_counts,
        "review_origins": dict(Counter(item.review_provenance.origin for item in items)),
        "local_model_review_inference_ms": {
            "count": len(sorted_review_times),
            "mean": sum(sorted_review_times) / len(sorted_review_times) if sorted_review_times else None,
            "median": sorted_review_times[len(sorted_review_times) // 2] if sorted_review_times else None,
            "max": max(sorted_review_times) if sorted_review_times else None,
            "owner_review_seconds": 0,
        },
        "local_model_qualification": qualification,
        "training_started": False,
        "data_uploaded": False,
    }


def build_catalog(root: Path, run_id: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    current_items, current_catalog_id = _current_catalog_items(root)
    new_items = _accepted_run_items(root, run_id)
    by_id = {item.asset_id: item for item in current_items}
    by_id.update({item.asset_id: item for item in new_items})
    items = list(by_id.values())
    if not items or len(items) > 10_000:
        raise IntakeError("A prototype gallery must contain between 1 and 10,000 items")
    if len(items) < 500 and len(items) > 32:
        # Small V2 staging galleries are valid and useful, but are reported as partial.
        pass
    root_counts = Counter(item.root_asset_id for item in items)
    existing_root_counts = Counter(item.root_asset_id for item in current_items)
    new_root_counts = Counter(item.root_asset_id for item in new_items)
    over_limit_roots = {
        root_id
        for root_id, count in root_counts.items()
        if count > 3 and count > existing_root_counts.get(root_id, 0)
    }
    if over_limit_roots:
        raise IntakeError("Catalog would add a fourth variant to an existing root image")
    active_snapshot = _active_catalog_path(root)
    prior_revision = 0
    if active_snapshot:
        try:
            active_document = json.loads(active_snapshot.read_text(encoding="utf-8"))
            if active_document.get("schema_version") == 2:
                prior_revision = int(active_document.get("availability_revision", 0))
        except (OSError, ValueError, TypeError):
            pass
    item_documents = [item.model_dump(mode="json") for item in items]
    availability_revision = prior_revision + int(item_documents != [item.model_dump(mode="json") for item in _current_catalog_items(root)[0]])
    identity = {"availability_revision": availability_revision, "items": item_documents}
    catalog_id = "gallery-" + hashlib.sha256(_json(identity).encode()).hexdigest()[:24]
    catalog_body = {
        "schema_version": 2,
        "catalog_kind": "drawable-prototype-gallery",
        "purpose": "local-educational-prototype",
        "created_at": _now(),
        "created_by": "codex-local-curation",
        "availability_revision": availability_revision,
        "items": item_documents,
    }
    directory = root / "prototype/catalogs"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    relative = f"prototype/catalogs/{catalog_id}.json"
    path = ensure_inside(root, relative)
    if path.exists():
        catalog = PrototypeGalleryCatalogV2.model_validate_json(path.read_bytes())
    else:
        catalog = PrototypeGalleryCatalogV2(
            catalog_id=catalog_id,
            **{**catalog_body, "created_at": datetime.fromisoformat(catalog_body["created_at"])},
        )
    encoded = json.dumps(catalog.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise IntakeError("Immutable V2 catalog ID collides with different content")
    else:
        path.write_text(encoded, encoding="utf-8")
        os.chmod(path, 0o600)
    checks = {
        "catalog_id": catalog_id,
        "catalog_path": relative,
        "catalog_sha256": sha256_file(path),
        "item_count": len(items),
        "new_item_count": len(new_items),
        "preserved_catalog_id": current_catalog_id,
        "roots": len(root_counts),
        "preexisting_variant_cap_exceptions": sum(
            1 for root_id, count in existing_root_counts.items() if count > 3
        ),
        "added_root_variants": dict(new_root_counts),
        "originals": sum(item.parent_asset_id is None for item in items),
        "crops": sum(item.parent_asset_id is not None for item in items),
        "works": len({item.source_work_id for item in items if item.source_work_id}),
        "clips": len({item.clip_id for item in items if item.clip_id}),
        "sources": dict(Counter(item.source_id for item in items)),
        "styles": dict(Counter(item.labels.style for item in items)),
        "scopes": dict(Counter(item.labels.primary_scope for item in items)),
        "review_origins": dict(Counter(item.review_provenance.origin for item in items)),
        "active_build_unchanged": True,
        "research_family_assignments": 0,
    }
    report_dir = root / "reports/private/reference-collections"
    report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    report_path = report_dir / f"{catalog_id}.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "drawable-reference-collection",
                "created_at": _now(),
                **checks,
                **_collection_report_metrics(root, run_id, items),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(report_path, 0o600)
    checks["collection_report"] = report_path.relative_to(root).as_posix()
    return checks


def prepare_release_audit(root: Path, run_id: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    catalog_result = build_catalog(root, run_id)
    catalog_path = ensure_inside(root, catalog_result["catalog_path"])
    catalog = PrototypeGalleryCatalogV2.model_validate_json(catalog_path.read_bytes())
    buckets: dict[tuple[str, str, bool], list[PrototypeGalleryItemV2]] = defaultdict(list)
    for item in catalog.items:
        buckets[(item.source_id, item.labels.primary_scope, item.parent_asset_id is not None)].append(item)
    for key, bucket in buckets.items():
        bucket.sort(key=lambda item: hashlib.sha256(f"release-audit:{catalog.catalog_id}:{key}:{item.asset_id}".encode()).hexdigest())
    keys = sorted(buckets, key=lambda key: hashlib.sha256(f"release-strata:{key}".encode()).hexdigest())
    ordered: list[PrototypeGalleryItemV2] = []
    offset = 0
    while any(offset < len(buckets[key]) for key in keys):
        for key in keys:
            if offset < len(buckets[key]):
                ordered.append(buckets[key][offset])
        offset += 1
    selected: list[PrototypeGalleryItemV2] = []
    roots: set[str] = set()
    for item in ordered:
        if item.root_asset_id in roots:
            continue
        selected.append(item)
        roots.add(item.root_asset_id)
        if len(selected) >= min(50, len(catalog.items)):
            break
    # If a small collection has fewer independent roots, fill remaining audit seats with
    # distinct assets only after the root-stratified pass.
    if len(selected) < min(50, len(catalog.items)):
        selected_ids = {item.asset_id for item in selected}
        selected.extend(item for item in ordered if item.asset_id not in selected_ids and len(selected) < min(50, len(catalog.items)))
    items = [{
        "candidate_id": item.asset_id,
        "source_id": item.source_id,
        "relative_path": item.relative_path,
        "image_sha256": item.sha256,
        "source_work_id": item.source_work_id,
        "clip_id": item.clip_id,
        "root_asset_id": item.root_asset_id,
        "asset_revision": item.asset_revision,
        "review_origin": item.review_provenance.origin,
        "reviewer_id": item.review_provenance.reviewer_id,
        "style": item.labels.style,
        "scope": item.labels.primary_scope,
        "is_crop": item.parent_asset_id is not None,
        "assistant_review": None,
    } for item in selected]
    db = connect(_db_path(root))
    try:
        pair_rows = db.execute(
            """SELECT p.*,l.source_id AS left_source_id,l.relative_path AS left_path,
                      r.source_id AS right_source_id,r.relative_path AS right_path
               FROM reference_pairs p
               JOIN reference_candidates l ON l.candidate_id=p.left_candidate_id
               JOIN reference_candidates r ON r.candidate_id=p.right_candidate_id
               WHERE p.run_id=? AND p.state IN ('same','distinct','uncertain')""",
            (run_id,),
        ).fetchall()
    finally:
        db.close()
    pairs_by_state: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for pair in pair_rows:
        pairs_by_state[pair["state"]].append(pair)
    for rows in pairs_by_state.values():
        rows.sort(
            key=lambda pair: hashlib.sha256(
                f"release-pair-audit:{catalog.catalog_id}:{pair['pair_id']}".encode()
            ).hexdigest()
        )
    pair_keys = sorted(pairs_by_state, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    selected_pairs: list[sqlite3.Row] = []
    offset = 0
    while len(selected_pairs) < 12 and any(offset < len(pairs_by_state[key]) for key in pair_keys):
        for key in pair_keys:
            if offset < len(pairs_by_state[key]) and len(selected_pairs) < 12:
                selected_pairs.append(pairs_by_state[key][offset])
        offset += 1
    pair_items = []
    for pair in selected_pairs:
        evidence = json.loads(pair["evidence_json"])
        pair_items.append({
            "pair_id": pair["pair_id"],
            "split": pair["state"],
            "model_state": pair["state"],
            "model_relation": evidence.get("relation", "uncertain"),
            "left": {
                "candidate_id": pair["left_candidate_id"],
                "source_id": pair["left_source_id"],
                "relative_path": pair["left_path"],
                "image_sha256": pair["left_image_sha256"],
            },
            "right": {
                "candidate_id": pair["right_candidate_id"],
                "source_id": pair["right_source_id"],
                "relative_path": pair["right_path"],
                "image_sha256": pair["right_image_sha256"],
            },
            "assistant_review": None,
        })
    audit_id = "audit-" + hashlib.sha256(f"{catalog.catalog_id}:{len(items)}".encode()).hexdigest()[:24]
    directory = root / "prototype/reviewer/release-audits" / audit_id
    contact_items = [{**item, "category": f"{item['scope']} · {'crop' if item['is_crop'] else 'original'}"} for item in items]
    sheets = _write_contact_sheets(root, contact_items, directory / "contact-sheets")
    pair_sheets = _write_pair_contact_sheets(root, pair_items, directory / "pair-contact-sheets") if pair_items else []
    report = {
        "schema_version": 1,
        "kind": "drawable-reference-release-audit",
        "audit_id": audit_id,
        "catalog_id": catalog.catalog_id,
        "catalog_path": catalog_result["catalog_path"],
        "catalog_sha256": catalog_result["catalog_sha256"],
        "required_count": min(50, len(catalog.items)),
        "required_usable_rate": 0.95,
        "required_critical_failures": 0,
        "contact_sheets": sheets,
        "required_pair_count": len(pair_items),
        "pair_contact_sheets": pair_sheets,
        "items": items,
        "pair_items": pair_items,
        "assistant_judgments_complete": False,
        "assistant_pair_judgments_complete": not pair_items,
    }
    path = directory / "audit.json"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return {"status": "awaiting_assistant_audit", "audit_path": path.relative_to(root).as_posix(), "audit_id": audit_id, "selected_count": len(items), "contact_sheets": sheets, "catalog": catalog_result}


def _audit_results(root: Path, audit_relative: str, expected_catalog_sha256: str) -> dict[str, Any]:
    path = ensure_inside(root, audit_relative)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise IntakeError("Release audit must be a regular bounded file inside the private data root")
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError("Release audit is malformed") from error
    if audit.get("kind") != "drawable-reference-release-audit" or audit.get("schema_version") != 1:
        raise IntakeError("Release audit schema is unsupported")
    catalog_path = ensure_inside(root, str(audit.get("catalog_path", "")))
    if sha256_file(catalog_path) != expected_catalog_sha256 or audit.get("catalog_sha256") != expected_catalog_sha256:
        raise IntakeError("Release audit is stale against the staged catalog")
    items = audit.get("items")
    if not isinstance(items, list) or len(items) != audit.get("required_count") or len(items) > 50:
        raise IntakeError("Release audit has the wrong item count")
    if any(item.get("assistant_review") is None for item in items):
        return {"status": "awaiting_assistant_audit", "unreviewed": sum(item.get("assistant_review") is None for item in items), "audit_path": audit_relative}
    pair_items = audit.get("pair_items", [])
    if (
        not isinstance(pair_items, list)
        or len(pair_items) != audit.get("required_pair_count", 0)
        or len(pair_items) > 12
    ):
        raise IntakeError("Release pair audit has the wrong bounded sample size")
    unreviewed_pairs = sum(item.get("assistant_review") is None for item in pair_items)
    if unreviewed_pairs:
        return {
            "status": "awaiting_assistant_audit",
            "unreviewed": 0,
            "unreviewed_pairs": unreviewed_pairs,
            "audit_path": audit_relative,
        }
    accepted = critical = 0
    failures: list[str] = []
    for item in items:
        image = _safe_image_path(root, item["relative_path"])
        if sha256_file(image) != item["image_sha256"]:
            raise IntakeError(f"Release audit bytes changed for {item['candidate_id']}")
        reviewed = LocalVisualReview.model_validate(item["assistant_review"])
        if _review_is_acceptable(reviewed):
            accepted += 1
        else:
            failures.append(item["candidate_id"])
        if not reviewed.safe_content or reviewed.essential_anatomy_clipped or reviewed.text_crosses_important_anatomy or reviewed.damaged_or_unreadable:
            critical += 1
    rate = accepted / max(1, len(items))
    pair_exact = pair_false_merges = 0
    pair_false_merge_ids: list[str] = []
    duplicate_relations = {"same_image", "adjacent_frame_duplicate"}
    for pair in pair_items:
        for side in ("left", "right"):
            endpoint = pair[side]
            image = _safe_image_path(root, endpoint["relative_path"])
            if sha256_file(image) != endpoint["image_sha256"]:
                raise IntakeError(f"Release pair audit bytes changed for {pair['pair_id']}")
        reviewed_pair = LocalPairReview.model_validate(pair["assistant_review"])
        model_relation = pair.get("model_relation", "uncertain")
        pair_exact += int(model_relation == reviewed_pair.relation)
        if model_relation in duplicate_relations and reviewed_pair.relation not in duplicate_relations:
            pair_false_merges += 1
            pair_false_merge_ids.append(pair["pair_id"])
    pair_accuracy = pair_exact / max(1, len(pair_items)) if pair_items else None
    pair_passed = not pair_items or (pair_accuracy >= 0.90 and pair_false_merges == 0)
    return {
        "status": "passed" if rate >= 0.95 and critical == 0 and pair_passed else "failed",
        "audit_path": audit_relative,
        "catalog_id": audit["catalog_id"],
        "accepted_count": accepted,
        "audited_count": len(items),
        "usable_rate": rate,
        "critical_failures": critical,
        "failed_asset_ids": failures,
        "pair_audited_count": len(pair_items),
        "pair_exact_relation_accuracy": pair_accuracy,
        "pair_false_duplicate_merges": pair_false_merges,
        "pair_false_merge_ids": pair_false_merge_ids,
        "pair_audit_passed": pair_passed,
        "assistant_review_origin": "assistant",
        "human_gold_created": False,
    }


def _write_availability(root: Path, catalog_path: str) -> dict[str, Any]:
    catalog = PrototypeGalleryCatalogV2.model_validate_json(ensure_inside(root, catalog_path).read_bytes())
    document = {
        "schema_version": 1,
        "catalog_id": catalog.catalog_id,
        "revision": catalog.availability_revision,
        "items": {item.asset_id: {"sha256": item.sha256, "reviewed_revision": item.asset_revision} for item in catalog.items},
    }
    path = root / "prototype/search/availability" / f"{catalog.catalog_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "revision": catalog.availability_revision, "items": len(catalog.items)}


def finalize_collection(root: Path, run_id: str, *, audit_relative: str | None = None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    catalog = build_catalog(root, run_id)
    run_status = status_run(root, run_id)
    if catalog["item_count"] < int(run_status["target"]):
        return {
            "status": "collection_incomplete",
            "target": run_status["target"],
            "current_count": catalog["item_count"],
            "accepted_this_run": run_status["accepted_this_run"],
            "remaining": max(0, int(run_status["target"]) - int(catalog["item_count"])),
            "run_id": run_id,
        }
    if not audit_relative:
        return prepare_release_audit(root, run_id)
    result = _audit_results(root, audit_relative, catalog["catalog_sha256"])
    if result["status"] == "awaiting_assistant_audit":
        return result
    report_path = ensure_inside(root, catalog["collection_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["release_audit"] = result
    if result["status"] != "passed":
        audit_document = json.loads(ensure_inside(root, audit_relative).read_text(encoding="utf-8"))
        db = connect(_db_path(root))
        rejected = 0
        try:
            db.execute("BEGIN IMMEDIATE")
            for failed_asset in result["failed_asset_ids"]:
                rows = db.execute(
                    "SELECT candidate_id,run_id,source_sha256,revision,metadata_json FROM reference_candidates WHERE status='accepted'"
                ).fetchall()
                for candidate in rows:
                    metadata = json.loads(candidate["metadata_json"])
                    if metadata.get("asset_id") != failed_asset:
                        continue
                    audited = next((item for item in audit_document["items"] if item.get("candidate_id") == failed_asset), {})
                    review = audited.get("assistant_review") or {}
                    evidence = {
                        "audit_id": audit_document.get("audit_id"),
                        "reason": review.get("reason", "Assistant release audit rejected the item"),
                        "fresh_review_required": True,
                    }
                    db.execute(
                        "INSERT INTO reference_reviews(review_id,candidate_id,reviewed_revision,origin,reviewer_id,profile_id,model_sha256,runtime_sha256,prompt_version,rubric_version,run_id,image_sha256,evidence_json,decision_json,inference_ms,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (str(uuid4()), candidate["candidate_id"], candidate["revision"], "assistant", "assistant-codex", None, None, None, "release-audit-v1", RUBRIC_VERSION, candidate["run_id"], candidate["source_sha256"], _json(evidence), _json({"decision": "reject", **evidence}), None, _now()),
                    )
                    db.execute("UPDATE reference_candidates SET status='rejected',revision=revision+1 WHERE candidate_id=?", (candidate["candidate_id"],))
                    rejected += 1
            db.execute("UPDATE reference_runs SET status='audit_failed',updated_at=? WHERE run_id=?", (_now(), run_id))
            db.execute("COMMIT")
        except Exception:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()
        availability = (
            withdraw_availability(root, result["failed_asset_ids"])
            if result["failed_asset_ids"]
            else None
        )
        report["status"] = "audit_failed"
        report["availability_withdrawal"] = availability
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"status": "audit_failed_rejected_items", "audit": result, "rejected_item_count": rejected, "run_id": run_id, "collection_report": report_path.relative_to(root).as_posix()}

    previous = None
    pointer_path = root / "prototype/search/active.json"
    if pointer_path.is_file() and not pointer_path.is_symlink():
        previous = json.loads(pointer_path.read_text(encoding="utf-8")).get("build_id")
    db = connect(_db_path(root))
    try:
        db.execute("BEGIN IMMEDIATE")
        if previous:
            previous_manifest_path = root / "prototype/search/builds" / previous / "manifest.json"
            if previous_manifest_path.is_file() and not previous_manifest_path.is_symlink():
                previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
                previous_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                db.execute(
                    "INSERT OR IGNORE INTO reference_build_history(build_id,manifest_sha256,state,previous_build_id,catalog_path,created_at,metadata_json) VALUES(?,?,?,?,?,?,?)",
                    (
                        previous,
                        previous_pointer.get("manifest_sha256", sha256_file(previous_manifest_path)),
                        "retired",
                        None,
                        previous_manifest["snapshot_path"],
                        previous_manifest.get("created_at", _now()),
                        _json({"legacy_or_prior_active_build": True}),
                    ),
                )
        db.execute(
            "INSERT OR IGNORE INTO reference_build_history(build_id,manifest_sha256,state,previous_build_id,catalog_path,created_at,metadata_json) VALUES(?,?,?,?,?,?,?)",
            ("staging-" + catalog["catalog_id"], "0" * 64, "staging", previous, catalog["catalog_path"], _now(), _json({"catalog_sha256": catalog["catalog_sha256"], "audit": result})),
        )
        db.execute("COMMIT")
    except Exception:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise
    finally:
        db.close()
    mask = _write_availability(root, catalog["catalog_path"])
    executable = REPOSITORY_ROOT / "ml/.venv/bin/python"
    worker = [str(executable), "-u", "-m", "drawable_ml.prototype_worker"]
    prepare_command = [*worker, "prepare", "--data-root", str(root), "--snapshot", catalog["catalog_path"], "--semantic-only", "--stage-only"]
    prepared = subprocess_result = None
    try:
        subprocess_result = subprocess.run(prepare_command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False, timeout=86_400)
        if subprocess_result.returncode != 0:
            raise RuntimeError(subprocess_result.stderr[-3000:] or "Vector build failed")
        prepared = json.loads(subprocess_result.stdout.strip().splitlines()[-1])
        build_id = prepared["build_id"]
        validate_result = subprocess.run([*worker, "validate", "--data-root", str(root), "--build-id", build_id, "--staged"], cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False, timeout=1800)
        if validate_result.returncode != 0:
            raise RuntimeError(validate_result.stderr[-3000:] or "Staged build validation failed")
        validation = json.loads(validate_result.stdout.strip().splitlines()[-1])
        if validation.get("ready") is not True or validation.get("gallery_size") != catalog["item_count"]:
            raise RuntimeError("Staged search build did not load the complete V2 catalog")
        activation = subprocess.run([*worker, "activate", "--data-root", str(root), "--build-id", build_id], cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False, timeout=60)
        if activation.returncode != 0:
            raise RuntimeError(activation.stderr[-3000:] or "Staged build activation failed")
        activated = json.loads(activation.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
        if previous:
            report["active_build_id"] = previous
        report["status"] = "build_failed_previous_build_preserved"
        report["build_error"] = str(error)
        report["availability_mask"] = mask
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        db = connect(_db_path(root))
        try:
            db.execute("UPDATE reference_build_history SET state='failed',metadata_json=? WHERE build_id=?", (_json({"error": str(error)}), "staging-" + catalog["catalog_id"]))
            db.execute("UPDATE reference_runs SET status='build_failed',updated_at=? WHERE run_id=?", (_now(), run_id))
        finally:
            db.close()
        return {"status": "build_failed", "error": str(error), "previous_build_id": previous, "collection_report": report_path.relative_to(root).as_posix()}
    report.update({
        "status": "active",
        "build_id": activated["build_id"],
        "active_manifest_sha256": activated["manifest_sha256"],
        "availability_mask": mask,
        "release_audit": result,
        "build_validation": validation,
        "indexing_measurements": json.loads(
            ensure_inside(
                root,
                f"prototype/search/builds/{activated['build_id']}/manifest.json",
            ).read_text(encoding="utf-8")
        ).get("measurements", {}),
        "search_latency": {
            "status": "not_measured",
            "target_visible_cards_ms": 1500,
            "warm_measurements_ms": [],
            "retrieval_quality_evaluated": False,
        },
        "previous_build_id": previous if previous != activated["build_id"] else None,
    })
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    db = connect(_db_path(root))
    try:
        db.execute("BEGIN IMMEDIATE")
        existing_build = db.execute(
            "SELECT previous_build_id FROM reference_build_history WHERE build_id=?",
            (activated["build_id"],),
        ).fetchone()
        previous_for_build = (
            previous
            if previous != activated["build_id"]
            else (existing_build["previous_build_id"] if existing_build else None)
        )
        db.execute(
            "DELETE FROM reference_build_history WHERE build_id=?",
            ("staging-" + catalog["catalog_id"],),
        )
        db.execute(
            """INSERT INTO reference_build_history(
                   build_id,manifest_sha256,state,previous_build_id,catalog_path,created_at,metadata_json
               ) VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(build_id) DO UPDATE SET
                   manifest_sha256=excluded.manifest_sha256,
                   state='active',
                   previous_build_id=excluded.previous_build_id,
                   catalog_path=excluded.catalog_path,
                   created_at=excluded.created_at,
                   metadata_json=excluded.metadata_json""",
            (
                activated["build_id"],
                activated["manifest_sha256"],
                "active",
                previous_for_build,
                catalog["catalog_path"],
                _now(),
                _json({"catalog_id": catalog["catalog_id"], "availability_mask": mask, "audit": result}),
            ),
        )
        if previous and previous != activated["build_id"]:
            db.execute("UPDATE reference_build_history SET state='retired' WHERE build_id=?", (previous,))
        db.execute("UPDATE reference_runs SET status='complete',updated_at=? WHERE run_id=?", (_now(), run_id))
        db.execute("COMMIT")
    except Exception:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise
    finally:
        db.close()
    return {"status": "active", "build_id": activated["build_id"], "catalog_id": catalog["catalog_id"], "gallery_size": catalog["item_count"], "collection_report": report_path.relative_to(root).as_posix(), "release_audit": result}


def rollback_build(root: Path, build_id: str | None = None) -> dict[str, Any]:
    """Validate and atomically reactivate a previously recorded prototype build."""
    root = root.resolve(strict=True)
    migrate(_db_path(root))
    db = connect(_db_path(root))
    try:
        if build_id is None:
            row = db.execute(
                "SELECT build_id FROM reference_build_history WHERE state='retired' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise LookupError("No retired prototype build is available to roll back to")
            build_id = str(row["build_id"])
        history = db.execute(
            "SELECT * FROM reference_build_history WHERE build_id=?", (build_id,)
        ).fetchone()
        if history is None or history["state"] not in {"retired", "active"}:
            raise LookupError("Rollback target is not a recorded active or retired build")
    finally:
        db.close()
    executable = REPOSITORY_ROOT / "ml/.venv/bin/python"
    worker = [str(executable), "-u", "-m", "drawable_ml.prototype_worker"]
    validation = subprocess.run(
        [*worker, "validate", "--data-root", str(root), "--build-id", build_id],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if validation.returncode != 0:
        raise RuntimeError(validation.stderr[-3000:] or "Rollback target failed staged validation")
    validated = json.loads(validation.stdout.strip().splitlines()[-1])
    if validated.get("ready") is not True:
        raise RuntimeError("Rollback target is not ready for serving")
    activation = subprocess.run(
        [*worker, "activate", "--data-root", str(root), "--build-id", build_id],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if activation.returncode != 0:
        raise RuntimeError(activation.stderr[-3000:] or "Rollback activation failed")
    activated = json.loads(activation.stdout.strip().splitlines()[-1])
    db = connect(_db_path(root))
    try:
        db.execute("BEGIN IMMEDIATE")
        db.execute("UPDATE reference_build_history SET state='retired' WHERE state='active' AND build_id<>?", (build_id,))
        db.execute(
            "UPDATE reference_build_history SET state='active',manifest_sha256=?,metadata_json=? WHERE build_id=?",
            (activated["manifest_sha256"], _json({"rollback_activated_at": _now(), "rollback_validation": validated}), build_id),
        )
        db.execute("COMMIT")
    except Exception:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise
    finally:
        db.close()
    return {"status": "active", "build_id": build_id, "gallery_size": validated["gallery_size"], "manifest_sha256": activated["manifest_sha256"]}
