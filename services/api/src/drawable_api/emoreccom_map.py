from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .intake import (
    IntakeError,
    _emoreccom_annotations,
    ensure_inside,
    sha256_file,
    write_json_atomic,
    write_json_create,
)

OFFICIAL_COMICS_INDEX_URL = "https://obj.umiacs.umd.edu/comics/COMICS_ocr_file.csv"
MAX_INDEX_BYTES = 500 * 1024 * 1024


def fetch_official_book_index(data_root: Path, *, resume: bool = False) -> dict[str, Any]:
    """Download only the official COMICS OCR CSV needed for book-ID cross-checks."""
    root = data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = ensure_inside(root, "raw/comics-metadata/COMICS_ocr_file.csv")
    receipt_dir = ensure_inside(root, "manifests/source-metadata/comics-index")
    if destination.is_symlink() or receipt_dir.is_symlink():
        raise IntakeError("Symbolic links are not permitted in COMICS index storage")
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise IntakeError("Existing COMICS OCR index is unsafe")
        digest = sha256_file(destination)
        receipt_path = receipt_dir / f"{digest}.json"
        if receipt_path.is_file() and not receipt_path.is_symlink():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                receipt.get("sha256") == digest
                and receipt.get("official_url") == OFFICIAL_COMICS_INDEX_URL
            ):
                return {
                    "status": "already_verified",
                    "index_path": destination.relative_to(root).as_posix(),
                    "sha256": digest,
                    "byte_length": destination.stat().st_size,
                }
        raise IntakeError(
            "A COMICS OCR index already exists without its official acquisition receipt; do not overwrite it"
        )
    partial = destination.with_name(f"{destination.name}.part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset and not resume:
        raise IntakeError("A partial official COMICS index exists; pass --resume to continue")
    headers = {
        "User-Agent": "drawable-research-data-intake/0.1",
        "Accept": "text/csv,application/octet-stream;q=0.9",
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(OFFICIAL_COMICS_INDEX_URL, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        raise IntakeError(
            "Official COMICS OCR index request failed; any partial bytes were retained"
        ) from error
    with response:
        final = urllib.parse.urlparse(response.geturl())
        if (
            final.scheme != "https"
            or (final.hostname or "").casefold().rstrip(".") != "obj.umiacs.umd.edu"
        ):
            raise IntakeError(
                "Official COMICS OCR index redirected outside the university object host"
            )
        status = getattr(response, "status", response.getcode())
        append = bool(
            offset
            and status == 206
            and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
        )
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        written = offset
        try:
            with partial.open(mode) as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_INDEX_BYTES:
                        raise IntakeError(
                            "Official COMICS OCR index exceeds the 500 MiB safety limit"
                        )
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise IntakeError(
                "Could not write the official COMICS index; the partial file was retained"
            ) from error
        length = response.headers.get("Content-Length")
        if length and written != offset + int(length):
            raise IntakeError(
                "Official COMICS index length differs from Content-Length; partial file retained"
            )
    book_ids, row_count = _official_book_index(partial, root)
    digest = sha256_file(partial)
    try:
        os.link(partial, destination)
    except FileExistsError as error:
        raise IntakeError(
            "COMICS OCR index destination appeared during download; it was not overwritten"
        ) from error
    partial.unlink()
    receipt = {
        "schema_version": 1,
        "source_id": "comics-official-ocr-index",
        "official_url": OFFICIAL_COMICS_INDEX_URL,
        "final_url": response.geturl(),
        "relative_path": destination.relative_to(root).as_posix(),
        "byte_length": destination.stat().st_size,
        "sha256": digest,
        "row_count": row_count,
        "distinct_comic_no": len(book_ids),
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "purpose": "Metadata-only cross-check for reviewed EmoRecCom filename-prefix-to-COMICS-book mapping; no COMICS panel archive is downloaded.",
    }
    write_json_create(receipt_dir / f"{digest}.json", receipt)
    return {
        "status": "downloaded_and_verified",
        "index_path": receipt["relative_path"],
        "sha256": digest,
        "byte_length": receipt["byte_length"],
        "distinct_comic_no": len(book_ids),
        "receipt_path": f"manifests/source-metadata/comics-index/{digest}.json",
    }


def _expected_prefixes(root: Path) -> dict[str, set[str]]:
    csv_root = ensure_inside(root, "visemocomic_csvs")
    if (
        not (csv_root / "emoreccom_train.csv").is_file()
        or not (csv_root / "emoreccom_test.csv").is_file()
    ):
        raise IntakeError(
            "Both official EmoRecCom train and test annotation CSVs are required for book mapping"
        )
    panels = _emoreccom_annotations(root)
    result: dict[str, set[str]] = {"source_train": set(), "source_test_locked": set()}
    for panel in panels.values():
        result[panel["publication_partition"]].add(panel["provisional_book_prefix"])
    if not result["source_train"] or not result["source_test_locked"]:
        raise IntakeError(
            "Both official EmoRecCom partitions must contain at least one panel prefix"
        )
    return result


def create_book_mapping_template(data_root: Path, output_path: Path) -> dict[str, Any]:
    root = data_root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    if output.resolve(strict=False) == root or root not in output.resolve(strict=False).parents:
        raise IntakeError("EmoRecCom mapping templates must remain inside the private data root")
    expected = _expected_prefixes(root)
    all_prefixes = sorted(expected["source_train"] | expected["source_test_locked"])
    template = {
        "schema_version": 1,
        "source_id": "emoreccom-panels",
        "status": "mapping_template_unreviewed",
        "instructions": "Manually map every filename prefix to the COMICS comic_no after checking the official OCR metadata and source naming. Do not infer a book ID from a number-like prefix alone. Supply one mapping per prefix and an evidence note.",
        "official_index_url": OFFICIAL_COMICS_INDEX_URL,
        "partitions_for_leakage_check_only": {
            "source_train_prefixes": sorted(expected["source_train"]),
            "locked_test_prefixes": sorted(expected["source_test_locked"]),
        },
        "mappings": [
            {"filename_prefix": prefix, "comic_no": "", "book_title": "", "evidence_note": ""}
            for prefix in all_prefixes
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, template)
    return {
        "status": template["status"],
        "prefix_count": len(all_prefixes),
        "template_path": output.relative_to(root).as_posix(),
    }


def _official_book_index(path: Path, root: Path) -> tuple[set[str], int]:
    if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
        raise IntakeError(
            "Official COMICS OCR index must be a regular file inside the private data root"
        )
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_INDEX_BYTES:
        raise IntakeError("Official COMICS OCR index is empty or exceeds the 500 MiB safety limit")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if (
                not reader.fieldnames
                or len(reader.fieldnames) != len(set(reader.fieldnames))
                or "comic_no" not in reader.fieldnames
            ):
                raise IntakeError(
                    "Official COMICS OCR index must have a unique header containing comic_no"
                )
            book_ids: set[str] = set()
            row_count = 0
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise IntakeError(
                        f"Official COMICS OCR index row {row_number} has extra fields"
                    )
                value = (row.get("comic_no") or "").strip()
                if not value or not value.isdigit():
                    raise IntakeError(
                        f"Official COMICS OCR index row {row_number} has an invalid comic_no"
                    )
                book_ids.add(str(int(value)))
                row_count += 1
    except (OSError, UnicodeError, csv.Error) as error:
        if isinstance(error, IntakeError):
            raise
        raise IntakeError("Official COMICS OCR index could not be parsed") from error
    if not book_ids:
        raise IntakeError("Official COMICS OCR index contains no book IDs")
    return book_ids, row_count


def _verify_official_index_receipt(root: Path, index_path: Path) -> dict[str, Any]:
    digest = sha256_file(index_path)
    receipt_path = ensure_inside(root, f"manifests/source-metadata/comics-index/{digest}.json")
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise IntakeError("Official COMICS OCR index acquisition receipt is missing")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError("Official COMICS OCR index receipt is malformed") from error
    if (
        receipt.get("source_id") != "comics-official-ocr-index"
        or receipt.get("official_url") != OFFICIAL_COMICS_INDEX_URL
        or receipt.get("relative_path") != index_path.relative_to(root).as_posix()
        or receipt.get("sha256") != digest
        or receipt.get("byte_length") != index_path.stat().st_size
    ):
        raise IntakeError("Official COMICS OCR index receipt does not match the local bytes")
    return receipt


def approve_book_mapping(
    data_root: Path,
    mapping_path: Path,
    official_index_path: Path,
    *,
    reviewer_id: str,
    output_path: Path,
) -> dict[str, Any]:
    root = data_root.resolve()
    for path, label in ((mapping_path, "mapping"), (official_index_path, "official index")):
        resolved = path if path.is_absolute() else root / path
        if (
            resolved.is_symlink()
            or not resolved.is_file()
            or root not in resolved.resolve().parents
        ):
            raise IntakeError(
                f"EmoRecCom {label} file must be a regular file inside the private data root"
            )
    source_mapping_path = mapping_path if mapping_path.is_absolute() else root / mapping_path
    index_path = (
        official_index_path if official_index_path.is_absolute() else root / official_index_path
    )
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", reviewer_id):
        raise IntakeError(
            "Reviewer ID must contain 3 to 80 ASCII letters, digits, underscores, or hyphens"
        )
    try:
        mapping_template = json.loads(source_mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError("EmoRecCom mapping input is missing or malformed") from error
    if (
        mapping_template.get("schema_version") != 1
        or mapping_template.get("source_id") != "emoreccom-panels"
        or not isinstance(mapping_template.get("mappings"), list)
    ):
        raise IntakeError("EmoRecCom mapping input has an unsupported schema or source identity")
    official_ids, index_rows = _official_book_index(index_path, root)
    index_receipt = _verify_official_index_receipt(root, index_path)
    expected = _expected_prefixes(root)
    submitted: dict[str, dict[str, str]] = {}
    for row in mapping_template["mappings"]:
        if not isinstance(row, dict):
            raise IntakeError("Every EmoRecCom book mapping must be an object")
        prefix = row.get("filename_prefix")
        raw_book_id = str(row.get("comic_no", "")).strip()
        evidence = str(row.get("evidence_note", "")).strip()
        if (
            not isinstance(prefix, str)
            or prefix in submitted
            or not re.fullmatch(r"[A-Za-z0-9.-]{1,100}", prefix)
        ):
            raise IntakeError("EmoRecCom mapping contains an invalid or duplicate filename prefix")
        if not raw_book_id.isdigit() or str(int(raw_book_id)) not in official_ids:
            raise IntakeError(
                f"Filename prefix {prefix} maps to a comic_no absent from the official OCR index"
            )
        if len(evidence) < 8:
            raise IntakeError(
                f"Filename prefix {prefix} needs a concise evidence note documenting how identity was checked"
            )
        submitted[prefix] = {
            "comic_no": str(int(raw_book_id)),
            "book_title": str(row.get("book_title", "")).strip(),
            "evidence_note": evidence,
        }
    expected_prefix_set = expected["source_train"] | expected["source_test_locked"]
    if set(submitted) != expected_prefix_set:
        missing = sorted(expected_prefix_set - set(submitted))
        extra = sorted(set(submitted) - expected_prefix_set)
        raise IntakeError(
            f"Every observed train/test prefix must map exactly once (missing={len(missing)}, extra={len(extra)})"
        )
    train_books = {submitted[prefix]["comic_no"] for prefix in expected["source_train"]}
    test_books = {submitted[prefix]["comic_no"] for prefix in expected["source_test_locked"]}
    overlap = sorted(train_books & test_books)
    if overlap:
        raise IntakeError(
            "Mapped EmoRecCom train and locked-test partitions share an official comic_no"
        )
    mapping_rows = [{"filename_prefix": key, **submitted[key]} for key in sorted(submitted)]
    payload = {
        "schema_version": 1,
        "source_id": "emoreccom-panels",
        "status": "reviewed_official_book_mapping",
        "official_index_url": OFFICIAL_COMICS_INDEX_URL,
        "official_index_relative_path": index_path.relative_to(root).as_posix(),
        "official_index_sha256": sha256_file(index_path),
        "official_index_byte_length": index_path.stat().st_size,
        "official_index_rows": index_rows,
        "official_index_receipt_sha256": sha256_file(
            ensure_inside(
                root, f"manifests/source-metadata/comics-index/{index_receipt['sha256']}.json"
            )
        ),
        "official_comic_count": len(official_ids),
        "mapping_input_sha256": sha256_file(source_mapping_path),
        "source_train_prefix_count": len(expected["source_train"]),
        "locked_test_prefix_count": len(expected["source_test_locked"]),
        "source_train_official_book_count": len(train_books),
        "locked_test_official_book_count": len(test_books),
        "train_test_book_overlap_count": 0,
        "reviewer_id": reviewer_id,
        "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "review_attestation": "Reviewer verified each filename-prefix-to-comic_no association against the official COMICS OCR metadata. This is a provenance check, not an image-quality or content approval.",
        "mappings": mapping_rows,
    }
    destination = output_path if output_path.is_absolute() else root / output_path
    if destination.is_symlink() or root not in destination.resolve(strict=False).parents:
        raise IntakeError("Approved EmoRecCom book maps must remain inside the private data root")
    cursor = root
    for component in destination.relative_to(root).parts[:-1]:
        cursor = cursor / component
        if cursor.is_symlink():
            raise IntakeError("Symbolic links are not allowed in approved book-map storage")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload["mapping_sha256"] = hashlib.sha256(
        (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    ).hexdigest()
    write_json_create(destination, payload)
    return {
        "status": payload["status"],
        "mapping_sha256": payload["mapping_sha256"],
        "mapped_prefixes": len(mapping_rows),
        "train_official_books": len(train_books),
        "locked_test_official_books": len(test_books),
        "train_test_book_overlap": 0,
        "output_path": destination.relative_to(root).as_posix(),
    }


def load_approved_book_mapping(data_root: Path, mapping_path: Path) -> dict[str, dict[str, Any]]:
    root = data_root.resolve()
    path = mapping_path if mapping_path.is_absolute() else root / mapping_path
    if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
        raise IntakeError("Approved EmoRecCom book mapping is missing or unsafe")
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError("Approved EmoRecCom book mapping is malformed") from error
    if (
        mapping.get("schema_version") != 1
        or mapping.get("source_id") != "emoreccom-panels"
        or mapping.get("status") != "reviewed_official_book_mapping"
    ):
        raise IntakeError("EmoRecCom book mapping is not an approved version-1 reviewer artifact")
    if not mapping.get("reviewer_id") or not mapping.get("reviewed_at"):
        raise IntakeError("Reviewed EmoRecCom book mapping lacks reviewer identity or timestamp")
    recorded_mapping_hash = mapping.get("mapping_sha256")
    without_hash = {key: value for key, value in mapping.items() if key != "mapping_sha256"}
    actual_mapping_hash = hashlib.sha256(
        (
            json.dumps(without_hash, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if not isinstance(recorded_mapping_hash, str) or recorded_mapping_hash != actual_mapping_hash:
        raise IntakeError("EmoRecCom book mapping checksum is invalid")
    index_path = ensure_inside(root, str(mapping.get("official_index_relative_path", "")))
    if (
        index_path.is_symlink()
        or not index_path.is_file()
        or sha256_file(index_path) != mapping.get("official_index_sha256")
    ):
        raise IntakeError(
            "Official COMICS OCR index no longer matches the reviewed mapping evidence"
        )
    _verify_official_index_receipt(root, index_path)
    official_ids, _ = _official_book_index(index_path, root)
    result: dict[str, dict[str, Any]] = {}
    rows = mapping.get("mappings")
    if not isinstance(rows, list):
        raise IntakeError("Reviewed EmoRecCom mapping has no rows")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("filename_prefix"), str):
            raise IntakeError("Reviewed EmoRecCom mapping contains a malformed row")
        prefix = row["filename_prefix"]
        book_id = str(row.get("comic_no", ""))
        if (
            prefix in result
            or not book_id.isdigit()
            or str(int(book_id)) not in official_ids
            or not row.get("evidence_note")
        ):
            raise IntakeError(
                "Reviewed EmoRecCom mapping has duplicate, unsupported, or unsubstantiated identity data"
            )
        result[prefix] = {
            **row,
            "_mapping_sha256": recorded_mapping_hash,
            "_official_index_sha256": mapping.get("official_index_sha256"),
            "_official_index_receipt_sha256": mapping.get("official_index_receipt_sha256"),
            "_reviewer_id": mapping.get("reviewer_id"),
            "_reviewed_at": mapping.get("reviewed_at"),
        }
    expected = _expected_prefixes(root)
    if set(result) != expected["source_train"] | expected["source_test_locked"]:
        raise IntakeError(
            "Reviewed EmoRecCom mapping is stale against the current annotation prefixes"
        )
    if {result[prefix]["comic_no"] for prefix in expected["source_train"]} & {
        result[prefix]["comic_no"] for prefix in expected["source_test_locked"]
    }:
        raise IntakeError(
            "Reviewed EmoRecCom mapping now crosses the protected train/test boundary"
        )
    return result
