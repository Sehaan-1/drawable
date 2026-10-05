from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
import zlib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .settings import REPOSITORY_ROOT

REGISTRY_PATH = REPOSITORY_ROOT / "config" / "sources.json"
MAX_ARCHIVE_ENTRIES = 100_000
MAX_EXTRACTED_BYTES = 100 * 1024**3
MAX_COMPRESSION_RATIO = 1_000
MAX_TERMS_SNAPSHOT_BYTES = 10 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
CSV_COLUMNS = {
    "filename",
    "body_xmin",
    "body_ymin",
    "body_xmax",
    "body_ymax",
    "face_xmin",
    "face_ymin",
    "face_xmax",
    "face_ymax",
}


class IntakeError(RuntimeError):
    """A source did not meet safe, reproducible intake requirements."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError(f"Cannot read source registry: {error}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or not isinstance(value.get("sources"), list)
    ):
        raise IntakeError("Unsupported or malformed source registry")
    if any(not isinstance(item, dict) for item in value["sources"]):
        raise IntakeError("Every source registry entry must be an object")
    ids = [item.get("id") for item in value["sources"]]
    if any(not isinstance(item, str) for item in ids) or len(ids) != len(set(ids)):
        raise IntakeError("Source registry IDs must be unique strings")
    permission_keys = {
        "local_display",
        "training",
        "evaluation",
        "derive",
        "trace",
        "export",
        "public_demo",
        "redistribute",
        "cloud_storage",
        "cloud_processing",
    }
    for source in value["sources"]:
        if not isinstance(source.get("authority_url"), str):
            raise IntakeError(f"Source {source['id']} has no authoritative HTTPS URL")
        authority = urllib.parse.urlparse(source["authority_url"])
        if (
            authority.scheme != "https"
            or not authority.hostname
            or authority.username
            or authority.password
        ):
            raise IntakeError(f"Source {source['id']} has an invalid authority URL")
        terms_url = source.get("terms_url")
        if terms_url is not None:
            parsed_terms = urllib.parse.urlparse(str(terms_url))
            if (
                parsed_terms.scheme != "https"
                or (parsed_terms.hostname or "").casefold() != authority.hostname.casefold()
                or parsed_terms.username
                or parsed_terms.password
            ):
                raise IntakeError(
                    f"Source {source['id']} terms URL is outside its declared HTTPS authority"
                )
        hosts = source.get("download_hosts")
        if not isinstance(hosts, list) or any(
            not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9.-]+", host) for host in hosts
        ):
            raise IntakeError(f"Source {source['id']} has an invalid download-host allowlist")
        archive_path = source.get("archive_path")
        download_url = source.get("download_url")
        if archive_path is not None:
            if (
                not isinstance(archive_path, str)
                or not archive_path
                or archive_path.startswith("/")
                or "\\" in archive_path
                or any(part in {"", ".", ".."} for part in PurePosixPath(archive_path).parts)
            ):
                raise IntakeError(f"Source {source['id']} has an unsafe archive path")
            parsed_download = urllib.parse.urlparse(str(download_url or ""))
            download_host = (parsed_download.hostname or "").casefold().rstrip(".")
            try:
                download_port = parsed_download.port
            except ValueError as error:
                raise IntakeError(f"Source {source['id']} has an invalid download port") from error
            if (
                parsed_download.scheme != "https"
                or not download_host
                or parsed_download.username
                or parsed_download.password
                or download_port not in {None, 443}
                or not any(
                    download_host == host.casefold()
                    or download_host.endswith(f".{host.casefold()}")
                    for host in hosts
                )
            ):
                raise IntakeError(
                    f"Source {source['id']} has a download URL outside its HTTPS host allowlist"
                )
        elif download_url is not None:
            raise IntakeError(f"API source {source['id']} must not declare a bulk archive URL")
        permissions = source.get("use_permissions")
        if (
            not isinstance(permissions, dict)
            or set(permissions) != permission_keys
            or any(type(flag) is not bool for flag in permissions.values())
        ):
            raise IntakeError(f"Source {source['id']} has malformed use-permission flags")
        if permissions.get("cloud_processing") and not permissions.get("cloud_storage"):
            raise IntakeError(
                f"Source {source['id']} enables cloud processing without separate cloud-storage permission"
            )
        if source.get("rights_status") not in {
            "review_required",
            "approved",
            "rejected",
            "quarantined",
            "unavailable",
        }:
            raise IntakeError(f"Source {source['id']} has an invalid rights status")
        review = source.get("terms_review")
        if not isinstance(review, dict) or review.get("status") not in {
            "review_required",
            "approved",
            "rejected",
        }:
            raise IntakeError(f"Source {source['id']} has no valid terms-review record")
        digest = source.get("terms_snapshot_sha256")
        if digest is not None and (
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise IntakeError(f"Source {source['id']} has a malformed terms snapshot hash")
        if source.get("rights_status") == "approved" and (
            review.get("status") != "approved"
            or not review.get("reviewer_id")
            or not review.get("reviewed_at")
            or review.get("reviewed_sha256") != digest
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise IntakeError(
                f"Approved source {source['id']} lacks a matching reviewer, timestamp, and terms snapshot hash"
            )
    return value


def source_by_id(source_id: str, path: Path = REGISTRY_PATH) -> dict[str, Any]:
    for source in load_registry(path)["sources"]:
        if source["id"] == source_id:
            return source
    raise IntakeError(f"Unknown source ID: {source_id}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_create(path: Path, value: Any) -> None:
    """Atomically create a JSON artifact without ever replacing an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise IntakeError(
                "Immutable artifact already exists; choose a new output path"
            ) from error
        except OSError as error:
            if path.exists():
                raise IntakeError(
                    "Immutable artifact already exists; choose a new output path"
                ) from error
            raise
    finally:
        temporary.unlink(missing_ok=True)


def ensure_inside(root: Path, relative: str) -> Path:
    candidate = root / relative
    resolved = candidate.resolve(strict=False)
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        raise IntakeError("Configured path escapes the dataset root")
    return candidate


def terms_snapshot_receipt_path(root: Path, source_id: str, digest: str) -> Path:
    """Resolve the distinct receipt path, with read compatibility for legacy snapshots."""
    receipt = ensure_inside(root, f"manifests/terms/{source_id}/{digest}.receipt.json")
    if receipt.is_file() and not receipt.is_symlink():
        return receipt
    return ensure_inside(root, f"manifests/terms/{source_id}/{digest}.json")


def _safe_member_name(name: str) -> str:
    normalized = unicodedata.normalize("NFC", name)
    if "\x00" in normalized or "\\" in normalized:
        raise IntakeError("Archive member contains a forbidden path separator or NUL")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise IntakeError("Archive contains an absolute path")
    parts = PurePosixPath(normalized).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise IntakeError("Archive contains a traversal or empty path component")
    return "/".join(parts)


def inspect_zip(path: Path, *, max_expanded_bytes: int = MAX_EXTRACTED_BYTES) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise IntakeError("Archive is not a regular file")
    seen: set[str] = set()
    total = 0
    files = 0
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > MAX_ARCHIVE_ENTRIES:
                raise IntakeError("Archive is empty or exceeds the entry-count limit")
            for entry in entries:
                safe_name = (
                    _safe_member_name(entry.filename.rstrip("/"))
                    if entry.filename.rstrip("/")
                    else ""
                )
                if not safe_name:
                    raise IntakeError("Archive contains an empty member name")
                collision_key = safe_name.casefold()
                if collision_key in seen:
                    raise IntakeError("Archive contains duplicate or case-colliding paths")
                seen.add(collision_key)
                mode = (entry.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise IntakeError("Archive contains a symbolic link")
                if entry.flag_bits & 0x1:
                    raise IntakeError("Encrypted archive members are not supported")
                if entry.is_dir():
                    continue
                if entry.file_size < 0 or entry.compress_size < 0:
                    raise IntakeError("Archive contains invalid size metadata")
                total += entry.file_size
                files += 1
                if total > max_expanded_bytes:
                    raise IntakeError("Archive exceeds the configured expanded-size limit")
                if entry.file_size > 0 and entry.compress_size == 0:
                    raise IntakeError("Archive contains suspicious compression metadata")
                if (
                    entry.compress_size > 0
                    and entry.file_size / entry.compress_size > MAX_COMPRESSION_RATIO
                ):
                    raise IntakeError("Archive contains a member above the compression-ratio limit")
            # Read every member to verify CRCs and detect truncation before declaring the ZIP valid.
            for entry in entries:
                if entry.is_dir():
                    continue
                with archive.open(entry, "r") as stream:
                    for _chunk in iter(lambda: stream.read(COPY_CHUNK_BYTES), b""):
                        pass
    except IntakeError:
        raise
    except (
        zipfile.BadZipFile,
        OSError,
        EOFError,
        RuntimeError,
        NotImplementedError,
        zlib.error,
    ) as error:
        raise IntakeError("Archive integrity or CRC verification failed") from error
    return {"valid": True, "files": files, "entries": len(entries), "expanded_bytes": total}


def _receipt_path(root: Path, source_id: str) -> Path:
    return root / "manifests" / "source-receipts" / f"{source_id}.json"


def verify_archive_receipt(
    root: Path, source: dict[str, Any], archive_path: Path, receipt: dict[str, Any]
) -> None:
    """Bind a receipt to the declared official URL, safe redirect hosts, and current bytes."""
    if not isinstance(receipt, dict):
        raise IntakeError("Archive receipt must be a JSON object")
    requested_url = source.get("download_url")
    final_url = urllib.parse.urlparse(str(receipt.get("final_url", "")))
    final_host = (final_url.hostname or "").casefold().rstrip(".")
    allowed_hosts = source.get("download_hosts", [])
    host_allowed = isinstance(allowed_hosts, list) and any(
        isinstance(allowed, str)
        and (final_host == allowed.casefold() or final_host.endswith(f".{allowed.casefold()}"))
        for allowed in allowed_hosts
    )
    expected_path = source.get("archive_path")
    actual_path = (
        archive_path.relative_to(root.resolve()).as_posix()
        if root.resolve() in archive_path.resolve().parents
        else None
    )
    archive_validation = receipt.get("archive_validation")
    if (
        not requested_url
        or receipt.get("source_id") != source.get("id")
        or receipt.get("authority_url") != source.get("authority_url")
        or receipt.get("requested_url") != requested_url
        or final_url.scheme != "https"
        or not host_allowed
        or receipt.get("archive_path") != expected_path
        or actual_path != expected_path
        or not isinstance(archive_validation, dict)
        or archive_validation.get("valid") is not True
        or not isinstance(receipt.get("retrieved_at"), str)
        or not isinstance(receipt.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"])
        or archive_path.is_symlink()
        or not archive_path.is_file()
        or receipt.get("byte_length") != archive_path.stat().st_size
        or sha256_file(archive_path) != receipt.get("sha256")
    ):
        raise IntakeError(
            "Archive receipt is missing, not bound to the declared official source, or mismatched with current bytes"
        )


def _archive_candidates(root: Path, source: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    expected = source.get("archive_filename")
    if not expected:
        return [], []
    canonical = ensure_inside(root, source["archive_path"])
    legacy = ensure_inside(root, expected)
    partial_names = {f"{expected}.part", f".{expected}.part"}
    archives: list[Path] = []
    partials: list[Path] = []
    for candidate in (canonical, legacy):
        if candidate.is_symlink():
            partials.append(candidate)
        elif candidate.is_file():
            archives.append(candidate)
        for partial_name in partial_names:
            partial = candidate.parent / partial_name
            if partial.is_symlink() or partial.is_file():
                partials.append(partial)
    # A verified raw location has precedence, but conflicting copies remain visible.
    return list(dict.fromkeys(archives)), list(dict.fromkeys(partials))


def _tree_summary(root: Path, relative: str) -> dict[str, Any] | None:
    tree = ensure_inside(root, relative)
    if not tree.is_dir() or tree.is_symlink():
        return None
    files = 0
    bytes_total = 0
    modality_counts: dict[str, int] = {"sketch": 0, "color": 0, "composition": 0, "other": 0}
    for current, dirnames, filenames in os.walk(tree, followlinks=False):
        current_path = Path(current)
        dirnames[:] = sorted(name for name in dirnames if not (current_path / name).is_symlink())
        for name in filenames:
            path = current_path / name
            if path.is_symlink():
                raise IntakeError("Extracted data tree contains a symbolic link")
            files += 1
            bytes_total += path.stat().st_size
            searchable = "/".join(path.relative_to(tree).parts).casefold()
            modality = next(
                (item for item in ("sketch", "color", "composition") if item in searchable), "other"
            )
            modality_counts[modality] += 1
    return {
        "status": "extracted_unverified",
        "files": files,
        "bytes": bytes_total,
        "modalities": modality_counts,
    }


def _anita_work_records(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = path or REPOSITORY_ROOT / "config" / "anita-work-terms.json"
    try:
        records = json.loads(path.read_text(encoding="utf-8"))["works"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise IntakeError("Per-work Anita terms registry is missing or invalid") from error
    indexed = {record.get("folder_id", "").casefold(): record for record in records}
    if len(indexed) != len(records) or any(not key for key in indexed):
        raise IntakeError("Anita work folder IDs must be unique and nonempty")
    permission_keys = {
        "local_display",
        "training",
        "evaluation",
        "derive",
        "trace",
        "export",
        "public_demo",
        "redistribute",
        "cloud_storage",
        "cloud_processing",
    }
    for record in records:
        permissions = record.get("use_permissions")
        if (
            not isinstance(permissions, dict)
            or set(permissions) != permission_keys
            or any(type(flag) is not bool for flag in permissions.values())
        ):
            raise IntakeError(
                f"Anita work {record['folder_id']} has incomplete per-work use permissions"
            )
        if record.get("rights_status") not in {
            "review_required",
            "approved",
            "rejected",
            "quarantined",
            "unavailable",
        }:
            raise IntakeError(f"Anita work {record['folder_id']} has an invalid rights status")
        if record.get("reviewer_decision") not in {"review_required", "approved", "rejected"}:
            raise IntakeError(f"Anita work {record['folder_id']} has an invalid terms decision")
        license_id = record.get("license")
        if license_id not in {"CC BY 4.0", "CC BY-NC-SA 4.0"}:
            raise IntakeError(f"Anita work {record['folder_id']} has an unknown license identifier")
        if record.get("non_commercial") != (license_id == "CC BY-NC-SA 4.0") or record.get(
            "share_alike"
        ) != (license_id == "CC BY-NC-SA 4.0"):
            raise IntakeError(
                f"Anita work {record['folder_id']} has inconsistent license obligations"
            )
        digest = record.get("terms_snapshot_sha256")
        if digest is not None and (
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise IntakeError(
                f"Anita work {record['folder_id']} has a malformed terms snapshot hash"
            )
        if record.get("rights_status") == "approved" and (
            record.get("reviewer_decision") != "approved"
            or not record.get("reviewer_id")
            or not record.get("reviewed_at")
            or record.get("reviewed_sha256") != digest
            or not digest
            or not record.get("attribution_text")
        ):
            raise IntakeError(
                f"Approved Anita work {record['folder_id']} lacks matching terms evidence, reviewer, timestamp, or attribution"
            )
    return indexed


def _emoreccom_annotations(root: Path) -> dict[str, dict[str, Any]]:
    """Read all published character rows, indexed by their panel filename.

    A panel commonly appears more than once because each row describes a different
    character. Only a panel crossing the publisher's train/test partition is an
    overlap; repeated rows in one partition are retained verbatim.
    """
    directory = ensure_inside(root, "visemocomic_csvs")
    mapping: dict[str, dict[str, Any]] = {}
    for split, filename in (
        ("source_train", "emoreccom_train.csv"),
        ("source_test_locked", "emoreccom_test.csv"),
    ):
        path = directory / filename
        if not path.is_file():
            continue
        csv_sha256 = sha256_file(path)
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if (
                    not reader.fieldnames
                    or len(reader.fieldnames) != len(set(reader.fieldnames))
                    or not CSV_COLUMNS.issubset(set(reader.fieldnames))
                ):
                    raise IntakeError(f"EmoRecCom {filename} has an invalid or incomplete header")
                for row_number, row in enumerate(reader, start=2):
                    if None in row:
                        raise IntakeError(
                            f"EmoRecCom {filename} row {row_number} has extra CSV fields"
                        )
                    name = (row.get("filename") or "").strip()
                    if not name or Path(name).name != name or ".." in name or "\\" in name:
                        raise IntakeError(f"EmoRecCom {filename} contains an unsafe panel path")
                    normalized_name = name.casefold()
                    prefix = name.split("_", maxsplit=1)[0]
                    panel = mapping.get(normalized_name)
                    if panel is None:
                        panel = {
                            "filename": name,
                            "publication_partition": split,
                            "provisional_book_prefix": prefix,
                            "book_identity_verified": False,
                            "annotation_source_file": filename,
                            "annotation_source_sha256": csv_sha256,
                            "annotations": [],
                        }
                        mapping[normalized_name] = panel
                    elif panel["publication_partition"] != split:
                        raise IntakeError("A panel appears in both official EmoRecCom partitions")

                    numeric_box: dict[str, float] = {}
                    for column in CSV_COLUMNS - {"filename"}:
                        try:
                            value = float(row.get(column) or "")
                        except (TypeError, ValueError) as error:
                            raise IntakeError(
                                f"EmoRecCom {filename} row {row_number} has an invalid {column}"
                            ) from error
                        if not math.isfinite(value):
                            raise IntakeError(
                                f"EmoRecCom {filename} row {row_number} has a non-finite {column}"
                            )
                        numeric_box[column] = value
                    boxes: dict[str, dict[str, float]] = {}
                    for kind in ("body", "face"):
                        box = {
                            axis: numeric_box[f"{kind}_{axis}"]
                            for axis in ("xmin", "ymin", "xmax", "ymax")
                        }
                        if box["xmin"] >= box["xmax"] or box["ymin"] >= box["ymax"]:
                            raise IntakeError(
                                f"EmoRecCom {filename} row {row_number} has invalid {kind} box geometry"
                            )
                        boxes[kind] = box
                    source_row_id = (row.get("Unnamed: 0") or "").strip() or str(row_number - 2)
                    source_fields = {
                        str(key): (value or "").strip()
                        for key, value in row.items()
                        if key is not None
                    }
                    canonical = json.dumps(
                        source_fields, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                    )
                    annotation_id = hashlib.sha256(
                        f"{filename.casefold()}\0{split}\0{row_number}\0{canonical}".encode()
                    ).hexdigest()
                    panel["annotations"].append(
                        {
                            "annotation_id": annotation_id,
                            "source_row_id": source_row_id,
                            "source_row_number": row_number,
                            "boxes": boxes,
                            "source_fields": source_fields,
                        }
                    )
        except (OSError, UnicodeError, csv.Error) as error:
            if isinstance(error, IntakeError):
                raise
            raise IntakeError(
                f"Cannot read EmoRecCom annotations in {filename}: {error}"
            ) from error
    return mapping


def _emoreccom_partitions(root: Path) -> dict[str, dict[str, Any]]:
    """Compatibility wrapper for callers that need partition plus annotations."""
    return _emoreccom_annotations(root)


def validate_emoreccom(root: Path, *, book_mapping_path: Path | None = None) -> dict[str, Any]:
    directory = ensure_inside(root, "visemocomic_csvs")
    if not directory.is_dir():
        return {"status": "annotations_missing"}
    csv_paths = sorted(directory.glob("*.csv"), key=lambda item: item.name)
    if (
        not (directory / "emoreccom_train.csv").is_file()
        or not (directory / "emoreccom_test.csv").is_file()
    ):
        return {"status": "required_csv_missing", "csv_files": [path.name for path in csv_paths]}
    try:
        panels = _emoreccom_annotations(root)
    except IntakeError as error:
        return {"status": "invalid_annotations", "error": str(error)}
    book_mappings: dict[str, dict[str, Any]] = {}
    if book_mapping_path is not None:
        from .emoreccom_map import load_approved_book_mapping

        try:
            book_mappings = load_approved_book_mapping(root, book_mapping_path)
        except IntakeError as error:
            return {"status": "invalid_book_mapping", "error": str(error)}
    train_panels = [
        panel for panel in panels.values() if panel["publication_partition"] == "source_train"
    ]
    test_panels = [
        panel for panel in panels.values() if panel["publication_partition"] == "source_test_locked"
    ]
    train_rows = sum(len(panel["annotations"]) for panel in train_panels)
    test_rows = sum(len(panel["annotations"]) for panel in test_panels)
    train_names = {panel["filename"] for panel in train_panels}
    test_names = {panel["filename"] for panel in test_panels}
    train_groups = {
        book_mappings[panel["provisional_book_prefix"]]["comic_no"]
        if book_mappings
        else panel["provisional_book_prefix"]
        for panel in train_panels
    }
    test_groups = {
        book_mappings[panel["provisional_book_prefix"]]["comic_no"]
        if book_mappings
        else panel["provisional_book_prefix"]
        for panel in test_panels
    }
    image_root = ensure_inside(root, "emoreccom_panels")
    available: dict[str, list[str]] = {}
    if image_root.is_dir():
        for path in image_root.rglob("*"):
            if path.is_symlink():
                return {"status": "panel_tree_symlink"}
            if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg"}:
                available.setdefault(path.name.casefold(), []).append(
                    path.relative_to(image_root).as_posix()
                )
    ambiguous = sorted(name for name, paths in available.items() if len(paths) != 1)
    available_names = set(available)
    missing_train = sorted(name for name in train_names if name.casefold() not in available_names)
    missing_test = sorted(name for name in test_names if name.casefold() not in available_names)
    referenced = train_names | test_names
    unreferenced = len(available_names - {name.casefold() for name in referenced})
    overlap = len(
        {name.casefold() for name in train_names} & {name.casefold() for name in test_names}
    )
    group_overlap = len(train_groups & test_groups)
    book_identity_verified = bool(book_mappings)
    validation_gates = {
        "required_columns_present": True,
        "box_geometry_valid": True,
        "official_filename_split_disjoint": overlap == 0,
        "filename_prefix_split_disjoint": len(
            {panel["provisional_book_prefix"] for panel in train_panels}
            & {panel["provisional_book_prefix"] for panel in test_panels}
        )
        == 0,
        "official_book_split_disjoint": group_overlap == 0,
        "all_panels_found": not missing_train and not missing_test,
        "panel_basenames_unambiguous": not ambiguous,
        "authoritative_book_mapping": book_identity_verified,
    }
    status = "validated" if all(validation_gates.values()) else "review_required"
    return {
        "status": status,
        "train_rows": train_rows,
        "train_unique_panels": len(train_panels),
        "test_rows": test_rows,
        "test_unique_panels": len(test_panels),
        "train_multi_character_panels": sum(
            len(panel["annotations"]) > 1 for panel in train_panels
        ),
        "test_multi_character_panels": sum(len(panel["annotations"]) > 1 for panel in test_panels),
        "official_character_annotations_preserved": True,
        "filename_overlap": overlap,
        "candidate_prefix_group_overlap": group_overlap,
        "book_identity_verified": book_identity_verified,
        "verified_train_book_count": len(train_groups) if book_identity_verified else 0,
        "verified_locked_test_book_count": len(test_groups) if book_identity_verified else 0,
        "official_book_overlap": group_overlap if book_identity_verified else None,
        "book_mapping_sha256": next(iter(book_mappings.values())).get("_mapping_sha256")
        if book_mappings
        else None,
        "validation_gates": validation_gates,
        "missing_train_panels": len(missing_train),
        "missing_test_panels": len(missing_test),
        "ambiguous_panel_basenames": ambiguous,
        "unreferenced_panels": unreferenced,
        "excluded_csvs": sorted(
            path.name
            for path in csv_paths
            if path.name not in {"emoreccom_train.csv", "emoreccom_test.csv"}
        ),
        "policy": "Multiple character rows within one panel are preserved. Filename prefixes remain provisional unless bound through a reviewer-recorded crosswalk to the official COMICS OCR book index. Locked-test prefixes are read only for split-leakage checks; test artwork and labels remain excluded from curation and development.",
    }


def audit_sources(
    root: Path, *, registry_path: Path = REGISTRY_PATH, verify_archives: bool = True
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    items: list[dict[str, Any]] = []
    for source in registry["sources"]:
        record: dict[str, Any] = {
            "source_id": source["id"],
            "title": source["title"],
            "authority_url": source["authority_url"],
            "rights_status": source["rights_status"],
            "cloud_storage_allowed": source["use_permissions"].get("cloud_storage", False),
            "cloud_processing_allowed": source["use_permissions"].get("cloud_processing", False),
        }
        terms_digest = source.get("terms_snapshot_sha256")
        if isinstance(terms_digest, str) and re.fullmatch(r"[0-9a-f]{64}", terms_digest):
            terms_receipt_path = terms_snapshot_receipt_path(root, source["id"], terms_digest)
            try:
                terms_receipt = json.loads(terms_receipt_path.read_text(encoding="utf-8"))
                snapshot_path = ensure_inside(root, terms_receipt.get("snapshot_path", ""))
                terms_verified = (
                    terms_receipt.get("sha256") == terms_digest
                    and snapshot_path.is_file()
                    and not snapshot_path.is_symlink()
                    and sha256_file(snapshot_path) == terms_digest
                )
            except (OSError, json.JSONDecodeError, IntakeError):
                terms_verified = False
            record["terms_snapshot_status"] = (
                "hash_verified_review_still_required"
                if terms_verified and source.get("rights_status") != "approved"
                else "hash_verified"
                if terms_verified
                else "snapshot_missing_or_mismatched"
            )
        else:
            record["terms_snapshot_status"] = "not_snapshotted"
        if source.get("archive_filename"):
            archives, partials = _archive_candidates(root, source)
            if len(archives) > 1:
                record.update(status="conflicting_archive_copies", archive_count=len(archives))
            elif not archives:
                record["status"] = "partial_download_present" if partials else "archive_missing"
            else:
                archive = archives[0]
                receipt_path = _receipt_path(root, source["id"])
                receipt = None
                if receipt_path.is_file():
                    try:
                        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        record["status"] = "receipt_corrupt"
                if "status" not in record:
                    try:
                        verified = inspect_zip(archive) if verify_archives else None
                        digest = sha256_file(archive) if verify_archives else None
                        relative = archive.relative_to(root).as_posix()
                        if not receipt:
                            status = "archive_present_unverified"
                        elif not verify_archives:
                            status = "archive_present_not_rehashed"
                        else:
                            verify_archive_receipt(root, source, archive, receipt)
                            if verified and not verified["valid"]:
                                raise IntakeError("Archive ZIP validation failed")
                            status = "archive_verified"
                        record.update(
                            status=status, archive_path=relative, byte_length=archive.stat().st_size
                        )
                        if digest:
                            record["sha256"] = digest
                        if verified:
                            record["archive_validation"] = verified
                    except (OSError, IntakeError) as error:
                        record.update(status="archive_invalid", error=str(error))
            tree = _tree_summary(root, source["extracted_path"])
            if tree:
                record["extracted_tree"] = tree
        else:
            record["status"] = "api_source_not_acquired"
        items.append(record)
    emoreccom = validate_emoreccom(root)
    return {
        "schema_version": 1,
        "audited_at": utc_now(),
        "policy": "Read-only audit. It does not extract, rename, delete, activate, train, or upload source data.",
        "sources": items,
        "emoreccom_annotations": emoreccom,
    }


def _safe_download_response(response, source: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
    final = urllib.parse.urlparse(response.geturl())
    if final.scheme != "https":
        raise IntakeError("The source redirected to a non-HTTPS address")
    host = (final.hostname or "").casefold().rstrip(".")
    permitted = any(
        host == allowed or host.endswith(f".{allowed}") for allowed in source["download_hosts"]
    )
    if not permitted:
        raise IntakeError("The official download redirected outside its declared host allowlist")
    content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().casefold()
    if content_type in {"text/html", "application/xhtml+xml"}:
        raise IntakeError("The provider returned a web page instead of the requested archive")


def capture_terms_snapshot(
    root: Path, source_id: str, *, registry_path: Path = REGISTRY_PATH
) -> dict[str, Any]:
    """Capture an official source terms page for review; never promotes source permissions."""
    source = source_by_id(source_id, registry_path)
    url = source.get("terms_url") or source.get("authority_url")
    parsed = urllib.parse.urlparse(url or "")
    authority = urllib.parse.urlparse(source.get("authority_url", ""))
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.hostname.casefold() != (authority.hostname or "").casefold()
    ):
        raise IntakeError("Terms snapshots must use the source's declared HTTPS authority host")
    if not isinstance(url, str):
        raise IntakeError("Source terms URL is missing")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "drawable-research-terms-snapshot/0.1",
            "Accept": "text/html, text/plain, application/json;q=0.9",
        },
    )
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.URLError as error:
        raise IntakeError(f"Official terms page could not be retrieved: {error.reason}") from error
    with response:
        final = urllib.parse.urlparse(response.geturl())
        if (
            final.scheme != "https"
            or (final.hostname or "").casefold() != parsed.hostname.casefold()
        ):
            raise IntakeError("Terms page redirected outside the declared official authority host")
        content_type = (
            (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().casefold()
        )
        if content_type and content_type not in {
            "text/html",
            "application/xhtml+xml",
            "text/plain",
            "application/json",
        }:
            raise IntakeError("Official terms endpoint did not return a text document")
        body = response.read(MAX_TERMS_SNAPSHOT_BYTES + 1)
        if not body or len(body) > MAX_TERMS_SNAPSHOT_BYTES:
            raise IntakeError(
                "Official terms document is empty or exceeds the 10 MiB snapshot limit"
            )
        metadata = {
            name.lower(): response.headers.get(name)
            for name in ("Content-Type", "ETag", "Last-Modified")
            if response.headers.get(name)
        }
    digest = hashlib.sha256(body).hexdigest()
    directory = ensure_inside(root, f"manifests/terms/{source_id}")
    directory.mkdir(parents=True, exist_ok=True)
    extension = ".json" if content_type == "application/json" else ".html"
    snapshot = directory / f"{digest}{extension}"
    if snapshot.exists():
        if sha256_file(snapshot) != digest:
            raise IntakeError("A content-addressed terms snapshot exists with a mismatched hash")
    else:
        temporary = directory / f".{digest}.{uuid.uuid4().hex}.part"
        try:
            with temporary.open("xb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, snapshot)
            except FileExistsError:
                if sha256_file(snapshot) != digest:
                    raise IntakeError("A concurrent terms snapshot has a mismatched hash")
        finally:
            temporary.unlink(missing_ok=True)
    receipt = {
        "schema_version": 1,
        "source_id": source_id,
        "authority_url": source["authority_url"],
        "requested_url": url,
        "final_url": response.geturl(),
        "retrieved_at": utc_now(),
        "content_type": content_type or None,
        "byte_length": len(body),
        "sha256": digest,
        "http": metadata,
        "snapshot_path": snapshot.relative_to(root).as_posix(),
        "review_status": "required",
        "permissions_changed": False,
    }
    write_json_atomic(directory / f"{digest}.receipt.json", receipt)
    return receipt


def acquire_archive(
    root: Path, source_id: str, *, resume: bool = False, registry_path: Path = REGISTRY_PATH
) -> dict[str, Any]:
    source = source_by_id(source_id, registry_path)
    url = source.get("download_url")
    archive_relative = source.get("archive_path")
    if not url or not archive_relative:
        raise IntakeError("This source uses an API workflow and has no archive download")
    target = ensure_inside(root, archive_relative)
    partial = target.with_name(f"{target.name}.part")
    if target.exists():
        raise IntakeError(
            "A destination archive already exists; audit it before choosing another path"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if partial.exists() and not resume:
        raise IntakeError("A partial download exists; pass --resume to continue it")
    headers = {
        "User-Agent": "drawable-research-data-intake/0.1",
        "Accept": "application/zip, application/octet-stream;q=0.9",
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=45)
    except urllib.error.URLError as error:
        raise IntakeError(f"Official source download failed: {error.reason}") from error
    with response:
        _safe_download_response(response, source)
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
                    chunk = response.read(COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    stream.write(chunk)
                    written += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise IntakeError(
                "Could not write the partial archive; the .part file was retained"
            ) from error
        content_length = response.headers.get("Content-Length")
        if content_length and not append and written != int(content_length):
            raise IntakeError(
                "Downloaded byte length differs from Content-Length; partial file retained"
            )
        if content_length and append and written != offset + int(content_length):
            raise IntakeError(
                "Resumed byte length differs from Content-Length; partial file retained"
            )
        if not zipfile.is_zipfile(partial):
            raise IntakeError(
                "Downloaded content is not a complete ZIP archive; partial file retained"
            )
        validation = inspect_zip(partial)
        digest = sha256_file(partial)
        try:
            os.link(partial, target)
        except FileExistsError as error:
            raise IntakeError(
                "The final archive appeared during download; it was not overwritten"
            ) from error
        partial.unlink()
    relative = target.relative_to(root).as_posix()
    receipt = {
        "schema_version": 1,
        "receipt_id": str(uuid.uuid4()),
        "source_id": source_id,
        "authority_url": source["authority_url"],
        "requested_url": url,
        "final_url": response.geturl(),
        "archive_path": relative,
        "byte_length": target.stat().st_size,
        "sha256": digest,
        "retrieved_at": utc_now(),
        "http": {
            name.lower(): response.headers.get(name)
            for name in ("Content-Type", "Content-Length", "ETag", "Last-Modified")
            if response.headers.get(name)
        },
        "terms_snapshot_sha256": source.get("terms_snapshot_sha256"),
        "terms_review": "required_before_activation",
        "archive_validation": validation,
    }
    write_json_atomic(_receipt_path(root, source_id), receipt)
    return receipt


def build_tree_manifest(
    root: Path,
    source_id: str,
    *,
    tree_path: Path | None = None,
    anita_work_ids: set[str] | None = None,
    registry_path: Path = REGISTRY_PATH,
    anita_terms_path: Path | None = None,
    emoreccom_book_map_path: Path | None = None,
) -> dict[str, Any]:
    source = source_by_id(source_id, registry_path)
    base = tree_path or ensure_inside(root, source["extracted_path"])
    if not base.is_absolute():
        base = ensure_inside(root, base.as_posix())
    if root.resolve() not in base.resolve().parents:
        raise IntakeError("Manifest input tree must be inside the data root")
    if not base.is_dir() or base.is_symlink():
        raise IntakeError("Expected extracted source directory is missing or unsafe")
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise IntakeError(
            "Install the locked Python environment to verify and fingerprint images"
        ) from error
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    anita_terms_hash: str | None = None
    emoreccom_book_mappings: dict[str, dict[str, Any]] = {}
    emoreccom_book_map_sha256: str | None = None
    if source_id == "emoreccom-panels" and emoreccom_book_map_path is not None:
        from .emoreccom_map import load_approved_book_mapping

        emoreccom_book_mappings = load_approved_book_mapping(root, emoreccom_book_map_path)
        resolved_book_map = (
            emoreccom_book_map_path
            if emoreccom_book_map_path.is_absolute()
            else root / emoreccom_book_map_path
        )
        emoreccom_book_map_sha256 = sha256_file(resolved_book_map)
    for current, dirnames, filenames in os.walk(base, followlinks=False):
        current_path = Path(current)
        dirnames[:] = sorted(dirnames)
        for dirname in dirnames:
            if (current_path / dirname).is_symlink():
                raise IntakeError("Extracted source contains a symbolic-link directory")
        for name in sorted(filenames):
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise IntakeError("Extracted source contains a non-regular file")
            relative = path.relative_to(base).as_posix()
            if source_id == "anita-dataset" and anita_work_ids is not None:
                selected_path_parts = PurePosixPath(relative).parts
                if not selected_path_parts or selected_path_parts[0].casefold() not in anita_work_ids:
                    continue
            normalized = unicodedata.normalize("NFC", relative).casefold()
            if normalized in seen_paths:
                raise IntakeError("Extracted source contains duplicate normalized paths")
            seen_paths.add(normalized)
            item: dict[str, Any] = {
                "relative_path": relative,
                "byte_length": path.stat().st_size,
                "sha256": sha256_file(path),
                "media_type": None,
                "width": None,
                "height": None,
                "channels": None,
                "has_alpha": None,
                "decoder_status": "not_image",
            }
            if path.suffix.casefold() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".bmp",
                ".tif",
                ".tiff",
            }:
                try:
                    with Image.open(path) as image:
                        width, height = image.size
                        if width <= 0 or height <= 0 or width * height > 40_000_000:
                            raise IntakeError(
                                f"Image dimensions exceed the 40 megapixel intake limit: {relative}"
                            )
                        image.verify()
                    with Image.open(path) as image:
                        image.load()
                        item.update(
                            media_type=Image.MIME.get(image.format or ""),
                            width=width,
                            height=height,
                            channels=len(image.getbands()),
                            has_alpha="A" in image.getbands() or "transparency" in image.info,
                            decoder_status="decoded",
                        )
                except IntakeError:
                    raise
                except (OSError, UnidentifiedImageError, ValueError) as error:
                    item.update(decoder_status="decode_failed", decode_error=type(error).__name__)
            item.update(
                asset_id=f"asset-{source_id}-{hashlib.sha256(relative.encode('utf-8')).hexdigest()[:24]}",
                work_id=None,
                artist_group_id=None,
                group_id=None,
                rights_status="review_required",
                use_permissions={
                    key: False
                    for key in (
                        "local_display",
                        "training",
                        "evaluation",
                        "derive",
                        "trace",
                        "export",
                        "public_demo",
                        "redistribute",
                        "cloud_storage",
                        "cloud_processing",
                    )
                },
                review_status="pending",
                learning_split="unassigned",
                gallery_member=False,
                gold_member=False,
                labels={
                    "style": None,
                    "primary_scope": None,
                    "secondary_scopes": [],
                    "person_count": None,
                    "sfw": "pending",
                    "quality": None,
                },
            )
            entries.append(item)
    entries.sort(key=lambda item: item["relative_path"])
    if source_id == "anita-dataset":
        terms_path = anita_terms_path or REPOSITORY_ROOT / "config" / "anita-work-terms.json"
        terms = _anita_work_records(terms_path)
        if anita_work_ids is not None and not anita_work_ids.issubset(terms):
            raise IntakeError("Selected Anita work IDs are absent from the work-terms registry")
        anita_terms_hash = sha256_file(terms_path)
        for item in entries:
            path_parts = PurePosixPath(item["relative_path"])
            folder = path_parts.parts[0].casefold() if path_parts.parts else ""
            work = terms.get(folder)
            if not work:
                item.update(candidate_type="excluded_or_unmapped", rights_status="review_required")
                continue
            scene = next(
                (
                    part
                    for part in path_parts.parts[1:]
                    if re.fullmatch(r"scene_\d+", part, flags=re.IGNORECASE)
                ),
                None,
            )
            artist_key = (
                f"anita-artist-{hashlib.sha256(work['artist'].encode('utf-8')).hexdigest()[:16]}"
            )
            item.update(
                work_id=work["folder_id"],
                work_group_id=f"anita:work:{work['folder_id']}",
                work_title=work["work_title"],
                artist=work["artist"],
                artist_group_id=artist_key,
                scene_id=scene,
                scene_group_id=f"anita:{work['folder_id']}:{scene or 'scene-unmapped'}",
                # Split at the artist level so different works by the same artist
                # cannot leak through train/validation/test boundaries.
                group_id=f"anita:artist:{artist_key}",
                license_id=work["license"],
                attribution_text=work.get("attribution_text"),
                terms_snapshot_sha256=work.get("terms_snapshot_sha256"),
                work_review_status=work.get("reviewer_decision", "review_required"),
                work_terms_reviewer_id=work.get("reviewer_id"),
                work_terms_reviewed_at=work.get("reviewed_at"),
                work_terms_reviewed_sha256=work.get("reviewed_sha256"),
                non_commercial=work.get("non_commercial", "NC" in work["license"]),
                share_alike=work.get("share_alike", "SA" in work["license"]),
                use_permissions=work.get("use_permissions", item["use_permissions"]),
                rights_status=work["rights_status"],
                candidate_type=next(
                    (kind for kind in ("sketch", "color", "composition") if kind in path_parts.parts),
                    "other",
                ),
            )
    elif source_id == "emoreccom-panels":
        partitions = _emoreccom_partitions(root)
        for item in entries:
            name = PurePosixPath(item["relative_path"]).name
            partition = partitions.get(name.casefold())
            if partition:
                split = partition["publication_partition"]
                prefix = partition["provisional_book_prefix"]
                annotations = partition["annotations"]
                book = emoreccom_book_mappings.get(prefix)
                bounds = []
                if item.get("width") is not None and item.get("height") is not None:
                    for annotation in annotations:
                        annotation_bounds: dict[str, bool] = {}
                        for box_name, box in annotation["boxes"].items():
                            annotation_bounds[box_name] = (
                                box["xmin"] >= 0
                                and box["ymin"] >= 0
                                and box["xmax"] <= item["width"]
                                and box["ymax"] <= item["height"]
                            )
                        bounds.append(
                            {
                                "annotation_id": annotation["annotation_id"],
                                "boxes_within_panel": annotation_bounds,
                            }
                        )
                item.update(
                    publication_partition=split,
                    provisional_book_prefix=prefix,
                    book_identity_verified=book is not None,
                    work_id=f"comics:{book['comic_no']}" if book else None,
                    comic_no=book["comic_no"] if book else None,
                    book_title=book.get("book_title") if book else None,
                    book_mapping_sha256=book.get("_mapping_sha256") if book else None,
                    official_book_index_sha256=book.get("_official_index_sha256") if book else None,
                    official_book_index_receipt_sha256=book.get("_official_index_receipt_sha256")
                    if book
                    else None,
                    book_mapping_reviewer_id=book.get("_reviewer_id") if book else None,
                    book_mapping_reviewed_at=book.get("_reviewed_at") if book else None,
                    annotation_source_file=partition["annotation_source_file"],
                    annotation_source_sha256=partition["annotation_source_sha256"],
                    character_annotations=annotations,
                    annotation_box_bounds=bounds,
                    group_id=f"emoreccom:book:{book['comic_no']}" if book else None,
                    candidate_type="panel_with_official_character_boxes",
                    rights_status="review_required",
                )
            else:
                item.update(
                    publication_partition="unreferenced",
                    candidate_type="unreferenced_panel",
                    rights_status="review_required",
                )
    tree_digest = hashlib.sha256()
    for item in entries:
        tree_digest.update(item["relative_path"].encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(item["byte_length"]).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(item["sha256"].encode("ascii"))
        tree_digest.update(b"\n")
    return {
        "schema_version": 1,
        "source_id": source_id,
        "source_archive_sha256": None,
        "source_registry_sha256": sha256_file(registry_path),
        "source_registry_row_sha256": _sha256_bytes(
            json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "anita_work_terms_sha256": anita_terms_hash,
        "anita_work_ids": sorted(anita_work_ids) if anita_work_ids is not None else None,
        "emoreccom_book_map_sha256": emoreccom_book_map_sha256,
        "emoreccom_book_map_path": (
            (
                emoreccom_book_map_path
                if emoreccom_book_map_path.is_absolute()
                else root / emoreccom_book_map_path
            )
            .resolve()
            .relative_to(root.resolve())
            .as_posix()
            if emoreccom_book_map_path is not None
            else None
        ),
        "relative_root": base.resolve().relative_to(root.resolve()).as_posix(),
        "generated_at": utc_now(),
        "tree_sha256": tree_digest.hexdigest(),
        "file_count": len(entries),
        "entries": entries,
        "status": "integrity_manifest_only_rights_and_review_pending",
    }


def extract_verified_archive(
    root: Path, source_id: str, *, registry_path: Path = REGISTRY_PATH
) -> dict[str, Any]:
    source = source_by_id(source_id, registry_path)
    archives, _partials = _archive_candidates(root, source)
    if len(archives) != 1:
        raise IntakeError("Exactly one official source archive must be present")
    archive_path = archives[0]
    receipt_path = _receipt_path(root, source_id)
    if not receipt_path.is_file():
        raise IntakeError("An official acquisition receipt is required before extraction")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError("Source receipt is corrupt") from error
    verify_archive_receipt(root, source, archive_path, receipt)
    digest = sha256_file(archive_path)
    inspected = inspect_zip(archive_path)
    destination = root / "extracted" / source_id / digest
    if destination.exists():
        manifest_path = destination / "tree-manifest.json"
        if not manifest_path.is_file():
            raise IntakeError("Existing hash-addressed extraction lacks its completion manifest")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{digest[:12]}-", dir=destination.parent))
    try:
        content_root = staged / "content"
        content_root.mkdir()
        with zipfile.ZipFile(archive_path) as zip_archive:
            for member in zip_archive.infolist():
                if member.is_dir():
                    continue
                safe_name = _safe_member_name(member.filename)
                out = content_root.joinpath(*PurePosixPath(safe_name).parts)
                resolved = out.resolve(strict=False)
                if content_root.resolve() not in resolved.parents:
                    raise IntakeError("Archive member escapes the staged extraction")
                out.parent.mkdir(parents=True, exist_ok=True)
                with (
                    zip_archive.open(member, "r") as source_stream,
                    out.open("xb") as target_stream,
                ):
                    shutil.copyfileobj(source_stream, target_stream, length=COPY_CHUNK_BYTES)
        manifest = {
            "schema_version": 1,
            "source_id": source_id,
            "archive_path": archive_path.relative_to(root).as_posix(),
            "archive_sha256": digest,
            "extracted_at": utc_now(),
            "limits": {
                "maximum_expanded_bytes": MAX_EXTRACTED_BYTES,
                "maximum_archive_entries": MAX_ARCHIVE_ENTRIES,
            },
            "archive_validation": inspected,
            "status": "extracted_pending_rights_and_review",
        }
        write_json_atomic(staged / "extraction-receipt.json", manifest)
        os.rename(staged, destination)
        return {**manifest, "relative_extraction_path": destination.relative_to(root).as_posix()}
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def _hash_tree_files(base: Path) -> dict[str, tuple[int, str]]:
    if base.is_symlink() or not base.is_dir():
        raise IntakeError("Extraction tree is missing or unsafe")
    result: dict[str, tuple[int, str]] = {}
    for current, dirnames, filenames in os.walk(base, followlinks=False):
        current_path = Path(current)
        dirnames[:] = sorted(dirnames)
        if any((current_path / dirname).is_symlink() for dirname in dirnames):
            raise IntakeError("Extraction tree contains a symbolic-link directory")
        for name in sorted(filenames):
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise IntakeError("Extraction tree contains a non-regular file")
            relative = path.relative_to(base).as_posix()
            normalized = unicodedata.normalize("NFC", relative)
            if normalized in result:
                raise IntakeError("Extraction tree has colliding normalized paths")
            result[normalized] = (path.stat().st_size, sha256_file(path))
    return result


def reconcile_archived_source(
    root: Path, source_id: str, *, registry_path: Path = REGISTRY_PATH
) -> dict[str, Any]:
    """Compare archive members, the fresh hash-addressed extraction, and the configured source tree."""
    source = source_by_id(source_id, registry_path)
    if not source.get("archive_path"):
        raise IntakeError("Only official ZIP sources have archive-extraction reconciliation")
    receipt_path = _receipt_path(root, source_id)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError("Official archive acquisition receipt is missing or invalid") from error
    archive_path = ensure_inside(root, str(receipt.get("archive_path", "")))
    verify_archive_receipt(root, source, archive_path, receipt)
    archive_hash = str(receipt["sha256"])
    extraction_receipt_path = ensure_inside(
        root, f"extracted/{source_id}/{archive_hash}/extraction-receipt.json"
    )
    content_root = ensure_inside(root, f"extracted/{source_id}/{archive_hash}/content")
    try:
        extraction_receipt = json.loads(extraction_receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError("Hash-addressed clean extraction receipt is missing or invalid") from error
    if (
        extraction_receipt.get("schema_version") != 1
        or extraction_receipt.get("source_id") != source_id
        or extraction_receipt.get("archive_sha256") != archive_hash
        or extraction_receipt.get("archive_path") != source.get("archive_path")
        or extraction_receipt.get("status") != "extracted_pending_rights_and_review"
    ):
        raise IntakeError("Clean extraction receipt does not match the verified official archive")
    if content_root.is_symlink() or not content_root.is_dir():
        raise IntakeError("Hash-addressed clean extraction content is missing or unsafe")

    expected: dict[str, tuple[int, str]] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                relative = _safe_member_name(member.filename)
                digest = hashlib.sha256()
                byte_count = 0
                with archive.open(member, "r") as stream:
                    for block in iter(lambda: stream.read(COPY_CHUNK_BYTES), b""):
                        byte_count += len(block)
                        digest.update(block)
                if byte_count != member.file_size:
                    raise IntakeError(f"Archive member length changed during reconciliation: {relative}")
                expected[relative] = (byte_count, digest.hexdigest())
    except IntakeError:
        raise
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile, zlib.error) as error:
        raise IntakeError("Could not read the verified archive during extraction reconciliation") from error
    clean_actual = _hash_tree_files(content_root)
    clean_differences = _tree_differences(expected, clean_actual)
    if expected:
        first_parts = {PurePosixPath(path).parts[0] for path in expected}
        has_single_archive_root = len(first_parts) == 1 and all(
            len(PurePosixPath(path).parts) > 1 for path in expected
        )
    else:
        has_single_archive_root = False
    archive_root = next(iter({PurePosixPath(path).parts[0] for path in expected}), "")
    legacy_expected = {
        "/".join(PurePosixPath(path).parts[1:]) if has_single_archive_root else path: item
        for path, item in expected.items()
    }
    legacy_root = ensure_inside(root, str(source["extracted_path"]))
    legacy_status = "missing"
    legacy_differences: list[str] = ["configured_extraction_tree_missing"]
    legacy_file_count = 0
    legacy_actual: dict[str, tuple[int, str]] = {}
    if legacy_root.is_dir() and not legacy_root.is_symlink():
        legacy_actual = _hash_tree_files(legacy_root)
        legacy_file_count = len(legacy_actual)
        legacy_differences = _tree_differences(legacy_expected, legacy_actual)
        legacy_status = "reconciled" if not legacy_differences else "mismatch"
    clean_status = "reconciled" if not clean_differences else "mismatch"
    entries = [
        {"path": path, "bytes": size, "sha256": digest}
        for path, (size, digest) in sorted(expected.items())
    ]
    tree_digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    clean_entries = [
        {"path": path, "bytes": size, "sha256": digest}
        for path, (size, digest) in sorted(clean_actual.items())
    ]
    legacy_entries = [
        {"path": path, "bytes": size, "sha256": digest}
        for path, (size, digest) in sorted(legacy_actual.items())
    ]
    clean_tree_sha256 = hashlib.sha256(
        json.dumps(clean_entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    configured_tree_sha256 = hashlib.sha256(
        json.dumps(legacy_entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "schema_version": 1,
        "artifact_type": "drawable-source-extraction-reconciliation",
        "source_id": source_id,
        "archive_sha256": archive_hash,
        "archive_path": archive_path.relative_to(root.resolve()).as_posix(),
        "archive_receipt_path": receipt_path.relative_to(root.resolve()).as_posix(),
        "clean_extraction_receipt_path": extraction_receipt_path.relative_to(root.resolve()).as_posix(),
        "clean_extraction_path": content_root.relative_to(root.resolve()).as_posix(),
        "configured_tree_path": legacy_root.relative_to(root.resolve()).as_posix(),
        "configured_tree_archive_prefix_removed": archive_root if has_single_archive_root else None,
        "archive_files": len(expected),
        "clean_extraction_files": len(clean_actual),
        "configured_tree_files": legacy_file_count,
        "expected_tree_sha256": tree_digest,
        "clean_extraction_tree_sha256": clean_tree_sha256,
        "configured_tree_sha256": configured_tree_sha256,
        "clean_extraction_status": clean_status,
        "configured_tree_status": legacy_status,
        "clean_extraction_differences": clean_differences[:100],
        "configured_tree_differences": legacy_differences[:100],
        "clean_extraction_entries": clean_entries,
        "configured_tree_entries": legacy_entries,
        "status": "reconciled" if clean_status == "reconciled" and legacy_status == "reconciled" else "blocked",
        "training_started": False,
        "data_uploaded": False,
    }
    report_path = ensure_inside(
        root,
        f"reports/source-reconciliation/{source_id}-{archive_hash[:12]}-{clean_tree_sha256[:8]}-{configured_tree_sha256[:8]}.json",
    )
    if report_path.is_file() and not report_path.is_symlink():
        try:
            existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise IntakeError("Existing extraction-reconciliation report is corrupt") from error
        if existing_report != result:
            raise IntakeError("Existing extraction-reconciliation report has different evidence")
    else:
        try:
            write_json_create(report_path, result)
        except IntakeError:
            # Another local review process may have created the same immutable report.
            try:
                existing_report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise IntakeError("Concurrent extraction-reconciliation report is corrupt") from error
            if existing_report != result:
                raise IntakeError("Concurrent extraction-reconciliation report has different evidence")
    return {**result, "report_path": report_path.relative_to(root.resolve()).as_posix()}


def _tree_differences(
    expected: dict[str, tuple[int, str]], actual: dict[str, tuple[int, str]]
) -> list[str]:
    differences = [f"missing:{path}" for path in sorted(expected.keys() - actual.keys())]
    differences.extend(f"unexpected:{path}" for path in sorted(actual.keys() - expected.keys()))
    differences.extend(
        f"bytes_or_hash_mismatch:{path}"
        for path in sorted(expected.keys() & actual.keys())
        if expected[path] != actual[path]
    )
    return differences


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_data_package(
    root: Path,
    manifest_path: Path,
    *,
    purpose: str,
    out_path: Path,
    registry_path: Path = REGISTRY_PATH,
    anita_terms_path: Path | None = None,
) -> dict[str, Any]:
    """Create a deterministic, local-only package after explicit source/cloud review."""
    if purpose != "inference-readiness":
        raise IntakeError("This phase supports inference-readiness packages only")
    manifest_path = manifest_path.resolve()
    if root.resolve() not in manifest_path.parents or not manifest_path.is_file():
        raise IntakeError("Manifest must be an existing file inside the data root")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError("Input manifest is invalid") from error
    registry = load_registry(registry_path)
    sources = {item["id"]: item for item in registry["sources"]}
    source_id = manifest.get("source_id")
    if not isinstance(source_id, str):
        raise IntakeError("Manifest source ID is missing")
    source = sources.get(source_id)
    if not source:
        raise IntakeError("Manifest source is absent from the reviewed registry")
    anita_work_terms: dict[str, dict[str, Any]] = {}
    if source_id == "anita-dataset":
        terms_config_path = anita_terms_path or REPOSITORY_ROOT / "config" / "anita-work-terms.json"
        if manifest.get("anita_work_terms_sha256") != sha256_file(terms_config_path):
            raise IntakeError(
                "Manifest Anita work-terms registry hash is stale; rebuild it after the per-work review"
            )
        anita_work_terms = _anita_work_records(terms_config_path)
    required = source["use_permissions"]
    if (
        not source.get("rights_status") == "approved"
        or not required.get("export")
        or not required.get("cloud_storage")
        or not required.get("cloud_processing")
    ):
        raise IntakeError(
            "Source rights plus separate export, cloud-storage, and cloud-processing permissions must be approved before packaging"
        )
    registry_hash = sha256_file(registry_path)
    if manifest.get("source_registry_sha256") != registry_hash:
        raise IntakeError(
            "Manifest source-registry hash is stale; rebuild it after the rights review"
        )
    receipt_path = _receipt_path(root, source_id)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError(
            "Verified official archive receipt is required before packaging"
        ) from error
    archive_path = ensure_inside(root, str(receipt.get("archive_path", "")))
    verify_archive_receipt(root, source, archive_path, receipt)
    if receipt.get("sha256") != manifest.get("source_archive_sha256"):
        raise IntakeError("Manifest does not bind to the currently verified original archive")
    if not isinstance(source.get("terms_snapshot_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", source["terms_snapshot_sha256"]
    ):
        raise IntakeError(
            "Current authoritative terms page must be snapshotted and reviewed before packaging"
        )
    terms_digest = source["terms_snapshot_sha256"]
    terms_receipt_path = terms_snapshot_receipt_path(root, source_id, terms_digest)
    try:
        terms_receipt = json.loads(terms_receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError(
            "The registry's reviewed terms snapshot receipt is missing or corrupt"
        ) from error
    terms_snapshot_path = ensure_inside(root, terms_receipt.get("snapshot_path", ""))
    if (
        terms_receipt.get("source_id") != source_id
        or terms_receipt.get("sha256") != terms_digest
        or not terms_snapshot_path.is_file()
        or terms_snapshot_path.is_symlink()
        or sha256_file(terms_snapshot_path) != terms_digest
    ):
        raise IntakeError(
            "The authoritative terms snapshot bytes do not match the reviewed source-registry hash"
        )
    if manifest.get("status") not in {"validated", "manifest_ready_for_rights_and_human_review"}:
        raise IntakeError("Manifest status is not eligible for a transfer package")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise IntakeError("Transfer manifest contains no assets")
    out_path = out_path.resolve()
    if root.resolve() not in out_path.parents:
        raise IntakeError("Package output must remain inside the ignored data root")
    if out_path.exists():
        raise IntakeError("Package destination already exists; refusing to overwrite")
    # ZIP timestamps, ordering, paths, and JSON metadata are fixed for reproducible bundles.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staged = out_path.with_name(f".{out_path.name}.{uuid.uuid4().hex}.tmp")
    package_items: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(
            staged, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as bundle:
            for item in sorted(entries, key=lambda row: row["relative_path"]):
                relative = item.get("relative_path")
                if not isinstance(relative, str):
                    raise IntakeError("Manifest asset path is malformed")
                safe_relative = _safe_member_name(relative)
                if item.get("review_status") != "approved":
                    raise IntakeError(
                        f"Asset has not passed its explicit human review: {safe_relative}"
                    )
                if item.get("learning_split") not in {"train", "validation", "pilot_development"}:
                    raise IntakeError(
                        f"Unassigned or locked-test asset cannot enter the inference readiness bundle: {safe_relative}"
                    )
                item_permissions = item.get("use_permissions")
                if (
                    not isinstance(item_permissions, dict)
                    or not item_permissions.get("local_display")
                    or not item_permissions.get("export")
                    or not item_permissions.get("cloud_storage")
                    or not item_permissions.get("cloud_processing")
                ):
                    raise IntakeError(
                        f"Asset lacks local-display, export, or separate cloud storage/processing permission: {safe_relative}"
                    )
                if item.get("rights_status") != "approved":
                    raise IntakeError(f"Asset rights review is incomplete: {safe_relative}")
                if item.get("gallery_member") is not True:
                    raise IntakeError(
                        f"Only individually approved gallery assets can enter this inference-readiness bundle: {safe_relative}"
                    )
                labels = item.get("labels")
                if (
                    not isinstance(labels, dict)
                    or labels.get("sfw") != "approved"
                    or labels.get("quality") not in {2, 3}
                    or labels.get("style")
                    not in {
                        "manga_anime",
                        "western_ink",
                        "realistic_academic",
                        "cartoon",
                        "gesture_sketch",
                    }
                ):
                    raise IntakeError(
                        f"Asset lacks approved SFW, quality 2–3, or style labels: {safe_relative}"
                    )
                if labels.get("primary_scope") not in {
                    "eye",
                    "eyebrow",
                    "mouth",
                    "face_head",
                    "hair",
                    "hand",
                    "foot",
                    "upper_body_clothing",
                    "full_body",
                    "multi_character",
                }:
                    raise IntakeError(
                        f"Asset lacks an approved primary-scope label: {safe_relative}"
                    )
                if source_id == "anita-dataset":
                    work = anita_work_terms.get(str(item.get("work_id", "")).casefold())
                    if (
                        not work
                        or item.get("candidate_type") != "sketch"
                        or item.get("work_review_status") != "approved"
                        or work.get("rights_status") != "approved"
                        or work.get("terms_snapshot_sha256") != terms_digest
                        or item.get("terms_snapshot_sha256") != terms_digest
                        or item.get("attribution_text") != work.get("attribution_text")
                        or item.get("use_permissions") != work.get("use_permissions")
                        or item.get("license_id") != work.get("license")
                    ):
                        raise IntakeError(
                            f"Anita asset is not an approved native sketch with current work-level terms and attribution: {safe_relative}"
                        )
                if source_id == "emoreccom-panels" and (
                    item.get("publication_partition") != "source_train"
                    or item.get("book_identity_verified") is not True
                ):
                    raise IntakeError(
                        f"EmoRecCom asset is not verified development data with authoritative book identity: {safe_relative}"
                    )
                original_root = ensure_inside(root, manifest["relative_root"])
                asset_path = ensure_inside(original_root, safe_relative)
                if asset_path.is_symlink() or not asset_path.is_file():
                    raise IntakeError(f"Manifest asset is missing or unsafe: {safe_relative}")
                actual_digest = sha256_file(asset_path)
                actual_bytes = asset_path.stat().st_size
                if actual_digest != item.get("sha256") or actual_bytes != item.get("byte_length"):
                    raise IntakeError(f"Asset changed since manifest creation: {safe_relative}")
                if item.get("decoder_status") != "decoded":
                    raise IntakeError(
                        f"Non-decoded files cannot enter an inference readiness bundle: {safe_relative}"
                    )
                archive_name = f"assets/{source_id}/{actual_digest}{asset_path.suffix.casefold()}"
                info = zipfile.ZipInfo(archive_name, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100444 << 16
                with asset_path.open("rb") as stream, bundle.open(info, "w") as target:
                    shutil.copyfileobj(stream, target, length=COPY_CHUNK_BYTES)
                package_items.append(
                    {
                        "asset_id": item.get("asset_id"),
                        "archive_path": archive_name,
                        "byte_length": actual_bytes,
                        "sha256": actual_digest,
                    }
                )
            metadata = {
                "schema_version": 1,
                "purpose": purpose,
                "source_id": source_id,
                "manifest_sha256": sha256_file(manifest_path),
                "registry_sha256": registry_hash,
                "created_at": "reproducible-build-timestamp",
                "assets": package_items,
                "training_authorized": False,
                "test_partition_included": False,
            }
            payload = (
                json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                + "\n"
            ).encode("utf-8")
            info = zipfile.ZipInfo("bundle-manifest.json", date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100444 << 16
            bundle.writestr(info, payload)
        os.link(staged, out_path)
        staged.unlink()
    finally:
        staged.unlink(missing_ok=True)
    digest = sha256_file(out_path)
    return {
        "status": "created_pending_owner_transfer",
        "relative_path": out_path.relative_to(root.resolve()).as_posix(),
        "byte_length": out_path.stat().st_size,
        "sha256": digest,
        "assets": len(package_items),
        "training_authorized": False,
    }
