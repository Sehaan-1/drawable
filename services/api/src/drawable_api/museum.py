from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .intake import (
    IntakeError,
    ensure_inside,
    load_registry,
    sha256_file,
    terms_snapshot_receipt_path,
    write_json_atomic,
    write_json_create,
)
from .settings import REPOSITORY_ROOT

MAX_METADATA_RECORDS = 200
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_EDGE = 8192
DEFAULT_QUERIES = (
    "figure drawing",
    "figure study",
    "anatomical study",
    "gesture drawing",
    "costume study",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _url_host(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
    ):
        raise IntakeError("Museum API returned a malformed or non-HTTPS URL")
    return parsed.hostname.casefold().rstrip(".")


def _read_response(
    url: str, *, allowed_hosts: set[str], max_bytes: int, user_agent: str, aic: bool = False
) -> tuple[bytes, str, str]:
    if _url_host(url) not in allowed_hosts:
        raise IntakeError("Museum URL is outside the configured official-host allowlist")
    headers = {"User-Agent": user_agent, "Accept": "application/json,image/*;q=0.8,*/*;q=0.1"}
    if aic:
        headers["AIC-User-Agent"] = "Drawable academic prototype"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            final_url = response.geturl()
            if _url_host(final_url) not in allowed_hosts:
                raise IntakeError(
                    "Museum request redirected outside the configured official-host allowlist"
                )
            payload = response.read(max_bytes + 1)
            if len(payload) > max_bytes:
                raise IntakeError("Museum response exceeds its configured size limit")
            content_type = (
                (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().casefold()
            )
            return payload, content_type, final_url
    except IntakeError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        raise IntakeError(f"Museum request failed: {type(error).__name__}") from error


def _request_json(
    url: str, *, allowed_hosts: set[str], source_id: str
) -> tuple[dict[str, Any], str]:
    payload, content_type, final_url = _read_response(
        url,
        allowed_hosts=allowed_hosts,
        max_bytes=10 * 1024 * 1024,
        user_agent="Drawable academic prototype (local museum candidate intake)",
        aic=source_id == "aic-open-access",
    )
    if content_type not in {"application/json", "application/ld+json"}:
        raise IntakeError("Museum metadata endpoint returned a non-JSON response")
    try:
        body = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IntakeError("Museum metadata endpoint returned malformed JSON") from error
    if not isinstance(body, dict):
        raise IntakeError("Museum metadata response must be an object")
    return body, final_url


def _met_records(
    queries: list[str], limit: int, *, sleep: Callable[[float], None]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hosts = {"collectionapi.metmuseum.org"}
    object_ids: list[int] = []
    search_log: list[dict[str, Any]] = []
    seen: set[int] = set()
    for query in queries:
        offset = 0
        while len(object_ids) < limit:
            params = urllib.parse.urlencode(
                {
                    "q": query,
                    "hasImages": "true",
                    "offset": offset,
                    "limit": min(100, limit - len(object_ids)),
                }
            )
            url = f"https://collectionapi.metmuseum.org/public/collection/v1.1/search?{params}"
            body, final_url = _request_json(
                url, allowed_hosts=hosts, source_id="the-met-open-access"
            )
            ids = body.get("objectIDs") or []
            if not isinstance(ids, list):
                raise IntakeError("Met paginated search returned malformed object IDs")
            for value in ids:
                if type(value) is int and value > 0 and value not in seen:
                    seen.add(value)
                    object_ids.append(value)
                    if len(object_ids) == limit:
                        break
            search_log.append(
                {
                    "query": query,
                    "endpoint": final_url,
                    "offset": offset,
                    "page_count": len(ids),
                    "total": body.get("total"),
                }
            )
            offset += len(ids)
            if not ids or len(ids) < min(100, limit) or offset >= 10_000:
                break
            sleep(0.1)
        if len(object_ids) >= limit:
            break

    records: list[dict[str, Any]] = []
    for index, object_id in enumerate(object_ids):
        if index:
            sleep(0.1)
        url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
        record, final_url = _request_json(url, allowed_hosts=hosts, source_id="the-met-open-access")
        if record.get("objectID") != object_id:
            raise IntakeError("Met object detail response does not match the requested object ID")
        record["source_record_url"] = final_url
        records.append(record)
    return records, search_log


def _aic_records(
    queries: list[str], limit: int, *, sleep: Callable[[float], None]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    hosts = {"api.artic.edu", "www.artic.edu"}
    fields = "id,title,artist_display,date_display,medium,dimensions,credit_line,is_public_domain,image_id,api_link,thumbnail"
    seen: set[int] = set()
    records: list[dict[str, Any]] = []
    search_log: list[dict[str, Any]] = []
    iiif_base = "https://www.artic.edu/iiif/2"
    for query in queries:
        params_value = {"q": query, "query": {"term": {"is_public_domain": True}}}
        params = urllib.parse.urlencode(
            {
                "params": json.dumps(params_value, separators=(",", ":")),
                "limit": min(100, limit),
                "fields": fields,
            }
        )
        url = f"https://api.artic.edu/api/v1/artworks/search?{params}"
        body, final_url = _request_json(url, allowed_hosts=hosts, source_id="aic-open-access")
        config = body.get("config", {})
        base = config.get("iiif_url") if isinstance(config, dict) else None
        if isinstance(base, str):
            if _url_host(base) not in hosts or not base.endswith("/iiif/2"):
                raise IntakeError("AIC response supplied an unexpected IIIF base URL")
            iiif_base = base
        data = body.get("data") or []
        if not isinstance(data, list):
            raise IntakeError("AIC search response contains malformed artwork records")
        accepted = 0
        for record in data:
            if (
                not isinstance(record, dict)
                or type(record.get("id")) is not int
                or record["id"] in seen
            ):
                continue
            seen.add(record["id"])
            record["source_record_url"] = (
                record.get("api_link") or f"https://api.artic.edu/api/v1/artworks/{record['id']}"
            )
            record["search_endpoint"] = final_url
            records.append(record)
            accepted += 1
            if len(records) >= limit:
                break
        search_log.append(
            {
                "query": query,
                "endpoint": final_url,
                "page_count": len(data),
                "total": (body.get("pagination") or {}).get("total"),
                "unique_records_added": accepted,
            }
        )
        if len(records) >= limit:
            break
        sleep(0.1)
    return records, search_log, iiif_base


def _save_image(
    root: Path,
    source_id: str,
    object_id: str,
    url: str,
    *,
    source: str,
    sleep: Callable[[float], None],
    downloaded_count: int,
) -> dict[str, Any]:
    allowed = {"images.metmuseum.org"} if source == "the-met-open-access" else {"www.artic.edu"}
    if downloaded_count:
        sleep(1.0)
    payload, content_type, final_url = _read_response(
        url,
        allowed_hosts=allowed,
        max_bytes=MAX_IMAGE_BYTES,
        user_agent="Drawable academic prototype",
        aic=source == "aic-open-access",
    )
    if not content_type.startswith("image/"):
        raise IntakeError("Museum image endpoint returned a non-image response")
    digest = hashlib.sha256(payload).hexdigest()
    try:
        from io import BytesIO

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
                if (
                    width <= 0
                    or height <= 0
                    or max(width, height) > MAX_IMAGE_EDGE
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise IntakeError(
                        "Museum image exceeds the configured dimension or pixel limit"
                    )
                image.verify()
                media_type = Image.MIME.get(image.format or "")
    except IntakeError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise IntakeError("Museum image could not be safely decoded") from error
    if media_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise IntakeError("Museum image is not a supported JPEG, PNG, or WebP asset")
    relative_root = f"museum/{source_id}/objects"
    relative_path = (
        f"{object_id}-{digest[:20]}.jpg"
        if media_type == "image/jpeg"
        else f"{object_id}-{digest[:20]}.{media_type.removeprefix('image/')}"
    )
    destination = ensure_inside(root, f"{relative_root}/{relative_path}")
    cursor = root
    for component in destination.relative_to(root).parts[:-1]:
        cursor = cursor / component
        if cursor.is_symlink():
            raise IntakeError("Symbolic links are not allowed in museum image storage")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.parent.is_symlink():
        raise IntakeError("Symbolic links are not allowed in museum image storage")
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or sha256_file(destination) != digest
        ):
            raise IntakeError("Content-addressed museum image path contains unexpected bytes")
    else:
        temporary = destination.with_name(f".{destination.name}.{os.urandom(8).hex()}.part")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if sha256_file(destination) != digest:
                    raise IntakeError(
                        "Concurrent museum image acquisition produced conflicting bytes"
                    )
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "relative_root": relative_root,
        "relative_path": relative_path,
        "sha256": digest,
        "byte_length": len(payload),
        "media_type": media_type,
        "width": width,
        "height": height,
        "decoder_status": "decoded",
        "image_url": final_url,
    }


def fetch_museum_candidates(
    data_root: Path,
    source_id: str,
    *,
    queries: list[str] | None = None,
    metadata_limit: int = MAX_METADATA_RECORDS,
    image_limit: int = 50,
    registry_path: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Fetch a small, sequential, quarantine-only set of official museum records/images.

    This function is only called from the explicit CLI execution path. It never
    approves rights, labels, review status, gallery eligibility, or use permissions.
    """
    if source_id not in {"the-met-open-access", "aic-open-access"}:
        raise IntakeError(
            "Museum acquisition only supports the Met and Art Institute of Chicago registries"
        )
    if not 1 <= metadata_limit <= MAX_METADATA_RECORDS or not 1 <= image_limit <= 100:
        raise IntakeError(
            "Museum intake limits must be between 1 and 200 metadata records and 1 and 100 images"
        )
    root = data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    registry_file = registry_path or REPOSITORY_ROOT / "config" / "sources.json"
    registry = load_registry(registry_file)
    source_config = next((item for item in registry["sources"] if item["id"] == source_id), None)
    if source_config is None:
        raise IntakeError("Museum source is not present in the source registry")
    query_list = [item.strip() for item in (queries or list(DEFAULT_QUERIES)) if item.strip()]
    if not query_list or len(query_list) > 20 or any(len(item) > 120 for item in query_list):
        raise IntakeError("Provide 1 to 20 non-empty search phrases of at most 120 characters")

    if source_id == "the-met-open-access":
        records, search_log = _met_records(query_list, metadata_limit, sleep=sleep)
        iiif_base = None
    else:
        records, search_log, iiif_base = _aic_records(query_list, metadata_limit, sleep=sleep)

    entries: list[dict[str, Any]] = []
    full_records: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    seen_hashes: set[str] = set()
    image_count = 0
    download_attempts = 0
    for record in records:
        if source_id == "the-met-open-access":
            object_id = record.get("objectID")
            is_public_domain = record.get("isPublicDomain") is True
            image_url = record.get("primaryImage")
            record_url = record.get("source_record_url")
            title = record.get("title")
            artist: str | None = "; ".join(
                str(item.get("name"))
                for item in record.get("constituents", [])
                if isinstance(item, dict) and item.get("role") == "Artist" and item.get("name")
            )
            medium = record.get("medium")
            date = record.get("objectDate")
            credit = record.get("creditLine")
            image_id = None
        else:
            object_id = record.get("id")
            is_public_domain = record.get("is_public_domain") is True
            image_id = record.get("image_id")
            image_url = (
                f"{iiif_base}/{urllib.parse.quote(str(image_id), safe='')}/full/843,/0/default.jpg"
                if image_id
                else None
            )
            record_url = record.get("source_record_url")
            title = record.get("title")
            artist_value = record.get("artist_display")
            artist = artist_value if isinstance(artist_value, str) else None
            medium = record.get("medium")
            date = record.get("date_display")
            credit = record.get("credit_line")
        if type(object_id) is not int or object_id <= 0:
            rejected["invalid_object_id"] = rejected.get("invalid_object_id", 0) + 1
            continue
        if not is_public_domain:
            rejected["not_marked_public_domain"] = rejected.get("not_marked_public_domain", 0) + 1
            continue
        if not isinstance(image_url, str) or not image_url or not isinstance(record_url, str):
            rejected["missing_primary_image_or_record_url"] = (
                rejected.get("missing_primary_image_or_record_url", 0) + 1
            )
            continue
        if download_attempts >= image_limit:
            rejected["image_limit_reached"] = rejected.get("image_limit_reached", 0) + 1
            continue
        download_attempts += 1
        image = _save_image(
            root,
            source_id,
            str(object_id),
            image_url,
            source=source_id,
            sleep=sleep,
            downloaded_count=image_count,
        )
        if image["sha256"] in seen_hashes:
            rejected["duplicate_image_bytes"] = rejected.get("duplicate_image_bytes", 0) + 1
            continue
        seen_hashes.add(image["sha256"])
        image_count += 1
        object_group = f"{source_id}:object:{object_id}"
        entry = {
            **image,
            "asset_id": f"asset-{source_id}-{object_id}",
            "object_id": str(object_id),
            "work_id": f"{source_id}:{object_id}",
            "group_id": object_group,
            "museum_object_group_id": object_group,
            "source_record_url": record_url,
            "is_public_domain": True,
            "terms_snapshot_sha256": source_config.get("terms_snapshot_sha256"),
            "primary_image": image_url if source_id == "the-met-open-access" else None,
            "image_id": image_id,
            "title": title,
            "artist": artist,
            "medium": medium,
            "date": date,
            "credit_line": credit,
            "candidate_type": "museum_artwork_candidate",
            "rights_status": source_config["rights_status"]
            if source_config["rights_status"] == "approved"
            else "review_required",
            "use_permissions": dict(source_config["use_permissions"])
            if source_config["rights_status"] == "approved"
            else {key: False for key in source_config["use_permissions"]},
            "review_status": "pending",
            "learning_split": "unassigned",
            "gallery_member": False,
            "gold_member": False,
            "labels": {
                "style": None,
                "primary_scope": None,
                "secondary_scopes": [],
                "person_count": None,
                "sfw": "pending",
                "quality": None,
            },
        }
        entries.append(entry)
        full_records.append(
            {"record": record, "entry_asset_id": entry["asset_id"], "image_sha256": image["sha256"]}
        )

    if not entries:
        raise IntakeError(
            "No valid public-domain museum images survived the bounded candidate intake"
        )
    tree_digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["relative_path"]):
        tree_digest.update(entry["relative_path"].encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(entry["byte_length"]).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(entry["sha256"].encode("ascii"))
        tree_digest.update(b"\n")
    now = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    metadata_snapshot = {
        "schema_version": 1,
        "source_id": source_id,
        "authority_url": next(
            item["authority_url"] for item in registry["sources"] if item["id"] == source_id
        ),
        "captured_at": now,
        "queries": query_list,
        "search_log": search_log,
        "records_inspected": len(records),
        "images_saved": len(entries),
        "rejected_counts": rejected,
        "records": full_records,
        "research_note": "Metadata and image bytes are acquisition candidates only; all assets remain quarantined pending item-level review and current source/work terms checks.",
    }
    snapshot_bytes = _canonical_json_bytes(metadata_snapshot)
    snapshot_hash = hashlib.sha256(snapshot_bytes).hexdigest()
    snapshot_path = root / "manifests" / "source-metadata" / source_id / f"{snapshot_hash}.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot_path.exists():
        write_json_create(snapshot_path, metadata_snapshot)
    manifest = {
        "schema_version": 1,
        "source_id": source_id,
        "status": "candidate_images_acquired_quarantined_pending_human_review",
        "created_at": now,
        "relative_root": f"museum/{source_id}/objects",
        "source_registry_sha256": sha256_file(registry_file),
        "source_registry_row_sha256": hashlib.sha256(
            _canonical_json_bytes(
                next(item for item in registry["sources"] if item["id"] == source_id)
            )
        ).hexdigest(),
        "source_metadata_snapshot_sha256": snapshot_hash,
        "metadata_snapshot_path": snapshot_path.relative_to(root).as_posix(),
        "queries": query_list,
        "file_count": len(entries),
        "tree_sha256": tree_digest.hexdigest(),
        "entries": entries,
        "gallery_activated": False,
        "training_authorized": False,
    }
    manifest_path = root / "manifests" / f"{source_id}-tree.json"
    write_json_atomic(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "source_id": source_id,
        "records_inspected": len(records),
        "images_saved": len(entries),
        "rejected_counts": rejected,
        "metadata_snapshot_sha256": snapshot_hash,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "gallery_activated": False,
        "training_authorized": False,
    }


def refresh_museum_manifest(
    data_root: Path,
    source_id: str,
    *,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Bind already acquired museum bytes to newly reviewed source terms offline."""
    if source_id not in {"the-met-open-access", "aic-open-access"}:
        raise IntakeError("Museum manifest refresh only supports the Met and AIC sources")
    root = data_root.resolve()
    registry_file = registry_path or REPOSITORY_ROOT / "config" / "sources.json"
    registry = load_registry(registry_file)
    source = next((item for item in registry["sources"] if item["id"] == source_id), None)
    if source is None:
        raise IntakeError("Museum source is not present in the source registry")
    terms_hash = source.get("terms_snapshot_sha256")
    review = source.get("terms_review", {})
    if (
        source.get("rights_status") != "approved"
        or review.get("status") != "approved"
        or review.get("reviewed_sha256") != terms_hash
        or not isinstance(review.get("reviewer_id"), str)
        or not review.get("reviewer_id")
        or source.get("use_permissions", {}).get("local_display") is not True
        or not isinstance(terms_hash, str)
    ):
        raise IntakeError(
            "Current source terms, reviewer, and local-display permission must be approved before manifest refresh"
        )
    receipt_path = terms_snapshot_receipt_path(root, source_id, terms_hash)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise IntakeError("Approved museum terms snapshot receipt is missing")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        snapshot_path = ensure_inside(root, str(receipt.get("snapshot_path", "")))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError("Approved museum terms snapshot receipt is malformed") from error
    if (
        receipt.get("source_id") != source_id
        or receipt.get("sha256") != terms_hash
        or snapshot_path.is_symlink()
        or not snapshot_path.is_file()
        or sha256_file(snapshot_path) != terms_hash
    ):
        raise IntakeError("Approved museum terms snapshot does not match the registry hash")

    manifest_path = ensure_inside(root, f"manifests/{source_id}-tree.json")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise IntakeError("Museum candidate manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError("Museum candidate manifest is malformed") from error
    if (
        manifest.get("source_id") != source_id
        or not isinstance(manifest.get("entries"), list)
        or not manifest["entries"]
    ):
        raise IntakeError("Museum candidate manifest has invalid identity or no entries")
    root_value = manifest.get("relative_root")
    if not isinstance(root_value, str):
        raise IntakeError("Museum candidate manifest has no relative image root")
    tree_root = ensure_inside(root, root_value)
    seen_ids: set[str] = set()
    digest = hashlib.sha256()
    for entry in sorted(manifest["entries"], key=lambda item: str(item.get("relative_path", ""))):
        if not isinstance(entry, dict) or entry.get("is_public_domain") is not True:
            raise IntakeError("Only explicitly public-domain museum candidates can be refreshed")
        asset_id = entry.get("asset_id")
        if not isinstance(asset_id, str) or asset_id in seen_ids:
            raise IntakeError("Museum manifest contains invalid or duplicate asset IDs")
        seen_ids.add(asset_id)
        path = ensure_inside(tree_root, str(entry.get("relative_path", "")))
        if path.is_symlink() or not path.is_file():
            raise IntakeError("A museum source image is missing or unsafe")
        if path.stat().st_size != entry.get("byte_length") or sha256_file(path) != entry.get(
            "sha256"
        ):
            raise IntakeError("A museum source image no longer matches the candidate manifest")
        entry["rights_status"] = "approved"
        entry["use_permissions"] = dict(source["use_permissions"])
        entry["terms_snapshot_sha256"] = terms_hash
        entry["review_status"] = "pending"
        entry["gallery_member"] = False
        entry["gold_member"] = False
        entry["learning_split"] = "unassigned"
        entry["labels"] = {
            "style": None,
            "primary_scope": None,
            "secondary_scopes": [],
            "person_count": None,
            "sfw": "pending",
            "quality": None,
        }
        digest.update(str(entry["relative_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["byte_length"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    manifest["source_registry_sha256"] = sha256_file(registry_file)
    manifest["source_registry_row_sha256"] = hashlib.sha256(
        _canonical_json_bytes(source)
    ).hexdigest()
    manifest["terms_snapshot_sha256"] = terms_hash
    manifest["tree_sha256"] = digest.hexdigest()
    manifest["status"] = "source_terms_approved_assets_still_pending_human_review"
    manifest["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    write_json_atomic(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "source_id": source_id,
        "assets_refreshed": len(seen_ids),
        "terms_snapshot_sha256": terms_hash,
        "source_registry_sha256": manifest["source_registry_sha256"],
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "gallery_activated": False,
        "training_authorized": False,
    }
