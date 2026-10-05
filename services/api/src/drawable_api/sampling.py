from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from drawable_research_contracts.readiness import REQUIREMENTS_REVISION

from .intake import (
    IntakeError,
    _anita_work_records,
    _safe_member_name,
    audit_sources,
    ensure_inside,
    load_registry,
    sha256_file,
    verify_archive_receipt,
    write_json_create,
)
from .settings import REPOSITORY_ROOT

RECIPE_PATH = REPOSITORY_ROOT / "config" / "recipes" / "pilot-v1.json"
YIELD_SOURCES = (
    "anita-dataset",
    "emoreccom-panels",
    "the-met-open-access",
    "aic-open-access",
)


def _private_rows_snapshot(root: Path, kind: str, key: str, rows: list[dict[str, Any]]) -> tuple[str, str]:
    payload = {"schema_version": 1, "rows": rows}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    identity = hashlib.sha256(canonical).hexdigest()
    relative = f"reports/private/{kind}/{key}-{identity}.json"
    path = ensure_inside(root, relative)
    if path.exists():
        if path.is_symlink() or not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != payload:
            raise IntakeError("Selected registry snapshot path exists with different or unsafe bytes")
    else:
        write_json_create(path, payload)
    return relative, sha256_file(path)


def _stable_key(seed: str, source_id: str, entry: dict[str, Any]) -> str:
    value = "\0".join(
        (seed, source_id, str(entry.get("group_id") or ""), str(entry.get("asset_id") or ""))
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _base_candidate(entry: dict[str, Any]) -> bool:
    width, height = entry.get("width"), entry.get("height")
    return (
        entry.get("decoder_status") == "decoded"
        and type(width) is int
        and type(height) is int
        and min(width, height) >= 256
        and width * height <= 40_000_000
        and isinstance(entry.get("asset_id"), str)
        and bool(entry.get("asset_id"))
        and isinstance(entry.get("relative_path"), str)
        and type(entry.get("byte_length")) is int
        and entry["byte_length"] > 0
        and isinstance(entry.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None
        and isinstance(entry.get("group_id"), str)
        and bool(entry.get("group_id"))
        and isinstance(entry.get("use_permissions"), dict)
        and entry["use_permissions"].get("local_display") is True
        and entry.get("rights_status") == "approved"
        and entry.get("review_status", "pending") == "pending"
    )


def _eligible_entry(
    source_id: str, entry: dict[str, Any], work_records: dict[str, dict[str, Any]]
) -> bool:
    if not _base_candidate(entry):
        return False
    if source_id == "anita-dataset":
        work = work_records.get(str(entry.get("work_id", "")).casefold())
        return bool(
            work
            and entry.get("candidate_type") == "sketch"
            and entry.get("work_review_status") == "approved"
            and work.get("rights_status") == "approved"
            and work.get("reviewer_decision") == "approved"
            and work.get("terms_snapshot_sha256") == entry.get("terms_snapshot_sha256")
            and work.get("reviewed_sha256") == entry.get("terms_snapshot_sha256")
            and entry.get("attribution_text") == work.get("attribution_text")
            and entry.get("license_id") == work.get("license")
            and entry.get("use_permissions") == work.get("use_permissions")
        )
    if source_id == "emoreccom-panels":
        return (
            entry.get("candidate_type") == "panel_with_official_character_boxes"
            and entry.get("publication_partition") == "source_train"
            and entry.get("book_identity_verified") is True
        )
    if source_id == "the-met-open-access":
        record_url = urllib.parse.urlparse(str(entry.get("source_record_url", "")))
        allowed = {"collectionapi.metmuseum.org", "www.metmuseum.org", "metmuseum.org"}
        return (
            entry.get("is_public_domain") is True
            and bool(entry.get("primary_image"))
            and record_url.scheme == "https"
            and (record_url.hostname or "").casefold() in allowed
        )
    if source_id == "aic-open-access":
        record_url = urllib.parse.urlparse(str(entry.get("source_record_url", "")))
        return (
            entry.get("is_public_domain") is True
            and bool(entry.get("image_id"))
            and record_url.scheme == "https"
            and (record_url.hostname or "").casefold() in {"api.artic.edu", "www.artic.edu"}
        )
    return False


def _pick_distinct(
    entries: list[dict[str, Any]],
    count: int,
    seed: str,
    source_id: str,
    group_field: str = "group_id",
) -> list[dict[str, Any]]:
    ranked = sorted(entries, key=lambda entry: _stable_key(seed, source_id, entry))
    selected: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    seen_hashes: set[str] = set()
    for entry in ranked:
        group = entry.get(group_field)
        digest = entry.get("sha256")
        if (
            not isinstance(group, str)
            or not group
            or group in seen_groups
            or not isinstance(digest, str)
            or digest in seen_hashes
        ):
            continue
        seen_groups.add(group)
        seen_hashes.add(digest)
        selected.append(entry)
        if len(selected) == count:
            break
    return selected


def _manifest_blockers(
    root: Path, manifest_path: Path, source_id: str, manifest: dict[str, Any]
) -> list[str]:
    blockers: list[str] = []
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or root not in manifest_path.resolve().parents
    ):
        blockers.append("source_manifest_not_a_regular_private_file")
    if (
        manifest.get("source_id") != source_id
        or not isinstance(manifest.get("entries"), list)
        or not isinstance(manifest.get("relative_root"), str)
        or not manifest.get("relative_root")
    ):
        return [*blockers, "source_manifest_schema_or_identity_invalid"]
    try:
        tree_root = ensure_inside(root, manifest["relative_root"])
        cursor = tree_root
        has_symlink = False
        while cursor != root:
            if cursor.is_symlink():
                has_symlink = True
                break
            cursor = cursor.parent
        if has_symlink or not tree_root.is_dir():
            blockers.append("source_tree_missing_or_unsafe")
    except (OSError, IntakeError):
        blockers.append("source_tree_missing_or_unsafe")
    entries = manifest["entries"]
    if any(not isinstance(entry, dict) for entry in entries):
        return [*blockers, "source_manifest_contains_invalid_entry"]
    paths: set[str] = set()
    asset_ids: set[str] = set()
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item.get("relative_path", ""))):
        relative = entry.get("relative_path")
        asset_id = entry.get("asset_id")
        sha256 = entry.get("sha256")
        byte_length = entry.get("byte_length")
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(asset_id, str)
            or not asset_id
            or not isinstance(sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", sha256)
            or type(byte_length) is not int
            or byte_length < 0
        ):
            blockers.append("source_manifest_entry_schema_invalid")
            continue
        try:
            if _safe_member_name(relative) != relative:
                blockers.append("source_manifest_path_not_canonical")
        except IntakeError:
            blockers.append("source_manifest_path_unsafe")
        normalized_path = unicodedata.normalize("NFC", relative).casefold()
        if normalized_path in paths or asset_id in asset_ids:
            blockers.append("source_manifest_paths_or_asset_ids_not_unique")
        paths.add(normalized_path)
        asset_ids.add(asset_id)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(byte_length).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
    if (
        type(manifest.get("file_count")) is not int
        or manifest["file_count"] != len(entries)
        or manifest.get("tree_sha256") != digest.hexdigest()
    ):
        blockers.append("source_manifest_tree_digest_or_file_count_mismatch")
    return sorted(set(blockers))


def _verified_asset_path(root: Path, relative_root: str, relative_path: str) -> Path:
    base = ensure_inside(root, relative_root)
    candidate = ensure_inside(base, relative_path)
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            raise IntakeError("Selected asset path traverses a symbolic link")
        cursor = cursor.parent
    if base not in candidate.parents:
        raise IntakeError("Selected asset path escapes its manifest root")
    return candidate


def _source_blockers(
    root: Path,
    source: dict[str, Any],
    manifest: dict[str, Any],
    audit_row: dict[str, Any],
    registry_hash: str,
    source_row_hash: str,
) -> list[str]:
    blockers: list[str] = []
    if (
        source.get("rights_status") != "approved"
        or source.get("terms_review", {}).get("status") != "approved"
    ):
        blockers.append("source_terms_not_approved")
    if source.get("use_permissions", {}).get("local_display") is not True:
        blockers.append("source_local_display_not_permitted")
    if audit_row.get("terms_snapshot_status") != "hash_verified":
        blockers.append("current_terms_snapshot_not_hash_verified")
    recorded_row_hash = manifest.get("source_registry_row_sha256")
    if recorded_row_hash != source_row_hash and manifest.get("source_registry_sha256") != registry_hash:
        blockers.append("manifest_source_registry_hash_stale")
    if source.get("archive_path"):
        receipt_path = ensure_inside(root, f"manifests/source-receipts/{source['id']}.json")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            archive = ensure_inside(root, str(receipt.get("archive_path", "")))
            verify_archive_receipt(root, source, archive, receipt)
            if manifest.get("source_archive_sha256") != receipt.get("sha256"):
                blockers.append("manifest_source_archive_hash_mismatch")
        except (OSError, json.JSONDecodeError, IntakeError):
            blockers.append("verified_official_archive_receipt_missing_or_invalid")
    else:
        snapshot_hash = manifest.get("source_metadata_snapshot_sha256")
        if not isinstance(snapshot_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", snapshot_hash):
            blockers.append("official_metadata_snapshot_missing")
        else:
            snapshot_path = ensure_inside(
                root, f"manifests/source-metadata/{source['id']}/{snapshot_hash}.json"
            )
            if (
                not snapshot_path.is_file()
                or snapshot_path.is_symlink()
                or sha256_file(snapshot_path) != snapshot_hash
            ):
                blockers.append("official_metadata_snapshot_hash_mismatch")
    return blockers


def build_pilot_yield_sample(
    data_root: Path,
    *,
    profile_path: Path,
    registry_path: Path | None = None,
    terms_path: Path | None = None,
    recipe_path: Path = RECIPE_PATH,
) -> dict[str, Any]:
    """Create a deterministic candidate list for human source-yield review.

    This never marks an asset reviewed, eligible for the gallery, or authorized for
    training. Only selected source strata are considered; failed strata stay
    blocked independently and do not pull unselected registry rows into the sample.
    """
    root = data_root.resolve()
    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError("The pilot-v1 sampling recipe is missing or invalid") from error
    expected_targets = {
        "anita-dataset": 70,
        "emoreccom-panels": 60,
        "the-met-open-access": 50,
        "aic-open-access": 50,
    }
    expected_styles = {
        "manga_anime": 60,
        "western_ink": 45,
        "realistic_academic": 55,
        "cartoon": 45,
        "gesture_sketch": 45,
    }
    expected_gallery_sources = {
        "anita-dataset": 105,
        "emoreccom-panels": 70,
        "the-met-open-access": 35,
        "aic-open-access": 40,
    }
    if not isinstance(recipe, dict):
        raise IntakeError("Unsupported pilot source-yield recipe")
    yield_targets = recipe.get("source_yield")
    style_targets = (
        recipe.get("pilot_gallery", {}).get("styles")
        if isinstance(recipe.get("pilot_gallery"), dict)
        else None
    )
    if (
        recipe.get("schema_version") != 1
        or recipe.get("recipe_id") != "pilot-v1"
        or recipe.get("seed") != "drawable-pilot-v1"
        or not isinstance(yield_targets, dict)
        or {
            source_id: row.get("items")
            for source_id, row in yield_targets.items()
            if isinstance(row, dict)
        }
        != expected_targets
        or not isinstance(style_targets, dict)
        or not all(type(count) is int and count > 0 for count in style_targets.values())
        or style_targets != expected_styles
        or recipe.get("pilot_gallery", {}).get("sources") != expected_gallery_sources
        or recipe.get("pilot_gallery", {}).get("target_items") != 250
        or yield_targets.get("anita-dataset", {}).get("per_work") != 5
        or yield_targets.get("emoreccom-panels", {}).get("distinct_verified_book_groups") != 60
    ):
        raise IntakeError("Unsupported pilot source-yield recipe")
    registry_file = registry_path or REPOSITORY_ROOT / "config" / "sources.json"
    terms_file = terms_path or REPOSITORY_ROOT / "config" / "anita-work-terms.json"
    registry = load_registry(registry_file)
    sources = {source["id"]: source for source in registry["sources"]}
    registry_hash = sha256_file(registry_file)
    work_records = _anita_work_records(terms_file)
    from .m0 import _load_source_selection_profile

    work_registry = json.loads(terms_file.read_text(encoding="utf-8"))
    profile, profile_relative, profile_digest, profile_evidence = _load_source_selection_profile(
        root, profile_path, registry["sources"], work_registry["works"]
    )
    selected_by_id = {item.source_id: item for item in profile.sources}
    selected_source_rows = [sources[item.source_id] for item in profile.sources]
    source_snapshot_path, source_snapshot_hash = _private_rows_snapshot(
        root,
        "selected-source-registry",
        profile_digest,
        selected_source_rows,
    )
    selected_work_ids = {
        work_id for item in profile.sources for work_id in item.work_ids
    }
    selected_work_rows = [
        row
        for row in work_registry["works"]
        if str(row.get("work_id", row.get("folder_id", ""))) in selected_work_ids
    ]
    work_snapshot_path, work_snapshot_hash = _private_rows_snapshot(
        root,
        "selected-work-registry",
        profile_digest,
        selected_work_rows,
    )
    profile_evidence[source_snapshot_path] = source_snapshot_hash
    profile_evidence[work_snapshot_path] = work_snapshot_hash
    audit = audit_sources(root, registry_path=registry_file, verify_archives=False)
    audit_rows = {row["source_id"]: row for row in audit["sources"]}
    plans: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    any_blocked = False
    for source_id, selection in selected_by_id.items():
        if "local_display" not in selection.operations:
            plans.append(
                {
                    "source_id": source_id,
                    "target": 0,
                    "status": "blocked",
                    "candidate_count": 0,
                    "selected": [],
                    "blockers": ["source_profile_does_not_include_local_display"],
                }
            )
            any_blocked = True
            continue
        if source_id not in YIELD_SOURCES:
            plans.append(
                {
                    "source_id": source_id,
                    "target": 0,
                    "status": "blocked",
                    "candidate_count": 0,
                    "selected": [],
                    "blockers": ["source_yield_sampler_not_configured"],
                }
            )
            any_blocked = True
            continue
        target = recipe["source_yield"][source_id]["items"]
        if source_id == "anita-dataset":
            target = recipe["source_yield"][source_id]["per_work"] * len(selection.work_ids)
        source = sources.get(source_id)
        if source is None:
            plans.append(
                {
                    "source_id": source_id,
                    "target": target,
                    "status": "blocked",
                    "candidate_count": 0,
                    "selected": [],
                    "blockers": ["source_missing_from_registry"],
                }
            )
            any_blocked = True
            continue
        manifest_path = root / "manifests" / f"{source_id}-tree.json"
        try:
            if manifest_path.is_symlink():
                raise IntakeError("Source manifest is a symbolic link")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            plans.append(
                {
                    "source_id": source_id,
                    "target": target,
                    "status": "blocked",
                    "candidate_count": 0,
                    "selected": [],
                    "blockers": ["source_manifest_missing_or_invalid"],
                }
            )
            any_blocked = True
            continue
        if not isinstance(manifest, dict):
            plans.append(
                {
                    "source_id": source_id,
                    "target": target,
                    "status": "blocked",
                    "candidate_count": 0,
                    "selected": [],
                    "blockers": ["source_manifest_schema_or_identity_invalid"],
                }
            )
            any_blocked = True
            continue
        blockers = _manifest_blockers(root, manifest_path, source_id, manifest)
        if not isinstance(manifest.get("entries"), list) or any(
            not isinstance(entry, dict) for entry in manifest.get("entries", [])
        ):
            entries: list[dict[str, Any]] = []
        else:
            entries = manifest["entries"]
        blockers.extend(
            _source_blockers(
                root,
                source,
                manifest,
                audit_rows.get(source_id, {}),
                registry_hash,
                hashlib.sha256(
                    json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            )
        )
        if blockers:
            plans.append(
                {
                    "source_id": source_id,
                    "target": target,
                    "status": "blocked",
                    "candidate_count": 0,
                    "selected": [],
                    "blockers": sorted(set(blockers)),
                }
            )
            any_blocked = True
            continue

        eligible = [entry for entry in entries if _eligible_entry(source_id, entry, work_records)]
        selected: list[dict[str, Any]] = []
        selection_blockers: list[str] = []
        if source_id == "anita-dataset":
            per_work = recipe["source_yield"][source_id]["per_work"]
            used_hashes: set[str] = set()
            for work_id in sorted(selection.work_ids):
                if work_id.casefold() not in work_records:
                    selection_blockers.append(f"selected_work_missing_from_terms:{work_id}")
                    continue
                pool = [
                    entry
                    for entry in eligible
                    if str(entry.get("work_id", "")).casefold() == work_id
                    and entry["sha256"] not in used_hashes
                ]
                work_sample = (
                    _pick_distinct(
                        pool, per_work, recipe["seed"], f"{source_id}:{work_id}", "scene_group_id"
                    )
                    if all(isinstance(entry.get("scene_group_id"), str) for entry in pool)
                    else []
                )
                if len(work_sample) < per_work:
                    selection_blockers.append(
                        f"insufficient_approved_nonadjacent_scene_groups:{work_id}:{len(work_sample)}/{per_work}"
                    )
                    continue
                selected.extend(work_sample)
                used_hashes.update(entry["sha256"] for entry in work_sample)
            if len(selection.work_ids) * per_work != target:
                selection_blockers.append("recipe_work_allocation_does_not_match_target")
        else:
            group_field = "group_id"
            if source_id in {"the-met-open-access", "aic-open-access"}:
                group_field = "museum_object_group_id"
                for entry in eligible:
                    entry.setdefault(group_field, entry.get("object_id"))
            distinct_groups = {
                entry.get(group_field)
                for entry in eligible
                if isinstance(entry.get(group_field), str) and entry.get(group_field)
            }
            if len(distinct_groups) < target:
                selection_blockers.append(
                    f"insufficient_distinct_approved_groups:{len(distinct_groups)}/{target}"
                )
            else:
                selected = _pick_distinct(eligible, target, recipe["seed"], source_id, group_field)
        if selection_blockers or len(selected) != target:
            plans.append(
                {
                    "source_id": source_id,
                    "target": target,
                    "status": "blocked",
                    "candidate_count": len(eligible),
                    "selected": [],
                    "blockers": selection_blockers
                    or [f"selection_shortfall:{len(selected)}/{target}"],
                }
            )
            any_blocked = True
            continue

        prior_hashes = {item["sha256"] for item in all_selected}
        if any(entry["sha256"] in prior_hashes for entry in selected):
            plans.append(
                {
                    "source_id": source_id,
                    "target": target,
                    "status": "blocked",
                    "candidate_count": len(eligible),
                    "selected": [],
                    "blockers": ["exact_duplicate_across_source_yield_samples"],
                }
            )
            any_blocked = True
            continue

        serialized: list[dict[str, Any]] = []
        try:
            for entry in selected:
                asset_path = _verified_asset_path(
                    root, manifest["relative_root"], entry["relative_path"]
                )
                if (
                    not asset_path.is_file()
                    or sha256_file(asset_path) != entry["sha256"]
                    or asset_path.stat().st_size != entry["byte_length"]
                ):
                    raise IntakeError("A selected candidate no longer matches its source manifest")
                serialized.append(
                    {
                        "asset_id": entry["asset_id"],
                        "relative_path": f"{manifest['relative_root']}/{entry['relative_path']}",
                        "sha256": entry["sha256"],
                        "group_id": entry.get("group_id") or entry.get("museum_object_group_id"),
                        "work_id": entry.get("work_id"),
                        "candidate_type": entry.get("candidate_type"),
                    }
                )
        except (OSError, KeyError, IntakeError):
            plans.append(
                {
                    "source_id": source_id,
                    "target": target,
                    "status": "blocked",
                    "candidate_count": len(eligible),
                    "selected": [],
                    "blockers": ["selected_asset_changed_or_unsafe"],
                }
            )
            any_blocked = True
            continue
        all_selected.extend({"source_id": source_id, **item} for item in serialized)
        plans.append(
            {
                "source_id": source_id,
                "target": target,
                "status": "candidate_sample_pending_human_review",
                "candidate_count": len(eligible),
                "selected": serialized,
                "blockers": [],
            }
        )
        plans[-1]["manifest_sha256"] = sha256_file(manifest_path)
        plans[-1]["source_archive_sha256"] = manifest.get("source_archive_sha256")
        plans[-1]["source_metadata_snapshot_sha256"] = manifest.get(
            "source_metadata_snapshot_sha256"
        )

    return {
        "schema_version": 2,
        "requirements_revision": REQUIREMENTS_REVISION,
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": sha256_file(recipe_path),
        "seed": recipe["seed"],
        "source_registry_sha256": source_snapshot_hash,
        "source_registry_snapshot_path": source_snapshot_path,
        "anita_work_terms_sha256": work_snapshot_hash,
        "anita_work_terms_snapshot_path": work_snapshot_path,
        "selected_source_registry_sha256": source_snapshot_hash,
        "source_selection_profile_id": profile.profile_id,
        "source_selection_profile_path": profile_relative,
        "source_selection_profile_sha256": profile_digest,
        "source_selection_profile_hashes": {"source_selection": profile_digest},
        "profile_evidence": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(profile_evidence.items())
        ],
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": (
            "blocked"
            if not all_selected
            else "partial_sample_pending_human_review"
            if any_blocked
            else "candidate_sample_pending_human_review"
        ),
        "human_review_required": True,
        "gallery_activation": False,
        "training_authorization": "pending",
        "selected_count": len(all_selected),
        "selected": all_selected,
        "sources": plans,
    }
