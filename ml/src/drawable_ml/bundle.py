from __future__ import annotations

import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError

from .contracts import SAFE_ID, SHA256
from .io import MLError, sha256_file

MAX_BUNDLE_METADATA_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_IMAGE_BYTES = 25 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def inspect_training_bundle(
    path: Path,
    *,
    expected_dataset_sha256: str,
    expected_recipe_sha256: str,
    expected_asset_hashes: dict[str, str] | None = None,
    expected_prefix_hashes: dict[str, str] | None = None,
    expected_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MLError("Training bundle must be a regular, non-symlink file")
    if not SHA256.fullmatch(expected_dataset_sha256) or not SHA256.fullmatch(expected_recipe_sha256):
        raise MLError("Expected dataset and recipe identities must be SHA-256 digests")
    bundle_digest = sha256_file(path)
    if expected_bundle_sha256 is not None and bundle_digest != expected_bundle_sha256:
        raise MLError("Transferred bundle bytes differ from the locally reviewed package hash")
    if not zipfile.is_zipfile(path):
        raise MLError("Training bundle is not a valid ZIP archive")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names: set[str] = set()
            for info in infos:
                pure = PurePosixPath(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if (
                    info.is_dir()
                    or info.filename in names
                    or pure.is_absolute()
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or "\\" in info.filename
                    or mode == stat.S_IFLNK
                ):
                    raise MLError("Training bundle contains duplicate, noncanonical, or unsafe member paths")
                names.add(info.filename)
                maximum = MAX_BUNDLE_METADATA_BYTES if info.filename == "bundle.json" else MAX_BUNDLE_IMAGE_BYTES
                if info.file_size > maximum:
                    raise MLError("Training bundle contains a member beyond the supported size limit")
                if info.flag_bits & 0x1:
                    raise MLError("Encrypted ZIP members are not supported")
            if "bundle.json" not in names:
                raise MLError("Training bundle has no bundle.json manifest")
            if len(names) > 100_001:
                raise MLError("Training bundle contains too many members")
            raw_metadata = archive.read("bundle.json")
            try:
                bundle = json.loads(raw_metadata)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise MLError("Training bundle metadata is not valid UTF-8 JSON") from error
            if not isinstance(bundle, dict) or not isinstance(bundle.get("assets"), list) or not isinstance(bundle.get("drawings"), list):
                raise MLError("Training bundle metadata must contain asset and drawing lists")
            if (
                bundle.get("schema_version") != 2
                or bundle.get("format") != "drawable-training-bundle"
                or bundle.get("purpose") != "training-experiment"
            ):
                raise MLError("Training bundle format or purpose is unsupported")
            if bundle.get("dataset_manifest_sha256") != expected_dataset_sha256:
                raise MLError("Training bundle does not match the current dataset manifest")
            if bundle.get("experiment_recipe_sha256") != expected_recipe_sha256:
                raise MLError("Training bundle does not match the current frozen recipe")
            if (
                not isinstance(bundle.get("dataset_recipe_sha256"), str)
                or not SHA256.fullmatch(bundle["dataset_recipe_sha256"])
                or not isinstance(bundle.get("source_registry_sha256"), str)
                or not SHA256.fullmatch(bundle["source_registry_sha256"])
            ):
                raise MLError("Training bundle dataset/source-registry recipe hashes are malformed")
            expected_members = {"bundle.json"}
            asset_ids: set[str] = set()
            training_asset_ids: set[str] = set()
            evaluation_asset_ids: set[str] = set()
            for item in bundle.get("assets", []):
                if not isinstance(item, dict):
                    raise MLError("Training bundle asset rows must be objects")
                asset_id = item.get("asset_id")
                member = item.get("path")
                digest = item.get("sha256")
                if not isinstance(asset_id, str) or not SAFE_ID.fullmatch(asset_id) or asset_id in asset_ids or member != f"assets/{asset_id}.png":
                    raise MLError("Training bundle contains a duplicate or malformed asset row")
                if item.get("learning_split") == "locked_test":
                    raise MLError("Training bundle contains a protected test asset")
                if (
                    item.get("rights_status") != "approved"
                    or item.get("review_status") != "approved"
                    or item.get("sfw") != "approved"
                    or not isinstance(item.get("permissions"), dict)
                    or item["permissions"].get("export") is not True
                    or item["permissions"].get("cloud_storage") is not True
                    or item["permissions"].get("cloud_processing") is not True
                ):
                    raise MLError("Reference asset rights/review/transfer permissions are incomplete")
                if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                    raise MLError("Training bundle asset checksum is malformed")
                if expected_asset_hashes is not None and expected_asset_hashes.get(asset_id) != digest:
                    raise MLError("Training bundle reference bytes differ from the validated local manifest")
                asset_ids.add(asset_id)
                expected_members.add(member)
            drawing_ids: set[str] = set()
            prefix_ids: set[str] = set()
            referenced_assets: set[str] = set()
            query_hashes: dict[str, str] = {}
            for drawing in bundle.get("drawings", []):
                if not isinstance(drawing, dict):
                    raise MLError("Training bundle drawing rows must be objects")
                recording_id = drawing.get("recording_id")
                if not isinstance(recording_id, str) or recording_id in drawing_ids:
                    raise MLError("Training bundle contains a duplicate or malformed drawing row")
                if drawing.get("learning_split") not in {"train", "model_selection"}:
                    raise MLError("Protected or unknown drawing splits may not enter a training bundle")
                consent = drawing.get("consent", {})
                if not isinstance(consent, dict):
                    raise MLError("Bundle drawing consent record is malformed")
                purpose_consent = "training" if drawing.get("learning_split") == "train" else "evaluation"
                if (
                    consent.get("research") is not True
                    or consent.get(purpose_consent) is not True
                    or consent.get("cloud_storage") is not True
                    or consent.get("cloud_processing") is not True
                ):
                    raise MLError("Bundle drawing consent does not authorize the selected cloud operation")
                reference = drawing.get("reference", {})
                if not isinstance(reference, dict) or not isinstance(reference.get("permissions", {}), dict):
                    raise MLError("Bundle reference provenance/permission record is malformed")
                reference_kind = reference.get("kind")
                if reference_kind in {"assigned-gallery", "external"}:
                    reference_permissions = reference["permissions"]
                    purpose_permission = "training" if drawing["learning_split"] == "train" else "evaluation"
                    if (
                        reference_permissions.get(purpose_permission) is not True
                        or reference_permissions.get("cloud_storage") is not True
                        or reference_permissions.get("cloud_processing") is not True
                    ):
                        raise MLError("Referenced artwork lacks explicit purpose-specific cloud permission")
                if reference_kind == "assigned-gallery":
                    reference_asset_id = reference.get("asset_id")
                    if not isinstance(reference_asset_id, str) or reference_asset_id not in asset_ids:
                        raise MLError("Assigned reference is missing from the transferred asset set")
                    referenced_assets.add(reference_asset_id)
                    if drawing["learning_split"] == "train":
                        training_asset_ids.add(reference_asset_id)
                    else:
                        evaluation_asset_ids.add(reference_asset_id)
                elif reference_kind == "external":
                    if not isinstance(reference.get("source_url"), str) or not reference["source_url"].startswith("https://"):
                        raise MLError("External reference provenance is missing its HTTPS source URL")
                    if reference.get("asset_id") is not None:
                        raise MLError("External references may not masquerade as an assigned asset")
                elif reference_kind == "none":
                    if reference.get("asset_id") is not None or reference.get("source_url") is not None:
                        raise MLError("A no-reference record may not carry a hidden asset or source URL")
                else:
                    raise MLError("A no-reference record may not carry a hidden asset or source URL")
                drawing_ids.add(recording_id)
                if not isinstance(drawing.get("prefixes"), list):
                    raise MLError("Training bundle drawing has no valid prefix list")
                for prefix in drawing.get("prefixes", []):
                    if not isinstance(prefix, dict):
                        raise MLError("Training bundle prefix rows must be objects")
                    prefix_id = prefix.get("prefix_id")
                    member = prefix.get("path")
                    digest = prefix.get("sha256")
                    if not isinstance(prefix_id, str) or not SAFE_ID.fullmatch(prefix_id) or prefix_id in prefix_ids or member != f"queries/{prefix_id}.png":
                        raise MLError("Training bundle contains a duplicate or malformed prefix row")
                    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                        raise MLError("Training bundle query checksum is malformed")
                    if expected_prefix_hashes is not None and expected_prefix_hashes.get(prefix_id) != digest:
                        raise MLError("Training bundle query bytes differ from the validated local manifest")
                    prefix_ids.add(prefix_id)
                    prior = query_hashes.setdefault(digest, drawing["learning_split"])
                    if prior != drawing["learning_split"]:
                        raise MLError("Identical query-prefix pixels cross bundled train/model-selection splits")
                    expected_members.add(member)
                    positive_ids = prefix.get("positive_asset_ids", [])
                    unusable_ids = prefix.get("judged_unusable_asset_ids", [])
                    judgments = prefix.get("judgments", [])
                    if (
                        not isinstance(positive_ids, list)
                        or not isinstance(unusable_ids, list)
                        or not isinstance(judgments, list)
                        or any(not isinstance(value, str) for value in positive_ids + unusable_ids)
                        or any(not isinstance(value, dict) for value in judgments)
                    ):
                        raise MLError("Training bundle reviewed relevance labels are malformed")
                    ids = positive_ids + unusable_ids
                    if len(set(ids)) != len(ids) or not set(ids) <= asset_ids:
                        raise MLError("Bundle judgment references a duplicate or missing asset")
                    referenced_assets.update(ids)
                    if drawing["learning_split"] == "train":
                        training_asset_ids.update(ids)
                    else:
                        evaluation_asset_ids.update(ids)
                    grades = {row.get("asset_id"): row.get("grade") for row in judgments}
                    if set(grades) != set(ids) or len(grades) != len(judgments):
                        raise MLError("Bundle judgments do not exactly match the reviewed candidate pool")
                    if any(type(grades.get(asset_id)) is not int or grades.get(asset_id, -1) < 2 for asset_id in positive_ids):
                        raise MLError("Bundle positive labels do not meet the reviewed relevance threshold")
                    if any(type(grades.get(asset_id)) is not int or grades.get(asset_id, 4) >= 2 for asset_id in unusable_ids):
                        raise MLError("Bundle unusable labels do not match the reviewed relevance grades")
            if not drawing_ids or not asset_ids:
                raise MLError("Training bundle must contain reviewed drawings and reference assets")
            if (
                expected_asset_hashes is not None and asset_ids != set(expected_asset_hashes)
            ) or (
                expected_prefix_hashes is not None and prefix_ids != set(expected_prefix_hashes)
            ):
                raise MLError("Training bundle file membership differs from the validated local manifest")
            if referenced_assets != asset_ids:
                raise MLError("Training bundle contains missing or unreferenced reference assets")
            asset_rows = {row["asset_id"]: row for row in bundle["assets"]}
            for asset_id in training_asset_ids:
                if asset_rows[asset_id]["permissions"].get("training") is not True:
                    raise MLError("A training example references an asset without training permission")
            for asset_id in evaluation_asset_ids:
                if asset_rows[asset_id]["permissions"].get("evaluation") is not True:
                    raise MLError("A model-selection example references an asset without evaluation permission")
            if names != expected_members:
                raise MLError("Training bundle contains missing or unreferenced files")
            for member in sorted(names - {"bundle.json"}):
                entry = next(info for info in infos if info.filename == member)
                expected_hash = next(
                    row["sha256"]
                    for row in bundle["assets"]
                    if row["path"] == member
                ) if member.startswith("assets/") else next(
                    prefix["sha256"]
                    for drawing in bundle["drawings"]
                    for prefix in drawing["prefixes"]
                    if prefix["path"] == member
                )
                content = archive.read(entry)
                if not content.startswith(PNG_SIGNATURE):
                    raise MLError(f"Bundle image has an invalid PNG signature: {member}")
                try:
                    with Image.open(io.BytesIO(content)) as image:
                        if image.format != "PNG":
                            raise MLError(f"Bundle member is not a PNG image: {member}")
                        width, height = image.size
                        if width < 1 or height < 1 or width > 8192 or height > 8192 or width * height > 40_000_000:
                            raise MLError(f"Bundle PNG dimensions exceed the reviewed limits: {member}")
                        image.verify()
                except (OSError, UnidentifiedImageError) as error:
                    raise MLError(f"Bundle image failed PNG decoding: {member}") from error
                if hashlib.sha256(content).hexdigest() != expected_hash:
                    raise MLError(f"Bundle image checksum mismatch: {member}")
    except (OSError, zipfile.BadZipFile, RuntimeError, AttributeError, KeyError, TypeError, IndexError, StopIteration) as error:
        if isinstance(error, MLError):
            raise
        raise MLError(f"Could not safely inspect the training bundle: {error}") from error
    return {
        "status": "verified",
        "path": str(path),
        "sha256": bundle_digest,
        "bytes": path.stat().st_size,
        "drawings": len(drawing_ids),
        "assets": len(asset_ids),
        "protected_splits_included": False,
    }
