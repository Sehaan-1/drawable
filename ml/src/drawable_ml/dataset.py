from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .contracts import AssetRecord, DatasetManifest, DrawingRecord
from .io import MLError, load_json, resolve_private_path, sha256_file, sha256_json

MAX_IMAGE_EDGE = 8192
MAX_IMAGE_PIXELS = 40_000_000
TRAINING_SPLITS = {"train", "model_selection"}


def load_dataset(data_root: Path, manifest_path: Path, registry_path: Path) -> tuple[DatasetManifest, str]:
    root = data_root.resolve()
    try:
        relative_manifest = manifest_path.resolve(strict=True).relative_to(root).as_posix()
    except (OSError, ValueError) as error:
        raise MLError("Dataset manifest must be inside the private data root") from error
    manifest_path = resolve_private_path(root, relative_manifest)
    raw = load_json(manifest_path)
    if not isinstance(raw, dict) or not isinstance(raw.get("assets"), list) or not isinstance(raw.get("drawings"), list):
        raise MLError("Training dataset manifest must contain asset and drawing lists")
    if any(isinstance(row, dict) and row.get("learning_split") == "locked_test" for row in raw["assets"]):
        raise MLError("Locked-test assets must be held in a separate access-controlled manifest")
    if any(
        isinstance(row, dict)
        and row.get("learning_split") in {"threshold_selection", "calibration_audit", "locked_test"}
        for row in raw["drawings"]
    ):
        raise MLError("Threshold, calibration, and locked-test drawings must not be present in a training manifest")
    try:
        manifest = DatasetManifest.model_validate(raw)
    except Exception as error:
        raise MLError(f"Dataset manifest validation failed: {error}") from error
    registry = load_json(registry_path)
    if not isinstance(registry, dict) or registry.get("schema_version") != 1 or not isinstance(registry.get("sources"), list):
        raise MLError("Source registry schema is unsupported")
    registry_hash = sha256_file(registry_path)
    if registry_hash != manifest.source_registry_sha256:
        raise MLError("Source registry changed after the dataset manifest was reviewed")
    if not manifest.assets or not manifest.drawings:
        raise MLError("Dataset manifest must contain reviewed reference assets and drawings")
    if any(not isinstance(item, dict) for item in registry["sources"]):
        raise MLError("Source registry rows must be objects")
    sources = {item.get("id"): item for item in registry["sources"]}
    if len(sources) != len(registry["sources"]):
        raise MLError("Source registry IDs must be unique")
    anita_terms_path = registry_path.with_name("anita-work-terms.json")
    anita_terms: dict[str, dict[str, Any]] = {}
    if anita_terms_path.is_file():
        terms = load_json(anita_terms_path)
        if not isinstance(terms, dict) or not isinstance(terms.get("works"), list):
            raise MLError("Anita per-work terms registry schema is unsupported")
        if any(not isinstance(item, dict) for item in terms["works"]):
            raise MLError("Anita per-work terms rows must be objects")
        anita_terms = {
            str(item.get("work_id", item.get("folder_id", ""))).casefold(): item
            for item in terms["works"]
        }
    asset_ids: set[str] = set()
    path_hashes: dict[str, str] = {}
    asset_hashes: set[str] = set()
    group_splits: dict[tuple[str, str], str] = {}
    work_group_splits: dict[str, set[str]] = {}
    verified_assets: dict[str, AssetRecord] = {}
    for asset in manifest.assets:
        if asset.asset_id in asset_ids:
            raise MLError("Dataset asset IDs must be unique")
        asset_ids.add(asset.asset_id)
        source = sources.get(asset.source_id)
        if not source:
            raise MLError(f"Asset {asset.asset_id} names an unknown source")
        source_permissions = source.get("use_permissions", {})
        terms_review = source.get("terms_review")
        if (
            not isinstance(source_permissions, dict)
            or source.get("rights_status") != "approved"
            or not isinstance(terms_review, dict)
            or terms_review.get("status") != "approved"
        ):
            raise MLError(f"Source rights are not approved for {asset.source_id}")
        if source.get("terms_snapshot_sha256") != asset.terms_snapshot_sha256:
            raise MLError(f"Asset {asset.asset_id} refers to a stale source terms snapshot")
        if asset.source_registry_sha256 != registry_hash:
            raise MLError(f"Asset {asset.asset_id} refers to a stale source registry")
        if asset.rights_status != "approved" or asset.review_status != "approved" or asset.sfw != "approved":
            raise MLError(f"Asset {asset.asset_id} has not passed rights, human review, and SFW gates")
        for permission in ("training", "evaluation", "derive", "export", "cloud_storage", "cloud_processing"):
            if getattr(asset.permissions, permission) and not source_permissions.get(permission, False):
                raise MLError(f"Asset {asset.asset_id} claims a use not approved in its source registry: {permission}")
        if asset.learning_split == "locked_test" and (
            asset.permissions.training or asset.permissions.cloud_processing or asset.permissions.cloud_storage
        ):
            raise MLError("Locked-test assets may never carry training or cloud-transfer permissions")
        if asset.source_id == "anita-dataset":
            work = anita_terms.get(asset.source_work_group.casefold())
            if not work or work.get("rights_status") != "approved" or work.get("reviewer_decision") != "approved":
                raise MLError(f"Anita work-level rights are not approved for {asset.source_work_group}")
            if work.get("terms_snapshot_sha256") != asset.terms_snapshot_sha256:
                raise MLError(f"Anita work-level terms changed for {asset.source_work_group}")
            allowed = work.get("use_permissions", {})
            if any(getattr(asset.permissions, key) and not allowed.get(key, False) for key in asset.permissions.model_fields_set):
                raise MLError(f"Anita per-work terms do not grant all declared asset uses for {asset.asset_id}")
        group_key = (asset.source_id, asset.source_work_group)
        previous_split = group_splits.setdefault(group_key, asset.learning_split)
        if previous_split != asset.learning_split:
            raise MLError(f"A source work/book group crosses asset splits: {asset.source_work_group}")
        file_path = resolve_private_path(root, asset.relative_path)
        if sha256_file(file_path) != asset.sha256:
            raise MLError(f"Asset bytes changed after manifesting: {asset.asset_id}")
        _validate_image(file_path)
        previous_digest_split = path_hashes.setdefault(asset.sha256, asset.learning_split)
        if previous_digest_split != asset.learning_split:
            raise MLError("Exact duplicate image bytes cross a train/validation/test boundary")
        asset_hashes.add(asset.sha256)
        work_group_splits.setdefault(asset.source_work_group, set()).add(asset.learning_split)
        verified_assets[asset.asset_id] = asset
    drawing_ids: set[str] = set()
    sequence_groups: dict[str, str] = {}
    used_prefix_ids: set[str] = set()
    sequence_hash_splits: dict[str, str] = {}
    query_hash_splits: dict[str, str] = {}
    for drawing in manifest.drawings:
        if drawing.recording_id in drawing_ids:
            raise MLError("Drawing recording IDs must be unique")
        drawing_ids.add(drawing.recording_id)
        _verify_raw_sequence(root, drawing)
        prior_sequence_split = sequence_hash_splits.setdefault(drawing.raw_sequence_sha256, drawing.learning_split)
        if prior_sequence_split != drawing.learning_split:
            raise MLError("Exact duplicate drawing-sequence bytes cross learning splits")
        group_split = sequence_groups.setdefault(drawing.sequence_group_id, drawing.learning_split)
        if group_split != drawing.learning_split:
            raise MLError("Related drawing sequences cross train/validation/holdout splits")
        for group in drawing.referenced_source_work_groups:
            matching_splits = work_group_splits.get(group, set())
            if drawing.learning_split == "train" and matching_splits and matching_splits != {"train"}:
                raise MLError("A training drawing references a non-training source/work group")
            if drawing.learning_split == "model_selection" and "locked_test" in matching_splits:
                raise MLError("A model-selection drawing references a locked-test source/work group")
        if drawing.reference_asset_id is not None:
            reference_asset = verified_assets.get(drawing.reference_asset_id)
            if not reference_asset:
                raise MLError("Assigned reference asset is missing from the reviewed training manifest")
            if drawing.learning_split == "train" and reference_asset.learning_split != "train":
                raise MLError("A training drawing references a non-training gallery asset")
            if drawing.learning_split == "model_selection" and reference_asset.learning_split == "locked_test":
                raise MLError("A model-selection drawing references a locked-test gallery asset")
        for prefix in drawing.prefixes:
            if prefix.prefix_id in used_prefix_ids:
                raise MLError("Prefix IDs must be globally unique")
            used_prefix_ids.add(prefix.prefix_id)
            query = resolve_private_path(root, prefix.png_path)
            if sha256_file(query) != prefix.png_sha256:
                raise MLError(f"Prefix image bytes changed after review: {prefix.prefix_id}")
            _validate_image(query)
            if prefix.png_sha256 in asset_hashes:
                raise MLError("A query-prefix image is byte-identical to a gallery reference image")
            prior_query_split = query_hash_splits.setdefault(prefix.png_sha256, drawing.learning_split)
            if prior_query_split != drawing.learning_split:
                raise MLError("Exact duplicate query-prefix images cross drawing splits")
            judged = prefix.positive_asset_ids + prefix.judged_unusable_asset_ids
            if drawing.learning_split == "train" and not prefix.positive_asset_ids:
                # A scope-only example remains valid; the retrieval loss will skip it.
                pass
            for asset_id in judged:
                judged_asset = verified_assets.get(asset_id)
                if judged_asset is None:
                    raise MLError(f"Prefix {prefix.prefix_id} references an asset absent from this manifest")
                if drawing.learning_split == "train":
                    if judged_asset.learning_split != "train" or not judged_asset.permissions.training:
                        raise MLError("Training prefixes may reference only training-permitted training assets")
                elif drawing.learning_split == "model_selection":
                    if judged_asset.learning_split == "locked_test":
                        raise MLError("Model selection may not reference locked-test assets")
                    if not judged_asset.permissions.evaluation:
                        raise MLError("Model-selection candidates require explicit evaluation permission")
                else:
                    raise MLError("Only train and model-selection drawings may be supplied to this ML package")
    digest = sha256_json(raw)
    return manifest, digest


def package_training_bundle(
    data_root: Path,
    manifest_path: Path,
    registry_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    root = data_root.resolve()
    manifest, manifest_digest = load_dataset(root, manifest_path, registry_path)
    if output_path.exists() or output_path.is_symlink():
        raise MLError("Training bundles are immutable; choose a new output path")
    try:
        output_relative = output_path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as error:
        raise MLError("Training bundle destination must remain inside the private data root") from error
    output_path = resolve_private_path(root, output_relative, must_exist=False)
    selected_drawings = [row for row in manifest.drawings if row.learning_split in TRAINING_SPLITS]
    selected_ids = {
        asset_id
        for drawing in selected_drawings
        for asset_id in _drawing_asset_ids(drawing)
    }
    selected_assets = [row for row in manifest.assets if row.asset_id in selected_ids]
    if any(row.learning_split == "locked_test" for row in selected_assets):
        raise MLError("Protected test content cannot enter a training bundle")
    training_asset_ids = {
        asset_id
        for drawing in selected_drawings
        if drawing.learning_split == "train"
        for asset_id in _drawing_asset_ids(drawing)
    }
    evaluation_asset_ids = {
        asset_id
        for drawing in selected_drawings
        if drawing.learning_split == "model_selection"
        for asset_id in _drawing_asset_ids(drawing)
    }
    for asset in selected_assets:
        if not asset.permissions.export or not asset.permissions.cloud_storage or not asset.permissions.cloud_processing:
            raise MLError(f"Training bundle transfer is not explicitly permitted for asset {asset.asset_id}")
        if asset.asset_id in training_asset_ids and not asset.permissions.training:
            raise MLError(f"Training use is not explicitly permitted for asset {asset.asset_id}")
        if asset.asset_id in evaluation_asset_ids and not asset.permissions.evaluation:
            raise MLError(f"Model-selection use is not explicitly permitted for asset {asset.asset_id}")
    for drawing in selected_drawings:
        if not drawing.cloud_storage_consent or not drawing.cloud_processing_consent:
            raise MLError(f"Cloud transfer/processing consent is missing for sequence {drawing.recording_id}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    required_free_bytes = sum(resolve_private_path(root, row.relative_path).stat().st_size for row in selected_assets)
    required_free_bytes += sum(
        resolve_private_path(root, prefix.png_path).stat().st_size
        for drawing in selected_drawings
        for prefix in drawing.prefixes
    )
    if shutil.disk_usage(output_path.parent).free < required_free_bytes + 64 * 1024 * 1024:
        raise MLError("Insufficient free disk to stage the training bundle with a 64 MiB safety margin")
    temporary = output_path.with_name(f".{output_path.name}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise MLError("A previous partial package exists; inspect it before choosing another destination")
    records: list[dict[str, Any]] = []
    written: dict[str, str] = {}
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for asset in sorted(selected_assets, key=lambda row: row.asset_id):
                path = resolve_private_path(root, asset.relative_path)
                archive_name = f"assets/{asset.asset_id}.png"
                _write_member(archive, archive_name, path.read_bytes())
                written[archive_name] = asset.sha256
                records.append({
                    "asset_id": asset.asset_id,
                    "path": archive_name,
                    "sha256": asset.sha256,
                    "source_id": asset.source_id,
                    "source_work_group": asset.source_work_group,
                    "learning_split": asset.learning_split,
                    "rights_status": asset.rights_status,
                    "review_status": asset.review_status,
                    "sfw": asset.sfw,
                    "terms_snapshot_sha256": asset.terms_snapshot_sha256,
                    "source_registry_sha256": asset.source_registry_sha256,
                    "license_id": asset.license_id,
                    "attribution": asset.attribution,
                    "style": asset.style,
                    "primary_scope": asset.primary_scope,
                    "permissions": asset.permissions.model_dump(mode="json"),
                })
            sequence_rows: list[dict[str, Any]] = []
            for drawing in sorted(selected_drawings, key=lambda row: row.recording_id):
                prefix_rows: list[dict[str, Any]] = []
                for prefix in drawing.prefixes:
                    query_path = resolve_private_path(root, prefix.png_path)
                    member = f"queries/{prefix.prefix_id}.png"
                    _write_member(archive, member, query_path.read_bytes())
                    written[member] = prefix.png_sha256
                    prefix_rows.append({
                        "prefix_id": prefix.prefix_id,
                        "target_percent": prefix.target_percent,
                        "actual_percent": prefix.actual_percent,
                        "stroke_index": prefix.stroke_index,
                        "path": member,
                        "sha256": prefix.png_sha256,
                        "visible_scopes": prefix.visible_scopes,
                        "positive_asset_ids": prefix.positive_asset_ids,
                        "judged_unusable_asset_ids": prefix.judged_unusable_asset_ids,
                        "judgments": [item.model_dump(mode="json") for item in prefix.judgments],
                    })
                sequence_rows.append({
                    "recording_id": drawing.recording_id,
                    "contributor_group_id": drawing.contributor_group_id,
                    "sequence_group_id": drawing.sequence_group_id,
                    "task_kind": drawing.task_kind,
                    "collection_program": drawing.collection_program,
                    "learning_split": drawing.learning_split,
                    "style": drawing.style,
                    "final_scope": drawing.final_scope,
                    "consent": {
                        "research": drawing.research_consent,
                        "training": drawing.training_consent,
                        "evaluation": drawing.evaluation_consent,
                        "cloud_storage": drawing.cloud_storage_consent,
                        "cloud_processing": drawing.cloud_processing_consent,
                    },
                    "reference": {
                        "kind": drawing.reference_kind,
                        "asset_id": drawing.reference_asset_id,
                        "source_url": drawing.reference_source_url,
                        "permissions": drawing.reference_permissions.model_dump(mode="json"),
                        "referenced_source_work_groups": drawing.referenced_source_work_groups,
                    },
                    "prefixes": prefix_rows,
                })
            payload = {
                "schema_version": 2,
                "format": "drawable-training-bundle",
                "purpose": "training-experiment",
                "dataset_id": manifest.dataset_id,
                "dataset_manifest_sha256": manifest_digest,
                "source_registry_sha256": manifest.source_registry_sha256,
                "dataset_recipe_sha256": manifest.dataset_recipe_sha256,
                "experiment_recipe_sha256": manifest.experiment_recipe_sha256,
                "assets": records,
                "drawings": sequence_rows,
            }
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            _write_member(archive, "bundle.json", encoded)
            written["bundle.json"] = hashlib.sha256(encoded).hexdigest()
        # Re-open and hash every member before the immutable destination is published.
        with zipfile.ZipFile(temporary, "r") as archive:
            if archive.testzip() is not None:
                raise MLError("Training bundle failed its CRC readback")
            for name, expected in written.items():
                with archive.open(name) as stream:
                    digest = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                    observed = digest.hexdigest()
                if observed != expected:
                    raise MLError(f"Training bundle readback mismatch for {name}")
        output_path.hardlink_to(temporary)
        temporary.unlink()
        return {
            "schema_version": 1,
            "status": "created_pending_manual_transfer",
            "purpose": "training-experiment",
            "path": output_path.relative_to(root).as_posix(),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "drawings": len(selected_drawings),
            "assets": len(selected_assets),
            "training_authorized": False,
            "uploaded": False,
            "protected_splits_included": False,
        }
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _verify_raw_sequence(root: Path, drawing: DrawingRecord) -> None:
    path = resolve_private_path(root, drawing.raw_sequence_path)
    if sha256_file(path) != drawing.raw_sequence_sha256:
        raise MLError(f"Raw sequence changed after consent/review: {drawing.recording_id}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MLError(f"Raw sequence is unreadable: {drawing.recording_id}") from error
    consent = raw.get("consent", {})
    ref = raw.get("reference", {})
    perms = ref.get("permittedUses", {}) if isinstance(ref, dict) else {}
    if raw.get("format") != "drawable-sequence" or raw.get("formatVersion") != 3:
        raise MLError("Only reviewed V3 sequence records can enter a cloud training bundle")
    if raw.get("recordingId") != drawing.recording_id:
        raise MLError("Sequence identifier does not match the consented source record")
    task = raw.get("task", {})
    if task.get("collectionProgram") != drawing.collection_program:
        raise MLError("Collection program does not match the original sequence record")
    if task.get("intendedStyle") != drawing.style or task.get("intendedScope") != drawing.final_scope:
        raise MLError("Dataset style/scope metadata does not match the reviewed recording task")
    if raw.get("contributorGroupId") != drawing.contributor_group_id:
        raise MLError("Contributor group does not match the original pseudonymous recording")
    checks = {
        "creatorConfirmed": drawing.creator_confirmed,
        "contributorCanGrantPermissions": drawing.contributor_can_grant_permissions,
        "referenceRightsDeclared": drawing.reference_rights_declared,
        "research": drawing.research_consent,
        "training": drawing.training_consent,
        "evaluation": drawing.evaluation_consent,
        "cloudStorage": drawing.cloud_storage_consent,
        "cloudProcessing": drawing.cloud_processing_consent,
    }
    if any(consent.get(source_key) is not expected for source_key, expected in checks.items()):
        raise MLError("Dataset consent fields do not match the original sequence record")
    if drawing.reference_kind != ref.get("kind"):
        raise MLError("Dataset reference type does not match the original sequence record")
    if drawing.reference_asset_id != ref.get("assetId"):
        raise MLError("Assigned reference ID does not match the consented sequence")
    if drawing.reference_source_url != ref.get("sourceUrl"):
        raise MLError("External reference URL does not match the consented sequence")
    if drawing.reference_kind != "none" and any(perms.get(key) is not value for key, value in {
        "training": drawing.reference_permissions.training,
        "evaluation": drawing.reference_permissions.evaluation,
        "cloudStorage": drawing.reference_permissions.cloud_storage,
        "cloudProcessing": drawing.reference_permissions.cloud_processing,
    }.items()):
        raise MLError("Reference permissions do not match the original sequence record")


def _drawing_asset_ids(drawing: DrawingRecord) -> set[str]:
    asset_ids = {
        asset_id
        for prefix in drawing.prefixes
        for asset_id in prefix.positive_asset_ids + prefix.judged_unusable_asset_ids
    }
    if drawing.reference_asset_id is not None:
        asset_ids.add(drawing.reference_asset_id)
    return asset_ids


def _validate_image(path: Path) -> None:
    if path.stat().st_size > 25 * 1024 * 1024:
        raise MLError("A single prepared PNG may not exceed 25 MiB")
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise MLError("Prepared queries and references must be normalized PNG files")
            width, height = image.size
            if width < 1 or height < 1 or width > MAX_IMAGE_EDGE or height > MAX_IMAGE_EDGE or width * height > MAX_IMAGE_PIXELS:
                raise MLError("Image dimensions exceed the validated edge or megapixel limit")
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise MLError(f"Prepared PNG failed image decoding: {path.name}") from error


def _write_member(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content)
