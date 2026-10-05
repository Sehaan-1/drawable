from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from .contracts import SAFE_ID, SHA256
from .io import (
    MLError,
    load_json,
    resolve_private_path,
    sha256_file,
    sha256_json,
    write_json_atomic,
)

ModelPurpose = Literal["inference", "training"]


def capture_model_terms_snapshot(
    data_root: Path,
    registry_path: Path,
    model_id: str,
    purpose: ModelPurpose,
    source_url: str,
    input_path: Path,
) -> dict[str, Any]:
    """Import an owner-saved official terms page without network access or approval."""
    registry = load_json(registry_path)
    rows = {row.get("id"): row for row in registry.get("models", []) if isinstance(row, dict)}
    model = rows.get(model_id)
    if model is None:
        raise MLError("Requested model ID is absent from the registry")
    expected_url = model.get("terms_source_url")
    parsed = urlsplit(source_url)
    if (
        not isinstance(expected_url, str)
        or source_url != expected_url
        or parsed.scheme != "https"
        or parsed.hostname != "huggingface.co"
        or parsed.username
        or parsed.password
        or parsed.query
    ):
        raise MLError("Terms snapshot URL must exactly match the pinned official model-page URL")
    if purpose not in {"inference", "training"}:
        raise MLError("Model terms purpose must be inference or training")
    if input_path.is_symlink() or not input_path.is_file():
        raise MLError("Saved terms input must be a regular, non-symlink file")
    if input_path.stat().st_size < 1 or input_path.stat().st_size > 10 * 1024 * 1024:
        raise MLError("Saved terms page must contain between 1 byte and 10 MiB")
    content = input_path.read_bytes()
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "drawable-model-terms-snapshot",
        "model_id": model_id,
        "purpose": purpose,
        "source_url": source_url,
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "content_type": mimetypes.guess_type(input_path.name)[0] or "application/octet-stream",
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "content_base64": base64.b64encode(content).decode("ascii"),
    }
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    root = data_root.resolve()
    relative = f"manifests/model-terms/{model_id}/{digest}.json"
    destination = resolve_private_path(root, relative, must_exist=False)
    if destination.exists():
        if destination.is_symlink() or sha256_file(destination) != digest:
            raise MLError("Content-addressed model terms snapshot exists with different or unsafe bytes")
        status = "already_captured_pending_owner_review"
    else:
        write_json_atomic(destination, body)
        status = "captured_pending_owner_review"
    return {
        "schema_version": 1,
        "status": status,
        "model_id": model_id,
        "purpose": purpose,
        "source_url": source_url,
        "snapshot_path": relative,
        "terms_snapshot_sha256": digest,
        "source_content_sha256": body["content_sha256"],
        "network_used": False,
        "model_loaded": False,
        "training_started": False,
    }


def record_model_terms_review(
    data_root: Path,
    registry_path: Path,
    model_id: str,
    purpose: ModelPurpose,
    snapshot_sha256: str,
    decision: Literal["approved", "rejected"],
    reviewer_id: str,
    review_note: str,
    *,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Record the owner's explicit inference or training terms decision in the registry."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,99}", reviewer_id):
        raise MLError("Reviewer ID must be a stable pseudonymous code")
    if len(review_note.strip()) < 20 or len(review_note) > 1000:
        raise MLError("Terms review note must explain the decision in 20–1000 characters")
    if not SHA256.fullmatch(snapshot_sha256):
        raise MLError("Terms snapshot hash is malformed")
    root = data_root.resolve()
    snapshot_relative = f"manifests/model-terms/{model_id}/{snapshot_sha256}.json"
    snapshot_path = resolve_private_path(root, snapshot_relative)
    if sha256_file(snapshot_path) != snapshot_sha256:
        raise MLError("Terms snapshot bytes do not match the supplied hash")
    snapshot = load_json(snapshot_path)
    if (
        snapshot.get("artifact_type") != "drawable-model-terms-snapshot"
        or snapshot.get("model_id") != model_id
        or snapshot.get("purpose") != purpose
    ):
        raise MLError("Terms snapshot identity does not match model and review purpose")
    registry = load_json(registry_path)
    rows = {row.get("id"): row for row in registry.get("models", []) if isinstance(row, dict)}
    model = rows.get(model_id)
    if model is None:
        raise MLError("Requested model ID is absent from the registry")
    if snapshot.get("source_url") != model.get("terms_source_url"):
        raise MLError("Terms snapshot no longer matches the model registry's official URL")
    key = "terms_review" if purpose == "inference" else "training_terms_review"
    review = model.get(key)
    if not isinstance(review, dict):
        raise MLError("Model registry is missing its purpose-specific terms review record")
    if review.get("status") != "pending" and not replace_existing:
        raise MLError("An existing terms decision is immutable unless --replace-existing is supplied")
    review.update(
        {
            "status": decision,
            "reviewer_id": reviewer_id,
            "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "terms_snapshot_sha256": snapshot_sha256,
            "review_note": review_note.strip(),
        }
    )
    write_json_atomic(registry_path, registry, refuse_overwrite=False)
    return {
        "schema_version": 1,
        "status": "terms_decision_recorded",
        "model_id": model_id,
        "purpose": purpose,
        "decision": decision,
        "terms_snapshot_sha256": snapshot_sha256,
        "registry_path": str(registry_path),
        "model_loaded": False,
        "training_authorized": False,
        "training_started": False,
        "data_uploaded": False,
    }


def _review_blockers(root: Path, model_id: str, review: Any, label: str) -> list[str]:
    if not isinstance(review, dict) or review.get("status") != "approved":
        return [f"{label}_terms_not_reviewed"]
    reviewer = review.get("reviewer_id")
    reviewed_at = review.get("reviewed_at")
    digest = review.get("terms_snapshot_sha256")
    if not isinstance(reviewer, str) or not reviewer.strip():
        return [f"{label}_reviewer_missing"]
    try:
        timestamp = datetime.fromisoformat(str(reviewed_at))
    except (TypeError, ValueError):
        return [f"{label}_review_timestamp_invalid"]
    if timestamp.tzinfo is None or timestamp > datetime.now(UTC):
        return [f"{label}_review_timestamp_invalid"]
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        return [f"{label}_terms_snapshot_hash_missing"]
    relative = f"manifests/model-terms/{model_id}/{digest}.json"
    try:
        snapshot = resolve_private_path(root, relative)
        if sha256_file(snapshot) != digest:
            return [f"{label}_terms_snapshot_hash_mismatch"]
    except (MLError, OSError):
        return [f"{label}_terms_snapshot_missing"]
    return []


def _expected_source(model: dict[str, Any]) -> dict[str, Any] | None:
    source = model.get("artifact_source")
    if source is None:
        return None
    if not isinstance(source, dict):
        raise MLError("Model artifact source metadata must be an object")
    repository = source.get("repository")
    revision = source.get("revision")
    weights_file = source.get("weights_file")
    digest = source.get("weights_sha256")
    byte_length = source.get("weights_bytes")
    required_files = source.get("artifact_directory_files")
    if (
        source.get("provider") != "huggingface"
        or not isinstance(repository, str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repository)
        or not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or not isinstance(weights_file, str)
        or Path(weights_file).name != weights_file
        or not isinstance(digest, str)
        or not SHA256.fullmatch(digest)
        or type(byte_length) is not int
        or byte_length < 1
        or not isinstance(required_files, list)
        or not required_files
        or any(not isinstance(item, str) or Path(item).name != item for item in required_files)
        or weights_file not in required_files
    ):
        raise MLError("Pinned model artifact source metadata is malformed")
    return source


def _artifact_files(root: Path, model_id: str, manifest: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    files_root = resolve_private_path(root, f"models/{model_id}/files", must_exist=False)
    if not files_root.is_dir() or files_root.is_symlink():
        raise MLError("Model artifact payload directory is missing or unsafe")
    files: list[dict[str, Any]] = []
    declared_paths: set[str] = set()
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise MLError("Model artifact file manifest is empty or invalid")
    for item in entries:
        if not isinstance(item, dict):
            raise MLError("Model artifact file entries must be objects")
        relative = item.get("path")
        digest = item.get("sha256")
        byte_length = item.get("bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
            or relative in declared_paths
            or not isinstance(digest, str)
            or not SHA256.fullmatch(digest)
            or type(byte_length) is not int
            or byte_length < 1
        ):
            raise MLError("Model artifact file entry has malformed path, size, or checksum")
        declared_paths.add(relative)
        file_path = resolve_private_path(root, f"models/{model_id}/files/{relative}")
        if file_path.stat().st_size != byte_length or sha256_file(file_path) != digest:
            raise MLError(f"Model artifact file bytes do not match: {relative}")
        files.append({"path": relative, "bytes": byte_length, "sha256": digest})
    if files != sorted(files, key=lambda item: item["path"]):
        raise MLError("Model artifact file entries must be sorted by path")
    observed_paths: set[str] = set()
    for candidate in files_root.rglob("*"):
        if candidate.is_symlink():
            raise MLError("Model artifact payload may not contain symlinks")
        if candidate.is_file():
            observed_paths.add(candidate.relative_to(files_root).as_posix())
    if observed_paths != declared_paths:
        raise MLError("Model artifact directory contains unmanifested or missing files")
    return files_root, files


def verify_model_registry(
    data_root: Path,
    registry_path: Path,
    *,
    purpose: ModelPurpose = "training",
    model_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Check purpose-specific terms and pinned local bytes without loading weights or networking."""
    if purpose not in {"inference", "training"}:
        raise MLError("Model verification purpose must be inference or training")
    registry = load_json(registry_path)
    if registry.get("schema_version") != 2 or not isinstance(registry.get("models"), list):
        raise MLError("Model registry schema is unsupported")
    root = data_root.resolve()
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for model in registry["models"]:
        if not isinstance(model, dict):
            raise MLError("Each model registry row must be an object")
        model_id = model.get("id")
        if not isinstance(model_id, str) or not SAFE_ID.fullmatch(model_id) or model_id in seen:
            raise MLError("Model IDs must be unique, stable, and path-safe")
        seen.add(model_id)
        if model_ids is not None and model_id not in model_ids:
            continue
        training_required = purpose == "training"
        row: dict[str, Any] = {
            "model_id": model_id,
            "role": model.get("role"),
            "purpose": purpose,
            "training_required": training_required,
            "status": "blocked",
            "blockers": [],
        }
        try:
            source = _expected_source(model)
        except MLError as error:
            source = None
            row["blockers"].append(f"pinned_artifact_source_invalid:{error}")
        if model.get("state") != "acquired":
            row["blockers"].append("official_weights_not_acquired")
        if source is not None and model.get("revision") != source["revision"]:
            row["blockers"].append("local_model_revision_does_not_match_pinned_upstream_revision")
        revision = model.get("revision")
        expected_weight_hash = model.get("weights_sha256")
        expected_manifest_hash = model.get("artifact_manifest_sha256")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            row["blockers"].append("immutable_model_revision_missing")
        if not isinstance(expected_weight_hash, str) or not SHA256.fullmatch(expected_weight_hash):
            row["blockers"].append("registry_weight_hash_missing")
        if not isinstance(expected_manifest_hash, str) or not SHA256.fullmatch(expected_manifest_hash):
            row["blockers"].append("registry_artifact_manifest_hash_missing")
        if source is not None and expected_weight_hash != source["weights_sha256"]:
            row["blockers"].append("registry_weight_hash_differs_from_pinned_upstream_file")

        row["blockers"].extend(_review_blockers(root, model_id, model.get("terms_review"), "inference"))
        if training_required:
            if model.get("inference_only") is True:
                row["blockers"].append("registry_still_marks_model_inference_only")
            row["blockers"].extend(
                _review_blockers(root, model_id, model.get("training_terms_review"), "training")
            )
            if model.get("training_authorized") is not True:
                row["blockers"].append("model_training_not_explicitly_authorized")
            if model.get("cloud_training_authorized") is not True:
                row["blockers"].append("cloud_model_processing_not_explicitly_authorized")
        try:
            artifact_path = resolve_private_path(root, f"models/{model_id}/artifact.json")
            artifact = load_json(artifact_path)
            if (
                artifact.get("schema_version") != 1
                or artifact.get("model_id") != model_id
                or artifact.get("revision") != revision
                or artifact.get("weights_file") != (source or {}).get("weights_file")
            ):
                raise MLError("model artifact identity, revision, or weight-file name is invalid")
            files_root, files = _artifact_files(root, model_id, artifact)
            by_path = {item["path"]: item for item in files}
            weights_file = str((source or {}).get("weights_file") or "")
            weight_entry = by_path.get(weights_file)
            if weight_entry is None:
                raise MLError("pinned model weight file is absent from the local artifact")
            if (
                weight_entry["sha256"] != (source or {}).get("weights_sha256")
                or weight_entry["bytes"] != (source or {}).get("weights_bytes")
                or artifact.get("weights_sha256") != weight_entry["sha256"]
            ):
                raise MLError("Local model weight bytes do not match the immutable upstream pin")
            if not set((source or {}).get("artifact_directory_files", [])).issubset(by_path):
                raise MLError("Required local model configuration files are missing")
            artifact_digest = sha256_json({
                "schema_version": 1,
                "model_id": model_id,
                "revision": revision,
                "weights_file": weights_file,
                "weights_sha256": weight_entry["sha256"],
                "files": files,
            })
            if (
                artifact.get("artifact_manifest_sha256") != artifact_digest
                or artifact_digest != expected_manifest_hash
            ):
                raise MLError("Model artifact manifest hash does not match the reviewed registry")
            row["verified_artifact_root"] = files_root.relative_to(root).as_posix()
            row["artifact_manifest_sha256"] = artifact_digest
            row["weights_sha256"] = weight_entry["sha256"]
            row["file_count"] = len(files)
            row["file_bytes"] = sum(item["bytes"] for item in files)
        except (MLError, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            row["blockers"].append(f"local_model_artifact_invalid:{error}")
        if not row["blockers"]:
            row["status"] = "complete"
        results.append(row)
    if model_ids is not None and model_ids - seen:
        raise MLError(f"Requested model IDs are absent from the registry: {', '.join(sorted(model_ids - seen))}")
    return {
        "schema_version": 1,
        "purpose": purpose,
        "registry_sha256": sha256_file(registry_path),
        "models": results,
        "status": "complete" if results and all(item["status"] == "complete" for item in results) else "blocked",
        "weights_loaded": False,
        "network_used": False,
    }


def create_model_artifact_manifest(
    data_root: Path,
    registry_path: Path,
    model_id: str,
) -> dict[str, Any]:
    """Hash a manually staged, exact-pinned model directory without loading or promoting it."""
    registry = load_json(registry_path)
    rows = {row.get("id"): row for row in registry.get("models", []) if isinstance(row, dict)}
    model = rows.get(model_id)
    if model is None:
        raise MLError("Requested model ID is absent from the registry")
    source = _expected_source(model)
    if source is None:
        raise MLError("This model does not have an immutable upstream artifact pin")
    root = data_root.resolve()
    files_root = resolve_private_path(root, f"models/{model_id}/files", must_exist=False)
    if not files_root.is_dir() or files_root.is_symlink():
        raise MLError("Stage the pinned model files beneath the private model directory first")
    files: list[dict[str, Any]] = []
    for candidate in sorted(files_root.rglob("*")):
        if candidate.is_symlink():
            raise MLError("Model artifact files may not contain symlinks")
        if candidate.is_file():
            relative = candidate.relative_to(files_root).as_posix()
            files.append({"path": relative, "bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)})
    by_path = {item["path"]: item for item in files}
    weight = by_path.get(source["weights_file"])
    if weight is None:
        raise MLError(f"Pinned weight file is missing: {source['weights_file']}")
    if weight["sha256"] != source["weights_sha256"] or weight["bytes"] != source["weights_bytes"]:
        raise MLError("Staged model weight bytes do not match the pinned upstream checksum and size")
    missing = sorted(set(source["artifact_directory_files"]) - set(by_path))
    if missing:
        raise MLError(f"Required model files are missing: {', '.join(missing)}")
    body: dict[str, Any] = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": source["revision"],
        "weights_file": source["weights_file"],
        "weights_sha256": source["weights_sha256"],
        "files": files,
    }
    body["artifact_manifest_sha256"] = sha256_json(body)
    artifact_path = root / "models" / model_id / "artifact.json"
    write_json_atomic(artifact_path, body)
    return {
        "schema_version": 1,
        "status": "manifest_created_pending_owner_review",
        "model_id": model_id,
        "artifact_path": artifact_path.relative_to(root).as_posix(),
        "artifact_manifest_sha256": body["artifact_manifest_sha256"],
        "weights_sha256": body["weights_sha256"],
        "file_count": len(files),
        "file_bytes": sum(item["bytes"] for item in files),
        "weights_loaded": False,
        "network_used": False,
        "training_started": False,
    }
