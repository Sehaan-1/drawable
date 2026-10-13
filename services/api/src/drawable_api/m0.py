from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any
from urllib.parse import quote

from drawable_research_contracts import (
    CurationSnapshotV1,
    ViewpointReviewV1,
    validate_curation_snapshot_files,
    validate_viewpoint_rubric,
)
from drawable_research_contracts.readiness import (
    REQUIREMENTS_REVISION,
    STAGES,
    EvidenceInputV1,
    GateReportV2,
    SourceSelectionProfileV1,
    validate_readiness,
)
from pydantic import ValidationError

from .curation import _source_permissions, _verify_terms_snapshot, viewpoint_rubric
from .effort import project_effort_plan
from .intake import (
    IntakeError,
    ensure_inside,
    load_registry,
    sha256_file,
    terms_snapshot_receipt_path,
    write_json_create,
)
from .settings import REPOSITORY_ROOT

FULL_PROGRAM_GOLD_REFERENCES = 2_000
USE_PERMISSION_KEYS = (
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
M0_NEXT_ACTIONS = {
    "official_archive_reconciliation": "Acquire only required official archives and reconcile their receipts and clean extractions.",
    "source_work_rights_review": "Review current source and per-work terms, record every use permission, and verify the captured terms snapshots.",
    "source_yield_review": "Review every item in the fixed sample, including eye-crop usefulness, viewpoint, decision, and active review time.",
    "ten_feasibility_sequences": "Collect ten consented freehand V3 recordings and replay them in pinned Chromium with matching render hashes.",
    "human_effort_storage": "Import the effort log, measure disk headroom, and disclose the full-program owner-hours projection.",
    "small_approved_gallery": "Build and validate a small staging gallery from individually reviewed, permitted assets.",
    "frozen_cpu_forward": "Review inference terms and measure both exact frozen encoders on CPU against the approved staging gallery.",
}


def _import_json(
    data_root: Path,
    input_path: Path,
    *,
    subdirectory: str,
    maximum_bytes: int,
    validator: Any,
) -> dict[str, Any]:
    root = data_root.resolve()
    if input_path.is_symlink() or not input_path.is_file():
        raise IntakeError("Import input must be a regular, non-symlink file")
    if input_path.stat().st_size > maximum_bytes:
        raise IntakeError(f"Import input exceeds the {maximum_bytes // (1024 * 1024)} MiB limit")
    raw = input_path.resolve(strict=True).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IntakeError("Import input must be valid UTF-8 JSON") from error
    summary = validator(value)
    import hashlib

    digest = hashlib.sha256(raw).hexdigest()
    relative = f"reports/private/{subdirectory}/{digest}.json"
    destination = _private_file(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or sha256_file(destination) != digest:
            raise IntakeError("Content-addressed import path exists with different or unsafe bytes")
        status = "already_imported"
    else:
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as target:
                target.write(raw)
                target.flush()
                os.fsync(target.fileno())
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        status = "imported"
    return {
        "schema_version": 1,
        "status": status,
        "path": relative,
        "sha256": digest,
        **summary,
        "network_used": False,
        "training_started": False,
        "data_uploaded": False,
    }


def _validate_effort_log(value: Any) -> dict[str, Any]:
    import re

    if (
        not isinstance(value, dict)
        or value.get("format") != "drawable-m0-effort-log"
        or value.get("formatVersion") != 1
        or not isinstance(value.get("chunks"), list)
        or len(value["chunks"]) > 1_000_000
    ):
        raise IntakeError("Effort log format, version, or chunk count is invalid")
    categories = {"source-review", "asset-review", "feasibility-drawing", "evidence-review"}
    chunk_ids: set[str] = set()
    time_ranges: list[tuple[float, float]] = []
    total_ms = 0
    category_ms: dict[str, int] = {category: 0 for category in categories}
    now = datetime.now(UTC).timestamp() * 1000
    for chunk in value["chunks"]:
        if not isinstance(chunk, dict):
            raise IntakeError("Effort log chunks must be JSON objects")
        chunk_id = chunk.get("chunkId")
        if (
            not isinstance(chunk_id, str)
            or not re.fullmatch(r"[0-9a-f-]{36}", chunk_id)
            or chunk_id in chunk_ids
            or not isinstance(chunk.get("sessionId"), str)
            or not re.fullmatch(r"[0-9a-f-]{36}", chunk["sessionId"])
            or chunk.get("category") not in categories
            or type(chunk.get("activeMs")) is not int
            or not 1 <= chunk["activeMs"] <= 86_400_000
            or chunk.get("schemaVersion") != 1
            or chunk.get("checkpoint") not in {"periodic", "paused"}
        ):
            raise IntakeError("Effort log chunk identity, category, or active duration is invalid")
        try:
            started = datetime.fromisoformat(str(chunk.get("startedAt")))
            ended = datetime.fromisoformat(str(chunk.get("endedAt")))
        except ValueError as error:
            raise IntakeError("Effort log chunk timestamps are invalid") from error
        if (
            started.utcoffset() is None
            or ended.utcoffset() is None
            or ended <= started
            or ended.timestamp() * 1000 > now + 60_000
            or chunk["activeMs"] > (ended.timestamp() - started.timestamp()) * 1000 + 2000
        ):
            raise IntakeError("Effort log timestamps or active duration are inconsistent")
        time_ranges.append((started.timestamp(), ended.timestamp()))
        chunk_ids.add(chunk_id)
        total_ms += chunk["activeMs"]
        category_ms[chunk["category"]] += chunk["activeMs"]
    time_ranges.sort()
    if any(
        current[0] < previous[1] - 1
        for previous, current in itertools.pairwise(time_ranges)
    ):
        raise IntakeError("Effort log contains overlapping timer intervals; resolve concurrent tabs before import")
    active_minutes = total_ms // 60_000
    if type(value.get("active_effort_minutes")) is not int or value["active_effort_minutes"] != active_minutes:
        raise IntakeError("Effort log minute total does not match its checkpoint chunks")
    return {
        "chunks": len(chunk_ids),
        "active_effort_minutes": active_minutes,
        "active_effort_milliseconds": total_ms,
        "annotation_effort_milliseconds": category_ms["source-review"] + category_ms["asset-review"],
        "other_owner_effort_milliseconds": category_ms["feasibility-drawing"] + category_ms["evidence-review"],
        "effort_milliseconds_by_category": category_ms,
    }


def import_effort_log(data_root: Path, input_path: Path) -> dict[str, Any]:
    return _import_json(
        data_root,
        input_path,
        subdirectory="m0-effort-logs",
        maximum_bytes=25 * 1024 * 1024,
        validator=_validate_effort_log,
    )


def _validate_recording_batch(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("format") != "drawable-sequence-batch"
        or value.get("formatVersion") != 1
        or not isinstance(value.get("records"), list)
        or len(value["records"]) > 2000
    ):
        raise IntakeError("Recording batch format, version, or record count is invalid")
    recording_ids: set[str] = set()
    accepted = 0
    for item in value["records"]:
        if (
            not isinstance(item, dict)
            or item.get("status") not in {"accepted", "needs_review"}
            or not isinstance(item.get("sequence"), dict)
            or not isinstance(item["sequence"].get("recordingId"), str)
        ):
            raise IntakeError("Recording batch contains a malformed entry")
        recording_id = item["sequence"]["recordingId"]
        if recording_id in recording_ids:
            raise IntakeError("Recording batch contains duplicate recording IDs")
        recording_ids.add(recording_id)
        accepted += item["status"] == "accepted"
    return {"records": len(recording_ids), "accepted_records": accepted}


def import_recording_batch(data_root: Path, input_path: Path) -> dict[str, Any]:
    return _import_json(
        data_root,
        input_path,
        subdirectory="recording-batches",
        maximum_bytes=250 * 1024 * 1024,
        validator=_validate_recording_batch,
    )


def _snapshot_receipt(root: Path, source_id: str, digest: str) -> tuple[str, str, list[tuple[str, str]]]:
    import re

    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise IntakeError(f"Reviewed terms hash is missing or malformed for {source_id}")
    receipt_path = terms_snapshot_receipt_path(root, source_id, digest)
    try:
        receipt_relative = receipt_path.relative_to(root).as_posix()
    except ValueError as error:
        raise IntakeError("Terms receipt must remain inside the private data root") from error
    receipt_path = _private_file(root, receipt_relative)
    receipt = _parse_json(receipt_path, f"{source_id} terms receipt")
    if receipt.get("source_id") != source_id or receipt.get("sha256") != digest:
        raise IntakeError(f"Terms receipt identity or hash is invalid for {source_id}")
    snapshot_relative = receipt.get("snapshot_path")
    if not isinstance(snapshot_relative, str):
        raise IntakeError(f"Terms receipt has no content snapshot path for {source_id}")
    snapshot_path = _private_file(root, snapshot_relative)
    if not snapshot_path.is_file() or sha256_file(snapshot_path) != digest:
        raise IntakeError(f"Current terms snapshot bytes do not match the reviewed hash for {source_id}")
    return receipt_relative, snapshot_relative, [
        (receipt_relative, sha256_file(receipt_path)),
        (snapshot_relative, digest),
    ]


def _valid_owner_review(reviewer_id: Any, reviewed_at: Any) -> bool:
    import re

    if not isinstance(reviewer_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", reviewer_id):
        return False
    if not isinstance(reviewed_at, str):
        return False
    try:
        timestamp = datetime.fromisoformat(reviewed_at)
    except ValueError:
        return False
    return timestamp.tzinfo is not None and timestamp <= datetime.now(UTC)


def _copy_repo_evidence(
    root: Path,
    source_path: Path,
    *,
    subdirectory: str = "m0-rights-registry-snapshots",
) -> tuple[str, str]:
    if source_path.is_symlink() or not source_path.is_file():
        raise IntakeError("Repository evidence must be a regular, non-symlink file")
    raw = source_path.read_bytes()
    import hashlib

    digest = hashlib.sha256(raw).hexdigest()
    relative = f"reports/private/{subdirectory}/{digest}.json"
    destination = _private_file(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or sha256_file(destination) != digest:
            raise IntakeError("Private registry snapshot path exists with unsafe or changed bytes")
    else:
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    return relative, digest


def _load_source_selection_profile(
    root: Path,
    profile_path: Path,
    sources: list[dict[str, Any]],
    works: list[dict[str, Any]],
) -> tuple[SourceSelectionProfileV1, str, str, dict[str, str]]:
    if profile_path.is_symlink() or not profile_path.is_file():
        raise IntakeError("Source selection profile must be a regular, non-symlink JSON file")
    try:
        profile = SourceSelectionProfileV1.model_validate_json(profile_path.read_bytes())
    except (OSError, ValidationError, ValueError) as error:
        raise IntakeError(f"Source selection profile is invalid: {error}") from error
    source_by_id = {row["id"]: row for row in sources if isinstance(row.get("id"), str)}
    work_by_id = {
        str(row.get("work_id", row.get("folder_id", ""))): row
        for row in works
        if isinstance(row, dict)
    }
    evidence: dict[str, str] = {}
    for selected in profile.sources:
        source = source_by_id.get(selected.source_id)
        if source is None:
            raise IntakeError(f"Selected source is absent from the reviewed registry: {selected.source_id}")
        configured_method = source.get("acquisition_method") or (
            "official_archive" if source.get("archive_path") else "official_api"
        )
        if selected.acquisition_method != configured_method:
            raise IntakeError(
                f"Selected acquisition method does not match the registry for {selected.source_id}"
            )
        if selected.source_id != "anita-dataset" and selected.work_ids:
            raise IntakeError("Only AnitaDataset currently has a per-work terms registry")
        if selected.source_id == "anita-dataset" and not selected.work_ids:
            raise IntakeError("Select at least one Anita work; the per-work permission decision is mandatory")
        for work_id in selected.work_ids:
            if work_id not in work_by_id:
                raise IntakeError(f"Selected Anita work is absent from the registry: {work_id}")
        if selected.acquisition_method == "official_archive" and selected.acquisition_evidence:
            raise IntakeError("Archive evidence is recorded by the archive receipt reconciliation")
        if selected.acquisition_method != "official_archive" and not selected.acquisition_evidence:
            raise IntakeError(
                f"Selected {selected.acquisition_method} source requires item-level provenance evidence"
            )
        for item in selected.acquisition_evidence:
            path = _private_file(root, item.path)
            if not path.is_file() or sha256_file(path) != item.sha256:
                raise IntakeError(f"Selected acquisition evidence changed or is missing: {item.path}")
            evidence[item.path] = item.sha256
    profile_relative, profile_digest = _copy_repo_evidence(
        root, profile_path, subdirectory="source-selection-profiles"
    )
    evidence[profile_relative] = profile_digest
    return profile, profile_relative, profile_digest, evidence


def build_source_work_rights_report(
    data_root: Path,
    *,
    source_profile: Path,
    output: Path | None = None,
    gate_output: Path | None = None,
) -> dict[str, Any]:
    """Validate actual source and per-work rights decisions without creating or inferring them."""
    root = data_root.resolve()
    source_registry_path = REPOSITORY_ROOT / "config" / "sources.json"
    work_registry_path = REPOSITORY_ROOT / "config" / "anita-work-terms.json"
    source_registry = load_registry(source_registry_path)
    work_registry = _parse_json(work_registry_path, "Repository per-work terms registry")
    if (
        not isinstance(source_registry, dict)
        or source_registry.get("schema_version") != 1
        or not isinstance(source_registry.get("sources"), list)
        or not isinstance(work_registry, dict)
        or work_registry.get("schema_version") != 1
        or not isinstance(work_registry.get("works"), list)
    ):
        raise IntakeError("Source or work registry schema is unsupported")
    profile, profile_relative, profile_digest, profile_inputs = _load_source_selection_profile(
        root, source_profile, source_registry["sources"], work_registry["works"]
    )
    selected_ids = {item.source_id for item in profile.sources}
    selected_work_ids = {work_id for item in profile.sources for work_id in item.work_ids}
    selected_sources = [row for row in source_registry["sources"] if row.get("id") in selected_ids]
    selected_works = [
        row
        for row in work_registry["works"]
        if str(row.get("work_id", row.get("folder_id", ""))) in selected_work_ids
    ]
    selected_source_rows = {item.source_id: item for item in profile.sources}
    blockers: list[str] = []
    snapshot_hashes: dict[str, str] = {}
    source_reviews: list[dict[str, Any]] = []
    work_reviews: list[dict[str, Any]] = []
    decisions = 0

    def evaluate_permissions(owner: str, permissions: Any, approved: bool) -> bool:
        nonlocal decisions
        if (
            not isinstance(permissions, dict)
            or set(permissions) != set(USE_PERMISSION_KEYS)
            or any(type(permissions.get(key)) is not bool for key in USE_PERMISSION_KEYS)
        ):
            blockers.append(f"permissions_incomplete:{owner}")
            return False
        if permissions["cloud_processing"] and not permissions["cloud_storage"]:
            blockers.append(f"cloud_processing_without_storage_permission:{owner}")
        if not approved and any(permissions.values()):
            blockers.append(f"denied_rights_row_grants_a_use:{owner}")
        decisions += len(USE_PERMISSION_KEYS)
        return True

    for source in selected_sources:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str):
            blockers.append("source_registry_contains_invalid_row")
            continue
        source_id = source["id"]
        rights = source.get("rights_status")
        review = source.get("terms_review")
        reviewed = (
            rights in {"approved", "rejected", "quarantined", "unavailable"}
            and isinstance(review, dict)
            and review.get("status") == ("approved" if rights == "approved" else "rejected")
            and _valid_owner_review(review.get("reviewer_id"), review.get("reviewed_at"))
            and isinstance(source.get("terms_snapshot_sha256"), str)
            and review.get("reviewed_sha256") == source.get("terms_snapshot_sha256")
        )
        if not reviewed:
            blockers.append(f"source_terms_decision_missing:{source_id}")
        permissions_complete = evaluate_permissions(
            f"source:{source_id}", source.get("use_permissions"), rights == "approved"
        )
        selected_operations = selected_source_rows[source_id].operations
        permissions = source.get("use_permissions", {})
        if any(permissions.get(operation) is not True for operation in selected_operations):
            blockers.append(f"selected_operations_not_permitted:source:{source_id}")
        if reviewed:
            try:
                _, _, snapshots = _snapshot_receipt(
                    root, source_id, str(source.get("terms_snapshot_sha256"))
                )
                for relative, digest in snapshots:
                    snapshot_hashes[relative] = digest
            except (IntakeError, OSError, ValueError) as error:
                blockers.append(f"source_terms_snapshot_invalid:{source_id}:{error}")
        source_reviews.append(
            {
                "source_id": source_id,
                "rights_status": rights,
                "decision_recorded": reviewed,
                "permissions_complete": permissions_complete,
                "use_permissions": source.get("use_permissions"),
                "terms_snapshot_sha256": source.get("terms_snapshot_sha256"),
            }
        )

    for work in selected_works:
        if not isinstance(work, dict) or not isinstance(work.get("folder_id"), str):
            blockers.append("work_registry_contains_invalid_row")
            continue
        work_id = work["folder_id"]
        rights = work.get("rights_status")
        decision = work.get("reviewer_decision")
        work_digest = work.get("terms_snapshot_sha256")
        decision_matches_status = (
            (rights == "approved" and decision == "approved")
            or (rights in {"rejected", "quarantined", "unavailable"} and decision == "rejected")
        )
        reviewed = (
            decision in {"approved", "rejected"}
            and rights in {"approved", "rejected", "quarantined", "unavailable"}
            and decision_matches_status
            and _valid_owner_review(work.get("reviewer_id"), work.get("reviewed_at"))
            and isinstance(work_digest, str)
            and work.get("reviewed_sha256") == work_digest
            and (rights != "approved" or (decision == "approved" and bool(work.get("attribution_text"))))
        )
        if not reviewed:
            blockers.append(f"work_terms_decision_missing:{work_id}")
        permissions_complete = evaluate_permissions(
            f"work:{work_id}",
            work.get("use_permissions"),
            rights == "approved" and decision == "approved",
        )
        selected_operations = selected_source_rows["anita-dataset"].operations
        permissions = work.get("use_permissions", {})
        if any(permissions.get(operation) is not True for operation in selected_operations):
            blockers.append(f"selected_operations_not_permitted:work:{work_id}")
        if reviewed:
            try:
                _, _, snapshots = _snapshot_receipt(root, "anita-dataset", str(work_digest))
                for relative, hash_value in snapshots:
                    snapshot_hashes[relative] = hash_value
            except (IntakeError, OSError, ValueError) as error:
                blockers.append(f"work_terms_snapshot_invalid:{work_id}:{error}")
        work_reviews.append(
            {
                "work_id": work_id,
                "rights_status": rights,
                "reviewer_decision": decision,
                "decision_recorded": reviewed,
                "permissions_complete": permissions_complete,
                "use_permissions": work.get("use_permissions"),
                "terms_snapshot_sha256": work_digest,
            }
        )

    source_scope = {"schema_version": 1, "profile_id": profile.profile_id, "sources": selected_sources}
    work_scope = {"schema_version": 1, "profile_id": profile.profile_id, "works": selected_works}
    profile_key = profile_digest
    source_copy = f"reports/private/selected-source-rows/{profile_key}.json"
    work_copy = f"reports/private/selected-work-rows/{profile_key}.json"
    write_json_create(_private_file(root, source_copy), source_scope)
    write_json_create(_private_file(root, work_copy), work_scope)
    source_copy_hash = sha256_file(_private_file(root, source_copy))
    work_copy_hash = sha256_file(_private_file(root, work_copy))
    complete = not blockers and len(source_reviews) == len(selected_sources) and len(work_reviews) == len(selected_works)
    body = {
        "schema_version": 2,
        "requirements_revision": REQUIREMENTS_REVISION,
        "profile_id": profile.profile_id,
        "source_selection_profile_path": profile_relative,
        "source_selection_profile_sha256": profile_digest,
        "artifact_type": "drawable-m0-source-work-rights-review",
        "status": "complete" if complete else "blocked",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_registry_sha256": source_copy_hash,
        "source_registry_snapshot_path": source_copy,
        "work_registry_sha256": work_copy_hash,
        "work_registry_snapshot_path": work_copy,
        "source_count": len(source_reviews),
        "selected_source_count": len(selected_sources),
        "work_count": len(work_reviews),
        "selected_work_count": len(selected_works),
        "unselected_source_count": len(source_registry["sources"]) - len(selected_sources),
        "unselected_work_count": len(work_registry["works"]) - len(selected_works),
        "permission_decision_count": decisions,
        "sources": source_reviews,
        "works": work_reviews,
        "terms_snapshot_inputs": [{"path": path, "sha256": digest} for path, digest in sorted(snapshot_hashes.items())],
        "blockers": sorted(set(blockers)),
        "training_authorized": False,
        "training_started": False,
        "data_uploaded": False,
    }
    if output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = Path(f"reports/private/m0-source-work-rights-v2-{timestamp}.json")
    report_path = output if output.is_absolute() else root / output
    try:
        report_relative = report_path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as error:
        raise IntakeError("Source/work rights report must remain inside the private data root") from error
    report_path = _private_file(root, report_relative)
    write_json_create(report_path, body)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": body["status"],
        "report_path": report_relative,
        "report_sha256": sha256_file(report_path),
        "sources_reviewed": sum(item["decision_recorded"] for item in source_reviews),
        "works_reviewed": sum(item["decision_recorded"] for item in work_reviews),
        "permission_decisions_recorded": decisions,
        "blockers": body["blockers"],
        "gate_report_written": False,
        "training_started": False,
        "data_uploaded": False,
    }
    if not complete:
        return result

    gate = GateReportV2(
        schema_version=2,
        requirements_revision=REQUIREMENTS_REVISION,
        artifact_type="drawable-gate-report",
        gate_id="source_work_rights_review",
        producer="drawable m0 rights-report",
        produced_at=datetime.now(UTC),
        checks={
            "source_terms_reviewed": True,
            "work_terms_reviewed": True,
            "actual_use_permissions_recorded": True,
            "terms_snapshots_hashed": True,
            "training_rights_decided": True,
            "local_display_rights_decided": True,
            "derive_rights_decided": True,
            "trace_rights_decided": True,
            "export_rights_decided": True,
            "publication_rights_decided": True,
            "demonstration_rights_decided": True,
            "permissions_recorded_per_source_and_work": True,
            "selected_profile_reviewed": True,
            "unselected_rows_quarantined": True,
        },
        measurements={
            "sources_reviewed": len(source_reviews),
            "works_reviewed": len(work_reviews),
            "permission_decisions_recorded": decisions,
        },
        profile_hashes={"source_selection": profile_digest},
        inputs=[
            EvidenceInputV1(path=path, sha256=digest)
            for path, digest in sorted(
                {
                    source_copy: source_copy_hash,
                    work_copy: work_copy_hash,
                    profile_relative: profile_digest,
                    report_relative: sha256_file(report_path),
                    **profile_inputs,
                    **snapshot_hashes,
                }.items()
            )
        ],
        summary=(
            f"Recorded source-specific rights and all ten use decisions for {len(source_reviews)} selected sources "
            f"and {len(work_reviews)} selected Anita works. Unselected rows remain quarantined; "
            "these decisions do not authorize cloud processing or training execution."
        ),
    )
    if gate_output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        gate_output = Path(f"reports/m0-source-work-rights-gate-v2-{timestamp}.json")
    gate_path = gate_output if gate_output.is_absolute() else root / gate_output
    try:
        gate_relative = gate_path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as error:
        raise IntakeError("Source/work rights gate report must remain inside the private data root") from error
    gate_path = _private_file(root, gate_relative)
    write_json_create(gate_path, gate.model_dump(mode="json"))
    result.update(
        {
            "gate_report_written": True,
            "gate_report_path": gate_relative,
            "gate_report_sha256": sha256_file(gate_path),
        }
    )
    return result


def build_small_approved_gallery_report(
    data_root: Path,
    *,
    snapshot_path: str,
    report_output: Path | None = None,
    gate_output: Path | None = None,
    manifest_output: Path | None = None,
) -> dict[str, Any]:
    """Recheck a small review snapshot and freeze a non-serving M0 staging manifest."""
    root = data_root.resolve()
    if not root.is_dir():
        raise IntakeError("Private data root must exist before validating a staging gallery")
    snapshot_file = _private_file(root, snapshot_path)
    _parse_json(snapshot_file, "M0 staging snapshot")
    try:
        snapshot = CurationSnapshotV1.model_validate_json(snapshot_file.read_bytes())
        file_summary = validate_curation_snapshot_files(snapshot, root)
    except (OSError, TypeError, ValueError, ValidationError) as error:
        raise IntakeError(f"M0 staging snapshot or image bytes failed validation: {error}") from error
    if not 1 <= len(snapshot.items) <= 32:
        raise IntakeError("M0 staging gallery must contain between 1 and 32 approved assets")

    source_registry_path = REPOSITORY_ROOT / "config" / "sources.json"
    work_registry_path = REPOSITORY_ROOT / "config" / "anita-work-terms.json"
    rubric_path = REPOSITORY_ROOT / "config" / "viewpoint-rubric-v1.json"
    source_registry = load_registry(source_registry_path)
    work_registry = _parse_json(work_registry_path, "Repository per-work terms registry")
    if (
        not isinstance(source_registry, dict)
        or not isinstance(source_registry.get("sources"), list)
        or not isinstance(work_registry, dict)
        or not isinstance(work_registry.get("works"), list)
    ):
        raise IntakeError("M0 source/work registry schema is invalid")
    source_map = {
        str(item.get("id")): item
        for item in source_registry["sources"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    work_map = {
        str(item.get("folder_id", "")).casefold(): item
        for item in work_registry["works"]
        if isinstance(item, dict)
    }
    source_registry_hash = sha256_file(source_registry_path)
    work_registry_hash = sha256_file(work_registry_path)
    rubric_hash = sha256_file(rubric_path)
    source_copy, source_copy_hash = _copy_repo_evidence(
        root, source_registry_path, subdirectory="m0-gallery-registry-snapshots"
    )
    work_copy, work_copy_hash = _copy_repo_evidence(
        root, work_registry_path, subdirectory="m0-gallery-registry-snapshots"
    )
    rubric_copy, rubric_copy_hash = _copy_repo_evidence(
        root, rubric_path, subdirectory="m0-gallery-registry-snapshots"
    )
    snapshot_hash = sha256_file(snapshot_file)
    blockers: list[str] = []
    if snapshot.viewpoint_rubric_sha256 != rubric_hash:
        blockers.append("staging_snapshot_viewpoint_rubric_is_stale")
    inputs: dict[str, str] = {
        snapshot_path: snapshot_hash,
        source_copy: source_copy_hash,
        work_copy: work_copy_hash,
        rubric_copy: rubric_copy_hash,
    }
    rows = _candidate_rows(root, [item.asset_id for item in snapshot.items])
    database = root / "metadata.sqlite3"
    if database.is_symlink() or not database.is_file():
        raise IntakeError("The local curation metadata database is missing or unsafe")
    uri = f"file:{quote(str(database.resolve()), safe='/:')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    approved_items: list[dict[str, Any]] = []
    families: set[tuple[str, str]] = set()
    try:
        for item in snapshot.items:
            row = rows.get(item.asset_id)
            if row is None:
                blockers.append(f"staging_asset_missing_from_current_catalog:{item.asset_id}")
                continue
            try:
                metadata = json.loads(row["metadata_json"])
            except (TypeError, json.JSONDecodeError):
                blockers.append(f"staging_asset_metadata_invalid:{item.asset_id}")
                continue
            source = source_map.get(item.source_id)
            if (
                row["source_id"] != item.source_id
                or row["sha256"] != item.sha256
                or row["state"] != "candidate"
                or row["revision"] != item.asset_revision
                or metadata.get("source_registry_sha256") != source_registry_hash
                or item.source_registry_sha256 != source_registry_hash
                or metadata.get("relative_root") is None
                or PurePosixPath(str(metadata.get("relative_root")), str(row["relative_path"])).as_posix().removeprefix("./")
                != item.relative_path
            ):
                blockers.append(f"staging_asset_identity_or_revision_changed:{item.asset_id}")
            review = metadata.get("curation_review")
            labels = metadata.get("labels")
            if (
                row["rights_status"] != "approved"
                or metadata.get("review_status") != "approved"
                or metadata.get("gallery_eligible_candidate") is not True
                or metadata.get("learning_split") != "pilot_development"
                or not isinstance(review, dict)
                or review.get("decision") != "keep"
                or review.get("review_id") != item.review_id
                or review.get("reviewer_id") != item.reviewer_id
                or not isinstance(labels, dict)
                or labels != item.labels.model_dump(mode="json")
                or metadata.get("viewpoint")
                != (item.viewpoint.model_dump(mode="json") if item.viewpoint is not None else None)
                or metadata.get("license_id") != item.license_id
                or metadata.get("attribution_text") != item.attribution
            ):
                blockers.append(f"staging_asset_owner_review_changed:{item.asset_id}")
            if (
                metadata.get("group_id") != item.source_family
                or not item.permissions.local_display
                or not item.permissions.evaluation
                or metadata.get("use_permissions") != item.permissions.model_dump(mode="python")
            ):
                blockers.append(f"staging_asset_family_or_requested_permission_invalid:{item.asset_id}")

            if source is None:
                blockers.append(f"staging_source_missing_from_registry:{item.source_id}")
            else:
                if (
                    source.get("rights_status") != "approved"
                    or source.get("terms_review", {}).get("status") != "approved"
                    or source.get("terms_snapshot_sha256") != item.terms_snapshot_sha256
                    or metadata.get("terms_snapshot_sha256") != item.terms_snapshot_sha256
                ):
                    blockers.append(f"staging_source_rights_or_terms_changed:{item.source_id}")
                try:
                    current_terms_hash = _verify_terms_snapshot(root, source)
                    _, _, terms_inputs = _snapshot_receipt(root, item.source_id, current_terms_hash)
                    for relative, digest in terms_inputs:
                        inputs[relative] = digest
                    permissions = _source_permissions(source, metadata, terms_path=work_registry_path)
                    if (
                        not permissions.get("local_display")
                        or not permissions.get("evaluation")
                        or not item.permissions.local_display
                        or not item.permissions.evaluation
                        or any(
                            allowed and not permissions.get(name, False)
                            for name, allowed in item.permissions.model_dump(mode="python").items()
                        )
                    ):
                        blockers.append(f"staging_source_use_permission_missing:{item.asset_id}")
                except (IntakeError, OSError, ValueError) as error:
                    blockers.append(f"staging_source_terms_invalid:{item.asset_id}:{error}")

            if item.source_id == "anita-dataset":
                work_id = str(item.source_work_id or "").casefold()
                work = work_map.get(work_id)
                if (
                    not work
                    or item.source_work_terms_sha256 != work_registry_hash
                    or metadata.get("anita_work_terms_sha256") != work_registry_hash
                    or metadata.get("work_id") != item.source_work_id
                    or work.get("rights_status") != "approved"
                    or work.get("reviewer_decision") != "approved"
                    or work.get("terms_snapshot_sha256") != item.terms_snapshot_sha256
                    or work.get("use_permissions", {}).get("local_display") is not True
                    or work.get("use_permissions", {}).get("evaluation") is not True
                ):
                    blockers.append(f"staging_anita_work_permission_missing:{item.asset_id}")

            group_id = str(item.source_family)
            family_rows = connection.execute(
                "SELECT asset_id, sha256, state, metadata_json FROM assets WHERE source_id = ? AND group_id = ? ORDER BY asset_id",
                (item.source_id, group_id),
            ).fetchall()
            assignment = connection.execute(
                "SELECT learning_split, membership_sha256 FROM family_split_assignments WHERE source_id = ? AND group_id = ? AND status = 'active'",
                (item.source_id, group_id),
            ).fetchone()
            family_membership = hashlib.sha256(
                json.dumps(
                    [{"asset_id": member["asset_id"], "sha256": member["sha256"]} for member in family_rows],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            family_splits_valid = bool(family_rows) and all(
                member["state"] != "locked_test"
                and json.loads(member["metadata_json"]).get("learning_split") == "pilot_development"
                for member in family_rows
            )
            if (
                not assignment
                or assignment["learning_split"] != "pilot_development"
                or assignment["membership_sha256"] != family_membership
                or not family_splits_valid
            ):
                blockers.append(f"staging_source_family_assignment_stale:{item.source_id}:{group_id}")
            families.add((item.source_id, group_id))
            inputs[item.relative_path] = item.sha256
            if (
                source is not None
                and metadata.get("rights_status") == "approved"
                and metadata.get("review_status") == "approved"
                and metadata.get("gallery_eligible_candidate") is True
                and item.permissions.local_display
                and item.permissions.evaluation
            ):
                approved_items.append(
                    {
                        "asset_id": item.asset_id,
                        "source_id": item.source_id,
                        "source_work_id": item.source_work_id,
                        "relative_path": item.relative_path,
                        "sha256": item.sha256,
                        "source_family": item.source_family,
                        "license_id": item.license_id,
                        "attribution": item.attribution,
                        "permissions": item.permissions.model_dump(mode="json"),
                    }
                )
    finally:
        connection.close()

    if not approved_items:
        blockers.append("staging_gallery_contains_no_currently_eligible_assets")
    if not families:
        blockers.append("staging_gallery_contains_no_verified_source_families")
    if not blockers and len(approved_items) == len(snapshot.items):
        if manifest_output is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            manifest_output = Path(f"reports/private/m0-staging-gallery-v1-{timestamp}.json")
        manifest_path = manifest_output if manifest_output.is_absolute() else root / manifest_output
        try:
            manifest_relative = manifest_path.resolve(strict=False).relative_to(root).as_posix()
            PurePosixPath(manifest_relative).relative_to("reports/private")
        except ValueError as error:
            raise IntakeError("M0 staging gallery manifest must remain beneath reports/private") from error
        manifest_path = _private_file(root, manifest_relative)
        manifest = {
            "schema_version": 1,
            "artifact_type": "drawable-m0-approved-staging-gallery",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "source_snapshot_path": snapshot_path,
            "source_snapshot_sha256": snapshot_hash,
            "viewpoint_rubric_sha256": rubric_hash,
            "approved_asset_count": len(approved_items),
            "source_family_count": len(families),
            "items": approved_items,
            "gallery_activated": False,
            "training_authorized": False,
            "training_started": False,
            "data_uploaded": False,
        }
        write_json_create(manifest_path, manifest)
        inputs[manifest_relative] = sha256_file(manifest_path)
    else:
        manifest_relative = None
        manifest_path = None

    if report_output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        report_output = Path(f"reports/private/m0-gallery-review-v1-{timestamp}.json")
    report_path = report_output if report_output.is_absolute() else root / report_output
    try:
        report_relative = report_path.resolve(strict=False).relative_to(root).as_posix()
        PurePosixPath(report_relative).relative_to("reports/private")
    except ValueError as error:
        raise IntakeError("M0 gallery validation report must remain beneath reports/private") from error
    report_path = _private_file(root, report_relative)
    report_body = {
        "schema_version": 1,
        "artifact_type": "drawable-m0-small-gallery-review",
        "status": "complete" if not blockers else "blocked",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "snapshot_path": snapshot_path,
        "snapshot_sha256": snapshot_hash,
        "approved_asset_count": len(approved_items),
        "source_family_count": len(families),
        "validation_summary": file_summary,
        "staging_manifest_path": manifest_relative,
        "staging_manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
        "inputs": [
            {"path": path, "sha256": digest} for path, digest in sorted(inputs.items())
        ],
        "blockers": sorted(set(blockers)),
        "gallery_activated": False,
        "training_authorized": False,
        "training_started": False,
        "data_uploaded": False,
    }
    write_json_create(report_path, report_body)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": report_body["status"],
        "report_path": report_relative,
        "report_sha256": sha256_file(report_path),
        "approved_asset_count": len(approved_items),
        "source_family_count": len(families),
        "blockers": report_body["blockers"],
        "gate_report_written": False,
        "gallery_activated": False,
        "training_authorized": False,
        "training_started": False,
        "data_uploaded": False,
    }
    if blockers or manifest_relative is None or manifest_path is None:
        return result

    gate = GateReportV2(
        schema_version=2,
        requirements_revision=REQUIREMENTS_REVISION,
        artifact_type="drawable-gate-report",
        gate_id="small_approved_gallery",
        producer="drawable m0 gallery-report",
        produced_at=datetime.now(UTC),
        checks={
            "rights_rechecked": True,
            "every_asset_reviewed": True,
            "checksums_verified": True,
            "staging_build_valid": True,
        },
        measurements={
            "approved_assets": len(approved_items),
            "source_families": len(families),
        },
        inputs=[
            EvidenceInputV1(path=path, sha256=digest)
            for path, digest in sorted(
                {
                    **inputs,
                    report_relative: sha256_file(report_path),
                }.items()
            )
        ],
        summary=(
            f"Rechecked {len(approved_items)} approved staging assets across {len(families)} current development families; "
            "all source/work permissions and source bytes were verified. The manifest is private and not activated."
        ),
    )
    if gate_output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        gate_output = Path(f"reports/m0-small-approved-gallery-gate-v2-{timestamp}.json")
    gate_path = gate_output if gate_output.is_absolute() else root / gate_output
    try:
        gate_relative = gate_path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as error:
        raise IntakeError("M0 gallery gate report must remain inside the private data root") from error
    gate_path = _private_file(root, gate_relative)
    write_json_create(gate_path, gate.model_dump(mode="json"))
    result.update(
        {
            "gate_report_written": True,
            "gate_report_path": gate_relative,
            "gate_report_sha256": sha256_file(gate_path),
            "staging_manifest_path": manifest_relative,
            "staging_manifest_sha256": sha256_file(manifest_path),
        }
    )
    return result


def build_feasibility_report(
    data_root: Path,
    *,
    ledger_path: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Assemble an immutable M0 status report from the owner's hash-verified ledger."""
    root = data_root.resolve()
    if not root.is_dir():
        raise IntakeError("Private data root must exist before writing an M0 feasibility report")
    if ledger_path is None:
        default_ledger = root / "reports" / "readiness-ledger-v2.json"
        ledger_path = default_ledger if default_ledger.is_file() else None
    resolved_ledger: Path | None = None
    ledger_relative: str | None = None
    inputs: dict[str, str] = {}
    if ledger_path is not None:
        candidate = ledger_path if ledger_path.is_absolute() else root / ledger_path
        try:
            ledger_relative = candidate.resolve(strict=True).relative_to(root).as_posix()
        except (OSError, ValueError) as error:
            raise IntakeError("M0 readiness ledger must be a regular file inside the private data root") from error
        resolved_ledger = _private_file(root, ledger_relative)
        if not resolved_ledger.is_file():
            raise IntakeError("M0 readiness ledger must be a regular private file")
        inputs[ledger_relative] = sha256_file(resolved_ledger)

    try:
        readiness = validate_readiness(root, "m0", resolved_ledger)
    except (OSError, TypeError, ValueError) as error:
        raise IntakeError(f"M0 readiness ledger could not be validated: {error}") from error
    verified = readiness["verified_evidence"]
    blockers = readiness["blockers"]
    gates: list[dict[str, Any]] = []
    for gate_id in STAGES["m0"]:
        evidence = verified.get(gate_id)
        if evidence is None:
            gates.append(
                {
                    "gate_id": gate_id,
                    "status": "blocked",
                    "blocker": blockers.get(gate_id, "owner-reviewed_evidence_missing"),
                    "next_action": M0_NEXT_ACTIONS.get(gate_id, "Produce and review the required gate evidence."),
                }
            )
            continue
        report_relative = str(evidence["report_path"])
        report_path = _private_file(root, report_relative)
        if not report_path.is_file() or sha256_file(report_path) != evidence["sha256"]:
            raise IntakeError(f"Owner-attested M0 evidence changed during report assembly: {gate_id}")
        gate_report = GateReportV2.model_validate_json(report_path.read_bytes())
        inputs[report_relative] = str(evidence["sha256"])
        gates.append(
            {
                "gate_id": gate_id,
                "status": "complete",
                "report_path": report_relative,
                "report_sha256": evidence["sha256"],
                "reviewed_by": evidence["reviewed_by"],
                "summary": gate_report.summary,
                "checks": gate_report.checks,
                "measurements": gate_report.measurements,
                "next_action": None,
            }
        )

    complete = readiness["status"] == "complete"
    body = {
        "schema_version": 2,
        "requirements_revision": REQUIREMENTS_REVISION,
        "artifact_type": "drawable-m0-feasibility",
        "status": "complete" if complete else "blocked",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "stage": "m0",
        "gate_count": len(gates),
        "completed_gate_count": sum(gate["status"] == "complete" for gate in gates),
        "gates": gates,
        "blockers": blockers,
        "inputs": [
            {"path": path, "sha256": digest} for path, digest in sorted(inputs.items())
        ],
        "recommendation": (
            "proceed_to_next_readiness_stage_only" if complete else "complete_blocked_m0_evidence"
        ),
        "training_authorized": False,
        "training_started": False,
        "data_uploaded": False,
    }
    if output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = Path(f"reports/private/feasibility-m0-v2-{timestamp}.json")
    output_path = output if output.is_absolute() else root / output
    try:
        output_relative = output_path.resolve(strict=False).relative_to(root).as_posix()
        PurePosixPath(output_relative).relative_to("reports/private")
    except ValueError as error:
        raise IntakeError("M0 feasibility report must remain beneath reports/private") from error
    output_path = _private_file(root, output_relative)
    write_json_create(output_path, body)
    return {
        "schema_version": 2,
        "requirements_revision": REQUIREMENTS_REVISION,
        "status": body["status"],
        "report_path": output_relative,
        "report_sha256": sha256_file(output_path),
        "completed_gate_count": body["completed_gate_count"],
        "required_gate_count": len(gates),
        "blockers": blockers,
        "training_authorized": False,
        "training_started": False,
        "data_uploaded": False,
    }


def _verify_m0_gate(root: Path, relative: str, gate_id: str) -> tuple[GateReportV2, Path]:
    path = _private_file(root, relative)
    report = GateReportV2.model_validate_json(path.read_bytes())
    if report.gate_id != gate_id:
        raise IntakeError(f"Expected {gate_id} evidence at {relative}")
    expected_checks = {
        "source_yield_review": ("sample_identity_verified", "owner_judgments_complete", "review_time_measured"),
        "ten_feasibility_sequences": ("consent_rechecked", "replay_verified", "render_hashes_verified"),
    }.get(gate_id, ())
    if any(report.checks.get(check) is not True for check in expected_checks):
        raise IntakeError(f"{gate_id} evidence has incomplete required checks")
    for item in report.inputs:
        source = _private_file(root, item.path)
        if not source.is_file() or sha256_file(source) != item.sha256:
            raise IntakeError(f"M0 input changed since {gate_id} report creation: {item.path}")
    return report, path


def _effort_protocol_blockers(
    plan: dict[str, Any],
    protocol: dict[str, Any],
    projection: dict[str, Any],
    replay_recording_ids: set[str],
) -> list[str]:
    blockers: list[str] = []
    if plan.get("protocol_id") != protocol.get("protocol_id"):
        blockers.append("effort_plan_does_not_name_the_authoritative_collection_protocol")
    if plan.get("enabled_assets_target") != protocol.get("enabled_assets_target"):
        blockers.append("enabled_asset_target_differs_from_the_authoritative_collection_protocol")
    screened_target = plan.get("screened_candidates_target")
    screened_minimum = protocol.get("screened_candidates_minimum")
    if type(screened_target) is not int or type(screened_minimum) is not int or screened_target < screened_minimum:
        blockers.append("screened_candidate_target_is_below_the_authoritative_minimum")

    expected_reference_targets: dict[str, int] = {
        f"{style}/{scope}": count
        for style, scopes in protocol.get("reference_gold_targets", {}).items()
        for scope, count in scopes.items()
        if type(count) is int
    }
    observed_reference_targets = {
        str(row.get("stratum_id")): row.get("gold_target")
        for row in plan.get("reference_strata", [])
        if isinstance(row, dict) and isinstance(row.get("stratum_id"), str)
    }
    if observed_reference_targets != expected_reference_targets:
        missing = sorted(set(expected_reference_targets) - set(observed_reference_targets))
        extra = sorted(set(observed_reference_targets) - set(expected_reference_targets))
        changed = sorted(
            key
            for key in expected_reference_targets.keys() & observed_reference_targets.keys()
            if expected_reference_targets[key] != observed_reference_targets[key]
        )
        if missing:
            blockers.append(f"reference_gold_strata_missing:{','.join(missing)}")
        if extra:
            blockers.append(f"reference_gold_strata_unplanned:{','.join(extra)}")
        if changed:
            blockers.append(f"reference_gold_strata_targets_changed:{','.join(changed)}")

    expected_query_counts = Counter(
        {
            (row["partition"], row["query_class"]): row["count"]
            for row in protocol.get("query_drawing_targets", [])
        }
    )
    observed_query_counts: Counter[tuple[str, str]] = Counter()
    for row in plan.get("drawings", []):
        if isinstance(row, dict) and row.get("split") in {
            "training", "model_selection", "threshold", "calibration_audit", "locked_test"
        }:
            observed_query_counts[(str(row["split"]), str(row.get("query_class")))] += 1
    if observed_query_counts != expected_query_counts:
        blockers.append("query_drawing_partition_counts_do_not_match_the_authoritative_protocol")

    drawings = [row for row in plan.get("drawings", []) if isinstance(row, dict)]
    pilot_target = protocol.get("pilot_development_drawings")
    pilot_count = sum(row.get("split") == "pilot" for row in drawings)
    if type(pilot_target) is not int or pilot_count != pilot_target:
        blockers.append("pilot_development_drawing_count_does_not_match_the_authoritative_protocol")
    replay_target = protocol.get("m0_feasibility_recordings")
    replay_rows = [row for row in drawings if row.get("drawing_id") in replay_recording_ids]
    represented_ids = {str(row["drawing_id"]) for row in replay_rows}
    if type(replay_target) is not int or len(replay_recording_ids) != replay_target or represented_ids != replay_recording_ids:
        blockers.append("m0_recordings_are_not_reconciled_once_in_the_unique_drawing_plan")
    for row in replay_rows:
        drawing_id = str(row["drawing_id"])
        reused = drawing_id in row.get("reused_evidence", [])
        if (row.get("split") == "pilot") != reused or row.get("split") not in {"pilot", "m0"}:
            blockers.append(f"m0_recording_reuse_classification_invalid:{drawing_id}")
    if any(
        value not in replay_recording_ids
        for row in drawings
        for value in row.get("reused_evidence", [])
    ):
        blockers.append("drawing_plan_references_unknown_m0_replay_evidence")

    for purpose, minimum in protocol.get("judgment_pair_minimums", {}).items():
        observed = projection["judgment_pairs_by_purpose"].get(purpose, {}).get(
            "unique_query_candidate_pairs", 0
        )
        if type(minimum) is not int or observed < minimum:
            blockers.append(f"judgment_pair_minimum_not_met:{purpose}:{observed}/{minimum}")
    return sorted(set(blockers))


def build_human_effort_report(
    data_root: Path,
    *,
    effort_log: str,
    source_yield_report: str,
    source_yield_gate: str,
    replay_report: str,
    replay_gate: str,
    effort_plan: Path,
    five_hour_checkpoint_reviewed: bool,
    annotation_stopped_at_budget: bool,
    output: Path | None = None,
    gate_output: Path | None = None,
) -> dict[str, Any]:
    """Measure current owner effort/storage and project the versioned collection plan."""
    root = data_root.resolve()
    if not root.is_dir():
        raise IntakeError("Private data root must exist before measuring M0 storage")
    effort_path = _private_file(root, effort_log)
    effort_value = _parse_json(effort_path, "Imported active-effort log")
    effort_summary = _validate_effort_log(effort_value)

    yield_path = _private_file(root, source_yield_report)
    source_yield = _parse_json(yield_path, "Source-yield review report")
    if source_yield.get("status") != "complete" or not isinstance(source_yield.get("judgments"), list):
        raise IntakeError("Source-yield review report must be complete before effort projection")
    yield_gate, yield_gate_path = _verify_m0_gate(root, source_yield_gate, "source_yield_review")
    yield_gate_inputs = {item.path: item.sha256 for item in yield_gate.inputs}
    if (
        source_yield_report not in yield_gate_inputs
        or yield_gate_inputs[source_yield_report] != sha256_file(yield_path)
    ):
        raise IntakeError("Source-yield gate report does not bind the supplied human-review report")
    source_profile_hash = yield_gate.profile_hashes.get("source_selection")
    if source_profile_hash is None:
        raise IntakeError("Source-yield gate report is missing the selected-source profile identity")

    replay_path = _private_file(root, replay_report)
    replay = _parse_json(replay_path, "Independent recording replay report")
    if replay.get("status") != "verified" or not isinstance(replay.get("outcomes"), list):
        raise IntakeError("Independent replay report must verify before effort projection")
    replay_gate_body, replay_gate_path = _verify_m0_gate(root, replay_gate, "ten_feasibility_sequences")
    replay_gate_inputs = {item.path: item.sha256 for item in replay_gate_body.inputs}
    if replay_report not in replay_gate_inputs or replay_gate_inputs[replay_report] != sha256_file(replay_path):
        raise IntakeError("Ten-sequence gate report does not bind the supplied replay report")
    outcomes = replay["outcomes"]
    if len(outcomes) != 10 or any(row.get("status") != "verified" for row in outcomes):
        raise IntakeError("Effort projection requires ten independently verified feasibility recordings")
    replay_recording_ids = {
        str(row.get("recording_id"))
        for row in outcomes
        if isinstance(row.get("recording_id"), str) and row.get("recording_id")
    }
    if len(replay_recording_ids) != 10:
        raise IntakeError("Verified replay report must contain ten distinct recording identities")

    plan_path = effort_plan if effort_plan.is_absolute() else root / effort_plan
    try:
        plan_relative = plan_path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as error:
        raise IntakeError("Effort plan must remain inside the private data root") from error
    plan_file = _private_file(root, plan_relative)
    plan = _parse_json(plan_file, "Versioned effort plan")
    protocol_source = REPOSITORY_ROOT / "config" / "recipes" / "full-program-effort-v3.json"
    protocol_relative, protocol_digest = _copy_repo_evidence(
        root, protocol_source, subdirectory="m0-effort-protocols"
    )
    protocol_file = _private_file(root, protocol_relative)
    protocol = _parse_json(protocol_file, "Authoritative full-program effort protocol")
    try:
        projection, projection_blockers = project_effort_plan(plan)
    except (TypeError, ValueError) as error:
        raise IntakeError(f"Versioned effort plan is invalid: {error}") from error
    projection_blockers.extend(
        _effort_protocol_blockers(plan, protocol, projection, replay_recording_ids)
    )

    plan_evidence: dict[str, str] = {}
    raw_evidence = plan.get("evidence_inputs", [])
    if not isinstance(raw_evidence, list):
        raise IntakeError("Effort plan evidence_inputs must be an array")
    for item in raw_evidence:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise IntakeError("Effort plan contains a malformed evidence input")
        evidence_path = _private_file(root, item["path"])
        digest = item.get("sha256")
        if not isinstance(digest, str) or sha256_file(evidence_path) != digest:
            raise IntakeError(f"Effort plan evidence changed or is missing: {item['path']}")
        plan_evidence[item["path"]] = digest

    measured_rows: list[dict[str, Any]] = []
    for key in ("reference_strata", "drawing_rates", "judgment_rates", "second_reviews", "adjudications"):
        rows = plan.get(key, [])
        if isinstance(rows, list):
            measured_rows.extend(row for row in rows if isinstance(row, dict) and row.get("source") == "measured")
    non_gold = plan.get("non_gold_acceptance")
    if isinstance(non_gold, dict) and non_gold.get("source") == "measured":
        measured_rows.append(non_gold)
    for row in measured_rows:
        evidence_path_value = row.get("evidence_path")
        if not isinstance(evidence_path_value, str) or evidence_path_value not in plan_evidence:
            raise IntakeError("Every measured effort rate must bind an item in effort-plan evidence_inputs")

    free = shutil.disk_usage(root)
    projected_storage_bytes = projection["projected_storage_bytes"]
    storage_headroom_bytes = free.free - projected_storage_bytes
    required_headroom_bytes = int(free.total * 0.20)
    enough_headroom = projected_storage_bytes > 0 and storage_headroom_bytes >= required_headroom_bytes
    annotation_ms = effort_summary["annotation_effort_milliseconds"]
    annotation_minutes = annotation_ms // 60_000
    checkpoint_valid = annotation_ms < 300 * 60_000 or five_hour_checkpoint_reviewed
    hard_stop_valid = annotation_ms < 600 * 60_000 or annotation_stopped_at_budget
    time_budget_valid = annotation_ms <= 600 * 60_000
    blockers = list(projection_blockers)
    if not enough_headroom:
        blockers.append("projected_download_expansion_and_build_storage_would_breach_20_percent_free_headroom")
    if not checkpoint_valid:
        blockers.append("five_hour_checkpoint_review_is_required_at_or_after_300_annotation_minutes")
    if not hard_stop_valid or not time_budget_valid:
        blockers.append("annotation_must_stop_by_600_minutes_and_record_the_hard_stop")

    projected_hours = projection["projected_full_program_hours"]["base"]
    if output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = Path(f"reports/private/m0-human-effort-storage-v2-{timestamp}.json")
    output_path = output if output.is_absolute() else root / output
    try:
        output_relative = output_path.resolve(strict=False).relative_to(root).as_posix()
        PurePosixPath(output_relative).relative_to("reports/private")
    except ValueError as error:
        raise IntakeError("Human-effort report must remain beneath reports/private") from error
    output_path = _private_file(root, output_relative)
    report_body = {
        "schema_version": 2,
        "requirements_revision": REQUIREMENTS_REVISION,
        "artifact_type": "drawable-m0-human-effort-storage",
        "status": "complete" if not blockers else "blocked",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "effort_plan_path": plan_relative,
        "effort_plan_sha256": sha256_file(plan_file),
        "effort_protocol_path": protocol_relative,
        "effort_protocol_sha256": protocol_digest,
        "effort_log_path": effort_log,
        "effort_log_sha256": sha256_file(effort_path),
        "effort_chunks": effort_summary["chunks"],
        "total_owner_effort_minutes": effort_summary["active_effort_minutes"],
        "owner_effort_milliseconds_by_category": effort_summary["effort_milliseconds_by_category"],
        "pilot_annotation_minutes": annotation_minutes,
        "pilot_annotation_milliseconds": annotation_ms,
        "five_hour_checkpoint_reviewed": five_hour_checkpoint_reviewed,
        "annotation_stopped_at_budget": annotation_stopped_at_budget,
        "storage": {
            "total_bytes": free.total,
            "free_bytes": free.free,
            "projected_program_bytes": projected_storage_bytes,
            "free_bytes_after_projection": storage_headroom_bytes,
            "required_headroom_bytes": required_headroom_bytes,
            "disk_free_gib": free.free / (1024**3),
            "storage_headroom_checked": enough_headroom,
        },
        "source_yield_report_path": source_yield_report,
        "source_yield_report_sha256": sha256_file(yield_path),
        "source_yield_gate_path": source_yield_gate,
        "source_yield_gate_sha256": sha256_file(yield_gate_path),
        "recording_replay_report_path": replay_report,
        "recording_replay_report_sha256": sha256_file(replay_path),
        "recording_gate_path": replay_gate,
        "recording_gate_sha256": sha256_file(replay_gate_path),
        "source_selection_profile_sha256": source_profile_hash,
        "projection": projection,
        "blockers": sorted(set(blockers)),
        "training_started": False,
        "data_uploaded": False,
    }
    write_json_create(output_path, report_body)
    result: dict[str, Any] = {
        "schema_version": 2,
        "status": report_body["status"],
        "requirements_revision": REQUIREMENTS_REVISION,
        "report_path": output_relative,
        "report_sha256": sha256_file(output_path),
        "pilot_annotation_minutes": annotation_minutes,
        "disk_free_gib": free.free / (1024**3),
        "projected_storage_bytes": projected_storage_bytes,
        "projected_review_hours_for_full_program": projected_hours,
        "hours_low_base_high": projection["projected_full_program_hours"],
        "unique_drawings": projection["unique_drawings"],
        "development_calibration_judgment_pairs": projection["development_calibration_judgment_pairs"],
        "final_unique_query_candidate_pairs": projection["final_unique_query_candidate_pairs"],
        "blockers": report_body["blockers"],
        "gate_report_written": False,
        "training_started": False,
        "data_uploaded": False,
    }
    if blockers:
        return result

    input_hashes: dict[str, str] = {
        effort_log: sha256_file(effort_path),
        source_yield_report: sha256_file(yield_path),
        source_yield_gate: sha256_file(yield_gate_path),
        replay_report: sha256_file(replay_path),
        replay_gate: sha256_file(replay_gate_path),
        plan_relative: sha256_file(plan_file),
        protocol_relative: protocol_digest,
        output_relative: sha256_file(output_path),
        **plan_evidence,
    }
    gate = GateReportV2(
        schema_version=2,
        requirements_revision=REQUIREMENTS_REVISION,
        artifact_type="drawable-gate-report",
        gate_id="human_effort_storage",
        producer="drawable m0 effort-report",
        produced_at=datetime.now(UTC),
        checks={
            "active_effort_measured": True,
            "storage_headroom_checked": enough_headroom,
            "five_hour_checkpoint_completed": five_hour_checkpoint_reviewed,
            "annotation_stopped_at_budget": annotation_stopped_at_budget,
        },
        measurements={
            "pilot_annotation_minutes": annotation_minutes,
            "active_effort_milliseconds": annotation_ms,
            "disk_free_gib": free.free / (1024**3),
            "projected_review_hours_for_full_program": projected_hours,
            "projected_storage_bytes": projected_storage_bytes,
            "unique_drawings": projection["unique_drawings"],
            "development_calibration_judgment_pairs": projection["development_calibration_judgment_pairs"],
            "final_unique_query_candidate_pairs": projection["final_unique_query_candidate_pairs"],
        },
        profile_hashes={"source_selection": source_profile_hash},
        inputs=[EvidenceInputV1(path=key, sha256=value) for key, value in sorted(input_hashes.items())],
        summary=(
            f"Measured {annotation_minutes} annotation minutes and projected {projected_hours:.1f} base owner-hours "
            f"({projection['projected_full_program_hours']['low']:.1f}–{projection['projected_full_program_hours']['high']:.1f} planning range) "
            f"from {projection['unique_drawings']} unique drawings and deduplicated judgment pools."
        ),
    )
    if gate_output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        gate_output = Path(f"reports/m0-human-effort-storage-gate-v2-{timestamp}.json")
    gate_path = gate_output if gate_output.is_absolute() else root / gate_output
    try:
        gate_relative = gate_path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as error:
        raise IntakeError("Human-effort gate report must remain inside the private data root") from error
    gate_path = _private_file(root, gate_relative)
    write_json_create(gate_path, gate.model_dump(mode="json"))
    result.update(
        {
            "gate_report_written": True,
            "gate_report_path": gate_relative,
            "gate_report_sha256": sha256_file(gate_path),
        }
    )
    return result


def _private_file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or "\\" in relative or any(part in {"", ".", ".."} for part in path.parts):
        raise IntakeError("M0 evidence paths must be normalized private-root-relative paths")
    candidate = ensure_inside(root, relative)
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            raise IntakeError("M0 evidence paths may not traverse symbolic links")
        cursor = cursor.parent
    return candidate


def _parse_json(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise IntakeError(f"{label} must be a regular private file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IntakeError(f"{label} is not valid UTF-8 JSON") from error


def _candidate_rows(root: Path, asset_ids: list[str]) -> dict[str, sqlite3.Row]:
    database = root / "metadata.sqlite3"
    if database.is_symlink() or not database.is_file():
        raise IntakeError("The local curation metadata database is missing or unsafe")
    uri = f"file:{quote(str(database.resolve()), safe='/:')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        result: dict[str, sqlite3.Row] = {}
        for offset in range(0, len(asset_ids), 400):
            portion = asset_ids[offset : offset + 400]
            marks = ",".join("?" for _ in portion)
            rows = connection.execute(
                f"SELECT asset_id, source_id, relative_path, sha256, byte_length, state, metadata_json FROM assets WHERE asset_id IN ({marks})",
                portion,
            ).fetchall()
            result.update({str(row["asset_id"]): row for row in rows})
        return result
    finally:
        connection.close()


def build_source_yield_report(
    data_root: Path,
    *,
    sample_path: str = "reports/source-yield-sample-pilot-v2.json",
    output: Path | None = None,
    gate_output: Path | None = None,
) -> dict[str, Any]:
    """Freeze fixed-sample judgments and emit a gate report only when all selected items are reviewed."""
    root = data_root.resolve()
    sample_relative = sample_path
    sample_file = _private_file(root, sample_relative)
    sample = _parse_json(sample_file, "Fixed source-yield sample")
    if (
        not isinstance(sample, dict)
        or sample.get("schema_version") != 2
        or sample.get("recipe_id") != "pilot-v1"
        or not isinstance(sample.get("selected"), list)
        or not isinstance(sample.get("sources"), list)
        or type(sample.get("selected_count")) is not int
        or sample["selected_count"] != len(sample["selected"])
        or not 1 <= sample["selected_count"] <= 250
    ):
        raise IntakeError("Fixed source-yield sample identity or selected count is invalid")

    rubric_path = REPOSITORY_ROOT / "config" / "viewpoint-rubric-v1.json"
    rubric_hash = sha256_file(rubric_path)
    source_snapshot_relative = sample.get("source_registry_snapshot_path")
    work_snapshot_relative = sample.get("anita_work_terms_snapshot_path")
    if not isinstance(source_snapshot_relative, str) or not isinstance(work_snapshot_relative, str):
        raise IntakeError("Fixed source-yield sample is missing its selected registry snapshots")
    source_snapshot_path = _private_file(root, source_snapshot_relative)
    work_snapshot_path = _private_file(root, work_snapshot_relative)
    registry_hash = sha256_file(source_snapshot_path)
    work_terms_hash = sha256_file(work_snapshot_path)
    if (
        sample.get("source_registry_sha256") != registry_hash
        or sample.get("anita_work_terms_sha256") != work_terms_hash
    ):
        raise IntakeError("Fixed source-yield sample selected registry snapshot changed")
    active_rubric = viewpoint_rubric()
    if active_rubric.get("rubric_sha256") != rubric_hash:
        raise IntakeError("Active viewpoint rubric fingerprint is inconsistent")

    selected = sample["selected"]
    identifiers: set[str] = set()
    for item in selected:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("asset_id"), str)
            or not item["asset_id"]
            or not isinstance(item.get("source_id"), str)
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
            or item["asset_id"] in identifiers
        ):
            raise IntakeError("Fixed source-yield sample contains malformed or duplicate identities")
        identifiers.add(item["asset_id"])

    rows = _candidate_rows(root, [str(item["asset_id"]) for item in selected])
    sample_hash = sha256_file(sample_file)
    profile_relative = sample.get("source_selection_profile_path")
    profile_digest = sample.get("source_selection_profile_sha256")
    if not isinstance(profile_relative, str) or not isinstance(profile_digest, str):
        raise IntakeError("Fixed source-yield sample is missing its selected-source identity")
    profile_file = _private_file(root, profile_relative)
    if sha256_file(profile_file) != profile_digest:
        raise IntakeError("Fixed source-yield sample selected-source profile changed")
    try:
        profile = SourceSelectionProfileV1.model_validate_json(profile_file.read_bytes())
        source_snapshot = _parse_json(source_snapshot_path, "Selected source registry snapshot")
        work_snapshot = _parse_json(work_snapshot_path, "Selected work registry snapshot")
    except (ValidationError, ValueError) as error:
        raise IntakeError("Fixed source-yield profile or selected registry snapshot is invalid") from error
    if (
        profile.profile_id != sample.get("source_selection_profile_id")
        or not isinstance(source_snapshot, dict)
        or not isinstance(source_snapshot.get("rows"), list)
        or {row.get("id") for row in source_snapshot["rows"] if isinstance(row, dict)}
        != {item.source_id for item in profile.sources}
        or not isinstance(work_snapshot, dict)
        or not isinstance(work_snapshot.get("rows"), list)
        or {str(row.get("work_id", row.get("folder_id", ""))) for row in work_snapshot["rows"] if isinstance(row, dict)}
        != {work_id for item in profile.sources for work_id in item.work_ids}
    ):
        raise IntakeError("Fixed source-yield profile does not match its selected registry snapshots")
    observations: list[dict[str, Any]] = []
    blockers: list[str] = []
    input_hashes: dict[str, str] = {
        sample_relative: sample_hash,
        profile_relative: profile_digest,
        source_snapshot_relative: registry_hash,
        work_snapshot_relative: work_terms_hash,
    }
    for evidence in sample.get("profile_evidence", []):
        if not isinstance(evidence, dict) or not isinstance(evidence.get("path"), str):
            raise IntakeError("Fixed source-yield sample contains malformed profile evidence")
        evidence_path = _private_file(root, evidence["path"])
        evidence_digest = evidence.get("sha256")
        if not isinstance(evidence_digest, str) or sha256_file(evidence_path) != evidence_digest:
            raise IntakeError("Selected-source acquisition evidence changed")
        input_hashes[evidence["path"]] = evidence_digest
    for item in selected:
        asset_id = item["asset_id"]
        row = rows.get(asset_id)
        if row is None:
            blockers.append(f"sample_asset_missing:{asset_id}")
            continue
        try:
            metadata = json.loads(row["metadata_json"])
        except (TypeError, json.JSONDecodeError):
            blockers.append(f"sample_asset_metadata_invalid:{asset_id}")
            continue
        if (
            row["source_id"] != item["source_id"]
            or row["sha256"] != item["sha256"]
            or metadata.get("sha256") != item["sha256"]
        ):
            blockers.append(f"sample_asset_identity_changed:{asset_id}")
            continue
        relative_path = item.get("relative_path")
        relative_root = metadata.get("relative_root")
        if not isinstance(relative_path, str) or not isinstance(relative_root, str):
            blockers.append(f"sample_asset_path_missing:{asset_id}")
            continue
        expected_path = PurePosixPath(relative_root, str(row["relative_path"])).as_posix().removeprefix("./")
        if relative_path != expected_path:
            blockers.append(f"sample_asset_path_changed:{asset_id}")
            continue
        asset_path = _private_file(root, relative_path)
        if (
            not asset_path.is_file()
            or asset_path.stat().st_size != row["byte_length"]
            or sha256_file(asset_path) != item["sha256"]
        ):
            blockers.append(f"sample_asset_bytes_changed:{asset_id}")
            continue
        input_hashes[relative_path] = item["sha256"]
        yield_review = metadata.get("source_yield_review")
        curation_review = metadata.get("curation_review")
        viewpoint_data = metadata.get("viewpoint")
        labels = metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}
        complete = True
        if (
            not isinstance(yield_review, dict)
            or yield_review.get("schema_version") != 1
            or yield_review.get("sample_sha256") != sample_hash
            or yield_review.get("asset_sha256") != item["sha256"]
            or yield_review.get("useful_eye_crop") not in {"yes", "no", "uncertain"}
        ):
            blockers.append(f"eye_crop_yield_judgment_missing_or_stale:{asset_id}")
            complete = False
        review_seconds = yield_review.get("review_seconds") if isinstance(yield_review, dict) else None
        if not (
            isinstance(review_seconds, (int, float))
            and not isinstance(review_seconds, bool)
            and math.isfinite(review_seconds)
            and 0 < review_seconds <= 86_400
        ):
            blockers.append(f"review_time_missing_or_invalid:{asset_id}")
            complete = False
        try:
            viewpoint = ViewpointReviewV1.model_validate(viewpoint_data)
            validate_viewpoint_rubric(viewpoint, rubric_hash)
            if viewpoint.review_state not in {"reviewed", "not_applicable"}:
                raise ValueError("viewpoint was not reviewed")
            if viewpoint.reviewer_id != yield_review.get("reviewer_id"):
                raise ValueError("viewpoint reviewer does not match the source-yield reviewer")
        except (ValidationError, ValueError, AttributeError):
            viewpoint = None
            blockers.append(f"viewpoint_judgment_missing_or_invalid:{asset_id}")
            complete = False
        if (
            not isinstance(curation_review, dict)
            or curation_review.get("reviewer_id") != (yield_review or {}).get("reviewer_id")
            or curation_review.get("decision") not in {"keep", "reject", "quarantine"}
        ):
            blockers.append(f"owner_review_record_missing:{asset_id}")
            complete = False
        observation = {
            "asset_id": asset_id,
            "source_id": item["source_id"],
            "asset_path": relative_path,
            "asset_sha256": item["sha256"],
            "reviewed": complete,
            "decision": curation_review.get("decision") if isinstance(curation_review, dict) else None,
            "gallery_eligible_candidate": metadata.get("gallery_eligible_candidate") is True,
            "reviewer_id": yield_review.get("reviewer_id") if isinstance(yield_review, dict) else None,
            "review_seconds": review_seconds,
            "useful_eye_crop": yield_review.get("useful_eye_crop") if isinstance(yield_review, dict) else None,
            "viewpoint_state": viewpoint.review_state if viewpoint else None,
            "viewpoint_identifiable": bool(
                viewpoint
                and viewpoint.review_state == "reviewed"
                and "indeterminate" not in (
                    viewpoint.horizontal_view,
                    viewpoint.facing,
                    viewpoint.elevation,
                )
            ),
            "viewpoint_ambiguous": bool(
                viewpoint
                and viewpoint.review_state == "reviewed"
                and "indeterminate" in (
                    viewpoint.horizontal_view,
                    viewpoint.facing,
                    viewpoint.elevation,
                )
            ),
            "labels_review_seconds": labels.get("review_seconds"),
        }
        observations.append(observation)

    measurements_ready = len(observations) == len(selected) and not blockers
    times = [float(item["review_seconds"]) for item in observations if type(item["review_seconds"]) in {int, float}]
    if measurements_ready and len(times) != len(selected):
        blockers.append("not_all_sample_review_times_measured")
        measurements_ready = False
    source_totals: list[dict[str, Any]] = []
    for source_row in sample["sources"]:
        if not isinstance(source_row, dict):
            continue
        source_id = source_row.get("source_id")
        source_observations = [item for item in observations if item["source_id"] == source_id]
        source_totals.append(
            {
                "source_id": source_id,
                "sample_target": source_row.get("target"),
                "sample_status": source_row.get("status"),
                "selected_count": len(source_observations),
                "reviewed_count": sum(item["reviewed"] for item in source_observations),
                "source_blockers": source_row.get("blockers", []),
            }
        )
    report_body = {
        "schema_version": 2,
        "requirements_revision": REQUIREMENTS_REVISION,
        "artifact_type": "drawable-m0-source-yield-review",
        "status": "complete" if measurements_ready else "blocked",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "recipe_id": sample["recipe_id"],
        "sample_path": sample_relative,
        "sample_sha256": sample_hash,
        "source_registry_sha256": registry_hash,
        "anita_work_terms_sha256": work_terms_hash,
        "viewpoint_rubric_sha256": rubric_hash,
        "selected_count": len(selected),
        "reviewed_count": len(observations),
        "sources": source_totals,
        "judgments": observations,
        "blockers": sorted(set(blockers)),
        "all_selected_sample_items_reviewed": measurements_ready,
        "network_used": False,
        "training_started": False,
        "data_uploaded": False,
    }
    if output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = Path(f"reports/private/source-yield-review-v2-{timestamp}.json")
    report_destination = output if output.is_absolute() else root / output
    try:
        report_relative = report_destination.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as error:
        raise IntakeError("Source-yield report must remain inside the private data root") from error
    report_destination = _private_file(root, report_relative)
    write_json_create(report_destination, report_body)
    response: dict[str, Any] = {
        "schema_version": 1,
        "status": report_body["status"],
        "report_path": report_relative,
        "report_sha256": sha256_file(report_destination),
        "selected_count": len(selected),
        "reviewed_count": len(observations),
        "blockers": report_body["blockers"],
        "gate_report_written": False,
        "training_started": False,
        "data_uploaded": False,
    }
    if not measurements_ready:
        return response

    report_relative_path = PurePosixPath(report_relative)
    try:
        report_relative_path.relative_to("reports/private")
    except ValueError as error:
        raise IntakeError("Source-yield review reports must be stored beneath reports/private") from error
    median_seconds = float(median(times))
    gate = GateReportV2(
        schema_version=2,
        requirements_revision=REQUIREMENTS_REVISION,
        artifact_type="drawable-gate-report",
        gate_id="source_yield_review",
        producer="drawable m0 source-yield-report",
        produced_at=datetime.now(UTC),
        checks={
            "sample_identity_verified": True,
            "owner_judgments_complete": True,
            "review_time_measured": True,
        },
        measurements={
            "reviewed_candidates": len(observations),
            "median_review_seconds_per_candidate": median_seconds,
            "useful_eye_crop_candidates": sum(item["useful_eye_crop"] == "yes" for item in observations),
            "viewpoint_identifiable_candidates": sum(item["viewpoint_identifiable"] for item in observations),
            "ambiguous_viewpoint_candidates": sum(item["viewpoint_ambiguous"] for item in observations),
        },
        profile_hashes={"source_selection": profile_digest},
        inputs=[
            EvidenceInputV1(path=relative, sha256=digest)
            for relative, digest in sorted(
                {
                    **input_hashes,
                    report_relative: sha256_file(report_destination),
                }.items()
            )
        ],
        summary=(
            f"Owner reviewed all {len(observations)} assets in the immutable pilot-v1 sample. "
            f"The fixed sample selected {len(selected)} items across {len(source_totals)} source strata; "
            "source-level shortfalls remain reported and were not filled by changing the sample."
        ),
    )
    if gate_output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        gate_output = Path(f"reports/m0-source-yield-gate-v2-{timestamp}.json")
    gate_destination = gate_output if gate_output.is_absolute() else root / gate_output
    try:
        gate_relative = gate_destination.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as error:
        raise IntakeError("Source-yield gate report must remain inside the private data root") from error
    gate_destination = _private_file(root, gate_relative)
    write_json_create(gate_destination, gate.model_dump(mode="json"))
    response.update(
        {
            "status": "complete",
            "gate_report_written": True,
            "gate_report_path": gate_relative,
            "gate_report_sha256": sha256_file(gate_destination),
            "measurements": gate.measurements,
        }
    )
    return response
