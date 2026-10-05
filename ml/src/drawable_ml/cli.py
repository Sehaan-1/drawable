from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from drawable_research_contracts import CurationSnapshotV1, validate_curation_snapshot_files
from pydantic import ValidationError

from .bundle import inspect_training_bundle
from .contracts import DatasetManifest
from .dataset import load_dataset, package_training_bundle
from .feasibility import measure_frozen_cpu
from .io import MLError, resolve_private_path, write_json_atomic
from .models import (
    capture_model_terms_snapshot,
    create_model_artifact_manifest,
    record_model_terms_review,
    verify_model_registry,
)
from .readiness import SPECS, STAGES, attest_gate, readiness_report
from .recipes import load_recipe

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path(os.environ.get("DRAWABLE_DATA_ROOT", "datasets")).resolve()
DEFAULT_SOURCE_REGISTRY = REPOSITORY_ROOT / "config" / "sources.json"
DEFAULT_MODEL_REGISTRY = REPOSITORY_ROOT / "config" / "models.json"
DEFAULT_RECIPE = REPOSITORY_ROOT / "ml" / "recipes" / "pilot-v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="drawable-ml",
        description="Offline dataset, model-artifact, and training-readiness checks (no training entry point).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    readiness = commands.add_parser(
        "readiness", help="validate evidence and report gated milestone status"
    )
    readiness_commands = readiness.add_subparsers(dest="readiness_command", required=True)
    readiness_report_parser = readiness_commands.add_parser("report")
    _add_data_root(readiness_report_parser)
    readiness_report_parser.add_argument(
        "--stage", choices=tuple(STAGES), default="training_preparation"
    )
    readiness_report_parser.add_argument("--ledger", type=Path)
    readiness_report_parser.add_argument("--output", type=Path)
    readiness_report_parser.set_defaults(handler=_readiness_report)

    readiness_attest = readiness_commands.add_parser(
        "attest", help="owner-review one immutable, hash-bound readiness measurement report"
    )
    _add_data_root(readiness_attest)
    readiness_attest.add_argument(
        "--gate", choices=tuple(spec.gate_id for spec in SPECS), required=True
    )
    readiness_attest.add_argument("--report", type=Path, required=True)
    readiness_attest.add_argument("--reviewer-id", required=True)
    readiness_attest.add_argument("--review-note", required=True)
    readiness_attest.add_argument("--ledger", type=Path)
    readiness_attest.add_argument("--output", type=Path, required=True)
    readiness_attest.set_defaults(handler=_readiness_attest)

    models = commands.add_parser(
        "models", help="verify local model artifact hashes without loading weights"
    )
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_verify = model_commands.add_parser("verify")
    _add_data_root(model_verify)
    model_verify.add_argument("--registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    model_verify.add_argument("--purpose", choices=("inference", "training"), default="inference")
    model_verify.set_defaults(handler=_models_verify)
    model_manifest = model_commands.add_parser(
        "manifest", help="hash manually staged pinned files without loading or approving a model"
    )
    _add_data_root(model_manifest)
    model_manifest.add_argument("--registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    model_manifest.add_argument("--model-id", required=True)
    model_manifest.set_defaults(handler=_models_manifest)
    model_terms_import = model_commands.add_parser(
        "terms-import", help="import a saved official terms page without network access or approval"
    )
    _add_data_root(model_terms_import)
    model_terms_import.add_argument("--registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    model_terms_import.add_argument("--model-id", required=True)
    model_terms_import.add_argument("--purpose", choices=("inference", "training"), required=True)
    model_terms_import.add_argument("--source-url", required=True)
    model_terms_import.add_argument("--input", type=Path, required=True)
    model_terms_import.set_defaults(handler=_models_terms_import)
    model_terms_review = model_commands.add_parser(
        "terms-review", help="record an explicit owner decision for captured model terms"
    )
    _add_data_root(model_terms_review)
    model_terms_review.add_argument("--registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    model_terms_review.add_argument("--model-id", required=True)
    model_terms_review.add_argument("--purpose", choices=("inference", "training"), required=True)
    model_terms_review.add_argument("--snapshot-sha256", required=True)
    model_terms_review.add_argument("--decision", choices=("approved", "rejected"), required=True)
    model_terms_review.add_argument("--reviewer-id", required=True)
    model_terms_review.add_argument("--review-note", required=True)
    model_terms_review.add_argument("--replace-existing", action="store_true")
    model_terms_review.set_defaults(handler=_models_terms_review)
    model_measure = model_commands.add_parser(
        "measure-cpu", help="measure reviewed frozen image encoders on CPU; no training or GPU probe"
    )
    _add_data_root(model_measure)
    model_measure.add_argument("--snapshot", type=Path, required=True)
    model_measure.add_argument("--registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    model_measure.add_argument("--source-registry", type=Path, default=DEFAULT_SOURCE_REGISTRY)
    model_measure.add_argument(
        "--source-profile",
        type=Path,
        required=True,
        help="Selected-source profile stored beneath the private data root",
    )
    model_measure.add_argument(
        "--anita-terms", type=Path, default=DEFAULT_SOURCE_REGISTRY.with_name("anita-work-terms.json")
    )
    model_measure.add_argument("--repeats", type=int, default=5)
    model_measure.add_argument("--output", type=Path, default=Path("reports/m0-frozen-cpu-forward-v2.json"))
    model_measure.set_defaults(handler=_models_measure_cpu)

    curation = commands.add_parser(
        "curation", help="validate immutable local review snapshots; never trains or uploads"
    )
    curation_commands = curation.add_subparsers(dest="curation_command", required=True)
    snapshot_validate = curation_commands.add_parser(
        "validate-snapshot",
        help="validate snapshot schema, family splits, rights flags, and referenced bytes",
    )
    _add_data_root(snapshot_validate)
    snapshot_validate.add_argument("--snapshot", type=Path, required=True)
    snapshot_validate.set_defaults(handler=_curation_snapshot_validate)

    data = commands.add_parser(
        "data", help="validate reviewed local dataset records and prepare manual-transfer bundles"
    )
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_validate = data_commands.add_parser("validate")
    _add_dataset_arguments(data_validate)
    data_validate.set_defaults(handler=_data_validate)
    data_package = data_commands.add_parser("package")
    _add_dataset_arguments(data_package)
    data_package.add_argument("--output", type=Path, required=True)
    data_package.set_defaults(handler=_data_package)
    data_inspect = data_commands.add_parser(
        "inspect-bundle", help="verify a manual Colab transfer against local review hashes"
    )
    _add_data_root(data_inspect)
    data_inspect.add_argument("--bundle", type=Path, required=True)
    data_inspect.add_argument("--expected-dataset-sha256", required=True)
    data_inspect.add_argument("--expected-recipe-sha256", required=True)
    data_inspect.add_argument("--expected-bundle-sha256", required=True)
    data_inspect.set_defaults(handler=_data_inspect_bundle)

    train = commands.add_parser(
        "train", help="review recipe/data/model readiness; execution is not implemented"
    )
    train_commands = train.add_subparsers(dest="train_command", required=True)
    prepare = train_commands.add_parser(
        "prepare", help="write an immutable, non-executing experiment readiness report"
    )
    _add_dataset_arguments(prepare)
    prepare.add_argument("--model-registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    prepare.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    prepare.add_argument("--bundle", type=Path)
    prepare.add_argument("--ledger", type=Path)
    prepare.add_argument("--output", type=Path)
    prepare.set_defaults(handler=_train_prepare)

    args = parser.parse_args()
    try:
        report = args.handler(args)
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return (
            0
            if report.get("status")
            in {
                "complete",
                "verified",
                "created_pending_manual_transfer",
                "ready_for_owner_review",
                "owner_attested_pending_stage_validation",
                "manifest_created_pending_owner_review",
                "measured_pending_owner_review",
                "captured_pending_owner_review",
                "already_captured_pending_owner_review",
                "terms_decision_recorded",
            }
            else 2
        )
    except MLError as error:
        print(f"drawable-ml: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"drawable-ml: readiness or artifact validation failed: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"drawable-ml: filesystem operation failed: {error}", file=sys.stderr)
        return 2


def _add_data_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    _add_data_root(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, default=DEFAULT_SOURCE_REGISTRY)


def _resolve_data_path(root: Path, path: Path) -> Path:
    root = root.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as error:
        raise MLError("Path must remain inside the private data root") from error
    return resolve_private_path(root, relative, must_exist=False)


def _data_validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _resolve_data_path(args.data_root, args.manifest)
    manifest, digest = load_dataset(args.data_root, manifest_path, args.source_registry)
    return {
        "schema_version": 1,
        "status": "complete",
        "dataset_id": manifest.dataset_id,
        "dataset_manifest_sha256": digest,
        "assets": len(manifest.assets),
        "drawings": len(manifest.drawings),
        "training_started": False,
        "data_uploaded": False,
    }


def _data_package(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _resolve_data_path(args.data_root, args.manifest)
    output = _resolve_data_path(args.data_root, args.output)
    return package_training_bundle(args.data_root, manifest_path, args.source_registry, output)


def _data_inspect_bundle(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _resolve_data_path(args.data_root, args.bundle)
    return inspect_training_bundle(
        bundle,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_recipe_sha256=args.expected_recipe_sha256,
        expected_bundle_sha256=args.expected_bundle_sha256,
    )


def _readiness_report(args: argparse.Namespace) -> dict[str, Any]:
    ledger = args.ledger
    if ledger is not None:
        ledger = _resolve_data_path(args.data_root, ledger)
    else:
        default_ledger = args.data_root.resolve() / "reports" / "readiness-ledger-v2.json"
        if default_ledger.is_file():
            ledger = default_ledger
    report = readiness_report(args.data_root, args.stage, ledger)
    if args.output is not None:
        output = _resolve_data_path(args.data_root, args.output)
        report["report_path"] = output.relative_to(args.data_root.resolve()).as_posix()
        write_json_atomic(output, report)
    return report


def _readiness_attest(args: argparse.Namespace) -> dict[str, Any]:
    root = args.data_root.resolve()
    report_path = args.report if args.report.is_absolute() else root / args.report
    existing_ledger = args.ledger
    if existing_ledger is not None and not existing_ledger.is_absolute():
        existing_ledger = root / existing_ledger
    if existing_ledger is None:
        default_ledger = root / "reports" / "readiness-ledger-v2.json"
        if default_ledger.is_file():
            existing_ledger = default_ledger
    ledger, source_report = attest_gate(
        root,
        args.gate,
        report_path.relative_to(root).as_posix() if report_path.is_relative_to(root) else "",
        args.reviewer_id,
        args.review_note,
        existing_ledger,
    )
    output = _resolve_data_path(root, args.output)
    write_json_atomic(output, ledger.model_dump(mode="json"), refuse_overwrite=False)
    return {
        "schema_version": 1,
        "status": "owner_attested_pending_stage_validation",
        "gate_id": args.gate,
        "report_path": report_path.relative_to(root).as_posix(),
        "report_summary": source_report["summary"],
        "ledger_path": output.relative_to(root).as_posix(),
        "training_authorized": False,
        "training_started": False,
    }


def _models_verify(args: argparse.Namespace) -> dict[str, Any]:
    report = verify_model_registry(args.data_root, args.registry, purpose=args.purpose)
    report.update({"weights_loaded": False, "network_used": False, "training_started": False})
    return report


def _models_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return create_model_artifact_manifest(args.data_root, args.registry, args.model_id)


def _models_terms_import(args: argparse.Namespace) -> dict[str, Any]:
    return capture_model_terms_snapshot(
        args.data_root,
        args.registry,
        args.model_id,
        args.purpose,
        args.source_url,
        args.input,
    )


def _models_terms_review(args: argparse.Namespace) -> dict[str, Any]:
    return record_model_terms_review(
        args.data_root,
        args.registry,
        args.model_id,
        args.purpose,
        args.snapshot_sha256,
        args.decision,
        args.reviewer_id,
        args.review_note,
        replace_existing=args.replace_existing,
    )


def _models_measure_cpu(args: argparse.Namespace) -> dict[str, Any]:
    root = args.data_root.resolve()
    snapshot = _resolve_data_path(root, args.snapshot)
    output = _resolve_data_path(root, args.output)
    return measure_frozen_cpu(
        root,
        snapshot.relative_to(root),
        args.registry,
        args.source_registry,
        args.anita_terms,
        _resolve_data_path(root, args.source_profile).relative_to(root),
        output.relative_to(root),
        repeats=args.repeats,
    )


def _curation_snapshot_validate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.data_root.resolve()
    snapshot_path = _resolve_data_path(root, args.snapshot)
    try:
        snapshot = CurationSnapshotV1.model_validate_json(snapshot_path.read_bytes())
        validation = validate_curation_snapshot_files(snapshot, root)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as error:
        raise MLError(f"Curation snapshot failed validation: {error}") from error
    body_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "status": "verified",
        "snapshot_id": snapshot_path.stem,
        "snapshot_file_sha256": body_sha256,
        "items": validation["items"],
        "viewpoint_rubric_id": snapshot.viewpoint_rubric_id,
        "viewpoint_rubric_sha256": snapshot.viewpoint_rubric_sha256,
        "viewpoint_annotations_reviewed": sum(
            item.viewpoint is not None and item.viewpoint.review_state == "reviewed"
            for item in snapshot.items
        ),
        "view_match_judgments": len(snapshot.view_matches),
        "training_started": False,
        "data_uploaded": False,
    }


def _train_prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.data_root.resolve()
    blockers: list[str] = []
    manifest: DatasetManifest | None = None
    dataset_manifest_sha256: str | None = None
    dataset_summary: dict[str, Any] | None = None
    recipe_sha256: str | None = None
    recipe_summary: dict[str, Any] | None = None
    bundle_summary: dict[str, Any] | None = None
    model_summary: dict[str, Any] | None = None
    gate_summary: dict[str, Any] | None = None

    try:
        manifest_path = _resolve_data_path(root, args.manifest)
        manifest, dataset_manifest_sha256 = load_dataset(root, manifest_path, args.source_registry)
        dataset_summary = {
            "dataset_id": manifest.dataset_id,
            "sha256": dataset_manifest_sha256,
            "assets": len(manifest.assets),
            "drawings": len(manifest.drawings),
        }
    except (MLError, OSError) as error:
        blockers.append(f"dataset_manifest_not_validated:{error}")

    try:
        recipe, recipe_sha256 = load_recipe(args.recipe)
        recipe_summary = {
            "recipe_id": recipe.recipe_id,
            "sha256": recipe_sha256,
            "automatic_training": recipe.automatic_training,
            "automatic_upload": recipe.automatic_upload,
            "effective_batch_size": recipe.effective_batch_size,
            "microbatch_size": recipe.microbatch_size,
            "session_budget_minutes": recipe.session_budget_minutes,
        }
        if manifest is not None and manifest.experiment_recipe_sha256 != recipe_sha256:
            blockers.append("dataset_manifest_is_not_bound_to_this_exact_experiment_recipe")
    except (MLError, OSError) as error:
        blockers.append(f"frozen_recipe_invalid:{error}")

    try:
        model_summary = verify_model_registry(root, args.model_registry, purpose="training")
        wanted = {recipe.semantic.model_id, recipe.structural.model_id} if recipe_summary else set()
        model_rows = {row.get("model_id"): row for row in model_summary.get("models", [])}
        for model_id in sorted(wanted):
            row = model_rows.get(model_id)
            if not row or row.get("status") != "complete":
                blockers.append(f"required_local_model_not_verified:{model_id}")
    except (MLError, OSError) as error:
        blockers.append(f"model_registry_or_artifacts_invalid:{error}")

    if args.bundle is None:
        blockers.append("manual_transfer_bundle_not_supplied")
    elif manifest is None or dataset_manifest_sha256 is None or recipe_sha256 is None:
        blockers.append("bundle_cannot_be_bound_until_dataset_and_recipe_validate")
    else:
        try:
            bundle_path = _resolve_data_path(root, args.bundle)
            selected_drawings = [
                drawing
                for drawing in manifest.drawings
                if drawing.learning_split in {"train", "model_selection"}
            ]
            expected_prefix_hashes = {
                prefix.prefix_id: prefix.png_sha256
                for drawing in selected_drawings
                for prefix in drawing.prefixes
            }
            selected_asset_ids = {
                asset_id
                for drawing in selected_drawings
                for asset_id in (
                    ([drawing.reference_asset_id] if drawing.reference_asset_id is not None else [])
                    + [
                        candidate_id
                        for prefix in drawing.prefixes
                        for candidate_id in prefix.positive_asset_ids
                        + prefix.judged_unusable_asset_ids
                    ]
                )
            }
            expected_asset_hashes = {
                asset.asset_id: asset.sha256
                for asset in manifest.assets
                if asset.asset_id in selected_asset_ids
            }
            bundle_summary = inspect_training_bundle(
                bundle_path,
                expected_dataset_sha256=dataset_manifest_sha256,
                expected_recipe_sha256=recipe_sha256,
                expected_asset_hashes=expected_asset_hashes,
                expected_prefix_hashes=expected_prefix_hashes,
            )
        except (MLError, OSError) as error:
            blockers.append(f"manual_transfer_bundle_invalid:{error}")

    ledger = args.ledger
    if ledger is not None:
        try:
            ledger = _resolve_data_path(root, ledger)
        except MLError as error:
            blockers.append(f"evidence_ledger_path_invalid:{error}")
            ledger = None
    else:
        default_ledger = root / "reports" / "readiness-ledger-v2.json"
        if default_ledger.is_file():
            ledger = default_ledger
    try:
        gate_summary = readiness_report(root, "training_preparation", ledger)
    except MLError as error:
        blockers.append(f"evidence_ledger_invalid:{error}")
        gate_summary = readiness_report(root, "training_preparation", None)
        gate_summary["blockers"]["evidence_ledger"] = str(error)
        gate_summary["status"] = "blocked"
    if gate_summary["status"] != "complete":
        blockers.append("required_m0_m1_pilot_m3_and_feasibility_evidence_incomplete")

    unique_blockers = list(dict.fromkeys(blockers))
    unique_blockers.append("training_execution_loop_and_resume_command_not_implemented")
    report = {
        "schema_version": 1,
        "report_kind": "non-executing-training-readiness",
        "generated_at": gate_summary["generated_at"],
        "status": "blocked",
        "blockers": unique_blockers,
        "dataset": dataset_summary,
        "recipe": recipe_summary,
        "models": model_summary,
        "bundle": bundle_summary,
        "required_evidence": gate_summary["required_gates"],
        "verified_evidence": gate_summary["verified_evidence"],
        "evidence_blockers": gate_summary["blockers"],
        "training_authorized": False,
        "training_started": False,
        "optimizer_created": False,
        "backward_passes_run": False,
        "weight_updates": 0,
        "data_uploaded": False,
        "gpu_status": "not_probed",
        "next_step": "Resolve the readiness blockers, implement/review the separately gated execution and resume path, then obtain a distinct owner authorization; this command never starts execution.",
    }
    if args.output is not None:
        output = _resolve_data_path(root, args.output)
        report["report_path"] = output.relative_to(root).as_posix()
        write_json_atomic(output, report)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
