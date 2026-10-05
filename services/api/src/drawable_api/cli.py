from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from drawable_research_contracts.readiness import (
    REQUIREMENTS_REVISION,
    EvidenceInputV1,
    GateReportV2,
    validate_readiness,
)

from . import intake, m0
from .catalog import catalog_manifest
from .database import migrate
from .emoreccom_map import (
    approve_book_mapping,
    create_book_mapping_template,
    fetch_official_book_index,
)
from .museum import fetch_museum_candidates, refresh_museum_manifest
from .readiness import build_training_readiness
from .sampling import build_pilot_yield_sample
from .settings import REPOSITORY_ROOT, Settings


def emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=Settings.from_env().data_root,
        help="Private dataset root (defaults to DRAWABLE_DATA_ROOT or ./datasets)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drawable", description="Local-only Drawable preparation and data intake tools"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser(
        "preflight",
        help="Report Python, disk, dependency, and explicitly requested accelerator status",
    )
    _root_arg(preflight)
    preflight.add_argument("--profile", choices=("local-cpu", "colab-cuda"), default="local-cpu")
    preflight.add_argument(
        "--probe-gpu",
        action="store_true",
        help="Explicitly probe the current machine; no GPU probe runs by default",
    )

    sources = commands.add_parser("sources", help="Acquire and audit official source archives")
    source_commands = sources.add_subparsers(dest="sources_command", required=True)
    audit = source_commands.add_parser(
        "audit", help="Read-only source archive/tree/annotation audit"
    )
    _root_arg(audit)
    audit.add_argument(
        "--no-verify", action="store_true", help="Skip full archive CRC and SHA-256 reads"
    )
    audit.add_argument("--write", type=Path, help="Write the report inside the private data root")
    reconcile = source_commands.add_parser(
        "reconcile", help="Compare official archive bytes with clean and configured extractions"
    )
    _root_arg(reconcile)
    reconcile.add_argument("--profile", type=Path, required=True, help="Selected-source profile JSON")
    reconcile.add_argument("--output", type=Path, default=Path("reports/m0-official-archive-reconciliation-v2.json"))
    acquire = source_commands.add_parser(
        "acquire", help="Download one declared official archive to a .part file then verify it"
    )
    _root_arg(acquire)
    acquire.add_argument("--source", required=True)
    acquire.add_argument("--resume", action="store_true")
    terms = source_commands.add_parser(
        "terms", help="Capture official source terms for human review"
    )
    terms_commands = terms.add_subparsers(dest="terms_command", required=True)
    snapshot = terms_commands.add_parser(
        "snapshot", help="Save a content-addressed official terms-page snapshot"
    )
    _root_arg(snapshot)
    snapshot.add_argument("--source", required=True)

    m0_commands = commands.add_parser("m0", help="Import and verify local M0 feasibility evidence")
    m0_subcommands = m0_commands.add_subparsers(dest="m0_command", required=True)
    effort_import = m0_subcommands.add_parser(
        "effort-import", help="Validate and content-address the browser-local active effort log"
    )
    _root_arg(effort_import)
    effort_import.add_argument("--input", type=Path, required=True)
    recording_import = m0_subcommands.add_parser(
        "recording-import", help="Validate and content-address an exported recording batch"
    )
    _root_arg(recording_import)
    recording_import.add_argument("--input", type=Path, required=True)
    source_yield = m0_subcommands.add_parser(
        "source-yield-report", help="Freeze complete fixed-sample yield judgments and report their evidence"
    )
    _root_arg(source_yield)
    source_yield.add_argument(
        "--sample",
        default="reports/source-yield-sample-pilot-v2.json",
        help="Selected-source sample report written by data sample",
    )
    source_yield.add_argument("--output", type=Path)
    source_yield.add_argument("--gate-output", type=Path)
    effort_report = m0_subcommands.add_parser(
        "effort-report", help="measure M0 time and disk and disclose a scoped full-program review projection"
    )
    _root_arg(effort_report)
    effort_report.add_argument("--effort-log", required=True, help="Imported private effort-log path")
    effort_report.add_argument("--source-yield-report", required=True)
    effort_report.add_argument("--source-yield-gate", required=True)
    effort_report.add_argument("--replay-report", required=True)
    effort_report.add_argument("--replay-gate", required=True)
    effort_report.add_argument(
        "--effort-plan",
        type=Path,
        required=True,
        help="Private versioned plan with unique drawings, judgment pools, strata, rates, and remaining work",
    )
    effort_report.add_argument("--five-hour-checkpoint-reviewed", action="store_true")
    effort_report.add_argument("--annotation-stopped-at-budget", action="store_true")
    effort_report.add_argument("--output", type=Path)
    effort_report.add_argument("--gate-output", type=Path)
    replay = m0_subcommands.add_parser(
        "replay-recordings", help="Independently replay ten selected recordings in pinned Chromium"
    )
    _root_arg(replay)
    replay.add_argument("--batch", type=Path, required=True)
    replay.add_argument("--recording-id", action="append", required=True)
    replay.add_argument("--output", type=Path, default=Path("reports/private/m0-recording-replay-v1.json"))
    replay.add_argument("--gate-output", type=Path, default=Path("reports/m0-ten-feasibility-sequences-v1.json"))
    rights = m0_subcommands.add_parser(
        "rights-report", help="validate recorded source/work permissions and captured terms snapshots"
    )
    _root_arg(rights)
    rights.add_argument("--profile", type=Path, required=True, help="Selected-source profile JSON")
    rights.add_argument("--output", type=Path)
    rights.add_argument("--gate-output", type=Path)
    feasibility = m0_subcommands.add_parser(
        "feasibility-report", help="assemble immutable M0 status from owner-attested gate evidence"
    )
    _root_arg(feasibility)
    feasibility.add_argument("--ledger", type=Path)
    feasibility.add_argument("--output", type=Path)
    gallery = m0_subcommands.add_parser(
        "gallery-report", help="recheck and freeze a non-serving M0 staging gallery manifest"
    )
    _root_arg(gallery)
    gallery.add_argument("--snapshot", required=True, help="Private immutable curation snapshot path")
    gallery.add_argument("--report-output", type=Path)
    gallery.add_argument("--gate-output", type=Path)
    gallery.add_argument("--manifest-output", type=Path)

    data = commands.add_parser("data", help="Validate, safely extract, and fingerprint source data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    validate = data_commands.add_parser(
        "validate", help="Check published EmoRecCom CSV/panel correspondence and split invariants"
    )
    _root_arg(validate)
    validate.add_argument(
        "--strict",
        action="store_true",
        help="Return failure until authoritative book IDs and all gates are verified",
    )
    validate.add_argument(
        "--book-map",
        type=Path,
        help="Optional reviewed official book mapping from data emoreccom mapping-approve",
    )
    sample = data_commands.add_parser(
        "sample", help="Create a deterministic, human-review-only pilot-v1 source-yield sample"
    )
    _root_arg(sample)
    sample.add_argument("--profile", type=Path, required=True, help="Selected-source profile JSON")
    sample.add_argument("--recipe", choices=("pilot-v1",), required=True)
    sample.add_argument(
        "--output",
        type=Path,
        help="Immutable output path inside --root; defaults to reports/source-yield-sample-pilot-v2.json",
    )
    manifest = data_commands.add_parser(
        "manifest", help="Build deterministic checksums and verified image metadata"
    )
    _root_arg(manifest)
    manifest.add_argument("--source", required=True)
    manifest.add_argument(
        "--path", type=Path, help="Existing extracted tree, constrained to --root"
    )
    manifest.add_argument(
        "--output",
        type=Path,
        help="Output path inside --root (defaults to manifests/<source>-tree.json)",
    )
    manifest.add_argument(
        "--inspect-existing",
        action="store_true",
        help="Explicitly fingerprint a legacy extraction without asserting archive provenance",
    )
    manifest.add_argument(
        "--emoreccom-book-map",
        type=Path,
        help="Include a verified, reviewer-recorded COMICS book mapping in an EmoRecCom manifest",
    )
    ingest = data_commands.add_parser(
        "ingest", help="Extract one receipt-verified ZIP to a versioned staging tree"
    )
    _root_arg(ingest)
    ingest.add_argument("--source", required=True)
    catalog = data_commands.add_parser(
        "catalog", help="Verify a tree manifest and catalog assets as quarantined local metadata"
    )
    _root_arg(catalog)
    catalog.add_argument("--manifest", type=Path, required=True)
    catalog.add_argument(
        "--database", type=Path, help="Metadata database (defaults to <root>/metadata.sqlite3)"
    )
    package = data_commands.add_parser(
        "package", help="Create a local cloud-readiness bundle after source and cloud-use approval"
    )
    _root_arg(package)
    package.add_argument("--purpose", choices=("inference-readiness",), required=True)
    package.add_argument("--manifest", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    emoreccom = data_commands.add_parser(
        "emoreccom", help="Validate official EmoRecCom-to-COMICS book identity"
    )
    emoreccom_commands = emoreccom.add_subparsers(dest="emoreccom_command", required=True)
    book_index = emoreccom_commands.add_parser(
        "fetch-book-index",
        help="Optionally fetch only the official COMICS OCR metadata CSV; no panel archive",
    )
    _root_arg(book_index)
    book_index.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly allow this official metadata-only download",
    )
    book_index.add_argument(
        "--resume", action="store_true", help="Resume an existing .part transfer"
    )
    map_template = emoreccom_commands.add_parser(
        "mapping-template",
        help="Write one blank crosswalk row per observed train/test filename prefix",
    )
    _root_arg(map_template)
    map_template.add_argument(
        "--output", type=Path, default=Path("manifests/emoreccom-book-map-template.json")
    )
    map_approve = emoreccom_commands.add_parser(
        "mapping-approve",
        help="Validate a human-checked prefix crosswalk against official COMICS OCR metadata",
    )
    _root_arg(map_approve)
    map_approve.add_argument("--mapping", type=Path, required=True)
    map_approve.add_argument(
        "--official-index", type=Path, default=Path("raw/comics-metadata/COMICS_ocr_file.csv")
    )
    map_approve.add_argument("--reviewer-id", required=True)
    map_approve.add_argument(
        "--output", type=Path, default=Path("manifests/emoreccom-book-map.json")
    )
    museum = data_commands.add_parser(
        "museum", help="Acquire bounded official museum reference candidates"
    )
    museum_commands = museum.add_subparsers(dest="museum_command", required=True)
    museum_fetch = museum_commands.add_parser(
        "fetch",
        help="Fetch up to 200 metadata records and 100 images, leaving all results quarantined",
    )
    _root_arg(museum_fetch)
    museum_fetch.add_argument(
        "--source", required=True, choices=("the-met-open-access", "aic-open-access")
    )
    museum_fetch.add_argument(
        "--query",
        action="append",
        help="Search phrase; defaults to the five pilot figure-study queries",
    )
    museum_fetch.add_argument("--metadata-limit", type=int, default=200, choices=range(1, 201))
    museum_fetch.add_argument("--image-limit", type=int, default=50, choices=range(1, 101))
    museum_fetch.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly allow network requests and local candidate writes",
    )
    museum_refresh = museum_commands.add_parser(
        "refresh-manifest",
        help="Bind existing museum bytes to newly approved source terms without downloading",
    )
    _root_arg(museum_refresh)
    museum_refresh.add_argument(
        "--source", required=True, choices=("the-met-open-access", "aic-open-access")
    )
    museum_refresh.add_argument(
        "--execute", action="store_true", help="Explicitly update the local candidate manifest"
    )

    database = commands.add_parser(
        "db", help="Create or inspect the versioned, local metadata database"
    )
    db_commands = database.add_subparsers(dest="db_command", required=True)
    migrate_cmd = db_commands.add_parser(
        "migrate", help="Apply versioned SQLite metadata migrations"
    )
    _root_arg(migrate_cmd)

    readiness = commands.add_parser(
        "readiness", help="Report evidence without promoting incomplete gates"
    )
    readiness_commands = readiness.add_subparsers(dest="readiness_command", required=True)
    report = readiness_commands.add_parser(
        "report", help="Summarize owner-reviewed evidence for one cumulative readiness stage"
    )
    _root_arg(report)
    report.add_argument(
        "--stage",
        choices=("m0", "m1", "pilot", "m2", "m3", "m5", "m6", "m7", "training_preparation", "training_execution"),
        default="training_preparation",
    )
    report.add_argument("--ledger", type=Path, help="Read an explicit ledger beneath --root")
    report.add_argument(
        "--output",
        type=Path,
        help="Write a new immutable readiness snapshot inside --root; existing files are never overwritten",
    )

    return parser


def _sources_reconcile(root: Path, output: Path, profile_path: Path) -> dict[str, Any]:
    registry = intake.load_registry()
    work_registry = json.loads(
        (REPOSITORY_ROOT / "config" / "anita-work-terms.json").read_text(encoding="utf-8")
    )
    profile, _profile_relative, profile_digest, profile_inputs = m0._load_source_selection_profile(
        root, profile_path, registry["sources"], work_registry["works"]
    )
    results: list[dict[str, Any]] = []
    blockers: dict[str, str] = {}
    for selection in profile.sources:
        source_id = selection.source_id
        if selection.acquisition_method != "official_archive":
            results.append(
                {
                    "source_id": source_id,
                    "acquisition_method": selection.acquisition_method,
                    "status": "reconciled",
                    "evidence": [item.path for item in selection.acquisition_evidence],
                }
            )
            continue
        try:
            result = intake.reconcile_archived_source(root, source_id)
            result["acquisition_method"] = "official_archive"
            results.append(result)
            if result["status"] != "reconciled":
                blockers[source_id] = result["status"]
        except (intake.IntakeError, OSError, ValueError) as error:
            blockers[source_id] = str(error)
    if blockers:
        return {
            "schema_version": 1,
            "status": "blocked",
            "gate_id": "official_archive_reconciliation",
            "sources": results,
            "blockers": blockers,
            "gate_report_written": False,
            "training_started": False,
            "data_uploaded": False,
        }

    inputs: dict[str, str] = dict(profile_inputs)
    for result in results:
        if result.get("acquisition_method") != "official_archive" and "source_id" in result:
            continue
        reconciliation_path = intake.ensure_inside(root, result["report_path"])
        body = json.loads(reconciliation_path.read_text(encoding="utf-8"))
        for relative in (
            body["archive_path"],
            body["archive_receipt_path"],
            body["clean_extraction_receipt_path"],
            body["report_path"],
        ):
            path = intake.ensure_inside(root, relative)
            inputs[relative] = intake.sha256_file(path)
        for tree_root, key in (
            (body["clean_extraction_path"], "clean_extraction_entries"),
            (body["configured_tree_path"], "configured_tree_entries"),
        ):
            for item in body[key]:
                relative = (Path(tree_root) / item["path"]).as_posix()
                path = intake.ensure_inside(root, relative)
                if path.is_symlink() or path.stat().st_size != item["bytes"]:
                    raise intake.IntakeError(f"Reconciled source file changed before report assembly: {relative}")
                digest = intake.sha256_file(path)
                if digest != item["sha256"]:
                    raise intake.IntakeError(f"Reconciled source file changed before report assembly: {relative}")
                inputs[relative] = digest
    gate_report = GateReportV2(
        schema_version=2,
        requirements_revision=REQUIREMENTS_REVISION,
        artifact_type="drawable-gate-report",
        gate_id="official_archive_reconciliation",
        producer="drawable sources reconcile",
        produced_at=datetime.now(UTC),
        checks={
            "receipts_verified": True,
            "archives_hash_verified": True,
            "clean_extractions_reconciled": True,
            "acquisition_methods_evidenced": True,
        },
        measurements={
            "archives_reconciled": sum(
                item.acquisition_method == "official_archive" for item in profile.sources
            )
        },
        profile_hashes={"source_selection": profile_digest},
        inputs=[EvidenceInputV1(path=path, sha256=digest) for path, digest in sorted(inputs.items())],
        summary=(
            f"Reconciled {sum(item.acquisition_method == 'official_archive' for item in profile.sources)} selected official archives "
            f"and recorded item-level evidence for {sum(item.acquisition_method != 'official_archive' for item in profile.sources)} "
            "selected non-archive inputs."
        ),
    )
    destination = output if output.is_absolute() else root / output
    if not _is_under(destination, root):
        raise intake.IntakeError("Archive reconciliation report must stay inside the private data root")
    destination = destination.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise intake.IntakeError("Archive reconciliation reports are immutable; choose a new output path")
    intake.write_json_create(destination, gate_report.model_dump(mode="json"))
    return {
        "schema_version": 1,
        "status": "complete",
        "gate_id": "official_archive_reconciliation",
        "gate_report_path": destination.relative_to(root).as_posix(),
        "gate_report_sha256": intake.sha256_file(destination),
        "archives_reconciled": sum(
            item.acquisition_method == "official_archive" for item in profile.sources
        ),
        "sources": results,
        "training_started": False,
        "data_uploaded": False,
    }


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _preflight(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    space = shutil.disk_usage(root)
    report: dict[str, Any] = {
        "schema_version": 1,
        "profile": args.profile,
        "python_version": ".".join(str(x) for x in sys.version_info[:3]),
        "python_compatible": sys.version_info[:2] == (3, 11),
        "uv_lock_present": (REPOSITORY_ROOT / "uv.lock").is_file(),
        "data_root_relative": root.relative_to(REPOSITORY_ROOT).as_posix()
        if _is_under(root, REPOSITORY_ROOT)
        else "external_data_root",
        "disk": {
            "total_bytes": space.total,
            "free_bytes": space.free,
            "required_headroom_bytes": int(space.total * 0.2),
        },
        "gpu": {
            "status": "not_probed",
            "reason": "GPU checks are opt-in; this report does not infer a driver failure.",
        },
        "training_authorized": False,
    }
    if args.probe_gpu:
        if args.profile == "colab-cuda" and not (
            os.environ.get("COLAB_RELEASE_TAG") or Path("/content").is_dir()
        ):
            report["gpu"] = {
                "status": "not_available_on_this_host",
                "reason": "The Colab profile was requested outside a Colab runtime.",
            }
        else:
            try:
                probe = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                report["gpu"] = {
                    "status": "available"
                    if probe.returncode == 0 and probe.stdout.strip()
                    else "unavailable",
                    "summary": probe.stdout.strip() or None,
                    "probe_exit_code": probe.returncode,
                }
            except (OSError, subprocess.TimeoutExpired):
                report["gpu"] = {
                    "status": "unavailable",
                    "reason": "Explicit GPU probe did not complete.",
                }
    elif args.profile == "colab-cuda":
        report["gpu"] = {
            "status": "not_probed",
            "reason": "Enable an actual Colab CUDA runtime and pass --probe-gpu to test it.",
        }
    emit(report)
    return 0 if report["python_compatible"] else 2


def _sources_audit(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    result = intake.audit_sources(root, verify_archives=not args.no_verify)
    if args.write:
        output = args.write if args.write.is_absolute() else root / args.write
        if not _is_under(output, root):
            raise intake.IntakeError("Audit reports must be written inside the private data root")
        intake.write_json_atomic(output, result)
        result["written_report"] = output.relative_to(root).as_posix()
    emit(result)
    return 0


def _data_validate(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    book_map = (
        args.book_map
        if args.book_map and args.book_map.is_absolute()
        else root / args.book_map
        if args.book_map
        else None
    )
    if book_map is not None and not _is_under(book_map, root):
        raise intake.IntakeError(
            "Reviewed EmoRecCom book mapping must remain inside the private data root"
        )
    result = intake.validate_emoreccom(root, book_mapping_path=book_map)
    emit(result)
    complete = result.get("status") == "validated" and result.get("book_identity_verified") is True
    return 1 if args.strict and not complete else 0


def _data_manifest(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    source = intake.source_by_id(args.source)
    if args.path:
        tree = args.path if args.path.is_absolute() else root / args.path
        if not _is_under(tree, root):
            raise intake.IntakeError("Manifest source path must be inside the private data root")
    else:
        tree = root / source["extracted_path"]
    receipt_file = root / "manifests" / "source-receipts" / f"{args.source}.json"
    if not receipt_file.is_file() and not args.inspect_existing:
        raise intake.IntakeError(
            "An acquisition receipt is required; use --inspect-existing only for a non-eligible legacy inventory"
        )
    book_map = (
        args.emoreccom_book_map
        if args.emoreccom_book_map and args.emoreccom_book_map.is_absolute()
        else root / args.emoreccom_book_map
        if args.emoreccom_book_map
        else None
    )
    if book_map is not None and (
        args.source != "emoreccom-panels" or not _is_under(book_map, root)
    ):
        raise intake.IntakeError(
            "A reviewed book map is only valid for EmoRecCom and must stay inside the private data root"
        )
    result = intake.build_tree_manifest(
        root, args.source, tree_path=tree, emoreccom_book_map_path=book_map
    )
    if receipt_file.is_file():
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        if receipt.get("sha256"):
            result["source_archive_sha256"] = receipt["sha256"]
            result["status"] = "manifest_ready_for_rights_and_human_review"
    else:
        result["status"] = "legacy_tree_only_provenance_unverified"
    output = args.output if args.output else Path("manifests") / f"{args.source}-tree.json"
    output = output if output.is_absolute() else root / output
    if not _is_under(output, root):
        raise intake.IntakeError("Tree manifests must remain inside the private data root")
    intake.write_json_atomic(output, result)
    emit(
        {
            "status": result["status"],
            "file_count": result["file_count"],
            "tree_sha256": result["tree_sha256"],
            "manifest_path": output.relative_to(root).as_posix(),
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            return _preflight(args)
        if args.command == "sources" and args.sources_command == "audit":
            return _sources_audit(args)
        if args.command == "sources" and args.sources_command == "reconcile":
            root = args.root.resolve()
            report = _sources_reconcile(root, args.output, args.profile)
            emit(report)
            return 0 if report.get("status") == "complete" else 2
        if args.command == "m0" and args.m0_command == "effort-import":
            emit(m0.import_effort_log(args.root.resolve(), args.input))
            return 0
        if args.command == "m0" and args.m0_command == "recording-import":
            emit(m0.import_recording_batch(args.root.resolve(), args.input))
            return 0
        if args.command == "m0" and args.m0_command == "source-yield-report":
            report = m0.build_source_yield_report(
                args.root.resolve(),
                sample_path=args.sample,
                output=args.output,
                gate_output=args.gate_output,
            )
            emit(report)
            return 0 if report.get("status") == "complete" else 2
        if args.command == "m0" and args.m0_command == "effort-report":
            report = m0.build_human_effort_report(
                args.root.resolve(),
                effort_log=args.effort_log,
                source_yield_report=args.source_yield_report,
                source_yield_gate=args.source_yield_gate,
                replay_report=args.replay_report,
                replay_gate=args.replay_gate,
                effort_plan=args.effort_plan,
                five_hour_checkpoint_reviewed=args.five_hour_checkpoint_reviewed,
                annotation_stopped_at_budget=args.annotation_stopped_at_budget,
                output=args.output,
                gate_output=args.gate_output,
            )
            emit(report)
            return 0 if report.get("status") == "complete" else 2
        if args.command == "m0" and args.m0_command == "replay-recordings":
            root = args.root.resolve()
            batch = args.batch if args.batch.is_absolute() else root / args.batch
            if not _is_under(batch, root) or batch.is_symlink() or not batch.is_file():
                raise intake.IntakeError("Recording batches must be imported regular files inside the private data root")
            script = REPOSITORY_ROOT / "tools" / "replay-m0-recordings.mjs"
            command = [
                "node",
                str(script),
                "--data-root",
                str(root),
                "--batch",
                batch.resolve().relative_to(root).as_posix(),
                "--report",
                args.output.as_posix(),
                "--gate-report",
                args.gate_output.as_posix(),
            ]
            for recording_id in args.recording_id:
                command.extend(("--recording-id", recording_id))
            process_result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
            return process_result.returncode
        if args.command == "m0" and args.m0_command == "rights-report":
            report = m0.build_source_work_rights_report(
                args.root.resolve(),
                source_profile=args.profile,
                output=args.output,
                gate_output=args.gate_output,
            )
            emit(report)
            return 0 if report.get("status") == "complete" else 2
        if args.command == "m0" and args.m0_command == "feasibility-report":
            report = m0.build_feasibility_report(
                args.root.resolve(), ledger_path=args.ledger, output=args.output
            )
            emit(report)
            return 0 if report.get("status") == "complete" else 2
        if args.command == "m0" and args.m0_command == "gallery-report":
            report = m0.build_small_approved_gallery_report(
                args.root.resolve(),
                snapshot_path=args.snapshot,
                report_output=args.report_output,
                gate_output=args.gate_output,
                manifest_output=args.manifest_output,
            )
            emit(report)
            return 0 if report.get("status") == "complete" else 2
        if args.command == "sources" and args.sources_command == "acquire":
            emit(intake.acquire_archive(args.root.resolve(), args.source, resume=args.resume))
            return 0
        if (
            args.command == "sources"
            and args.sources_command == "terms"
            and args.terms_command == "snapshot"
        ):
            emit(intake.capture_terms_snapshot(args.root.resolve(), args.source))
            return 0
        if args.command == "data" and args.data_command == "validate":
            return _data_validate(args)
        if args.command == "data" and args.data_command == "sample":
            root = args.root.resolve()
            sample_report = build_pilot_yield_sample(root, profile_path=args.profile)
            output = (
                args.output
                if args.output and args.output.is_absolute()
                else root / (args.output or Path("reports/source-yield-sample-pilot-v2.json"))
            )
            if not _is_under(output, root):
                raise intake.IntakeError(
                    "Pilot source-yield samples must remain inside the private data root"
                )
            output = output.resolve(strict=False)
            intake.write_json_create(output, sample_report)
            sample_result = {**sample_report, "written_report": output.relative_to(root).as_posix()}
            emit(sample_result)
            return 0
        if args.command == "data" and args.data_command == "manifest":
            return _data_manifest(args)
        if args.command == "data" and args.data_command == "ingest":
            emit(intake.extract_verified_archive(args.root.resolve(), args.source))
            return 0
        if args.command == "data" and args.data_command == "catalog":
            root = args.root.resolve()
            manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
            database = (
                args.database
                if args.database and args.database.is_absolute()
                else root / (args.database or Path("metadata.sqlite3"))
            )
            emit(catalog_manifest(root, database, manifest))
            return 0
        if args.command == "data" and args.data_command == "package":
            root = args.root.resolve()
            manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
            output = args.output if args.output.is_absolute() else root / args.output
            emit(intake.build_data_package(root, manifest, purpose=args.purpose, out_path=output))
            return 0
        if args.command == "data" and args.data_command == "emoreccom":
            root = args.root.resolve()
            if args.emoreccom_command == "fetch-book-index":
                if not args.execute:
                    emit(
                        {
                            "status": "not_run",
                            "url": "https://obj.umiacs.umd.edu/comics/COMICS_ocr_file.csv",
                            "purpose": "metadata-only book identity cross-check; no COMICS image archive",
                            "reason": "Pass --execute to download.",
                        }
                    )
                    return 0
                emit(fetch_official_book_index(root, resume=args.resume))
                return 0
            if args.emoreccom_command == "mapping-template":
                output = args.output if args.output.is_absolute() else root / args.output
                if not _is_under(output, root):
                    raise intake.IntakeError(
                        "EmoRecCom mapping template must remain inside the private data root"
                    )
                emit(create_book_mapping_template(root, output))
                return 0
            if args.emoreccom_command == "mapping-approve":
                mapping = args.mapping if args.mapping.is_absolute() else root / args.mapping
                official_index = (
                    args.official_index
                    if args.official_index.is_absolute()
                    else root / args.official_index
                )
                output = args.output if args.output.is_absolute() else root / args.output
                if not all(_is_under(path, root) for path in (mapping, official_index, output)):
                    raise intake.IntakeError(
                        "EmoRecCom mapping inputs and output must remain inside the private data root"
                    )
                emit(
                    approve_book_mapping(
                        root,
                        mapping,
                        official_index,
                        reviewer_id=args.reviewer_id,
                        output_path=output,
                    )
                )
                return 0
        if (
            args.command == "data"
            and args.data_command == "museum"
            and args.museum_command == "fetch"
        ):
            if not args.execute:
                emit(
                    {
                        "status": "not_run",
                        "source_id": args.source,
                        "metadata_limit": args.metadata_limit,
                        "image_limit": args.image_limit,
                        "queries": args.query or None,
                        "reason": "Pass --execute to make bounded official API and image requests and write quarantine-only candidate files.",
                    }
                )
                return 0
            emit(
                fetch_museum_candidates(
                    args.root.resolve(),
                    args.source,
                    queries=args.query,
                    metadata_limit=args.metadata_limit,
                    image_limit=args.image_limit,
                )
            )
            return 0
        if (
            args.command == "data"
            and args.data_command == "museum"
            and args.museum_command == "refresh-manifest"
        ):
            if not args.execute:
                emit(
                    {
                        "status": "not_run",
                        "source_id": args.source,
                        "reason": "Pass --execute to refresh the local manifest from approved terms; this makes no network requests.",
                    }
                )
                return 0
            emit(refresh_museum_manifest(args.root.resolve(), args.source))
            return 0
        if args.command == "db" and args.db_command == "migrate":
            version = migrate(args.root.resolve() / "metadata.sqlite3")
            emit({"schema_version": version, "status": "ready"})
            return 0
        if args.command == "readiness" and args.readiness_command == "report":
            root = args.root.resolve()
            ledger = args.ledger
            if ledger is not None:
                ledger = ledger if ledger.is_absolute() else root / ledger
                if not _is_under(ledger, root):
                    raise intake.IntakeError(
                        "Readiness ledgers must remain inside the private data root"
                    )
            if args.stage == "training_preparation":
                readiness_report = build_training_readiness(root, ledger_path=ledger)
                report_body = readiness_report.model_dump(mode="json", by_alias=True)
            else:
                if ledger is None:
                    default_ledger = root / "reports" / "readiness-ledger-v2.json"
                    ledger = default_ledger if default_ledger.is_file() else None
                report_body = validate_readiness(root, args.stage, ledger)
            if args.output:
                output = args.output if args.output.is_absolute() else root / args.output
                if not _is_under(output, root):
                    raise intake.IntakeError(
                        "Readiness snapshots must remain inside the private data root"
                    )
                output = output.resolve(strict=False)
                if output.exists():
                    raise intake.IntakeError(
                        "Readiness snapshots are immutable; choose a new output path rather than overwriting"
                    )
                intake.write_json_create(output, report_body)
                output_report = report_body
                output_report["written_report"] = output.relative_to(root).as_posix()
                emit(output_report)
            else:
                emit(report_body)
            return 0
    except intake.IntakeError as error:
        print(
            json.dumps({"error": "intake_failed", "message": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError, RuntimeError) as error:
        print(
            json.dumps({"error": "operation_failed", "message": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    return 2


def api_main() -> int:
    import uvicorn

    host = os.environ.get("DRAWABLE_API_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("The local API only supports loopback binding")
    uvicorn.run(
        "drawable_api.app:app",
        host=host,
        port=int(os.environ.get("DRAWABLE_API_PORT", "8000")),
        workers=1,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
