"""``linescout-manifest`` command line: validate, convert, schema, and synth.

Usage::

    linescout-manifest validate path/to/manifest.json [--data-root DIR] [--require-files]
    linescout-manifest convert path/to/old.json --from 1 --to 3 --out path/to/v3.json \\
        [--report path/to/report.json] [--pipeline-version V] [--label-version V]
    linescout-manifest migrate-v1 path/to/v1-manifest.json --out path/to/v3.json
    linescout-manifest schema [--out path/to/schema.json]
    linescout-manifest synth --out path/to/manifest.json [--count 24] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from linescout_ml.manifest import (
    ArtifactContract,
    Manifest,
    check_parent_integrity,
    check_split_integrity,
    dump_json_schema,
    learning_split_report,
)
from linescout_ml.migrate import (
    ConversionError,
    ConversionReport,
    convert_manifest,
    migrate_manifest,
)
from linescout_ml.synthetic import write_synthetic_dataset


def _version_hint(path: Path) -> str:
    """Name the conversion command when the document is an old schema version."""
    try:
        version = json.loads(path.read_text(encoding="utf-8")).get("schema_version")
    except (OSError, ValueError):
        return ""
    if version == 1:
        return (
            "If this is a schema v1 manifest, run: linescout-manifest convert "
            "<path> --from 1 --to 3"
        )
    if version == 2:
        return (
            "If this is a schema v2 manifest, run: linescout-manifest convert <path> "
            "--from 2 --to 3"
        )
    return ""


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.manifest)
    try:
        manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        print(f"INVALID {path}: {error}", file=sys.stderr)
        hint = _version_hint(path)
        if hint:
            print(hint, file=sys.stderr)
        return 1

    problems = check_split_integrity(manifest.records) + check_parent_integrity(manifest.records)
    if args.require_files:
        root = Path(args.data_root or path.parent)
        for record in manifest.servable_records:
            for label, rel in (
                ("original", record.original_path),
                ("line_art", record.line_art_path),
                ("thumbnail", record.thumbnail_path),
            ):
                if not (root / rel).is_file():
                    problems.append(f"{record.asset_id}: missing {label} file {rel}")

    if problems:
        for problem in problems:
            print(f"PROBLEM {problem}", file=sys.stderr)
        return 1

    servable = len(manifest.servable_records)
    print(
        f"OK {path}: {len(manifest.records)} records, {servable} servable, "
        f"dataset_version={manifest.dataset_version}, content_hash={manifest.content_hash()[:12]}"
    )
    if manifest.artifact_contract is not None:
        print(f"artifact_contract={manifest.artifact_contract.describe()}")
    else:
        print("artifact_contract=UNKNOWN (no record is verified current)", file=sys.stderr)
    print(json.dumps(learning_split_report(manifest.records), indent=None, sort_keys=True))
    return 0


def _write_output(
    manifest: Manifest | object,
    destination: Path,
    report: ConversionReport,
    report_path: str | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")  # type: ignore[union-attr]
    print(f"wrote {destination} ({report.records_converted} records)")
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    if report_path:
        Path(report_path).write_text(
            json.dumps(report.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {report_path}")
    if report.serving_eligibility_lost:
        print(
            f"NOTE {report.serving_eligibility_lost} previously enabled record(s) are NOT "
            "servable: record permission provenance and human SFW approval to restore "
            "serving. No asset gained permission or human approval.",
            file=sys.stderr,
        )
    if not report.contract_detected and report.schema_version_to == 3:
        print(
            "NOTE artifact_contract is unknown: every record is disabled until the "
            "current generation is declared (--pipeline-version/--label-version/"
            "--processing-revision) or the dataset is re-processed.",
            file=sys.stderr,
        )


def _cmd_convert(args: argparse.Namespace) -> int:
    source = Path(args.manifest)
    contract = None
    if args.pipeline_version is not None and args.label_version is not None:
        contract = ArtifactContract(
            pipeline_version=args.pipeline_version,
            label_version=args.label_version,
            processing_revision=args.processing_revision,
        )
    try:
        manifest, report = convert_manifest(
            json.loads(source.read_text(encoding="utf-8")),
            from_version=args.version_from,
            to_version=args.version_to,
            explicit_contract=contract,
        )
    except (OSError, ValueError, ValidationError, json.JSONDecodeError, ConversionError) as error:
        print(f"CONVERSION FAILED {source}: {error}", file=sys.stderr)
        return 1
    _write_output(manifest, Path(args.out), report, args.report)
    return 0


def _cmd_migrate_v1(args: argparse.Namespace) -> int:
    """Compatibility alias: ``migrate-v1`` converts straight to schema v3."""
    source = Path(args.manifest)
    try:
        manifest, report = migrate_manifest(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError, json.JSONDecodeError, ConversionError) as error:
        print(f"MIGRATION FAILED {source}: {error}", file=sys.stderr)
        return 1
    _write_output(manifest, Path(args.out), report, args.report)
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    text = dump_json_schema()
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    manifest_path = write_synthetic_dataset(Path(args.out), count=args.count, seed=args.seed)
    summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"wrote {manifest_path} with {len(summary['records'])} records")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="linescout-manifest")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a manifest JSON file")
    validate.add_argument("manifest")
    validate.add_argument(
        "--data-root", help="directory the manifest's relative paths resolve from"
    )
    validate.add_argument(
        "--require-files", action="store_true", help="also check servable asset files exist"
    )
    validate.set_defaults(func=_cmd_validate)

    convert = sub.add_parser("convert", help="convert between manifest schema versions")
    convert.add_argument("manifest", help="path to the source manifest JSON")
    convert.add_argument("--from", dest="version_from", type=int, required=True, choices=(1, 2, 3))
    convert.add_argument("--to", dest="version_to", type=int, required=True, choices=(2, 3))
    convert.add_argument("--out", required=True, help="output path for the converted manifest")
    convert.add_argument("--report", help="optional path for the JSON conversion report")
    convert.add_argument(
        "--pipeline-version", help="declare the current pipeline version (v3 only)"
    )
    convert.add_argument("--label-version", help="declare the current label version (v3 only)")
    convert.add_argument(
        "--processing-revision", type=int, default=1, help="current processing revision (v3 only)"
    )
    convert.set_defaults(func=_cmd_convert)

    migrate = sub.add_parser(
        "migrate-v1", help="convert a schema v1 manifest straight to the v3 contract"
    )
    migrate.add_argument("manifest", help="path to the v1 manifest JSON")
    migrate.add_argument("--out", required=True, help="output path for the v3 manifest")
    migrate.add_argument("--report", help="optional path for the JSON conversion report")
    migrate.set_defaults(func=_cmd_migrate_v1)

    schema = sub.add_parser("schema", help="print or write the manifest JSON schema")
    schema.add_argument("--out")
    schema.set_defaults(func=_cmd_schema)

    synth = sub.add_parser("synth", help="write a small synthetic dataset for smoke tests")
    synth.add_argument("--out", required=True, help="output directory")
    synth.add_argument("--count", type=int, default=24)
    synth.add_argument("--seed", type=int, default=7)
    synth.set_defaults(func=_cmd_synth)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
