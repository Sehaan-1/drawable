"""``linescout-manifest`` command line: validate, migrate, schema, and synth.

Usage::

    linescout-manifest validate path/to/manifest.json [--data-root DIR] [--require-files]
    linescout-manifest migrate-v1 path/to/v1-manifest.json --out path/to/v2.json
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
    Manifest,
    check_parent_integrity,
    check_split_integrity,
    dump_json_schema,
    learning_split_report,
)
from linescout_ml.migrate import migrate_manifest
from linescout_ml.synthetic import write_synthetic_dataset


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.manifest)
    try:
        manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        print(f"INVALID {path}: {error}", file=sys.stderr)
        hint = "If this is a schema v1 manifest, run: linescout-manifest migrate-v1"
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
    print(json.dumps(learning_split_report(manifest.records), indent=None, sort_keys=True))
    return 0


def _cmd_migrate_v1(args: argparse.Namespace) -> int:
    source = Path(args.manifest)
    try:
        manifest, report = migrate_manifest(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as error:
        print(f"MIGRATION FAILED {source}: {error}", file=sys.stderr)
        return 1

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination} ({report.records_migrated} records)")
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    if args.report:
        Path(args.report).write_text(
            json.dumps(report.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.report}")
    if report.serving_eligibility_lost:
        print(
            f"NOTE {report.serving_eligibility_lost} previously enabled record(s) are NOT "
            "servable under v2: record permission provenance and human SFW approval to "
            "restore serving. No asset gained permission or human approval.",
            file=sys.stderr,
        )
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

    migrate = sub.add_parser(
        "migrate-v1", help="migrate a schema v1 manifest to the frozen v2 contract"
    )
    migrate.add_argument("manifest", help="path to the v1 manifest JSON")
    migrate.add_argument("--out", required=True, help="output path for the v2 manifest")
    migrate.add_argument("--report", help="optional path for the JSON migration report")
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
