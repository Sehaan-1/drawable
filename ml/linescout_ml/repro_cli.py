"""``linescout-repro`` — the reproducibility toolbox for the Colab pipeline.

Five things a person needs when a run must be re-created, and each one is
deliberately a command rather than folklore:

``linescout-repro checkout --dir /path/to/drawable``
    Clone or reuse the repository at one immutable commit and verify HEAD, which
    is exactly what the notebook's environment cell does — so a workstation can
    be checked the same way a T4 runtime is.

``linescout-repro pin [--check]``
    What commit the notebook pins, whether it is still the placeholder, and
    whether the environment spec next to it is the one being installed.

``linescout-repro environment plan|install|report|record|drift``
    Which pinned packages a run needs, install them without touching torch, and
    record what the runtime actually turned out to be.

``linescout-repro checkpoints show|verify|record``
    Show the pinned model revisions and digests, verify what is in the cache, and
    pin an artifact the upstream publisher never hashed.

``linescout-repro selfcheck``
    The notebook ↔ package ↔ lockfile ↔ docs audit CI runs.

Everything here is stdlib + pydantic: no torch, no numpy, no GPU. Run it from a
``uv sync --frozen`` environment or from a Colab cell.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from linescout_ml.colab import repro
from linescout_ml.colab.checkpoints import (
    CheckpointError,
    CheckpointMismatchError,
    CheckpointUnpinnedError,
    CheckpointVerifier,
    load_model_lock,
    package_lock_path,
    sha256_file,
    summarize_lock,
)
from linescout_ml.colab.config import SourceConfigurationError, resolve_sources
from linescout_ml.colab.selfcheck import run_selfcheck


def _print_rows(rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    if not rows:
        print("(nothing)")
        return
    widths = {key: max(len(key), *(len(str(row.get(key, ""))) for row in rows)) for key in columns}
    print("  ".join(key.ljust(widths[key]) for key in columns))
    for row in rows:
        print("  ".join(str(row.get(key, "")).ljust(widths[key]) for key in columns))


# ----------------------------------------------------------------------- checkout


def _cmd_checkout(args: argparse.Namespace) -> int:
    try:
        record = repro.ensure_checkout(
            Path(args.dir),
            revision=args.rev,
            url=args.url,
            mirrors=tuple(args.mirror),
            on_note=lambda message: print(f"  note: {message}", file=sys.stderr),
        )
    except (repro.CheckoutError, repro.RevisionMismatchError) as error:
        print(f"CHECKOUT FAILED: {error}", file=sys.stderr)
        return 1
    payload = record.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for line in record.describe():
            print(line)
    return 0 if record.usable else 1


# ------------------------------------------------------------------------ pin


def _cmd_repro_pin(args: argparse.Namespace) -> int:
    """Report the repository pin the notebook uses, and how final it is.

    ``--check`` is for CI: it exits 1 while the pin is still the placeholder. The
    distinction matters because a notebook that clones an unreleased commit and a
    notebook that clones *whatever main is* fail in exactly the same way later — and
    only one of them is visible before someone spends an hour on a T4.
    """
    pin = str(args.pin or repro.COLAB_PIN)
    payload: dict[str, Any] = {
        "repo": repro.COLAB_REPO,
        "url": repro.COLAB_REPO_URL,
        "mirrors": list(repro.COLAB_MIRROR_URLS),
        "pin": pin,
        "immutable": repro.is_immutable_revision(pin),
        "placeholder": pin == repro.COLAB_PIN_PLACEHOLDER,
        "finalised": repro.pin_is_finalised(pin),
        "environment_spec": "/".join(repro.ENVIRONMENT_SPEC_PATH),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif payload["placeholder"]:
        print(
            f"pin for {repro.COLAB_REPO}: {pin}\n"
            "  PLACEHOLDER — no such commit yet, so this checkout cannot be verified.\n"
            "  Replace COLAB_PIN in ml/linescout_ml/colab/repro.py (and REPO_PIN in the\n"
            "  notebook) with the 40-hex commit that contains the finalized pipeline.\n"
            "  Nothing derives the pin from the current checkout: if it did, the repo\n"
            "  you happened to clone would decide what a stranger's GPU session runs."
        )
    else:
        mirrors = ", ".join(payload["mirrors"]) or "none"
        print(
            f"pin for {repro.COLAB_REPO}: {pin}\n"
            f"  immutable: {payload['immutable']}\n"
            f"  mirrors: {mirrors}\n"
            f"  environment: {payload['environment_spec']}"
        )
        if not payload["immutable"]:
            print("  WARNING: a branch or tag can move; a 40-hex commit cannot.")
    if args.check and not payload["finalised"]:
        print("PIN NOT FINALISED", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------- environment


def _resolve_preset_entries(presets: list[str], sources_root: Path | None) -> list[Any]:
    entries = [
        {"preset": preset, "license_id": "selfcheck-only"} for preset in (presets or ["synthetic"])
    ]
    return resolve_sources(entries, sources_root=sources_root or Path("/tmp/linescout-sources"))


def _cmd_environment_plan(args: argparse.Namespace) -> int:
    spec = repro.load_requirement_spec(Path(args.spec) if args.spec else None)
    try:
        sources = _resolve_preset_entries(
            args.preset, Path(args.sources_root) if args.sources_root else None
        )
    except SourceConfigurationError as error:
        print(f"SOURCE CONFIGURATION FAILED: {error}", file=sys.stderr)
        return 1
    plan = repro.plan_environment(
        sources,
        spec=spec,
        extract_line_art=not args.no_extract,
        label=not args.no_label,
        embed=not args.no_embed,
        embedders=args.embedder or ["mobileclip2_s2", "dinov2_vits14"],
    )
    if args.json:
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"environment spec: {spec.path if spec else 'MISSING'}")
        for line in plan.describe():
            print(line)
    return 0


def _cmd_environment_install(args: argparse.Namespace) -> int:
    spec = repro.load_requirement_spec(Path(args.spec) if args.spec else None)
    sources = _resolve_preset_entries(args.preset, None)
    plan = repro.plan_environment(
        sources,
        spec=spec,
        extract_line_art=not args.no_extract,
        label=not args.no_label,
        embed=not args.no_embed,
        embedders=args.embedder or ["mobileclip2_s2", "dinov2_vits14"],
    )
    targets = plan.to_install if args.only_missing else plan.packages
    if not targets:
        print("nothing to install; the runtime already satisfies the plan")
        return 0
    command = [args.python, "-m", "pip", "install", *targets]
    if args.constraints:
        command += ["--constraint", args.constraints]
    print("$ " + " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def _cmd_environment_report(args: argparse.Namespace) -> int:
    report = repro.runtime_report(snapshot=args.freeze)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _cmd_environment_record(args: argparse.Namespace) -> int:
    report = repro.runtime_report(snapshot=False)
    unrecorded = sorted(
        [name for name in repro.RUNTIME_PACKAGES if not report["packages"].get(name)]
        + (["cuda"] if not isinstance(report["cuda"], str) else [])
    )
    payload = {
        "schema_version": 1,
        "recorded_at": _today(),
        "recorded_on": f"{report['runtime']} · Python {report['python']} · {report['platform']}",
        "note": [
            "What a *specific* runtime had installed, recorded with `linescout-repro",
            "environment record`. The notebook compares against this file and prints",
            "the difference; it never assumes the runtime matches, and never installs",
            "torch to make the numbers agree. A CPU dev container records no torch and",
            "no CUDA level, which is exactly why the file is a report and not a spec.",
        ],
        "runtime": {
            "python": report["python"],
            "torch": (report["packages"].get("torch")),
            "cuda": (
                report["cuda"] if isinstance(report["cuda"], str) else report["cuda"].get("version")
            ),
        },
        "packages": report["packages"],
        "unrecorded": unrecorded,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
    return 0


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _cmd_environment_drift(args: argparse.Namespace) -> int:
    report = repro.runtime_report()
    baseline = repro.load_baseline(Path(args.baseline) if args.baseline else None)
    drift = repro.compare_runtime(report, baseline)
    if not drift:
        print("runtime matches the recorded baseline")
        return 0
    print("runtime differs from the recorded baseline:")
    for name, entry in sorted(drift.items()):
        print(f"  {name}: expected {entry['expected']}, found {entry['actual']}")
    return 1 if args.strict else 0


# ----------------------------------------------------------------------- checkpoints


def _cmd_checkpoints_show(args: argparse.Namespace) -> int:
    lock_path = Path(args.lock) if args.lock else package_lock_path()
    try:
        lock = load_model_lock(lock_path)
    except CheckpointError as error:
        print(f"CHECKPOINT LOCK UNREADABLE: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(
            json.dumps(
                {"lock": str(lock_path), "artifacts": lock.model_dump(mode="json")}, indent=2
            )
        )
        return 0
    print(f"lock: {lock_path}")
    _print_rows(
        [dict(row, sha256=row["sha256"] or "unpinned") for row in summarize_lock(lock)],
        ("group", "artifact", "revision", "sha256", "license"),
    )
    unpinned = [artifact.id for artifact in lock.unpinned()]
    if unpinned:
        print(f"\n{len(unpinned)} artifact(s) carry no digest: {', '.join(sorted(unpinned))}")
    return 0


def _cmd_checkpoints_verify(args: argparse.Namespace) -> int:
    lock = load_model_lock(Path(args.lock) if args.lock else None)
    verifier = CheckpointVerifier(
        lock,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        policy=args.policy,
    )
    groups = [args.group] if args.group else sorted(lock.groups())
    failures = 0
    for group in groups:
        for artifact in lock.for_group(group):
            candidate = verifier.cache_dir / "files" / artifact.id
            if not candidate.is_file():
                print(f"absent   {artifact.id} (not in {verifier.cache_dir})")
                continue
            try:
                verification = verifier.verify(artifact, candidate)
            except (CheckpointMismatchError, CheckpointUnpinnedError, CheckpointError) as error:
                print(f"FAILED   {artifact.id}: {error}", file=sys.stderr)
                failures += 1
                continue
            print(f"{verification.status:<9} {artifact.id} {verification.sha256 or ''}"[:120])
    if failures:
        print(f"{failures} artifact(s) failed verification", file=sys.stderr)
        return 1
    print("cache contents match the lock")
    return 0


def _cmd_checkpoints_record(args: argparse.Namespace) -> int:
    """Pin an artifact by hashing a file you already trust.

    This is the *only* supported way to fill a ``"sha256": null`` hole, and it
    writes a normal file edit: it shows up in ``git diff``, so a new digest has to
    pass review like any other change to a pin.
    """
    lock_path = Path(args.lock) if args.lock else package_lock_path()
    source = Path(args.file)
    if not source.is_file():
        print(f"no such file: {source}", file=sys.stderr)
        return 1
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", [])
    for entry in artifacts:
        if entry.get("id") != args.artifact:
            continue
        entry["sha256"] = sha256_file(source)
        entry["size_bytes"] = source.stat().st_size
        break
    else:
        print(f"{args.artifact!r} is not in {lock_path}", file=sys.stderr)
        return 1
    destination = Path(args.out) if args.out else lock_path
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"pinned {args.artifact} in {destination}; commit this change so the pin is reviewed")
    return 0


# ------------------------------------------------------------------------ selfcheck


def _cmd_selfcheck(args: argparse.Namespace) -> int:
    problems = run_selfcheck(Path(args.root) if args.root else None)
    if args.json:
        print(json.dumps({"problems": problems}, indent=2))
    elif problems:
        print(f"{len(problems)} consistency problem(s):")
        for problem in problems:
            print(f"  · {problem}")
    else:
        print("notebook, package, environment spec, checkpoint lock, and docs agree")
    return 1 if problems else 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="linescout-repro")
    sub = parser.add_subparsers(dest="command", required=True)

    checkout = sub.add_parser("checkout", help="clone or verify the repo at one immutable commit")
    checkout.add_argument("--dir", default="/content/drawable")
    checkout.add_argument("--url", default=repro.COLAB_REPO_URL)
    checkout.add_argument("--mirror", action="append", default=list(repro.COLAB_MIRROR_URLS))
    checkout.add_argument("--rev", default=repro.COLAB_PIN)
    checkout.add_argument("--json", action="store_true")
    checkout.set_defaults(handler=_cmd_checkout)

    pin = sub.add_parser("pin", help="the commit the notebook pins, and whether that is final")
    pin.add_argument("--pin", help="evaluate a candidate revision instead of COLAB_PIN")
    pin.add_argument(
        "--check", action="store_true", help="exit 1 unless the pin names a real commit"
    )
    pin.add_argument("--json", action="store_true")
    pin.set_defaults(handler=_cmd_repro_pin)

    environment = sub.add_parser("environment", help="the pinned environment, planned and recorded")
    env_sub = environment.add_subparsers(dest="action", required=True)

    def _stage_flags(command: argparse.ArgumentParser) -> None:
        command.add_argument("--preset", action="append", default=[], help="repeatable")
        command.add_argument("--embedder", action="append", default=[])
        command.add_argument("--no-extract", action="store_true")
        command.add_argument("--no-label", action="store_true")
        command.add_argument("--no-embed", action="store_true")

    plan = env_sub.add_parser("plan", help="which packages this run needs")
    plan.add_argument("--sources-root")
    plan.add_argument("--spec")
    plan.add_argument("--json", action="store_true")
    _stage_flags(plan)
    plan.set_defaults(handler=_cmd_environment_plan)

    install = env_sub.add_parser("install", help="pip install exactly that subset")
    install.add_argument("--python", default=sys.executable)
    install.add_argument("--constraints", help="constraints file, e.g. Colab's own torch build")
    install.add_argument("--spec")
    install.add_argument("--only-missing", action="store_true", default=True)
    install.add_argument("--all", dest="only_missing", action="store_false")
    install.add_argument("--dry-run", action="store_true")
    _stage_flags(install)
    install.set_defaults(handler=_cmd_environment_install)

    report = env_sub.add_parser("report", help="what versions this runtime actually has")
    report.add_argument("--freeze", action="store_true", help="include a full pip snapshot")
    report.set_defaults(handler=_cmd_environment_report)

    record = env_sub.add_parser("record", help="write a runtime baseline from this runtime")
    record.add_argument("--out", default=str(repro.runtime_baseline_path()))
    record.set_defaults(handler=_cmd_environment_record)

    drift = env_sub.add_parser("drift", help="compare this runtime against the baseline")
    drift.add_argument("--baseline")
    drift.add_argument("--strict", action="store_true", help="exit 1 on any difference")
    drift.set_defaults(handler=_cmd_environment_drift)

    checkpoints = sub.add_parser("checkpoints", help="pinned model code and weights")
    cp_sub = checkpoints.add_subparsers(dest="action", required=True)

    show = cp_sub.add_parser("show", help="the lock, as a table")
    show.add_argument("--lock")
    show.add_argument("--json", action="store_true")
    show.set_defaults(handler=_cmd_checkpoints_show)

    verify = cp_sub.add_parser("verify", help="check what the cache holds against the lock")
    verify.add_argument("--cache-dir")
    verify.add_argument("--lock")
    verify.add_argument("--group")
    verify.add_argument("--policy", choices=("strict", "record", "off"), default="record")
    verify.set_defaults(handler=_cmd_checkpoints_verify)

    rec = cp_sub.add_parser("record", help="pin an artifact from a file you have verified")
    rec.add_argument("--artifact", required=True)
    rec.add_argument("--file", required=True)
    rec.add_argument("--lock")
    rec.add_argument("--out")
    rec.set_defaults(handler=_cmd_checkpoints_record)

    selfcheck = sub.add_parser("selfcheck", help="notebook/package/lockfile/docs consistency audit")
    selfcheck.add_argument("--root")
    selfcheck.add_argument("--json", action="store_true")
    selfcheck.set_defaults(handler=_cmd_selfcheck)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Any = getattr(args, "handler", None)
    if handler is None:  # pragma: no cover - argparse requires a subcommand
        return 2
    return int(handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
