"""Bounded JSON-lines IPC for the isolated, persistent CPU inference process."""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from .prototype import PrototypeGallery, activate_build, prepare, prepare_reference_neighbors


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, allow_nan=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local educational prototype search (frozen CPU models)"
    )
    parser.add_argument(
        "command", choices=("prepare", "serve", "validate", "activate", "reference-neighbors")
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--snapshot")
    parser.add_argument(
        "--structural", action="store_true", help="Enable the experimental DINO comparison profile"
    )
    parser.add_argument("--semantic-only", action="store_true", help="Build only the frozen MobileCLIP baseline")
    parser.add_argument("--stage-only", action="store_true", help="Build without changing the active pointer")
    parser.add_argument("--build-id", help="Validated immutable build ID for validate/activate")
    parser.add_argument("--staged", action="store_true", help="Validate against the selected V2 catalog's private availability mask before activation")
    parser.add_argument("--candidate-manifest")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            with contextlib.redirect_stdout(sys.stderr):
                result = prepare(
                    args.data_root,
                    args.snapshot,
                    structural=args.structural and not args.semantic_only,
                    stage_only=args.stage_only,
                )
            emit(result)
            return 0
        if args.command == "activate":
            if not args.build_id:
                raise ValueError("activate requires --build-id")
            with contextlib.redirect_stdout(sys.stderr):
                result = activate_build(args.data_root, args.build_id)
            emit(result)
            return 0
        if args.command == "validate":
            if not args.build_id:
                raise ValueError("validate requires --build-id")
            with contextlib.redirect_stdout(sys.stderr):
                gallery = PrototypeGallery(
                    args.data_root,
                    build_id=args.build_id,
                    staged_validation=args.staged,
                )
                result = gallery.status()
            emit(result)
            return 0
        if args.command == "reference-neighbors":
            if not args.candidate_manifest or not args.output:
                raise ValueError("reference-neighbors requires --candidate-manifest and --output")
            with contextlib.redirect_stdout(sys.stderr):
                result = prepare_reference_neighbors(
                    args.data_root, args.candidate_manifest, args.output
                )
            emit(result)
            return 0
        with contextlib.redirect_stdout(sys.stderr):
            gallery = PrototypeGallery(
                args.data_root,
                structural=args.structural,
                tolerate_corrupt_assets=True,
            )
        emit(gallery.status())
        while line := sys.stdin.buffer.readline(6 * 1024 * 1024):
            if not line.endswith(b"\n"):
                raise ValueError("Inference IPC message is too large")
            request = json.loads(line)
            try:
                image = base64.b64decode(request["image"], validate=True)
                if len(image) > 4 * 1024 * 1024:
                    raise ValueError("Inference image is too large")
                with contextlib.redirect_stdout(sys.stderr):
                    result = gallery.search(image)
                emit({"id": request["id"], "result": result})
            except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
                print(f"Prototype inference failed: {error}", file=sys.stderr)
                emit(
                    {
                        "id": request.get("id"),
                        "error": "Local inference failed; restart the search workspace.",
                    }
                )
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        print(f"Local prototype search: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
