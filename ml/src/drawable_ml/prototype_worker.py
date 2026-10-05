"""Bounded JSON-lines IPC for the isolated, persistent CPU inference process."""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from .prototype import PrototypeGallery, prepare


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, allow_nan=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local educational prototype search (frozen CPU models)"
    )
    parser.add_argument("command", choices=("prepare", "serve"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--snapshot")
    parser.add_argument(
        "--structural", action="store_true", help="Enable the experimental DINO comparison profile"
    )
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            with contextlib.redirect_stdout(sys.stderr):
                result = prepare(args.data_root, args.snapshot)
            emit(result)
            return 0
        with contextlib.redirect_stdout(sys.stderr):
            gallery = PrototypeGallery(args.data_root, structural=args.structural)
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
