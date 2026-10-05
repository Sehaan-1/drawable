from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from drawable_api.app import create_app
from drawable_api.settings import Settings

root = Path(__file__).resolve().parents[1]
openapi_path = root / "packages" / "contracts" / "openapi.json"
settings = Settings(
    data_root=root / "datasets",
    api_mode="fixture",
    metadata_db=root / "datasets" / "metadata.sqlite3",
)
parser = argparse.ArgumentParser(
    description="Generate or check the OpenAPI contract without touching data."
)
parser.add_argument(
    "--check",
    action="store_true",
    help="Compare generated output without modifying tracked files",
)
args = parser.parse_args()
generated = json.dumps(create_app(settings).openapi(), indent=2, sort_keys=True) + "\n"
if args.check:
    current = openapi_path.read_text(encoding="utf-8") if openapi_path.is_file() else ""
    if current != generated:
        diff = difflib.unified_diff(
            current.splitlines(keepends=True),
            generated.splitlines(keepends=True),
            fromfile=str(openapi_path.relative_to(root)),
            tofile="generated OpenAPI",
        )
        print("".join(diff), end="")
        raise SystemExit(
            "OpenAPI contract is stale; run npm run api:openapi to regenerate it"
        )
    print(f"Verified {openapi_path.relative_to(root)} without writing files")
else:
    openapi_path.parent.mkdir(parents=True, exist_ok=True)
    openapi_path.write_text(generated, encoding="utf-8")
    print(f"Generated {openapi_path.relative_to(root)}")
