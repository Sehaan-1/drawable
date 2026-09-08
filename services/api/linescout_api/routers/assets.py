"""Serve thumbnails and trace-compatible line art for *eligible* gallery assets only.

Public serving enforces the canonical eligibility policy at request time, in
addition to the DB gate: every request re-verifies that the file still exists
inside the data root *and* that its bytes match the hash recorded in the
manifest. A tampered, replaced, or missing file fails closed with
``asset_unavailable`` and is dropped from this session's serving list — it is
never served from a stale cache position.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import FileResponse

from linescout_api.deps import State
from linescout_api.errors import not_found
from linescout_api.gallery import asset_file

router = APIRouter(tags=["assets"])

_KINDS: dict[str, Literal["thumbnail", "line_art"]] = {
    "thumbnail": "thumbnail",
    "line-art": "line_art",
}


def _verify_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@router.get("/assets/{asset_id}/{kind}", response_class=FileResponse)
def get_asset_file(state: State, asset_id: str, kind: str) -> FileResponse:
    if kind not in _KINDS or state.gallery is None:
        raise not_found("asset_not_found", "asset not found")
    file = asset_file(state.connection, asset_id, _KINDS[kind])
    if file is None:
        raise not_found("asset_not_found", "asset not found")
    path = (state.gallery.data_root / file.relative_path).resolve()
    if state.gallery.data_root.resolve() not in path.parents or not path.is_file():
        # Missing/corrupt gallery asset: drop it from this session's serving
        # list only. A missing file is runtime availability, never a change to
        # the asset's recorded permission or review state (the derived
        # ``enabled`` flag mirrors the manifest, nothing else).
        state.assets = [asset for asset in state.assets if asset.asset_id != asset_id]
        raise not_found("asset_unavailable", "asset file is missing this session")
    # Fail closed on tampered bytes: the served content must match the bytes
    # the manifest vouches for, or it is not the asset that was approved.
    try:
        digest = _verify_bytes(path)
    except OSError:
        state.assets = [asset for asset in state.assets if asset.asset_id != asset_id]
        raise not_found("asset_unavailable", "asset file is unreadable this session") from None
    if digest != file.sha256:
        state.assets = [asset for asset in state.assets if asset.asset_id != asset_id]
        raise not_found(
            "asset_unavailable", "asset file bytes changed since the manifest was written"
        )
    return FileResponse(
        path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400, immutable"}
    )
