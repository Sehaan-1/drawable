"""Isolated frozen CPU inference with one executor and bounded, replaceable queues."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from drawable_research_contracts import (
    CurationSnapshotV1,
    PrototypeGalleryCatalogV2,
)

from .search import ParsedSearchRequest, SearchInputError
from .settings import REPOSITORY_ROOT, Settings

logger = logging.getLogger(__name__)


def private_file(root: Path, relative: str) -> Path:
    if (
        not relative
        or Path(relative).is_absolute()
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError("Invalid prototype artifact path")
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("Prototype artifact path traverses a symbolic link")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Prototype artifact is missing")
    return path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(slots=True)
class Job:
    parsed: ParsedSearchRequest
    future: asyncio.Future[dict[str, Any]]
    queued_at: float


class PrototypeSearchRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.process: asyncio.subprocess.Process | None = None
        self.status: dict[str, Any] = {}
        self.assets: dict[str, Any] = {}
        self.catalog_id: str | None = None
        self._build_assets: OrderedDict[str, tuple[dict[str, Any], str | None]] = OrderedDict()
        self.error: str | None = "prototype_search_not_prepared"
        self.pending: OrderedDict[str, Job] = OrderedDict()
        self.wake = asyncio.Event()
        self.dispatcher: asyncio.Task[None] | None = None
        self.stderr_task: asyncio.Task[None] | None = None
        self.stderr_tail: deque[str] = deque(maxlen=12)
        self.generations: dict[str, tuple[int, float]] = {}

    @property
    def ready(self) -> bool:
        return self.error is None and self.process is not None and self.process.returncode is None

    async def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while line := await self.process.stderr.readline():
            self.stderr_tail.append(line.decode(errors="replace")[:1000])

    async def start(self) -> None:
        try:
            root = self.settings.data_root
            pointer = json.loads(private_file(root, "prototype/search/active.json").read_bytes())
            build_id = pointer["build_id"]
            if (
                not isinstance(build_id, str)
                or len(build_id) != 34
                or not build_id.startswith("prototype-")
                or any(char not in "0123456789abcdef" for char in build_id[10:])
            ):
                raise ValueError("Malformed active build ID")
            manifest_path = private_file(root, f"prototype/search/builds/{build_id}/manifest.json")
            if digest(manifest_path) != pointer["manifest_sha256"]:
                raise ValueError("Prototype build checksum mismatch")
            manifest = json.loads(manifest_path.read_bytes())
            snapshot_path = private_file(root, manifest["snapshot_path"])
            if digest(snapshot_path) != manifest["snapshot_sha256"]:
                raise ValueError("Prototype snapshot checksum mismatch")
            snapshot_document = json.loads(snapshot_path.read_bytes())
            snapshot = (
                CurationSnapshotV1.model_validate_json(snapshot_path.read_bytes())
                if snapshot_document.get("schema_version") == 1
                else PrototypeGalleryCatalogV2.model_validate_json(snapshot_path.read_bytes())
                if snapshot_document.get("schema_version") == 2
                else None
            )
            if snapshot is None:
                raise ValueError("Prototype catalog version is unsupported")
            if snapshot.purpose != "local-educational-prototype":
                raise ValueError("Prototype search requires an identified prototype snapshot")
            self.assets = {item.asset_id: item for item in snapshot.items}
            self.catalog_id = getattr(snapshot, "catalog_id", None)
            self._build_assets[build_id] = (self.assets, self.catalog_id)
            executable = os.environ.get(
                "DRAWABLE_ML_PYTHON", str(REPOSITORY_ROOT / "ml/.venv/bin/python")
            )
            env = {
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "TOKENIZERS_PARALLELISM": "false",
            }
            self.process = await asyncio.create_subprocess_exec(
                executable,
                "-u",
                "-m",
                "drawable_ml.prototype_worker",
                "serve",
                "--data-root",
                str(root),
                cwd=REPOSITORY_ROOT,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
            )
            self.stderr_task = asyncio.create_task(self._drain_stderr())
            assert self.process.stdout is not None
            response = await asyncio.wait_for(self.process.stdout.readline(), timeout=60)
            status = json.loads(response)
            if (
                status.get("build_id") != build_id
                or status.get("catalog_size", status.get("gallery_size")) != len(self.assets)
                or status.get("ready") is not True
            ):
                raise ValueError("Inference worker did not load the selected reviewed batch")
            self.status = status
            self.error = None
            self._startup_availability = None
            try:
                availability_relative = pointer.get("availability_path")
                if availability_relative and digest(private_file(root, availability_relative)) == pointer.get("availability_sha256"):
                    availability = json.loads(private_file(root, availability_relative).read_bytes())
                    if availability.get("catalog_id") == self.catalog_id:
                        self._startup_availability = availability
            except (OSError, ValueError, TypeError):
                self._startup_availability = None
            self.dispatcher = asyncio.create_task(self._dispatch())
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            logger.warning("Prototype search setup failed: %s %s", error, "".join(self.stderr_tail))
            self.error = "prototype_search_setup_failed_run_search_prepare"
            await self.close()

    async def close(self) -> None:
        if self.dispatcher is not None:
            self.dispatcher.cancel()
            await asyncio.gather(self.dispatcher, return_exceptions=True)
            self.dispatcher = None
        for job in self.pending.values():
            if not job.future.done():
                job.future.set_exception(
                    SearchInputError(503, "search_not_configured", "Local search is stopping.")
                )
        self.pending.clear()
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self.stderr_task is not None:
            await asyncio.gather(self.stderr_task, return_exceptions=True)
            self.stderr_task = None

    async def search(self, parsed: ParsedSearchRequest, *, started: float) -> dict[str, Any]:
        if not self.ready:
            raise SearchInputError(
                503,
                "search_not_configured",
                "Prepare local search with npm run search:prepare, then restart it.",
            )
        session = str(parsed.fields.session_id)
        previous = self.pending.get(session)
        if previous is None and len(self.pending) >= 4:
            raise SearchInputError(
                429, "search_queue_full", "Local search is busy; retry shortly.", retryable=True
            )
        if previous is not None and not previous.future.done():
            previous.future.set_exception(
                SearchInputError(
                    409, "superseded_search", "A newer drawing replaced this waiting search."
                )
            )
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        job = Job(parsed, future, monotonic())
        self.pending[session] = job
        self.wake.set()
        try:
            remaining = 10 - (monotonic() - started)
            if remaining <= 0:
                raise TimeoutError
            return await asyncio.wait_for(asyncio.shield(future), timeout=remaining)
        except TimeoutError as error:
            future.cancel()
            raise SearchInputError(
                504,
                "search_timeout",
                "CPU search took too long. Try again after it finishes.",
                retryable=True,
            ) from error
        except asyncio.CancelledError:
            future.cancel()
            raise
        finally:
            if self.pending.get(session) is job:
                self.pending.pop(session)

    async def _dispatch(self) -> None:
        while True:
            await self.wake.wait()
            if not self.pending:
                self.wake.clear()
                continue
            _, job = self.pending.popitem(last=False)
            if not self.pending:
                self.wake.clear()
            if job.future.done():
                continue
            latest = self.generations.get(str(job.parsed.fields.session_id))
            if latest is not None and latest[0] != job.parsed.fields.query_generation:
                job.future.set_exception(
                    SearchInputError(
                        409, "superseded_search", "A newer drawing replaced this waiting search."
                    )
                )
                continue
            try:
                process = self.process
                assert (
                    process is not None and process.stdin is not None and process.stdout is not None
                )
                identifier = str(uuid4())
                queue_ms = (monotonic() - job.queued_at) * 1000
                command = {
                    "id": identifier,
                    "image": base64.b64encode(job.parsed.image_png).decode("ascii"),
                }
                process.stdin.write(json.dumps(command).encode() + b"\n")
                await process.stdin.drain()
                # Hold the executor until inference actually finishes, even if its HTTP waiter times out.
                response = json.loads(await process.stdout.readline())
                if response.get("id") != identifier or "result" not in response:
                    raise RuntimeError("Inference worker returned an invalid response")
                result = response["result"]
                if result.get("build_id") != self.status["build_id"]:
                    raise RuntimeError("Inference worker mixed gallery builds")
                withdrawn_asset_ids = result.pop("withdrawn_asset_ids", [])
                if withdrawn_asset_ids:
                    try:
                        from .references import withdraw_availability

                        withdrawal = withdraw_availability(
                            self.settings.data_root, withdrawn_asset_ids
                        )
                        if withdrawal is None:
                            logger.warning(
                                "Could not update availability for changed references: %s",
                                withdrawn_asset_ids,
                            )
                    except (OSError, ValueError, RuntimeError):
                        logger.exception(
                            "Could not update availability for changed references: %s",
                            withdrawn_asset_ids,
                        )
                result["queue_ms"] = queue_ms
                if not job.future.done():
                    job.future.set_result(result)
            except asyncio.CancelledError:
                if not job.future.done():
                    job.future.set_exception(
                        SearchInputError(503, "search_not_configured", "Local search is stopping.")
                    )
                raise
            except (OSError, ValueError, RuntimeError, KeyError, TypeError):
                self.error = "prototype_search_worker_failed_restart_required"
                if not job.future.done():
                    job.future.set_exception(
                        SearchInputError(
                            503,
                            "search_worker_failed",
                            "Local search stopped; restart the search workspace.",
                        )
                    )

    def _availability(self) -> dict[str, Any] | None:
        fail_closed = {"items": {}} if self.catalog_id is not None else None
        try:
            pointer = json.loads(private_file(self.settings.data_root, "prototype/search/active.json").read_bytes())
            # Keep serving the build selected at startup while another build is activated.
            if pointer.get("build_id") != self.status.get("build_id"):
                relative = pointer.get("availability_path")
                expected = pointer.get("availability_sha256")
                if relative and isinstance(expected, str):
                    path = private_file(self.settings.data_root, relative)
                    if digest(path) != expected:
                        return fail_closed
                    active = json.loads(path.read_bytes())
                    rows = active.get("items")
                    if not isinstance(rows, dict):
                        return fail_closed
                    filtered = {
                        asset_id: row
                        for asset_id, row in rows.items()
                        if asset_id in self.assets
                        and isinstance(row, dict)
                        and row.get("sha256") == self.assets[asset_id].sha256
                        and row.get("reviewed_revision") == self.assets[asset_id].asset_revision
                    }
                    return {"schema_version": 1, "catalog_id": self.catalog_id, "items": filtered}
                return getattr(self, "_startup_availability", fail_closed) or fail_closed
            relative = pointer.get("availability_path")
            expected = pointer.get("availability_sha256")
            if not relative or not isinstance(expected, str):
                return fail_closed
            path = private_file(self.settings.data_root, relative)
            if digest(path) != expected:
                return fail_closed
            availability = json.loads(path.read_bytes())
            rows = availability.get("items")
            if (
                availability.get("catalog_id") != self.catalog_id
                or not isinstance(rows, dict)
                or not set(rows).issubset(self.assets)
                or any(
                    not isinstance(row, dict)
                    or row.get("sha256") != self.assets[asset_id].sha256
                    or row.get("reviewed_revision") != self.assets[asset_id].asset_revision
                    for asset_id, row in rows.items()
                )
            ):
                return fail_closed
            return availability
        except (OSError, ValueError, TypeError, KeyError):
            return getattr(self, "_startup_availability", fail_closed) or fail_closed

    def _assets_for_build(self, build_id: str) -> tuple[dict[str, Any], str | None]:
        if (
            not isinstance(build_id, str)
            or len(build_id) != 34
            or not build_id.startswith("prototype-")
            or any(char not in "0123456789abcdef" for char in build_id[10:])
        ):
            raise ValueError("Prototype build identity is invalid")
        cached = self._build_assets.get(build_id)
        if cached is not None:
            self._build_assets.move_to_end(build_id)
            return cached
        manifest_path = private_file(
            self.settings.data_root, f"prototype/search/builds/{build_id}/manifest.json"
        )
        manifest = json.loads(manifest_path.read_bytes())
        body = {key: value for key, value in manifest.items() if key != "build_id"}
        manifest_hash = hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if manifest.get("build_id") != build_id or f"prototype-{manifest_hash[:24]}" != build_id:
            raise ValueError("Pinned prototype build manifest identity is invalid")
        snapshot_path = private_file(self.settings.data_root, manifest["snapshot_path"])
        if digest(snapshot_path) != manifest["snapshot_sha256"]:
            raise ValueError("Pinned prototype catalog checksum is invalid")
        document = json.loads(snapshot_path.read_bytes())
        snapshot = (
            CurationSnapshotV1.model_validate_json(snapshot_path.read_bytes())
            if document.get("schema_version") == 1
            else PrototypeGalleryCatalogV2.model_validate_json(snapshot_path.read_bytes())
            if document.get("schema_version") == 2
            else None
        )
        if snapshot is None or snapshot.purpose != "local-educational-prototype":
            raise ValueError("Pinned prototype catalog is unsupported")
        result = ({item.asset_id: item for item in snapshot.items}, getattr(snapshot, "catalog_id", None))
        self._build_assets[build_id] = result
        while len(self._build_assets) > 4:
            self._build_assets.popitem(last=False)
        return result

    def _active_catalog_availability(
        self,
    ) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
        pointer = json.loads(
            private_file(self.settings.data_root, "prototype/search/active.json").read_bytes()
        )
        active_id = pointer.get("build_id")
        if (
            not isinstance(active_id, str)
            or len(active_id) != 34
            or not active_id.startswith("prototype-")
            or any(char not in "0123456789abcdef" for char in active_id[10:])
        ):
            raise ValueError("Active prototype build identity is invalid")
        active_manifest = private_file(
            self.settings.data_root, f"prototype/search/builds/{active_id}/manifest.json"
        )
        if digest(active_manifest) != pointer.get("manifest_sha256"):
            raise ValueError("Active prototype manifest checksum is invalid")
        assets, catalog_id = self._assets_for_build(active_id)
        if catalog_id is None:
            return None, None, assets
        relative = pointer.get("availability_path")
        expected = pointer.get("availability_sha256")
        if not relative or not isinstance(expected, str):
            raise ValueError("Active reference availability is missing")
        path = private_file(self.settings.data_root, relative)
        if digest(path) != expected:
            raise ValueError("Active reference availability checksum is invalid")
        availability = json.loads(path.read_bytes())
        rows = availability.get("items")
        if (
            availability.get("catalog_id") != catalog_id
            or not isinstance(rows, dict)
            or not set(rows).issubset(assets)
            or any(
                not isinstance(row, dict)
                or row.get("sha256") != assets[asset_id].sha256
                or row.get("reviewed_revision") != assets[asset_id].asset_revision
                for asset_id, row in rows.items()
            )
        ):
            raise ValueError("Active reference availability does not match its catalog")
        return availability, catalog_id, assets

    def _withdraw_changed_reference(self, asset_id: str) -> None:
        try:
            from .references import withdraw_availability

            result = withdraw_availability(self.settings.data_root, [asset_id])
            if result is None:
                logger.warning("Could not withdraw changed prototype reference %s", asset_id)
        except (OSError, ValueError, RuntimeError):
            logger.exception("Could not withdraw changed prototype reference %s", asset_id)

    def preview(self, asset_id: str, build_id: str, *, thumbnail: bool = False) -> tuple[Path, str]:
        if (
            not self.ready
            or not isinstance(build_id, str)
            or len(build_id) != 34
            or not build_id.startswith("prototype-")
            or any(char not in "0123456789abcdef" for char in build_id[10:])
        ):
            raise LookupError("Reference is absent from the active reviewed batch")
        try:
            assets, catalog_id = self._assets_for_build(build_id)
        except (OSError, ValueError, TypeError, KeyError):
            raise LookupError("Pinned reference build is unavailable") from None
        item = assets.get(asset_id)
        if item is None:
            raise LookupError("Reference is absent from the pinned build")
        try:
            if build_id == self.status.get("build_id"):
                availability = self._availability()
            else:
                current_availability, current_catalog_id, current_assets = (
                    self._active_catalog_availability()
                )
                if current_catalog_id is None:
                    current_item = current_assets.get(asset_id)
                    availability = (
                        {
                            "items": {
                                asset_id: {
                                    "sha256": current_item.sha256,
                                    "reviewed_revision": current_item.asset_revision,
                                }
                            }
                        }
                        if current_item is not None
                        else {"items": {}}
                    )
                else:
                    availability = current_availability
        except (OSError, ValueError, TypeError, KeyError):
            availability = None
        if catalog_id is not None and availability is None:
            raise LookupError("Reference availability could not be verified for this gallery")
        if availability is not None:
            row = availability.get("items", {}).get(asset_id)
            if not isinstance(row, dict) or row.get("sha256") != item.sha256 or row.get("reviewed_revision") != item.asset_revision:
                raise LookupError("Reference is no longer available in this reviewed build")
        relative = item.relative_path
        expected_hash = item.sha256
        media_type = item.media_type
        if thumbnail and getattr(item, "thumbnail_path", None):
            relative = item.thumbnail_path
            expected_hash = item.thumbnail_sha256
            media_type = "image/webp"
        try:
            path = private_file(self.settings.data_root, relative)
            changed = (
                (not thumbnail and path.stat().st_size != item.byte_length)
                or digest(path) != expected_hash
            )
        except (OSError, ValueError):
            self._withdraw_changed_reference(asset_id)
            raise ValueError("Reference bytes changed; prepare local search again") from None
        if changed:
            self._withdraw_changed_reference(asset_id)
            raise ValueError("Reference bytes changed; prepare local search again")
        return path, media_type
