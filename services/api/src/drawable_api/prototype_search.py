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

from drawable_research_contracts import CurationSnapshotV1, validate_curation_snapshot_files

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
            snapshot = CurationSnapshotV1.model_validate_json(snapshot_path.read_bytes())
            if snapshot.purpose != "local-educational-prototype":
                raise ValueError("Prototype search requires an identified prototype snapshot")
            validate_curation_snapshot_files(snapshot, root)
            self.assets = {item.asset_id: item for item in snapshot.items}
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
                or status.get("gallery_size") != len(self.assets)
                or status.get("ready") is not True
            ):
                raise ValueError("Inference worker did not load the selected reviewed batch")
            self.status = status
            self.error = None
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

    def preview(self, asset_id: str, build_id: str) -> tuple[Path, str]:
        if not self.ready or build_id != self.status.get("build_id") or asset_id not in self.assets:
            raise LookupError("Reference is absent from the active reviewed batch")
        item = self.assets[asset_id]
        path = private_file(self.settings.data_root, item.relative_path)
        if path.stat().st_size != item.byte_length or digest(path) != item.sha256:
            raise ValueError("Reference bytes changed; prepare local search again")
        return path, item.media_type
