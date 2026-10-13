from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from contextlib import AbstractContextManager
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .intake import sha256_file
from .settings import REPOSITORY_ROOT

REVIEW_CONFIG = REPOSITORY_ROOT / "config" / "references-reviewer-v1.json"
REVIEW_PROMPT = REPOSITORY_ROOT / "config" / "references-review-prompt-v1.txt"
RUNTIME_DIR = Path("prototype/reviewer")

Style = Literal["manga_anime", "western_ink", "realistic_academic", "cartoon", "gesture_sketch"]
Scope = Literal[
    "eye", "eyebrow", "mouth", "face_head", "hair", "hand", "foot",
    "upper_body_clothing", "full_body", "multi_character",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CropBox(StrictModel):
    x: float = Field(ge=0, le=1, allow_inf_nan=False)
    y: float = Field(ge=0, le=1, allow_inf_nan=False)
    width: float = Field(gt=0, le=1, allow_inf_nan=False)
    height: float = Field(gt=0, le=1, allow_inf_nan=False)
    purpose: str = Field(min_length=3, max_length=160)


class LocalVisualReview(StrictModel):
    useful_character_drawing: bool
    safe_content: bool
    style: Style | None
    primary_scope: Scope | None
    secondary_scopes: list[Scope] = Field(max_length=9)
    person_count: int | None = Field(default=None, ge=1)
    quality: int = Field(ge=1, le=3)
    text_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    text_crosses_important_anatomy: bool
    essential_anatomy_clipped: bool
    damaged_or_unreadable: bool
    uncertain: bool
    reason: str = Field(min_length=1, max_length=1800)
    crop_boxes: list[CropBox] = Field(max_length=3)


class LocalPairReview(StrictModel):
    relation: Literal["same_image", "adjacent_frame_duplicate", "related_distinct_pose", "distinct", "uncertain"]
    reason: str = Field(min_length=1, max_length=1200)
    pose_difference: bool
    expression_difference: bool
    facing_or_view_difference: bool
    perspective_or_silhouette_difference: bool
    useful_detail_difference: bool


class ReviewerError(RuntimeError):
    pass


def reviewer_config() -> dict[str, Any]:
    data = json.loads(REVIEW_CONFIG.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("models"), list):
        raise ReviewerError("Local reviewer pins are malformed")
    return data


def _rooted(root: Path, relative: str, *, must_exist: bool = False) -> Path:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or "\\" in relative or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ReviewerError("Reviewer artifact path is malformed")
    path = root.resolve()
    for part in parsed.parts:
        path = path / part
        if path.is_symlink():
            raise ReviewerError("Reviewer artifact path traverses a symbolic link")
    if must_exist and not path.is_file():
        raise ReviewerError("Reviewer artifact is missing")
    if not path.resolve(strict=False).is_relative_to(root.resolve()):
        raise ReviewerError("Reviewer artifact path escaped the private data root")
    return path


def _download(url: str, destination: Path, byte_count: int, expected_sha256: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        size = destination.stat().st_size
        digest = sha256_file(destination)
        if size == byte_count and digest == expected_sha256:
            return {"path": destination.name, "bytes": size, "sha256": digest, "reused": True}
        destination.unlink()

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > byte_count:
        partial.unlink()
        offset = 0
    if offset == byte_count:
        partial.unlink()
        offset = 0
    request = urllib.request.Request(url, headers={"User-Agent": "drawable-local-reference-curation/1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        response = urllib.request.urlopen(request, timeout=90)
    except (OSError, urllib.error.URLError) as error:
        raise ReviewerError(f"Pinned artifact download failed: {destination.name}: {error}") from error
    print(
        f"Downloading {destination.name}: {offset:,}/{byte_count:,} bytes already present",
        file=sys.stderr,
        flush=True,
    )
    next_progress = ((offset // (256 * 1024 * 1024)) + 1) * (256 * 1024 * 1024)
    with response, partial.open("ab" if offset else "wb") as output:
        if offset and getattr(response, "status", None) != 206:
            output.close()
            partial.unlink(missing_ok=True)
            return _download(url, destination, byte_count, expected_sha256)
        count = offset
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > byte_count:
                raise ReviewerError(f"Pinned artifact exceeded its recorded size: {destination.name}")
            output.write(block)
            if count >= next_progress:
                print(f"{destination.name}: {count:,}/{byte_count:,} bytes", file=sys.stderr, flush=True)
                next_progress += 256 * 1024 * 1024
        output.flush()
        os.fsync(output.fileno())
    if partial.stat().st_size != byte_count:
        raise ReviewerError(f"Pinned artifact size mismatch: {destination.name}; partial bytes were retained")
    if sha256_file(partial) != expected_sha256:
        partial.unlink(missing_ok=True)
        raise ReviewerError(f"Pinned artifact checksum mismatch: {destination.name}; retry setup to redownload")
    os.replace(partial, destination)
    os.chmod(destination, 0o600)
    return {"path": destination.name, "bytes": byte_count, "sha256": expected_sha256, "reused": False}


def _extract_runtime(archive: Path, destination: Path) -> Path:
    server_candidates: list[Path] = []
    staging = destination.with_name(".staging-" + str(os.getpid()))
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, mode=0o700)
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            selected: dict[tuple[str, ...], tarfile.TarInfo] = {}
            links: list[tuple[tarfile.TarInfo, tuple[str, ...]]] = []
            for member in members:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                    raise ReviewerError("Pinned runtime archive contains an unsafe path")
                if member.isdir():
                    continue
                if member.issym():
                    target = PurePosixPath(member.linkname)
                    if target.is_absolute() or any(part in {"", ".", ".."} for part in target.parts):
                        raise ReviewerError("Pinned runtime archive contains an unsafe symbolic link")
                    if ".so" not in Path(member.name).name:
                        raise ReviewerError("Pinned runtime archive contains an unexpected symbolic link")
                    target_relative = relative.parent.joinpath(*target.parts)
                    selected[relative.parts] = member
                    links.append((member, target_relative.parts))
                    continue
                if not member.isfile():
                    raise ReviewerError("Pinned runtime archive contains a non-regular file")
                if Path(member.name).name != "llama-server" and ".so" not in Path(member.name).name:
                    continue
                selected[relative.parts] = member
                if Path(member.name).name == "llama-server":
                    server_candidates.append(staging.joinpath(*relative.parts))
            for member, target_relative in links:
                if target_relative not in selected:
                    raise ReviewerError("Pinned runtime symbolic link points to an absent library")
            for member in selected.values():
                if not member.isfile():
                    continue
                relative = PurePosixPath(member.name)
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = bundle.extractfile(member)
                if source is None:
                    raise ReviewerError("Pinned runtime archive contains an unreadable file")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.chmod(target, 0o700 if Path(member.name).name == "llama-server" else 0o600)
            for member, _ in links:
                relative = PurePosixPath(member.name)
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.symlink(member.linkname, target)
        if len(server_candidates) != 1:
            raise ReviewerError("Pinned runtime archive did not contain exactly one llama-server")
        for link, target_relative in links:
            target = staging.joinpath(*target_relative)
            if not target.exists() or not target.resolve(strict=True).is_relative_to(staging.resolve()):
                raise ReviewerError("Pinned runtime symbolic link resolves outside its installation")
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
        return destination / server_candidates[0].relative_to(staging)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _extract_cuda_wheel(wheel: Path, destination: Path, library_prefix: str) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    extracted: list[dict[str, Any]] = []
    prefix = PurePosixPath(library_prefix)
    try:
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                relative = PurePosixPath(member.filename)
                if (
                    relative.is_absolute()
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or relative.as_posix().startswith("/")
                ):
                    raise ReviewerError("Pinned CUDA wheel contains an unsafe path")
                if not relative.as_posix().startswith(prefix.as_posix()) or ".so" not in relative.name:
                    continue
                mode = member.external_attr >> 16
                if mode and (mode & 0o170000) == 0o120000:
                    raise ReviewerError("Pinned CUDA wheel contains an unexpected symbolic link")
                target = destination.joinpath(*relative.parts)
                if not target.resolve(strict=False).is_relative_to(destination.resolve()):
                    raise ReviewerError("Pinned CUDA library escaped its private runtime directory")
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.chmod(target, 0o600)
                extracted.append(
                    {
                        "path": target.relative_to(destination.parent).as_posix(),
                        "bytes": target.stat().st_size,
                        "sha256": sha256_file(target),
                    }
                )
    except (OSError, zipfile.BadZipFile) as error:
        raise ReviewerError(f"Pinned CUDA runtime archive is malformed: {wheel.name}") from error
    if not extracted:
        raise ReviewerError(f"Pinned CUDA runtime archive has no libraries under {library_prefix}")
    return extracted


def _gpu_probe() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    fields = [field.strip() for field in result.stdout.strip().split(",")]

    def memory_number(index: int) -> int | None:
        try:
            return int(fields[index].split()[0])
        except (IndexError, ValueError):
            return None

    return {
        "available": result.returncode == 0 and bool(result.stdout.strip()),
        "summary": result.stdout.strip(),
        "memory_total_mb": memory_number(1),
        "memory_used_mb": memory_number(2),
        "memory_free_mb": memory_number(3),
        "error": result.stderr.strip() or None,
        "exit_code": result.returncode,
    }


def _runtime_environment(runtime_dir: Path) -> dict[str, str]:
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    library_dirs = sorted({str(path.parent) for path in runtime_dir.rglob("*.so*")})
    existing = environment.get("LD_LIBRARY_PATH")
    if existing:
        library_dirs.append(existing)
    if library_dirs:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs)
    return environment


def _cuda_devices(server: Path, runtime_dir: Path) -> list[str]:
    try:
        result = subprocess.run(
            [str(server), "--list-devices"],
            cwd=runtime_dir,
            env=_runtime_environment(runtime_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReviewerError(f"Pinned llama.cpp CUDA device probe failed: {error}") from error
    if result.returncode != 0:
        raise ReviewerError(f"Pinned llama.cpp CUDA device probe failed: {result.stderr[-800:]}")
    devices = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("Available devices"):
            name = line.split(":", 1)[0].strip()
            if name.upper().startswith("CUDA"):
                devices.append(line)
    return devices


def resize_preview(image: Image.Image, max_edge: int = 768) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return image


def _image_data_url(image: Image.Image) -> str:
    from io import BytesIO

    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


class LlamaReviewer(AbstractContextManager["LlamaReviewer"]):
    def __init__(
        self,
        root: Path,
        model_id: str,
        *,
        q8: bool = False,
        port: int | None = None,
        preferred_gpu_layers: int | None = None,
    ):
        self.root = root.resolve()
        self.config = reviewer_config()
        installed_path = _rooted(self.root, f"{RUNTIME_DIR.as_posix()}/installed.json", must_exist=True)
        try:
            installed = json.loads(installed_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewerError("Installed reviewer manifest is malformed") from error
        if (
            installed.get("schema_version") != 1
            or installed.get("runtime", {}).get("release") != self.config["runtime"]["release"]
            or installed.get("runtime", {}).get("sha256") != self.config["runtime"]["sha256"]
        ):
            raise ReviewerError("Installed reviewer runtime does not match the pinned release")
        self.installed = installed
        models = {row["id"]: row for row in self.config["models"]}
        if model_id not in models:
            raise ReviewerError(f"Unknown pinned local review model: {model_id}")
        self.model = models[model_id]
        self.q8 = q8
        self.preferred_gpu_layers = preferred_gpu_layers if preferred_gpu_layers is not None else (32 if q8 else 99)
        self.port = port or int(self.config["review_defaults"]["port"])
        self.runtime_archive = _rooted(self.root, f"{RUNTIME_DIR.as_posix()}/downloads/{self.config['runtime']['asset']}", must_exist=True)
        self.runtime_dir = _rooted(self.root, f"{RUNTIME_DIR.as_posix()}/runtime/b11425")
        self.server = self.runtime_dir / "bin" / "llama-server"
        if not self.server.is_file():
            found = list(self.runtime_dir.rglob("llama-server")) if self.runtime_dir.exists() else []
            if len(found) != 1:
                raise ReviewerError("llama.cpp runtime is not installed; run references:setup")
            self.server = found[0]
        relative_server = self.server.relative_to(self.root).as_posix()
        server_entry = next(
            (row for row in self.installed.get("runtime_files", []) if row.get("path") == relative_server),
            None,
        )
        if not server_entry or sha256_file(self.server) != server_entry.get("sha256"):
            raise ReviewerError("Installed llama-server binary failed its recorded checksum")
        self.cuda_libraries = list(self.installed.get("cuda_libraries", []))
        for entry in self.cuda_libraries:
            path = _rooted(self.root, str(entry.get("path", "")), must_exist=True)
            if path.stat().st_size != entry.get("bytes") or sha256_file(path) != entry.get("sha256"):
                raise ReviewerError("Installed CUDA runtime library failed its recorded checksum")
        self.cuda_devices = _cuda_devices(self.server, self.runtime_dir)
        if _gpu_probe().get("available") and not self.cuda_devices:
            raise ReviewerError("The NVIDIA GPU is visible to the driver but not to pinned llama.cpp")
        self.language_file = str(self.model["language_file"])
        self.language_sha256 = str(self.model["language_sha256"])
        self.language_bytes = int(self.model["language_bytes"])
        if q8:
            q8_config = self.model.get("q8")
            if not q8_config:
                raise ReviewerError("The pinned stronger-model Q8 artifact is not configured")
            self.language_file = str(q8_config["file"])
            self.language_sha256 = str(q8_config["sha256"])
            self.language_bytes = int(q8_config["bytes"])
        self.model_path = _rooted(
            self.root,
            f"{RUNTIME_DIR.as_posix()}/models/{model_id}/{self.language_file}",
            must_exist=True,
        )
        self.vision_path = _rooted(
            self.root,
            f"{RUNTIME_DIR.as_posix()}/models/{model_id}/{self.model['vision_file']}",
            must_exist=True,
        )
        if sha256_file(self.model_path) != self.language_sha256:
            raise ReviewerError("Local reviewer language weights failed their pinned checksum")
        if sha256_file(self.vision_path) != self.model["vision_sha256"]:
            raise ReviewerError("Local reviewer vision projector failed its pinned checksum")
        self.runtime_sha256 = str(self.config["runtime"]["sha256"])
        self.model_sha256 = hashlib.sha256(
            f"{self.language_sha256}:{self.model['vision_sha256']}".encode("ascii")
        ).hexdigest()
        self.prompt = REVIEW_PROMPT.read_text(encoding="utf-8")
        self.prompt_version = f"drawable-reference-review-v1:{sha256_file(REVIEW_PROMPT)}"
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path = _rooted(self.root, f"{RUNTIME_DIR.as_posix()}/logs/{model_id}.log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.gpu_layers: int | None = None
        self.startup_memory: dict[str, Any] = {}
        self._memory_stop = threading.Event()
        self._memory_thread: threading.Thread | None = None
        self.memory_samples: list[dict[str, Any]] = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _start(self, layers: int) -> None:
        arguments = [
            str(self.server),
            "--model", str(self.model_path),
            "--mmproj", str(self.vision_path),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--ctx-size", str(self.config["review_defaults"]["context_tokens"]),
            "--n-gpu-layers", str(layers),
            "--parallel", "1",
            "--jinja",
            "--chat-template-kwargs", '{"enable_thinking":false}',
            "--flash-attn", "on",
            "--no-webui",
        ]
        self._log_stream = self.log_path.open("ab")
        environment = _runtime_environment(self.runtime_dir)
        self.process = subprocess.Popen(
            arguments,
            cwd=self.runtime_dir,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.startup_memory = _gpu_probe()
        self.memory_samples = [self.startup_memory]
        self._memory_stop.clear()

        def sample_gpu() -> None:
            while not self._memory_stop.wait(0.75):
                self.memory_samples.append(_gpu_probe())

        self._memory_thread = threading.Thread(
            target=sample_gpu, name="reviewer-vram-monitor", daemon=True
        )
        self._memory_thread.start()
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self._stop()
                raise ReviewerError(f"llama-server exited during model load; inspect {self.log_path}")
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=2) as response:
                    if response.status == 200:
                        self.gpu_layers = layers
                        return
            except (OSError, urllib.error.URLError):
                time.sleep(0.5)
        self._stop()
        raise ReviewerError(f"llama-server readiness timed out; inspect {self.log_path}")

    def _stop(self) -> None:
        self._memory_stop.set()
        if self._memory_thread is not None:
            self._memory_thread.join(timeout=2)
            self._memory_thread = None
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        stream = getattr(self, "_log_stream", None)
        if stream is not None:
            stream.close()
            self._log_stream = None

    @property
    def peak_memory(self) -> dict[str, Any]:
        valid = [
            sample
            for sample in self.memory_samples
            if sample.get("available") and isinstance(sample.get("memory_used_mb"), int)
        ]
        if not valid:
            return {"available": False, "samples": len(self.memory_samples)}
        peak = max(valid, key=lambda sample: int(sample["memory_used_mb"]))
        return {
            "available": True,
            "samples": len(self.memory_samples),
            "peak_used_mb": peak["memory_used_mb"],
            "total_mb": peak.get("memory_total_mb"),
            "free_at_peak_mb": peak.get("memory_free_mb"),
        }

    def __enter__(self) -> Self:
        last_error: Exception | None = None
        candidates = [self.preferred_gpu_layers, 64, 48, 32, 24, 16, 8, 0]
        attempted: set[int] = set()
        for layers in candidates:
            if layers in attempted:
                continue
            attempted.add(layers)
            try:
                self._start(layers)
                return self
            except ReviewerError as error:
                last_error = error
                if "exited during model load" not in str(error):
                    break
        raise ReviewerError(str(last_error or "llama-server could not start"))

    def __exit__(self, *_: object) -> None:
        self._stop()

    def chat(self, image: Image.Image | list[Image.Image], *, prompt: str, schema: dict[str, Any], max_tokens: int) -> tuple[dict[str, Any], float]:
        if self.process is None or self.process.poll() is not None:
            raise ReviewerError("The pinned local review server is not running")
        images = image if isinstance(image, list) else [image]
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(entry)}}
            for entry in images[:2]
        )
        payload = {
            "model": self.model["id"],
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object", "schema": schema},
            "cache_prompt": False,
        }
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        timeout = int(self.config["review_defaults"]["inference_timeout_seconds"])
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read(2 * 1024 * 1024))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise ReviewerError(f"Local reviewer inference failed: {error}") from error
        duration_ms = (time.perf_counter() - started) * 1000
        try:
            content = result["choices"][0]["message"]["content"]
            if not isinstance(content, str) or len(content) > 32_000:
                raise ValueError("Response content is malformed or too large")
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise TypeError("Response is not a JSON object")
            return parsed, duration_ms
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as error:
            raise ReviewerError(f"Local reviewer returned a malformed response: {error}") from error


def setup_reviewer(root: Path, *, q8_model_id: str | None = None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config = reviewer_config()
    gpu_before = _gpu_probe()
    if not gpu_before["available"]:
        raise ReviewerError("The local CUDA GPU is not visible to nvidia-smi")
    required = 1024 * 1024 * 1024
    pinned_files: list[tuple[Path, int, str]] = []
    runtime = config["runtime"]
    pinned_files.append(
        (_rooted(root, f"{RUNTIME_DIR.as_posix()}/downloads/{runtime['asset']}"), int(runtime["bytes"]), str(runtime["sha256"]))
    )
    for model in config["models"]:
        folder = f"{RUNTIME_DIR.as_posix()}/models/{model['id']}"
        for key in ("language", "vision"):
            file_key = f"{key}_file"
            pinned_files.append((
                _rooted(root, f"{folder}/{model[file_key]}"),
                int(model[f"{key}_bytes"]),
                str(model[f"{key}_sha256"]),
            ))
    if q8_model_id:
        selected_q8 = next((row for row in config["models"] if row["id"] == q8_model_id), None)
        if not selected_q8 or not selected_q8.get("q8"):
            raise ReviewerError("The requested Q8 model does not have a pinned artifact")
        q8 = selected_q8["q8"]
        pinned_files.append((
            _rooted(root, f"{RUNTIME_DIR.as_posix()}/models/{selected_q8['id']}/{q8['file']}"),
            int(q8["bytes"]),
            str(q8["sha256"]),
        ))
    cuda_wheels = []
    for package in config.get("cuda_runtime", []):
        path = _rooted(root, f"{RUNTIME_DIR.as_posix()}/downloads/{package['asset']}")
        cuda_wheels.append((package, path))
        pinned_files.append((path, int(package["bytes"]), str(package["sha256"])))
    for path, expected_size, expected_hash in pinned_files:
        if not path.is_file() or path.is_symlink() or path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            required += expected_size
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < required:
        raise ReviewerError(f"Reviewer setup needs about {required} free bytes; found {free_bytes}")
    base = RUNTIME_DIR.as_posix()
    runtime = config["runtime"]
    archive_path = _rooted(root, f"{base}/downloads/{runtime['asset']}")
    runtime_download = _download(runtime["url"], archive_path, runtime["bytes"], runtime["sha256"])
    server_dir = _rooted(root, f"{base}/runtime/b11425")
    if not any(server_dir.rglob("llama-server")) if server_dir.exists() else True:
        server_path = _extract_runtime(archive_path, server_dir)
    else:
        server_path = next(server_dir.rglob("llama-server"))
    if not os.access(server_path, os.X_OK):
        raise ReviewerError("Verified llama-server binary is not executable")
    cuda_package_files = []
    cuda_libraries = []
    cuda_dir = server_dir / "cuda-12.8"
    for package, wheel_path in cuda_wheels:
        url = str(package["url"])
        downloaded = _download(
            url,
            wheel_path,
            int(package["bytes"]),
            str(package["sha256"]),
        )
        cuda_package_files.append({**downloaded, "package": package["package"], "version": package["version"]})
        extracted_libraries = _extract_cuda_wheel(wheel_path, cuda_dir, str(package["library_prefix"]))
        cuda_libraries.extend(
            {
                **library,
                "path": (server_dir.relative_to(root) / library["path"]).as_posix(),
            }
            for library in extracted_libraries
        )
    model_files: list[dict[str, Any]] = []
    for model in config["models"]:
        folder = f"{base}/models/{model['id']}"
        for key in ("language_file", "vision_file"):
            filename = model[key]
            url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{filename}?download=true"
            model_files.append(
                _download(
                    url,
                    _rooted(root, f"{folder}/{filename}"),
                    int(model[key.replace("file", "bytes")]),
                    str(model[key.replace("file", "sha256")]),
                )
            )
        q8 = model.get("q8")
        if q8_model_id == model["id"] and q8:
            filename = str(q8["file"])
            url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{filename}?download=true"
            model_files.append(
                _download(url, _rooted(root, f"{folder}/{filename}"), int(q8["bytes"]), str(q8["sha256"]))
            )
    installed = {
        "schema_version": 1,
        "runtime": {**runtime_download, "release": runtime["release"], "commit": runtime["commit"], "server_path": server_path.relative_to(root).as_posix()},
        "runtime_files": [
            {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(server_dir.rglob("*"))
            if path.is_file()
        ],
        "models": model_files,
        "cuda_packages": cuda_package_files,
        "cuda_libraries": cuda_libraries,
        "gpu": gpu_before,
        "setup_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = _rooted(root, f"{base}/installed.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path.write_text(json.dumps(installed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    return installed


def synthetic_preflight_image() -> Image.Image:
    image = Image.new("RGB", (384, 384), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((120, 36, 264, 180), outline="black", width=5)
    draw.arc((141, 67, 184, 102), 195, 345, fill="black", width=4)
    draw.arc((202, 67, 245, 102), 195, 345, fill="black", width=4)
    draw.arc((166, 103, 218, 153), 15, 165, fill="black", width=4)
    draw.line((147, 196, 237, 196, 264, 326, 120, 326, 147, 196), fill="black", width=5)
    draw.line((147, 226, 74, 285), fill="black", width=5)
    draw.line((237, 226, 310, 285), fill="black", width=5)
    return image


def preflight_reviewer(root: Path, model_id: str) -> dict[str, Any]:
    class PreflightOutput(StrictModel):
        visible_subject: bool

        @model_validator(mode="before")
        @classmethod
        def normalize_model_key(cls, value: Any) -> Any:
            if not isinstance(value, dict):
                return value
            if "visible_subject" in value:
                return {"visible_subject": value["visible_subject"]}
            for alias in ("visible", "is_character_figure"):
                if alias in value:
                    return {"visible_subject": value[alias]}
            return value

    before = _gpu_probe()
    started = time.perf_counter()
    with LlamaReviewer(root, model_id) as reviewer:
        output, elapsed = reviewer.chat(
            synthetic_preflight_image(),
            prompt=(
                "This is a synthetic line drawing of a person. Return the required JSON with "
                "visible_subject set to true only if the person is visible. Do not infer identity."
            ),
            schema=PreflightOutput.model_json_schema(),
            max_tokens=96,
        )
        parsed = PreflightOutput.model_validate(output)
        if not parsed.visible_subject:
            raise ReviewerError("The local model did not detect the synthetic drawn figure")
        layers = reviewer.gpu_layers
        peak_memory = reviewer.peak_memory
    after = _gpu_probe()
    return {
        "status": "complete",
        "model_id": model_id,
        "runtime_release": "b11425",
        "cuda_devices": reviewer.cuda_devices,
        "hardware_acceleration": bool(reviewer.gpu_layers),
        "gpu_before": before,
        "gpu_after": after,
        "gpu_peak_memory": peak_memory,
        "gpu_layers": layers,
        "inference_ms": elapsed,
        "wall_ms": (time.perf_counter() - started) * 1000,
        "response_valid": True,
        "visible_subject": parsed.visible_subject,
        "reason": "The synthetic drawn figure was detected.",
    }


def review_image(reviewer: LlamaReviewer, image: Image.Image) -> tuple[LocalVisualReview, float]:
    preview = resize_preview(image, int(reviewer.config["review_defaults"]["image_max_edge"]))
    schema = LocalVisualReview.model_json_schema()
    prompt = reviewer.prompt + "\n\nReview this image and return the required JSON object."
    first_error: Exception | None = None
    for attempt in range(2):
        try:
            extra = "" if attempt == 0 else "\nYour prior output failed independent validation. Re-evaluate and return one valid object only."
            output, duration_ms = reviewer.chat(
                preview,
                prompt=prompt + extra,
                schema=schema,
                max_tokens=int(reviewer.config["review_defaults"]["max_response_tokens"]),
            )
            return LocalVisualReview.model_validate(output), duration_ms
        except (ReviewerError, ValueError) as error:
            if first_error is not None:
                raise ReviewerError(f"Invalid reviewer output after one retry: {error}") from error
            first_error = error
    raise ReviewerError(f"Invalid reviewer output: {first_error}")


def review_pair(reviewer: LlamaReviewer, left: Image.Image, right: Image.Image) -> tuple[LocalPairReview, float]:
    prompt_path = REPOSITORY_ROOT / "config" / "references-pair-review-prompt-v1.txt"
    prompt = prompt_path.read_text(encoding="utf-8")
    previews = [resize_preview(left, 768), resize_preview(right, 768)]
    schema = LocalPairReview.model_json_schema()
    first_error: Exception | None = None
    for attempt in range(2):
        try:
            output, duration_ms = reviewer.chat(
                previews,
                prompt=prompt + "\n\nCompare image A (first) with image B (second).",
                schema=schema,
                max_tokens=256,
            )
            return LocalPairReview.model_validate(output), duration_ms
        except (ReviewerError, ValueError) as error:
            if first_error is not None:
                raise ReviewerError(f"Invalid pair-review output after one retry: {error}") from error
            first_error = error
    raise ReviewerError(f"Invalid pair-review output: {first_error}")
