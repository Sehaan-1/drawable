"""Typed runtime configuration.

Every setting has a free, local default so a fresh clone runs with no
environment at all. Overrides come from ``LINESCOUT_*`` environment variables
or a ``.env`` file in ``services/api``. Paths are resolved relative to the
repository root so no absolute machine path ever needs to be committed.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from linescout_api.preprocessing import MAX_COMPRESSED_BYTES, MAX_DECOMPRESSED_BYTES

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_MANIFEST = REPO_ROOT / "ml" / "fixtures" / "synthetic" / "manifest.json"


class DevicePolicy(StrEnum):
    AUTO = "auto"  # CUDA if available, otherwise CPU with a warning
    CUDA = "cuda"  # fail readiness if CUDA is missing
    CPU = "cpu"  # force CPU even when CUDA exists


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LINESCOUT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    cors_origins: list[str] = ["http://127.0.0.1:5173", "http://localhost:5173"]
    # Extra Host header values accepted alongside the loopback set. Exists for
    # tests (Starlette's TestClient targets "testserver"); production must
    # leave it empty so DNS-rebinding hostnames keep being rejected. The
    # test-only names live here — never hard-coded in the ASGI middleware.
    additional_allowed_hosts: list[str] = []

    # Storage (relative paths resolve from the repository root)
    data_dir: Path = Path("data")
    db_path: Path = Path("data/linescout.sqlite3")
    # In fixture mode this defaults to the committed synthetic gallery so a fresh
    # clone has references to serve; point it at data/gallery/<version>/manifest.json
    # once a real gallery exists. Set LINESCOUT_GALLERY_MANIFEST= (empty) to disable.
    gallery_manifest: Path | None = None

    # Runtime mode
    device: DevicePolicy = DevicePolicy.AUTO
    curation_mode: bool = False  # CURATION_MODE=1 in the spec; exposes /api/v1/curation/*
    fixture_mode: bool = True  # serve deterministic fixture results until models exist

    # Search contract limits (spec §5, API contracts).
    max_image_bytes: int = 4 * 1024 * 1024  # 4 MiB decoded snapshot PNG, as uploaded
    max_strokes_bytes: int = MAX_COMPRESSED_BYTES  # 256 KiB, gzip-compressed as uploaded
    max_strokes_decompressed_bytes: int = MAX_DECOMPRESSED_BYTES  # 1 MiB after gunzip
    max_text_hint_chars: int = 120
    min_points_for_search: int = 20
    min_ink_diagonal_ratio: float = 0.02
    canvas_logical_size: int = 2048
    snapshot_size: int = 512

    # Total request envelope — image + strokes + this multipart overhead slack
    # (part headers / delimiters / form fields). Enforced on the ASGI receive
    # channel by linescout_api.limits BEFORE multipart parsing, so a forged or
    # missing Content-Length cannot bypass it. Derived; not directly settable.
    multipart_overhead_bytes: int = 64 * 1024
    max_request_parts: int = 16  # /search declares 10 parts; small margin

    # Preference learning
    preference_half_life_days: float = 30.0

    @property
    def max_request_bytes(self) -> int:
        """Total raw request-body allowance: image + compressed strokes + overhead."""
        return self.max_image_bytes + self.max_strokes_bytes + self.multipart_overhead_bytes

    @field_validator("gallery_manifest", mode="before")
    @classmethod
    def _empty_string_disables(cls, value: object) -> object:
        return None if isinstance(value, str) and value.strip() == "" else value

    @field_validator("data_dir", "db_path", "gallery_manifest")
    @classmethod
    def _resolve_from_repo_root(cls, value: Path | None) -> Path | None:
        if value is None or value.is_absolute():
            return value
        return (REPO_ROOT / value).resolve()

    @model_validator(mode="after")
    def _default_fixture_gallery(self) -> Self:
        if (
            self.gallery_manifest is None
            and self.fixture_mode
            and "gallery_manifest" not in self.model_fields_set
        ):
            self.gallery_manifest = SYNTHETIC_MANIFEST
        return self

    @field_validator("cors_origins", "additional_allowed_hosts", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("cors_origins")
    @classmethod
    def _validate_origins(cls, value: list[str]) -> list[str]:
        """Each browser origin must be an exact ``http(s)://host[:port]`` origin.

        Wildcards, paths, queries, fragments, and embedded credentials are
        rejected: the loopback-Origin mutation check matches exact strings,
        and sloppy entries would silently never match (or, for CORS reads,
        over-broadly). Entries are normalized (lowercase host, no trailing
        slash) and deduplicated keeping order.
        """
        origins: list[str] = []
        for raw in value:
            origin = raw.strip()
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                msg = f"invalid CORS origin {origin!r}: expected http(s)://host[:port]"
                raise ValueError(msg)
            if (
                parsed.path not in ("", "/")
                or parsed.query
                or parsed.fragment
                or parsed.username
                or parsed.password
            ):
                msg = (
                    f"invalid CORS origin {origin!r}: must not contain a path, query, "
                    "fragment, or credentials"
                )
                raise ValueError(msg)
            normalized = f"{parsed.scheme}://{parsed.hostname}"
            if parsed.port is not None:
                normalized += f":{parsed.port}"
            if normalized not in origins:
                origins.append(normalized)
        return origins

    @field_validator("additional_allowed_hosts")
    @classmethod
    def _validate_additional_hosts(cls, value: list[str]) -> list[str]:
        hosts: list[str] = []
        for raw in value:
            host = raw.strip().lower()
            if not host or any(char in host for char in "/:@ \t"):
                msg = (
                    f"invalid additional host {raw!r}: expected a bare hostname "
                    '(no scheme, port, or path — e.g. "testserver")'
                )
                raise ValueError(msg)
            if host not in hosts:
                hosts.append(host)
        return hosts


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
