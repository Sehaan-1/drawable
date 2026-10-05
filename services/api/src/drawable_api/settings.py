from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ALLOWED_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
)


@dataclass(frozen=True, slots=True)
class Settings:
    data_root: Path
    api_mode: Literal["fixture", "real"]
    metadata_db: Path
    curation_mode: bool = False
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    source_registry: Path | None = None
    anita_work_terms: Path | None = None
    model_registry: Path | None = None
    local_prototype_mode: bool = False

    @property
    def source_registry_path(self) -> Path:
        return self.source_registry or REPOSITORY_ROOT / "config" / "sources.json"

    @property
    def anita_work_terms_path(self) -> Path:
        return self.anita_work_terms or REPOSITORY_ROOT / "config" / "anita-work-terms.json"

    @property
    def model_registry_path(self) -> Path:
        return self.model_registry or REPOSITORY_ROOT / "config" / "models.json"

    @classmethod
    def from_env(cls) -> Settings:
        data_root = Path(os.environ.get("DRAWABLE_DATA_ROOT", "datasets")).expanduser()
        if not data_root.is_absolute():
            data_root = REPOSITORY_ROOT / data_root
        local_prototype_mode = os.environ.get("DRAWABLE_LOCAL_PROTOTYPE", "0").strip().casefold() in {"1", "true", "yes", "on"}
        default_db = data_root / "prototype" / "metadata.sqlite3" if local_prototype_mode else data_root / "metadata.sqlite3"
        db = Path(os.environ.get("DRAWABLE_METADATA_DB", str(default_db))).expanduser()
        if not db.is_absolute():
            db = REPOSITORY_ROOT / db
        mode = os.environ.get("DRAWABLE_API_MODE", "fixture").lower()
        if mode not in {"fixture", "real"}:
            raise ValueError("DRAWABLE_API_MODE must be 'fixture' or 'real'")
        curation_mode = os.environ.get("CURATION_MODE", "0").strip().casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }
        origins = tuple(
            origin.strip().rstrip("/")
            for origin in os.environ.get("DRAWABLE_ALLOWED_ORIGINS", "").split(",")
            if origin.strip()
        )
        return cls(
            data_root.resolve(),
            cast(Literal["fixture", "real"], mode),
            db.resolve(),
            curation_mode,
            origins or DEFAULT_ALLOWED_ORIGINS,
            Path(os.environ["DRAWABLE_SOURCE_REGISTRY"]).expanduser().resolve()
            if os.environ.get("DRAWABLE_SOURCE_REGISTRY")
            else None,
            Path(os.environ["DRAWABLE_ANITA_WORK_TERMS"]).expanduser().resolve()
            if os.environ.get("DRAWABLE_ANITA_WORK_TERMS")
            else None,
            Path(os.environ["DRAWABLE_MODEL_REGISTRY"]).expanduser().resolve()
            if os.environ.get("DRAWABLE_MODEL_REGISTRY")
            else None,
            local_prototype_mode,
        )
