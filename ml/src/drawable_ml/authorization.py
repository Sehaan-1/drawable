from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from .contracts import SHA256, StrictRecord
from .io import MLError, load_json, resolve_private_path, sha256_json


class ExperimentAuthorizationV1(StrictRecord):
    schema_version: Literal[1]
    decision: Literal["approved"]
    approval_id: str = Field(min_length=8, max_length=120)
    authorized_by: str = Field(min_length=3, max_length=100)
    authorized_at: str
    expires_at: str
    purpose: Literal["backward_feasibility", "semantic_adaptation", "structural_projection"]
    runtime_profile: Literal["colab-cuda"]
    dataset_sha256: str
    recipe_sha256: str
    code_revision: str = Field(min_length=7, max_length=80)
    model_hashes: dict[str, str]
    maximum_steps: int = Field(ge=1, le=100_000)
    execution_budget_minutes: int = Field(ge=1, le=24 * 60)

    @field_validator("dataset_sha256", "recipe_sha256")
    @classmethod
    def valid_digest(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("Authorization hashes must be lowercase SHA-256 values")
        return value

    @field_validator("model_hashes")
    @classmethod
    def model_digests(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or any(not SHA256.fullmatch(digest) for digest in value.values()):
            raise ValueError("Authorization must bind every model to a lowercase SHA-256 digest")
        return value


def verify_authorization(
    path: Path,
    *,
    data_root: Path,
    purpose: str,
    dataset_sha256: str,
    recipe_sha256: str,
    model_hashes: dict[str, str],
    code_revision: str,
    steps: int,
    now: datetime | None = None,
) -> ExperimentAuthorizationV1:
    """Validate an owner-created authorization; this package never creates one."""
    root = data_root.resolve()
    try:
        relative = path.resolve(strict=True).relative_to(root).as_posix()
        path = resolve_private_path(root, relative)
    except (OSError, ValueError, MLError) as error:
        raise MLError("Authorization must be a regular file inside the private data root") from error
    if path.is_symlink() or not path.is_file():
        raise MLError("Authorization must be a regular, non-symlink file")
    try:
        approval = ExperimentAuthorizationV1.model_validate(load_json(path))
    except Exception as error:
        raise MLError(f"Authorization file is invalid: {error}") from error
    if approval.purpose != purpose:
        raise MLError("Authorization purpose does not match this training command")
    if approval.runtime_profile != "colab-cuda":
        raise MLError("Training execution is supported only in the explicitly approved Colab CUDA profile")
    expected = {
        "dataset": (approval.dataset_sha256, dataset_sha256),
        "recipe": (approval.recipe_sha256, recipe_sha256),
        "model hashes": (approval.model_hashes, model_hashes),
        "code revision": (approval.code_revision, code_revision),
    }
    for label, (approved, current) in expected.items():
        if approved != current:
            raise MLError(f"Authorization is stale: {label} changed after approval")
    if steps > approval.maximum_steps:
        raise MLError("Requested step count exceeds the approved experiment budget")
    try:
        start = datetime.fromisoformat(approval.authorized_at)
        expiry = datetime.fromisoformat(approval.expires_at)
    except ValueError as error:
        raise MLError("Authorization timestamps must be ISO-8601 with a timezone") from error
    current_time = now or datetime.now(UTC)
    if start.tzinfo is None or expiry.tzinfo is None or current_time < start or current_time >= expiry:
        raise MLError("Authorization is not currently valid")
    return approval


def authorization_digest(approval: ExperimentAuthorizationV1) -> str:
    return sha256_json(approval.model_dump(mode="json"))
