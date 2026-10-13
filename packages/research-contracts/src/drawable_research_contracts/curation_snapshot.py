from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .viewpoint import (
    SHA256_PATTERN,
    StrictModel,
    ViewMatchReviewV1,
    ViewpointReviewV1,
    validate_view_match_rubric,
    validate_viewpoint_rubric,
)

Scope = Literal[
    "eye", "eyebrow", "mouth", "face_head", "hair", "hand", "foot",
    "upper_body_clothing", "full_body", "multi_character",
]


class SnapshotLabelsV1(StrictModel):
    style: Literal["manga_anime", "western_ink", "realistic_academic", "cartoon", "gesture_sketch"]
    primary_scope: Scope
    secondary_scopes: list[Scope] = Field(default_factory=list, max_length=9)
    person_count: int | None = Field(ge=1)
    quality: int = Field(ge=1, le=3)
    sfw: Literal["safe", "unsafe", "uncertain"]
    character_useful: bool
    text_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    clipped: bool
    damaged: bool
    duplicate_of: str | None = None
    review_seconds: float = Field(ge=0, le=86_400, allow_inf_nan=False)
    keep_reject_reason: str = Field(min_length=1, max_length=2000)


class SnapshotPermissionsV1(StrictModel):
    local_display: bool
    training: bool = False
    evaluation: bool = False
    derive: bool = False
    trace: bool = False
    export: bool = False
    public_demo: bool = False
    redistribute: bool = False
    cloud_storage: bool = False
    cloud_processing: bool = False


class ReviewProvenanceV1(StrictModel):
    """Who reviewed these bytes, and which immutable evidence supported the decision."""

    schema_version: Literal[1] = 1
    origin: Literal["human", "assistant", "local_model", "unspecified"]
    reviewer_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{3,120}$")
    evidence: dict[str, str | int | float | bool | None] = Field(default_factory=dict, max_length=24)
    profile_id: str | None = Field(default=None, max_length=120)
    model_id: str | None = Field(default=None, max_length=160)
    model_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN.pattern)
    runtime_id: str | None = Field(default=None, max_length=120)
    runtime_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN.pattern)
    prompt_version: str = Field(min_length=1, max_length=120)
    rubric_version: str = Field(min_length=1, max_length=120)
    run_id: str = Field(min_length=1, max_length=120)
    image_sha256: str = Field(pattern=SHA256_PATTERN.pattern)
    reviewed_revision: int = Field(ge=0)
    inference_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def local_model_is_fully_bound(self) -> ReviewProvenanceV1:
        if self.origin == "local_model" and any(
            value is None
            for value in (
                self.profile_id,
                self.model_id,
                self.model_sha256,
                self.runtime_id,
                self.runtime_sha256,
            )
        ):
            raise ValueError("Local-model reviews must bind the qualified profile and artifact hashes")
        if self.origin == "assistant" and not self.evidence:
            raise ValueError("Assistant reviews must record evidence for the visual decision")
        if self.origin == "unspecified" and not self.evidence.get("legacy_record"):
            raise ValueError("Unspecified review origin is reserved for preserved legacy records")
        return self


class SnapshotAssetV1(StrictModel):
    asset_id: str = Field(min_length=1, max_length=200)
    source_id: str = Field(min_length=1, max_length=120)
    source_work_id: str | None = Field(default=None, max_length=200)
    source_work_terms_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN.pattern)
    relative_path: str = Field(min_length=1, max_length=1000)
    sha256: str = Field(pattern=SHA256_PATTERN.pattern)
    byte_length: int = Field(gt=0, le=50 * 1024 * 1024)
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    source_family: str = Field(min_length=1, max_length=200)
    learning_split: Literal["pilot_development"]
    rights_status: Literal["approved", "review_required"]
    authorization_basis: Literal["terms-reviewed", "owner-local-educational-prototype"] = "terms-reviewed"
    review_status: Literal["approved"]
    gallery_eligible_candidate: Literal[True]
    terms_snapshot_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN.pattern)
    source_registry_sha256: str = Field(pattern=SHA256_PATTERN.pattern)
    license_id: str = Field(min_length=1, max_length=200)
    attribution: str = Field(min_length=1, max_length=2000)
    permissions: SnapshotPermissionsV1
    labels: SnapshotLabelsV1
    viewpoint: ViewpointReviewV1 | None = None
    parent_asset_id: str | None = None
    crop_rectangle: dict[str, object] | None = None
    asset_revision: int = Field(ge=0)
    review_id: str = Field(min_length=1, max_length=100)
    reviewer_id: str = Field(pattern=r"^[A-Za-z0-9_-]{3,80}$")
    review_provenance: ReviewProvenanceV1 | None = None

    @field_validator("relative_path")
    @classmethod
    def relative_paths_only(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in value or "\x00" in value:
            raise ValueError("Snapshot asset paths must be normalized and relative to the private data root")
        return value

    @model_validator(mode="after")
    def local_development_only(self) -> SnapshotAssetV1:
        prototype = self.authorization_basis == "owner-local-educational-prototype"
        if prototype:
            if self.rights_status != "review_required" or self.terms_snapshot_sha256 is not None:
                raise ValueError("Prototype authorization must not claim reviewed source terms")
            if any(getattr(self.permissions, key) for key in ("training", "trace", "export", "public_demo", "redistribute", "cloud_storage", "cloud_processing")):
                raise ValueError("Prototype snapshots describe local review, cropping, and evaluation only")
        elif self.rights_status != "approved" or not self.terms_snapshot_sha256:
            raise ValueError("Reviewed snapshots require approved rights and a terms snapshot hash")
        if not self.permissions.local_display:
            raise ValueError("Snapshot assets require explicit local-display permission")
        if self.permissions.cloud_processing and not self.permissions.cloud_storage:
            raise ValueError("Cloud processing requires separate cloud-storage permission")
        if not prototype and self.source_id == "anita-dataset" and (
            not self.source_work_id or not self.source_work_terms_sha256
        ):
            raise ValueError("Anita snapshot assets require their reviewed source-work identity and terms hash")
        if self.labels.sfw != "safe" or self.labels.quality < 2 or not self.labels.character_useful:
            raise ValueError("Snapshot assets must meet the reviewed pilot relevance floor")
        if self.viewpoint is not None and self.viewpoint.review_state != "unreviewed":
            validate_viewpoint_rubric(self.viewpoint, self.viewpoint.rubric_sha256)
        return self


class PrototypeGalleryItemV2(StrictModel):
    """A prototype-serving item, deliberately independent of research split assignment."""

    asset_id: str = Field(min_length=1, max_length=200)
    source_id: str = Field(min_length=1, max_length=120)
    source_work_id: str | None = Field(default=None, max_length=200)
    relative_path: str = Field(min_length=1, max_length=1000)
    sha256: str = Field(pattern=SHA256_PATTERN.pattern)
    thumbnail_path: str | None = Field(default=None, min_length=1, max_length=1000)
    thumbnail_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN.pattern)
    byte_length: int = Field(gt=0, le=50 * 1024 * 1024)
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    source_family: str | None = Field(default=None, max_length=200)
    source_identity_verified: bool = False
    upstream_partition: Literal["source_train", "not_applicable"] = "not_applicable"
    research_family_id: str | None = Field(default=None, max_length=200)
    authorization_basis: Literal["owner-local-educational-prototype"]
    review_status: Literal["approved"]
    gallery_eligible_candidate: Literal[True]
    license_id: str = Field(min_length=1, max_length=200)
    attribution: str = Field(min_length=1, max_length=2000)
    permissions: SnapshotPermissionsV1
    labels: SnapshotLabelsV1
    parent_asset_id: str | None = Field(default=None, max_length=200)
    root_asset_id: str = Field(min_length=1, max_length=200)
    clip_id: str | None = Field(default=None, max_length=240)
    crop_rectangle: dict[str, object] | None = None
    asset_revision: int = Field(ge=0)
    review_id: str = Field(min_length=1, max_length=100)
    reviewer_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{3,120}$")
    review_provenance: ReviewProvenanceV1

    @field_validator("relative_path", "thumbnail_path")
    @classmethod
    def relative_paths_only(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = PurePosixPath(value)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in value or "\x00" in value:
            raise ValueError("Catalog asset paths must be normalized and relative to the private data root")
        return value

    @model_validator(mode="after")
    def local_catalog_only(self) -> PrototypeGalleryItemV2:
        if not self.permissions.local_display or any(
            getattr(self.permissions, key)
            for key in (
                "training",
                "trace",
                "export",
                "public_demo",
                "redistribute",
                "cloud_storage",
                "cloud_processing",
            )
        ):
            raise ValueError("Prototype gallery permissions must remain local-display only")
        if self.source_id == "emoreccom-panels" and self.upstream_partition != "source_train":
            raise ValueError("Only allowlisted EmoRecCom training panels can enter a prototype gallery")
        if self.research_family_id is not None:
            raise ValueError("Prototype gallery construction does not assign research family splits")
        if bool(self.thumbnail_path) != bool(self.thumbnail_sha256):
            raise ValueError("Thumbnail path and checksum must be present together")
        if self.review_provenance.image_sha256 != self.sha256:
            raise ValueError("Review provenance is stale against cataloged image bytes")
        if self.review_provenance.reviewed_revision != self.asset_revision:
            raise ValueError("Review provenance is stale against the reviewed asset revision")
        if self.labels.sfw != "safe" or self.labels.quality < 2 or not self.labels.character_useful:
            raise ValueError("Prototype gallery items must meet the reviewed usefulness floor")
        return self


class CurationSnapshotV1(StrictModel):
    schema_version: Literal[1] = 1
    snapshot_kind: Literal["drawable-curation-snapshot"] = "drawable-curation-snapshot"
    purpose: Literal["pilot-development-review-only", "local-educational-prototype"] = "pilot-development-review-only"
    created_at: datetime
    created_by: str = Field(pattern=r"^[A-Za-z0-9_-]{3,80}$")
    viewpoint_rubric_id: Literal["drawable-viewpoint-v1"] = "drawable-viewpoint-v1"
    viewpoint_rubric_sha256: str = Field(pattern=SHA256_PATTERN.pattern)
    items: list[SnapshotAssetV1] = Field(min_length=1, max_length=10_000)
    view_matches: list[ViewMatchReviewV1] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_snapshot_scope(self) -> CurationSnapshotV1:
        prototype = any(item.authorization_basis == "owner-local-educational-prototype" for item in self.items)
        if prototype != (self.purpose == "local-educational-prototype"):
            raise ValueError("Educational prototype items must be identified as a prototype snapshot")
        if self.created_at.utcoffset() is None or self.created_at > datetime.now(UTC):
            raise ValueError("Snapshot creation time must be timezone-aware and not in the future")
        ids = [item.asset_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("Snapshot contains duplicate asset IDs")
        group_splits: dict[tuple[str, str], set[str]] = {}
        for item in self.items:
            group_splits.setdefault((item.source_id, item.source_family), set()).add(item.learning_split)
            if item.viewpoint is not None and item.viewpoint.review_state != "unreviewed":
                validate_viewpoint_rubric(item.viewpoint, self.viewpoint_rubric_sha256)
        if any(len(splits) != 1 for splits in group_splits.values()):
            raise ValueError("A source-family group crosses learning splits")
        for review in self.view_matches:
            validate_view_match_rubric(review, self.viewpoint_rubric_sha256)
            if review.query_asset_id not in ids or review.reference_asset_id not in ids:
                raise ValueError("View-match judgments must refer to assets in the same immutable snapshot")
        return self


class PrototypeGalleryCatalogV2(StrictModel):
    """Scalable prototype catalog. V1 curation snapshots and search builds remain readable."""

    schema_version: Literal[2] = 2
    catalog_kind: Literal["drawable-prototype-gallery"] = "drawable-prototype-gallery"
    catalog_id: str = Field(pattern=r"^gallery-[0-9a-f]{24}$")
    purpose: Literal["local-educational-prototype"]
    created_at: datetime
    created_by: str = Field(pattern=r"^[A-Za-z0-9_-]{3,80}$")
    availability_revision: int = Field(ge=0)
    items: list[PrototypeGalleryItemV2] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_catalog(self) -> PrototypeGalleryCatalogV2:
        if self.created_at.utcoffset() is None or self.created_at > datetime.now(UTC):
            raise ValueError("Gallery catalog time must be timezone-aware and not in the future")
        ids = [item.asset_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("Gallery catalog contains duplicate asset IDs")
        return self


def validate_curation_snapshot_files(
    snapshot: CurationSnapshotV1 | PrototypeGalleryCatalogV2, data_root: Path
) -> dict[str, int | str]:
    root = data_root.resolve(strict=True)
    verified = 0
    for item in snapshot.items:
        relative = PurePosixPath(item.relative_path)
        path = root.joinpath(*relative.parts)
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"Snapshot asset {item.asset_id} traverses a symbolic link")
        resolved = path.resolve(strict=True)
        if root not in resolved.parents or not resolved.is_file():
            raise ValueError(f"Snapshot asset {item.asset_id} is missing or outside the private data root")
        if resolved.stat().st_size != item.byte_length:
            raise ValueError(f"Snapshot asset {item.asset_id} byte length changed")
        with resolved.open("rb") as stream:
            signature = stream.read(12)
        valid_signature = {
            "image/png": signature.startswith(b"\x89PNG\r\n\x1a\n"),
            "image/jpeg": signature.startswith(b"\xff\xd8\xff"),
            "image/webp": len(signature) >= 12 and signature[:4] == b"RIFF" and signature[8:12] == b"WEBP",
        }[item.media_type]
        if not valid_signature:
            raise ValueError(f"Snapshot asset {item.asset_id} bytes do not match the declared image type")
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item.sha256:
            raise ValueError(f"Snapshot asset {item.asset_id} checksum changed")
        verified += 1
        if isinstance(snapshot, PrototypeGalleryCatalogV2) and item.thumbnail_path:
            thumbnail_relative = PurePosixPath(item.thumbnail_path)
            thumbnail_path = root.joinpath(*thumbnail_relative.parts)
            cursor = root
            for part in thumbnail_relative.parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    raise ValueError(f"Thumbnail for {item.asset_id} traverses a symbolic link")
            resolved_thumbnail = thumbnail_path.resolve(strict=True)
            if root not in resolved_thumbnail.parents or not resolved_thumbnail.is_file():
                raise ValueError(f"Thumbnail for {item.asset_id} is missing or outside private storage")
            thumbnail_digest = hashlib.sha256(resolved_thumbnail.read_bytes()).hexdigest()
            if thumbnail_digest != item.thumbnail_sha256:
                raise ValueError(f"Thumbnail for {item.asset_id} checksum changed")
    return {"status": "verified", "items": verified, "training_started": False, "data_uploaded": False}
