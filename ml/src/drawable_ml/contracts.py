from __future__ import annotations

import re
from datetime import datetime
from typing import Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PermissionRecord(StrictRecord):
    training: bool = False
    evaluation: bool = False
    derive: bool = False
    export: bool = False
    cloud_storage: bool = False
    cloud_processing: bool = False

    @model_validator(mode="after")
    def cloud_pair_is_explicit(self) -> PermissionRecord:
        if self.cloud_processing and not self.cloud_storage:
            raise ValueError("Cloud processing permission requires separate cloud-storage permission")
        return self


class AssetRecord(StrictRecord):
    asset_id: str
    source_id: str
    relative_path: str
    sha256: str
    source_work_group: str
    learning_split: Literal["train", "validation", "locked_test"]
    rights_status: Literal["approved", "review_required", "rejected"]
    review_status: Literal["approved", "pending", "rejected"]
    terms_snapshot_sha256: str
    source_registry_sha256: str
    permissions: PermissionRecord
    license_id: str = Field(min_length=1, max_length=200)
    attribution: str = Field(min_length=1, max_length=2000)
    style: Literal["manga_anime", "western_ink", "realistic_academic", "cartoon", "gesture_sketch"]
    primary_scope: Literal[
        "eye", "eyebrow", "mouth", "face_head", "hair", "hand", "foot",
        "upper_body_clothing", "full_body", "multi_character",
    ]
    sfw: Literal["approved", "rejected", "uncertain"]

    @field_validator("asset_id", "source_id", "source_work_group")
    @classmethod
    def stable_ids(cls, value: str) -> str:
        if not SAFE_ID.fullmatch(value):
            raise ValueError("IDs must use stable, nonempty ASCII identifiers")
        return value

    @field_validator("sha256", "terms_snapshot_sha256", "source_registry_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("Expected a lowercase SHA-256 hex digest")
        return value


class AssetJudgment(StrictRecord):
    asset_id: str
    grade: int = Field(ge=0, le=3)

    @field_validator("asset_id")
    @classmethod
    def stable_asset_id(cls, value: str) -> str:
        if not SAFE_ID.fullmatch(value):
            raise ValueError("Judged asset IDs must be stable ASCII identifiers")
        return value


class PrefixRecord(StrictRecord):
    prefix_id: str
    target_percent: Literal[20, 40, 60, 80, 100]
    actual_percent: float = Field(gt=0, le=100)
    stroke_index: int = Field(ge=1)
    png_path: str
    png_sha256: str
    visible_scopes: list[Literal[
        "eye", "eyebrow", "mouth", "face_head", "hair", "hand", "foot",
        "upper_body_clothing", "full_body", "multi_character", "unknown",
    ]]
    scopes_reviewed: bool
    scope_reviewer: str | None = None
    positive_asset_ids: list[str] = Field(default_factory=list)
    judged_unusable_asset_ids: list[str] = Field(default_factory=list)
    judgments: list[AssetJudgment]
    relevance_reviewed: bool = False
    relevance_reviewer: str | None = None

    @field_validator("prefix_id")
    @classmethod
    def valid_prefix_id(cls, value: str) -> str:
        if not SAFE_ID.fullmatch(value):
            raise ValueError("Prefix ID is malformed")
        return value

    @field_validator("png_sha256")
    @classmethod
    def valid_png_hash(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("Prefix image checksum is malformed")
        return value

    @field_validator("scope_reviewer", "relevance_reviewer")
    @classmethod
    def reviewer_is_pseudonymous(cls, value: str | None) -> str | None:
        if value is not None and not SAFE_ID.fullmatch(value):
            raise ValueError("Reviewer identifiers must be pseudonymous stable IDs")
        return value

    @model_validator(mode="after")
    def labels_are_reviewed_and_disjoint(self) -> PrefixRecord:
        if not self.scopes_reviewed or not self.scope_reviewer:
            raise ValueError("Each prefix requires an explicit reviewed visible-scope label")
        if not self.visible_scopes or ("unknown" in self.visible_scopes and len(self.visible_scopes) != 1):
            raise ValueError("Unknown is an exclusive prefix label; at least one visible scope is required")
        if len(set(self.visible_scopes)) != len(self.visible_scopes):
            raise ValueError("Visible-scope labels must be unique")
        if not self.relevance_reviewed or not self.relevance_reviewer:
            raise ValueError("Each prefix requires a reviewed relevance snapshot, including empty judgments")
        if set(self.positive_asset_ids) & set(self.judged_unusable_asset_ids):
            raise ValueError("An asset cannot be both usable and unusable for the same prefix")
        if len(set(self.positive_asset_ids)) != len(self.positive_asset_ids) or len(set(self.judged_unusable_asset_ids)) != len(self.judged_unusable_asset_ids):
            raise ValueError("Prefix asset judgments must be unique")
        grade_by_asset = {item.asset_id: item.grade for item in self.judgments}
        if len(grade_by_asset) != len(self.judgments):
            raise ValueError("A candidate receives one shared grade per drawing prefix")
        if set(grade_by_asset) != set(self.positive_asset_ids) | set(self.judged_unusable_asset_ids):
            raise ValueError("Positive and unusable candidate lists must exactly match the reviewed judgment pool")
        if any(grade_by_asset.get(asset_id, 0) < 2 for asset_id in self.positive_asset_ids):
            raise ValueError("Positive candidates must have an explicit usefulness grade of 2 or 3")
        if any(grade_by_asset.get(asset_id, 3) >= 2 for asset_id in self.judged_unusable_asset_ids):
            raise ValueError("Unusable candidates must have an explicit grade of 0 or 1")
        return self


class DrawingRecord(StrictRecord):
    recording_id: str
    raw_sequence_path: str
    raw_sequence_sha256: str
    contributor_group_id: str
    sequence_group_id: str
    task_kind: Literal["in_domain", "unknown"]
    collection_program: Literal["full-program"]
    learning_split: Literal["train", "model_selection", "threshold_selection", "calibration_audit", "locked_test"]
    style: Literal["manga_anime", "western_ink", "realistic_academic", "cartoon", "gesture_sketch"]
    final_scope: Literal[
        "eye", "eyebrow", "mouth", "face_head", "hair", "hand", "foot",
        "upper_body_clothing", "full_body", "multi_character", "unknown",
    ]
    creator_confirmed: bool
    contributor_can_grant_permissions: bool
    reference_rights_declared: bool
    research_consent: bool
    training_consent: bool
    evaluation_consent: bool
    cloud_storage_consent: bool
    cloud_processing_consent: bool
    reference_kind: Literal["none", "assigned-gallery", "external"]
    reference_asset_id: str | None = None
    reference_source_url: str | None = Field(default=None, max_length=2048)
    reference_permissions: PermissionRecord = Field(default_factory=PermissionRecord)
    referenced_source_work_groups: list[str] = Field(default_factory=list)
    prefixes: list[PrefixRecord] = Field(min_length=1)

    @field_validator("recording_id", "contributor_group_id", "sequence_group_id")
    @classmethod
    def stable_sequence_ids(cls, value: str) -> str:
        if not SAFE_ID.fullmatch(value):
            raise ValueError("Sequence identifiers must be stable, pseudonymous ASCII IDs")
        return value

    @field_validator("raw_sequence_sha256")
    @classmethod
    def valid_sequence_hash(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("Raw sequence checksum must be a SHA-256 digest")
        return value

    @model_validator(mode="after")
    def consent_is_effective(self) -> DrawingRecord:
        if not self.creator_confirmed or not self.contributor_can_grant_permissions or not self.reference_rights_declared:
            raise ValueError("Creator, permission-authority, and reference-rights confirmations are required")
        if not self.research_consent:
            raise ValueError("Research consent is required")
        if self.cloud_processing_consent and not self.cloud_storage_consent:
            raise ValueError("Cloud processing consent requires cloud-storage consent")
        if self.learning_split == "train" and not self.training_consent:
            raise ValueError("Training split records require explicit training consent")
        if self.learning_split != "train" and not self.evaluation_consent:
            raise ValueError("Evaluation records require explicit evaluation consent")
        if self.reference_kind != "none" and self.learning_split == "train" and not self.reference_permissions.training:
            raise ValueError("The reference-use record does not permit training")
        if self.reference_kind != "none" and self.learning_split != "train" and not self.reference_permissions.evaluation:
            raise ValueError("The reference-use record does not permit evaluation")
        if self.learning_split in {"train", "model_selection"}:
            if not self.cloud_storage_consent or not self.cloud_processing_consent:
                raise ValueError("Cloud-selected records require both explicit cloud consents")
            if self.reference_kind != "none" and (
                not self.reference_permissions.cloud_storage or not self.reference_permissions.cloud_processing
            ):
                raise ValueError("Referenced material requires separate cloud-storage and processing permission")
        if len(set(self.referenced_source_work_groups)) != len(self.referenced_source_work_groups):
            raise ValueError("Referenced work groups must be unique")
        if self.reference_kind == "assigned-gallery" and not self.reference_asset_id:
            raise ValueError("Assigned gallery references require the stable reviewed asset ID")
        if self.reference_kind == "external":
            if not self.reference_source_url:
                raise ValueError("External reference provenance requires an HTTPS source URL")
            parsed = urlsplit(self.reference_source_url)
            sensitive = {"token", "access_token", "api_key", "key", "secret", "password", "auth", "signature", "sig", "session"}
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or any(key.casefold() in sensitive for key, _ in parse_qsl(parsed.query, keep_blank_values=True))
            ):
                raise ValueError("External reference URLs must be HTTPS and contain no credentials or secret query parameters")
        elif self.reference_source_url is not None:
            raise ValueError("Only external references may have a source URL")
        if [prefix.target_percent for prefix in self.prefixes] != [20, 40, 60, 80, 100]:
            raise ValueError("Each sequence needs ordered snapshots at 20, 40, 60, 80, and 100 percent")
        return self


class DatasetManifest(StrictRecord):
    schema_version: Literal[2]
    format: Literal["drawable-training-dataset"]
    dataset_id: str
    source_registry_sha256: str
    dataset_recipe_sha256: str
    experiment_recipe_sha256: str
    generated_at: str
    assets: list[AssetRecord]
    drawings: list[DrawingRecord]

    @field_validator("generated_at")
    @classmethod
    def manifest_timestamp_has_timezone(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("Manifest generation time must be ISO-8601") from error
        if parsed.tzinfo is None:
            raise ValueError("Manifest generation time must include a timezone")
        return value

    @field_validator("source_registry_sha256", "dataset_recipe_sha256", "experiment_recipe_sha256")
    @classmethod
    def manifest_hash(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("Manifest binding must be a SHA-256 digest")
        return value

    @field_validator("dataset_id")
    @classmethod
    def dataset_identifier(cls, value: str) -> str:
        if not SAFE_ID.fullmatch(value):
            raise ValueError("Dataset ID is malformed")
        return value
