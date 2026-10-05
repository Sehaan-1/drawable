from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RUBRIC_ID: Literal["drawable-viewpoint-v1"] = "drawable-viewpoint-v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


HorizontalView = Literal["front", "three_quarter", "profile", "rear", "indeterminate", "not_applicable"]
FacingDirection = Literal["image_left", "image_right", "center", "indeterminate", "not_applicable"]
Elevation = Literal["level", "view_from_above", "view_from_below", "indeterminate", "not_applicable"]
EvidenceBasis = Literal["visible_crop", "crop_with_parent_context", "visible_prefix", "not_applicable"]


class ViewpointReviewV1(StrictModel):
    schema_version: Literal[1] = 1
    rubric_id: Literal["drawable-viewpoint-v1"] = RUBRIC_ID
    rubric_sha256: str = Field(pattern=SHA256_PATTERN.pattern)
    review_state: Literal["unreviewed", "reviewed", "not_applicable"]
    evidence_basis: EvidenceBasis
    horizontal_view: HorizontalView
    facing: FacingDirection
    elevation: Elevation
    roll_degrees: float | None = Field(default=None, ge=-180, lt=180, allow_inf_nan=False)
    reviewer_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{3,80}$")
    reviewed_at: datetime | None = None
    rationale: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_review_state(self) -> ViewpointReviewV1:
        axes = (self.horizontal_view, self.facing, self.elevation)
        if self.review_state == "reviewed":
            if not self.reviewer_id or self.reviewed_at is None:
                raise ValueError("Reviewed viewpoint labels require reviewer_id and reviewed_at")
            if self.evidence_basis in {"not_applicable", "visible_prefix"} and self.evidence_basis == "not_applicable":
                raise ValueError("Reviewed labels require visible evidence or a visible prefix")
            if "not_applicable" in axes:
                raise ValueError("Use review_state=not_applicable when the entire view is not applicable")
        elif self.review_state == "unreviewed":
            if self.reviewer_id is not None or self.reviewed_at is not None:
                raise ValueError("Unreviewed viewpoint labels cannot carry reviewer identity or time")
            if any(axis != "indeterminate" for axis in axes):
                raise ValueError("Unreviewed viewpoint axes must remain explicitly indeterminate")
        else:
            if not self.reviewer_id or self.reviewed_at is None or len(self.rationale) < 10:
                raise ValueError("Not-applicable decisions require reviewer, time, and a rationale")
            if self.evidence_basis != "not_applicable" or any(axis != "not_applicable" for axis in axes):
                raise ValueError("Not-applicable decisions must mark every axis and evidence basis not_applicable")
        if self.roll_degrees is not None and not math.isfinite(self.roll_degrees):
            raise ValueError("roll_degrees must be finite")
        if self.reviewed_at is not None and self.reviewed_at.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        if self.reviewed_at is not None and self.reviewed_at > datetime.now(UTC):
            raise ValueError("reviewed_at cannot be in the future")
        return self


class ViewMatchReviewV1(StrictModel):
    schema_version: Literal[1] = 1
    rubric_id: Literal["drawable-viewpoint-v1"] = RUBRIC_ID
    rubric_sha256: str = Field(pattern=SHA256_PATTERN.pattern)
    query_asset_id: str = Field(min_length=1, max_length=200)
    reference_asset_id: str = Field(min_length=1, max_length=200)
    match: Literal["same", "near", "incompatible", "indeterminate", "not_applicable"]
    reviewer_id: str = Field(pattern=r"^[A-Za-z0-9_-]{3,80}$")
    reviewed_at: datetime
    rationale: str = Field(min_length=10, max_length=1000)

    @model_validator(mode="after")
    def validate_pair(self) -> ViewMatchReviewV1:
        if self.query_asset_id == self.reference_asset_id:
            raise ValueError("Query and reference assets must be different")
        if self.reviewed_at.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        if self.reviewed_at > datetime.now(UTC):
            raise ValueError("reviewed_at cannot be in the future")
        return self


def validate_viewpoint_rubric(value: ViewpointReviewV1, expected_sha256: str) -> None:
    if not SHA256_PATTERN.fullmatch(expected_sha256) or value.rubric_sha256 != expected_sha256:
        raise ValueError("Viewpoint review rubric hash does not match the active rubric")


def validate_view_match_rubric(value: ViewMatchReviewV1, expected_sha256: str) -> None:
    if not SHA256_PATTERN.fullmatch(expected_sha256) or value.rubric_sha256 != expected_sha256:
        raise ValueError("View-match review rubric hash does not match the active rubric")
