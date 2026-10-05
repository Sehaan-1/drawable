from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from drawable_research_contracts.viewpoint import (
    ViewMatchReviewV1,
    ViewpointReviewV1,
    validate_view_match_rubric,
    validate_viewpoint_rubric,
)

RUBRIC_HASH = "a" * 64


def test_unreviewed_viewpoint_is_explicitly_indeterminate() -> None:
    review = ViewpointReviewV1(
        rubric_sha256=RUBRIC_HASH,
        review_state="unreviewed",
        evidence_basis="visible_crop",
        horizontal_view="indeterminate",
        facing="indeterminate",
        elevation="indeterminate",
    )
    validate_viewpoint_rubric(review, RUBRIC_HASH)
    with pytest.raises(ValueError, match="rubric hash"):
        validate_viewpoint_rubric(review, "b" * 64)


def test_reviewed_viewpoint_requires_human_provenance_and_axes() -> None:
    payload = {
        "rubric_sha256": RUBRIC_HASH,
        "review_state": "reviewed",
        "evidence_basis": "crop_with_parent_context",
        "horizontal_view": "three_quarter",
        "facing": "image_left",
        "elevation": "level",
        "reviewer_id": "reviewer-1",
        "reviewed_at": datetime.now(UTC),
    }
    assert ViewpointReviewV1(**payload).horizontal_view == "three_quarter"
    with pytest.raises(ValidationError):
        ViewpointReviewV1(**{**payload, "reviewer_id": None})
    with pytest.raises(ValidationError):
        ViewpointReviewV1(**{**payload, "horizontal_view": "not_applicable"})


def test_not_applicable_requires_explicit_reason_and_pair_reviews_are_versioned() -> None:
    decision = ViewpointReviewV1(
        rubric_sha256=RUBRIC_HASH,
        review_state="not_applicable",
        evidence_basis="not_applicable",
        horizontal_view="not_applicable",
        facing="not_applicable",
        elevation="not_applicable",
        reviewer_id="reviewer-1",
        reviewed_at=datetime.now(UTC),
        rationale="The crop contains no visible eye or head orientation.",
    )
    validate_viewpoint_rubric(decision, RUBRIC_HASH)
    pair = ViewMatchReviewV1(
        rubric_sha256=RUBRIC_HASH,
        query_asset_id="query-1",
        reference_asset_id="reference-1",
        match="near",
        reviewer_id="reviewer-1",
        reviewed_at=datetime.now(UTC),
        rationale="Both depict a similar three-quarter view with mild elevation difference.",
    )
    validate_view_match_rubric(pair, RUBRIC_HASH)
    with pytest.raises(ValidationError):
        ViewMatchReviewV1(**{**pair.model_dump(), "reference_asset_id": "query-1"})
