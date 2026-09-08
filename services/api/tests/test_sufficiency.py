"""The two sufficiency rules, tested apart from HTTP.

Sufficiency (may this be searched at all) is a property of measured ink.
Vector availability (what the stroke branch can rank) is a property of the
delivered payload and the reported counts. Conflating them is what used to make
a substantive raster import "insufficient", so each rule is pinned here against
`InkStats` values that isolate it.
"""

from __future__ import annotations

import gzip
import json
import math

from linescout_api.preprocessing import (
    InkStats,
    decode_strokes,
    raster_sufficiency,
    vector_branch,
)

MIN_RATIO = 0.02
MIN_POINTS = 20
SIDE = 512


def stats_with(ink_pixels: int, bbox: tuple[int, int, int, int] | None) -> InkStats:
    return InkStats(width=SIDE, height=SIDE, ink_pixels=ink_pixels, bbox=bbox)


def strokes_payload(points_per_stroke: int, strokes: int):  # noqa: ANN201
    payload = {
        "version": 1,
        "canvas_width": 2048,
        "canvas_height": 2048,
        "strokes": [
            {
                "tool": "pressure",
                "pointer": "pen",
                "points": [
                    {"x": 1 + step, "y": 2, "p": 0.5, "t": step}
                    for step in range(points_per_stroke)
                ],
            }
            for _ in range(strokes)
        ],
    }
    return decode_strokes(gzip.compress(json.dumps(payload).encode("utf-8")), 1 << 20)


# --------------------------------------------------------------------- raster


def test_no_ink_is_blank() -> None:
    """A fully transparent import flattens to white: zero ink, therefore blank.

    `ink_stats` normalizes transparency onto white before measuring, so a
    transparent PNG and a solid white PNG both arrive here as "no ink pixels",
    and both are blank — the distinction the client needs so a blank import is
    never mistaken for content.
    """
    rule = raster_sufficiency(stats_with(0, None), MIN_RATIO)
    assert rule.blank and rule.insufficient and not rule.sufficient


def test_a_small_mark_is_insufficient_but_not_blank() -> None:
    rule = raster_sufficiency(stats_with(16, (250, 250, 254, 254)), MIN_RATIO)
    assert not rule.blank
    assert rule.insufficient and not rule.sufficient


def test_sparse_ink_that_spreads_the_canvas_is_sufficient() -> None:
    # A confident single-stroke sketch: hardly any coverage, plenty of spread.
    rule = raster_sufficiency(stats_with(40, (10, 10, 500, 500)), MIN_RATIO)
    assert rule.sufficient and not rule.blank


def test_diagonal_ratio_is_measured_against_the_snapshot_diagonal() -> None:
    bbox = (60, 60, 470, 470)
    assert math.isclose(
        stats_with(4000, bbox).bbox_diagonal_ratio,
        math.hypot(410, 410) / math.hypot(SIDE, SIDE),
    )


def test_vector_counts_are_not_an_input_to_sufficiency() -> None:
    """Identical pixels, every possible vector report: one verdict.

    This is the regression the old rule had — `point_count < min_points` made a
    fully drawn snapshot "insufficient". Nothing in the raster path may now
    depend on counts the drawing itself does not need.
    """
    figure = stats_with(4000, (60, 60, 470, 470))
    for _ in (0, 1, 19, 20, 5_000):
        assert raster_sufficiency(figure, MIN_RATIO).sufficient
    assert raster_sufficiency(stats_with(0, None), MIN_RATIO).blank


# ------------------------------------------------------------------- vectors


def test_no_vector_data_at_all_is_absent() -> None:
    vector = vector_branch(None, stroke_count=0, point_count=0, min_points=MIN_POINTS)
    assert vector.status == "absent"
    assert not vector.usable
    assert vector.degradation is not None
    assert vector.degradation[0] == "vector_absent"


def test_small_nonzero_vector_counts_are_sparse_not_absent() -> None:
    vector = vector_branch(None, stroke_count=1, point_count=19, min_points=MIN_POINTS)
    assert vector.status == "sparse"
    kind, detail = vector.degradation  # type: ignore[misc]
    assert kind == "vector_sparse"
    assert "19" in detail and "20" in detail


def test_delivered_payload_is_the_ground_truth_for_the_count() -> None:
    # The client claims 900 points; the payload it delivered carries 6, so 6 is
    # what the branch is judged on — and the detail says the count was measured.
    payload = strokes_payload(points_per_stroke=2, strokes=3)
    assert payload is not None
    vector = vector_branch(payload, stroke_count=3, point_count=900, min_points=MIN_POINTS)
    assert vector.status == "sparse"
    assert vector.point_count == 6
    assert vector.measured
    assert vector.degradation is not None
    assert "client-reported" not in vector.degradation[1]


def test_measured_payload_above_the_floor_is_usable() -> None:
    payload = strokes_payload(points_per_stroke=8, strokes=4)
    assert payload is not None
    vector = vector_branch(payload, stroke_count=4, point_count=32, min_points=MIN_POINTS)
    assert vector.usable
    assert vector.degradation is None


def test_reported_counts_without_a_payload_are_usable_but_unmeasured() -> None:
    # A live drawing whose client delivered no payload: the branch runs on the
    # estimate, and the estimate is labelled as one.
    vector = vector_branch(None, stroke_count=14, point_count=900, min_points=MIN_POINTS)
    assert vector.status == "usable"
    assert not vector.measured


def test_unmeasured_sparse_estimate_says_so() -> None:
    vector = vector_branch(None, stroke_count=1, point_count=19, min_points=MIN_POINTS)
    assert vector.degradation is not None
    assert "client-reported count" in vector.degradation[1]


def test_empty_strokes_in_a_payload_are_sparse_not_absent() -> None:
    # Strokes were delivered, they just carry no points: a different fact from
    # "no vector input was sent", and it reads that way on the wire.
    payload = strokes_payload(points_per_stroke=0, strokes=2)
    assert payload is not None
    vector = vector_branch(payload, stroke_count=2, point_count=0, min_points=MIN_POINTS)
    assert vector.status == "sparse"


def test_points_without_strokes_is_sparse() -> None:
    vector = vector_branch(None, stroke_count=0, point_count=900, min_points=MIN_POINTS)
    assert vector.status == "sparse"


# --------------------------------------------------------------- both at once


def test_substantive_raster_searches_even_with_an_empty_vector_branch() -> None:
    rule = raster_sufficiency(stats_with(4000, (60, 60, 470, 470)), MIN_RATIO)
    vector = vector_branch(None, stroke_count=0, point_count=0, min_points=MIN_POINTS)
    assert rule.sufficient
    assert vector.degradation is not None  # degraded, not rejected


def test_blank_canvas_reports_both_facts_separately() -> None:
    """Blank input and an empty vector branch are two items, not one.

    "Nothing was drawn" and "the stroke branch had nothing to rank" have
    different fixes; collapsing them into a single insufficient answer is what
    made transparent imports indistinguishable from sparse ones.
    """
    rule = raster_sufficiency(stats_with(0, None), MIN_RATIO)
    vector = vector_branch(None, stroke_count=0, point_count=0, min_points=MIN_POINTS)
    kinds = [item.kind for item in _input_degradations(rule, vector)]
    assert kinds == ["blank_raster", "vector_absent"]


def _input_degradations(raster, vector):  # noqa: ANN001, ANN201
    """`_degradations` against a healthy, non-fixture server: only the input items."""
    from types import SimpleNamespace

    from linescout_api.routers.search import _degradations

    state = SimpleNamespace(
        assets=[object()],
        settings=SimpleNamespace(fixture_mode=False),
        device=SimpleNamespace(is_cuda=True),
        disabled_branches=[],
    )
    return _degradations(state, raster=raster, vector=vector)  # type: ignore[arg-type]
