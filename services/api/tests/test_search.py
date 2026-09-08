from __future__ import annotations

import gzip
import io
import json
import math
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from PIL import Image

from linescout_api.main import create_app
from linescout_api.preprocessing import decode_snapshot, ink_stats, normalize_to_gray, tight_crop
from tests.conftest import (
    draw_arc,
    draw_eye,
    draw_figure,
    draw_two_figures,
    make_settings,
    png_bytes,
    post_search,
)

# ---------------------------------------------------------------- preprocessing


def test_normalization_handles_every_input_mode() -> None:
    for mode in ("RGB", "RGBA", "L", "LA", "P", "1"):
        image = Image.new(mode, (64, 64))
        gray = normalize_to_gray(image)
        assert gray.mode == "L" and gray.size == (64, 64)


def test_transparent_pixels_become_white_not_black() -> None:
    gray = decode_snapshot(png_bytes(None, size=64, mode="RGBA"), max_bytes=1 << 20)
    assert gray.getextrema() == (255, 255)
    assert ink_stats(gray).ink_pixels == 0


def test_tight_crop_preserves_geometry() -> None:
    gray = decode_snapshot(png_bytes(draw_eye, size=512), max_bytes=1 << 20)
    stats = ink_stats(gray)
    assert stats.bbox is not None
    crop = tight_crop(gray, stats)
    assert crop.width == crop.height  # letterboxed square
    left, top, right, bottom = stats.bbox
    assert crop.width >= max(right - left, bottom - top)  # padding never shrinks the box
    assert ink_stats(crop).ink_pixels == stats.ink_pixels  # no ink lost


# ---------------------------------------------------------------- contract validation


def test_blank_canvas_is_insufficient_not_an_error(client: TestClient, session_id: str) -> None:
    status, body = post_search(
        client, session_id, png_bytes(None), stroke_count=0, point_count=0, revision=3
    )
    assert status == 200
    assert body["mode"] == "insufficient"
    assert body["revision"] == 3
    assert body["groups"] == []


def test_small_vector_counts_do_not_veto_a_sufficient_raster(
    client: TestClient, session_id: str
) -> None:
    """A degraded stroke branch is a degradation, never a rejected drawing.

    The figure below is fully drawn in the snapshot but reports only 19
    sampled points — under the 20 the stroke branch wants. The drawing stays
    searchable; what shrinks is the vector branch, and that has to be visible
    structurally rather than by the response silently coming back empty.
    """
    status, body = post_search(
        client, session_id, png_bytes(draw_figure), point_count=19, stroke_count=1
    )
    assert status == 200
    assert body["mode"] != "insufficient"
    assert body["groups"]
    kinds = {item["kind"] for item in body["degradations"]}
    assert "vector_sparse" in kinds
    assert "blank_raster" not in kinds


def test_transparent_import_is_blank_not_substantive(client: TestClient, session_id: str) -> None:
    """A fully transparent PNG *is* an empty canvas once flattened onto white.

    It gets the same 200/insufficient answer as a canvas nothing was drawn on,
    and it says so structurally: `blank_raster` is what tells a client "your
    import had no ink" apart from "your drawing is too small to read".
    """
    transparent = png_bytes(None, size=512, mode="RGBA")
    status, body = post_search(client, session_id, transparent, stroke_count=0, point_count=0)
    assert status == 200
    assert body["mode"] == "insufficient"
    assert body["groups"] == []
    assert {item["kind"] for item in body["degradations"]} >= {"blank_raster"}
    # A substantive import of the same file size is *not* blank: same shape on
    # the wire, opposite verdict, decided purely by the measured ink.
    status, inked = post_search(
        client, session_id, png_bytes(draw_figure, mode="RGBA"), stroke_count=0, point_count=0
    )
    assert status == 200
    assert inked["mode"] != "insufficient"
    assert not any(item["kind"] == "blank_raster" for item in inked["degradations"])


def test_tiny_ink_bbox_is_insufficient(client: TestClient, session_id: str) -> None:
    dot = png_bytes(lambda d, s: d.ellipse((250, 250, 254, 254), fill="black"))
    status, body = post_search(client, session_id, dot, point_count=40)
    assert status == 200 and body["mode"] == "insufficient"


def test_malformed_image_is_400(client: TestClient, session_id: str) -> None:
    status, body = post_search(client, session_id, b"definitely not a png")
    assert status == 400
    assert body["error"]["code"] == "image_malformed"
    assert body["error"]["field"] == "image"
    assert body["schema_version"] == 1
    assert body["retryable"] is False
    UUID(str(body["request_id"]))


def test_non_png_image_is_422(client: TestClient, session_id: str) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, "JPEG")
    status, body = post_search(client, session_id, buffer.getvalue())
    assert status == 422 and body["error"]["code"] == "image_format"


def test_oversized_image_is_413(client: TestClient, session_id: str) -> None:
    status, body = post_search(client, session_id, b"\x89PNG" + b"\x00" * (4 * 1024 * 1024))
    assert status == 413 and body["error"]["code"] == "image_too_large"


def test_invalid_style_enum_is_422(client: TestClient, session_id: str) -> None:
    status, body = post_search(
        client, session_id, png_bytes(draw_figure), selected_style="watercolor"
    )
    assert status == 422
    assert body["error"]["field"] == "selected_style"


def test_unsupported_canvas_dimensions_are_422(client: TestClient, session_id: str) -> None:
    status, body = post_search(client, session_id, png_bytes(draw_figure), canvas_width=10)
    assert status == 422 and body["error"]["field"] == "canvas_width"
    status, body = post_search(client, session_id, png_bytes(draw_figure), canvas_width=512)
    assert status == 422
    assert body["error"]["code"] == "canvas_dimensions"
    assert body["error"]["field"] == "canvas_width"
    status, body = post_search(client, session_id, png_bytes(draw_figure), canvas_height=1024)
    assert status == 422
    assert body["error"]["code"] == "canvas_dimensions"
    assert body["error"]["field"] == "canvas_height"


def test_wrong_snapshot_size_is_422(client: TestClient, session_id: str) -> None:
    status, body = post_search(client, session_id, png_bytes(draw_figure, size=64))
    assert status == 422
    assert body["error"]["code"] == "image_dimensions"
    assert body["error"]["field"] == "image"


def test_long_text_hint_is_422(client: TestClient, session_id: str) -> None:
    status, body = post_search(client, session_id, png_bytes(draw_figure), text_hint="x" * 121)
    assert status == 422 and body["error"]["code"] == "text_hint_too_long"


def test_zero_revision_is_422(client: TestClient, session_id: str) -> None:
    status, _ = post_search(client, session_id, png_bytes(draw_figure), revision=0)
    assert status == 422


def test_invalid_session_id_is_422(client: TestClient, session_id: str) -> None:
    status, body = post_search(client, "not-a-uuid", png_bytes(draw_figure))
    assert status == 422 and body["error"]["field"] == "session_id"


def _stroke_payload(count: int) -> bytes:
    sequence = {
        "version": 1,
        "canvas_width": 2048,
        "canvas_height": 2048,
        "strokes": [
            {
                "tool": "pressure",
                "pointer": "pen",
                "points": [{"x": 1, "y": 2, "p": 0.5, "t": 0}, {"x": 3, "y": 4, "p": 0.6, "t": 8}],
            }
            for _ in range(count)
        ],
    }
    return gzip.compress(json.dumps(sequence).encode("utf-8"))


def test_stroke_sequence_round_trip_and_mismatch(client: TestClient, session_id: str) -> None:
    status, body = post_search(
        client,
        session_id,
        png_bytes(draw_figure),
        strokes=_stroke_payload(3),
        stroke_count=3,
        point_count=100,
    )
    assert status == 200 and body["mode"] in ("provisional", "confident")
    status, body = post_search(
        client, session_id, png_bytes(draw_figure), strokes=_stroke_payload(3), stroke_count=2
    )
    assert status == 422 and body["error"]["code"] == "stroke_count_mismatch"


def test_malformed_strokes_are_400(client: TestClient, session_id: str) -> None:
    status, body = post_search(client, session_id, png_bytes(draw_figure), strokes=b"not gzip")
    assert status == 400 and body["error"]["code"] == "strokes_malformed"
    bad_json = gzip.compress(b'{"version": 1, "strokes": "nope"}')
    status, body = post_search(client, session_id, png_bytes(draw_figure), strokes=bad_json)
    assert status == 400 and body["error"]["code"] == "strokes_malformed"


def test_non_finite_stroke_points_are_rejected(client: TestClient, session_id: str) -> None:
    for value in (math.nan, math.inf, -math.inf):
        sequence = {
            "version": 1,
            "canvas_width": 2048,
            "canvas_height": 2048,
            "strokes": [
                {
                    "tool": "pressure",
                    "pointer": "pen",
                    "points": [{"x": value, "y": 2, "p": 0.5, "t": 0}],
                }
            ],
        }
        payload = gzip.compress(json.dumps(sequence).encode("utf-8"))
        status, body = post_search(client, session_id, png_bytes(draw_figure), strokes=payload)
        assert status == 400
        assert body["error"]["code"] == "strokes_malformed"
        assert body["retryable"] is False


def test_oversized_strokes_are_413(client: TestClient, session_id: str) -> None:
    status, body = post_search(
        client,
        session_id,
        png_bytes(draw_figure),
        strokes=b"\x1f\x8b" + b"\x00" * (2 * 1024 * 1024),
    )
    assert status == 413 and body["error"]["code"] == "strokes_too_large"


# ---------------------------------------------------------------- response structure


def test_revision_is_echoed_unchanged(client: TestClient, session_id: str) -> None:
    for revision in (1, 42, 999_999):
        _, body = post_search(
            client,
            session_id,
            png_bytes(draw_figure),
            revision=revision,
            stroke_count=14,
            point_count=900,
        )
        assert body["revision"] == revision


def test_single_curve_is_provisional_with_scope_groups(client: TestClient, session_id: str) -> None:
    _, body = post_search(client, session_id, png_bytes(draw_arc), stroke_count=1, point_count=40)
    assert body["mode"] == "provisional"
    assert 1 <= len(body["groups"]) <= 3
    for group in body["groups"]:
        assert group["kind"] == "provisional_scope"
        assert group["scope"] is not None
        assert 1 <= len(group["results"]) <= 4


def test_full_figure_is_confident_with_best_match_first(
    client: TestClient, session_id: str
) -> None:
    _, body = post_search(
        client, session_id, png_bytes(draw_figure), stroke_count=14, point_count=900
    )
    assert body["mode"] == "confident"
    assert body["groups"][0]["id"] == "best_match"
    assert body["groups"][0]["kind"] == "best_match"
    assert 1 <= len(body["groups"][0]["results"]) <= 8
    style_groups = body["groups"][1:]
    assert all(group["kind"] == "style" for group in style_groups)
    assert all(len(group["results"]) <= 6 for group in style_groups)
    assert len({group["style"] for group in style_groups}) == len(style_groups)
    assert body["scope_predictions"][0]["label"] == "full_body"
    assert set(body["timing"]) == {
        "preprocessing_ms",
        "embedding_ms",
        "retrieval_ms",
        "reranking_ms",
        "total_ms",
    }


def test_two_figures_lean_toward_multi_character(client: TestClient, session_id: str) -> None:
    _, body = post_search(
        client, session_id, png_bytes(draw_two_figures), stroke_count=24, point_count=1600
    )
    labels = [prediction["label"] for prediction in body["scope_predictions"][:2]]
    assert "multi_character" in labels


def test_style_rows_only_contain_their_style(client: TestClient, session_id: str) -> None:
    _, body = post_search(
        client, session_id, png_bytes(draw_figure), stroke_count=14, point_count=900
    )
    for group in body["groups"][1:]:
        assert all(result["style"] == group["style"] for result in group["results"])


def test_results_never_include_disabled_assets(client: TestClient, session_id: str) -> None:
    disabled = "ls_synthetic_dd4dbf8306f50951"  # the rejected synthetic record
    for draw, strokes in ((draw_arc, 1), (draw_figure, 14), (draw_two_figures, 24)):
        _, body = post_search(
            client, session_id, png_bytes(draw), stroke_count=strokes, point_count=900
        )
        ids = {result["asset_id"] for group in body["groups"] for result in group["results"]}
        assert disabled not in ids


def test_trace_allowed_follows_native_origin(client: TestClient, session_id: str) -> None:
    _, body = post_search(
        client, session_id, png_bytes(draw_figure), stroke_count=14, point_count=900
    )
    seen_native = False
    seen_extracted = False
    for group in body["groups"]:
        for result in group["results"]:
            native = result["origin"] == "native_line_art"
            assert result["trace_allowed"] is native
            seen_native = seen_native or native
            seen_extracted = seen_extracted or not native
    assert seen_native and seen_extracted


def test_no_result_below_relevance_floor(client: TestClient, session_id: str) -> None:
    from linescout_api.fixture_ranker import RELEVANCE_FLOOR

    _, body = post_search(
        client, session_id, png_bytes(draw_figure), stroke_count=14, point_count=900
    )
    for group in body["groups"]:
        for result in group["results"]:
            assert result["relevance"] >= RELEVANCE_FLOOR


def test_search_is_deterministic(client: TestClient, session_id: str) -> None:
    a = post_search(client, session_id, png_bytes(draw_figure), stroke_count=14, point_count=900)[1]
    b = post_search(client, session_id, png_bytes(draw_figure), stroke_count=14, point_count=900)[1]
    # Timing and the per-request request_id are inherently non-deterministic;
    # everything else (results, identity, versions) must be stable.
    for payload in (a, b):
        payload.pop("timing")
        payload.pop("request_id")
    assert a == b


def test_explicit_style_moves_row_second_but_best_match_stays_first(
    client: TestClient, session_id: str
) -> None:
    _, body = post_search(
        client,
        session_id,
        png_bytes(draw_figure),
        stroke_count=14,
        point_count=900,
        selected_style="gesture_sketch",
    )
    assert body["groups"][0]["id"] == "best_match"
    assert body["groups"][1]["style"] == "gesture_sketch"
    best_ids = [result["asset_id"] for result in body["groups"][0]["results"]]
    _, baseline = post_search(
        client, session_id, png_bytes(draw_figure), stroke_count=14, point_count=900
    )
    assert best_ids == [result["asset_id"] for result in baseline["groups"][0]["results"]]


def test_not_ready_api_returns_structured_503(tmp_path: Path, session_id: str) -> None:
    settings = make_settings(tmp_path, gallery_manifest=tmp_path / "missing.json")
    with TestClient(create_app(settings)) as client:
        status, body = post_search(
            client, session_id, png_bytes(draw_figure), stroke_count=14, point_count=900
        )
        assert status == 503 and body["error"]["code"] == "not_ready"
        assert body["retryable"] is True
        assert body["schema_version"] == 1
        UUID(str(body["request_id"]))
        # Insufficient input short-circuits before readiness so a blank canvas still gets 200.
        status, body = post_search(
            client, session_id, png_bytes(None), stroke_count=0, point_count=0
        )
        assert status == 200 and body["mode"] == "insufficient"


def test_imported_raster_with_zero_points_is_searchable(
    client: TestClient, session_id: str
) -> None:
    """The substantive import: real line art, no vector geometry at all.

    Zero strokes and zero points is what an imported PNG/SVG flattens to, so
    the counts must not be read as "nothing was drawn". The stroke branch is
    off for this query and says so structurally; the search itself runs on the
    snapshot.
    """
    status, body = post_search(
        client, session_id, png_bytes(draw_figure), stroke_count=0, point_count=0
    )
    assert status == 200
    assert body["mode"] in ("provisional", "confident")
    assert body["groups"]
    assert body["stroke_status"] == "absent"
    assert body["counts_approximate"] is True
    vector = [item for item in body["degradations"] if item["kind"] == "vector_absent"]
    assert len(vector) == 1
    assert "stroke branch" in vector[0]["detail"]
    assert not any(item["kind"] == "blank_raster" for item in body["degradations"])


def test_zip_bomb_strokes_are_413(client: TestClient, session_id: str) -> None:
    bomb = gzip.compress(b"\x00" * (2 * 1024 * 1024))
    status, body = post_search(client, session_id, png_bytes(draw_figure), strokes=bomb)
    assert status == 413 and body["error"]["code"] == "strokes_too_large"


# ---------------------------------------------------------------- identity & bounds


def test_success_echoes_identity_and_versions(client: TestClient, session_id: str) -> None:
    status, body = post_search(
        client, session_id, png_bytes(draw_figure), stroke_count=14, point_count=900
    )
    assert status == 200
    UUID(str(body["request_id"]))
    assert isinstance(body["api_version"], str) and body["api_version"]
    assert body["canvas_width"] == 2048 and body["canvas_height"] == 2048
    assert isinstance(body["preprocessing_version"], str) and body["preprocessing_version"]
    assert body["dataset_version"]  # fixture gallery is loaded
    assert isinstance(body["index_version"], str) and body["index_version"]


def test_counts_and_stroke_status_report_exact_when_vector_payload_matches(
    client: TestClient, session_id: str
) -> None:
    status, body = post_search(
        client,
        session_id,
        png_bytes(draw_figure),
        strokes=_stroke_payload(3),
        stroke_count=3,
        point_count=6,  # exactly the delivered vector point total (2 points/stroke)
    )
    assert status == 200
    assert body["stroke_status"] == "present"
    assert body["counts_approximate"] is False


def test_raster_only_query_reports_approximate_counts(client: TestClient, session_id: str) -> None:
    status, body = post_search(
        client, session_id, png_bytes(draw_figure), stroke_count=0, point_count=0
    )
    assert status == 200
    assert body["stroke_status"] == "absent"
    assert body["counts_approximate"] is True


def test_point_count_discrepancy_is_flagged_not_trusted(
    client: TestClient, session_id: str
) -> None:
    # The reported point_count disagrees with the delivered vector payload; the
    # query is still searchable but the discrepancy is surfaced, never silently
    # trusted as exact.
    status, body = post_search(
        client,
        session_id,
        png_bytes(draw_figure),
        strokes=_stroke_payload(3),
        stroke_count=3,
        point_count=100,
    )
    assert status == 200
    assert body["stroke_status"] == "present"
    assert body["counts_approximate"] is True


def test_huge_integers_are_structured_422s(client: TestClient, session_id: str) -> None:
    for field, value in (
        ("revision", 10**12),
        ("stroke_count", 10**12),
        ("point_count", 10**12),
        ("canvas_width", 10**12),
        ("canvas_height", 10**12),
    ):
        status, body = post_search(client, session_id, png_bytes(draw_figure), **{field: value})
        assert status == 422, field
        assert body["schema_version"] == 1
        assert "error" in body
        UUID(str(body["request_id"]))
        assert (
            "errors" in (body.get("error", {}).get("details") or {})
            or body["error"].get("details") is None
        )


def test_not_ready_details_are_structured(tmp_path: Path, session_id: str) -> None:
    settings = make_settings(tmp_path, gallery_manifest=tmp_path / "missing.json")
    with TestClient(create_app(settings)) as client:
        status, body = post_search(
            client, session_id, png_bytes(draw_figure), stroke_count=14, point_count=900
        )
        assert status == 503
        assert body["error"]["code"] == "not_ready"
        assert body["retryable"] is True
        details = body["error"]["details"]
        assert details is not None
        assert details["warmup"] in ("pending", "complete", "skipped")
        assert details["gallery_loaded"] is False
        assert details["setup_error"] is not None
        # Readiness is a 503; blank input still short-circuits to 200 insufficient.
        status, body = post_search(
            client, session_id, png_bytes(None), stroke_count=0, point_count=0
        )
        assert status == 200 and body["mode"] == "insufficient"


# ---------------------------------------------------------------- assets


def test_thumbnail_and_line_art_are_served_for_enabled_assets(
    client: TestClient, session_id: str
) -> None:
    _, body = post_search(
        client, session_id, png_bytes(draw_figure), stroke_count=14, point_count=900
    )
    result = body["groups"][0]["results"][0]
    for url in (result["thumbnail_url"], result["asset_url"]):
        response = client.get(url.replace("http://testserver", ""))
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(response.content)).size[0] > 0


def test_disabled_asset_files_are_not_served(client: TestClient) -> None:
    response = client.get("/api/v1/assets/ls_synthetic_dd4dbf8306f50951/thumbnail")
    assert response.status_code == 404
    assert client.get("/api/v1/assets/ls_synthetic_dd4dbf8306f50951/line-art").status_code == 404
    assert client.get("/api/v1/assets/../../etc/passwd/thumbnail").status_code in (404, 422)
