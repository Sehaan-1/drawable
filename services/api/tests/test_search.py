from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from drawable_api.app import create_app
from drawable_api.search import MAX_BODY_BYTES, QUERY_PREPROCESSING_VERSION
from drawable_api.settings import Settings


def png(*, ink: bool) -> bytes:
    image = Image.new("RGB", (512, 512), "white")
    if ink:
        ImageDraw.Draw(image).line((100, 100, 220, 220), fill="black", width=8)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def search_data(*, session_id: str | None = None, generation: int = 1) -> dict[str, str]:
    return {
        "session_id": session_id or str(uuid4()),
        "document_id": str(uuid4()),
        "canvas_revision": "1",
        "query_generation": str(generation),
        "canvas_width": "2048",
        "canvas_height": "2048",
        "stroke_count": "0",
        "point_count": "0",
        "counts_approximate": "false",
        "stroke_status": "disabled",
        "preprocessing_version": QUERY_PREPROCESSING_VERSION,
    }


def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(Settings(tmp_path, "real", tmp_path / "metadata.sqlite3")))


def test_openapi_documents_the_bounded_search_multipart_contract(tmp_path: Path) -> None:
    schema = client(tmp_path).get("/api/openapi.json").json()
    operation = schema["paths"]["/api/v1/search"]["post"]
    body = operation["requestBody"]["content"]["multipart/form-data"]["schema"]

    assert body["additionalProperties"] is False
    assert body["properties"]["image"]["format"] == "binary"
    assert "strokes" not in body["required"]
    assert {"200", "400", "409", "413", "422", "429", "503"}.issubset(
        operation["responses"]
    )


def test_blank_query_returns_insufficient_without_starting_training(tmp_path: Path) -> None:
    response = client(tmp_path).post(
        "/api/v1/search",
        data=search_data(generation=7),
        files={"image": ("query.png", png(ink=False), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "insufficient"
    assert body["reason"] == "insufficient_visible_ink"
    assert body["query_generation"] == 7
    assert body["groups"] == []
    assert body["training_started"] is False
    assert body["data_uploaded"] is False
    assert response.headers["x-request-id"] == body["request_id"]


def test_sufficient_query_is_validated_but_search_remains_unavailable(tmp_path: Path) -> None:
    response = client(tmp_path).post(
        "/api/v1/search",
        data=search_data(),
        files={"image": ("query.png", png(ink=True), "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "search_not_configured"
    assert "tmp_path" not in response.text
    assert str(tmp_path) not in response.text


def test_search_rejects_stale_generation_and_preprocessing_recipe(tmp_path: Path) -> None:
    api = client(tmp_path)
    session_id = str(uuid4())
    files = {"image": ("query.png", png(ink=False), "image/png")}
    assert api.post(
        "/api/v1/search", data=search_data(session_id=session_id, generation=4), files=files
    ).status_code == 200
    stale = api.post(
        "/api/v1/search", data=search_data(session_id=session_id, generation=4), files=files
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stale_generation"

    wrong_recipe = search_data(generation=1)
    wrong_recipe["preprocessing_version"] = "old-client"
    mismatch = api.post(
        "/api/v1/search",
        data=wrong_recipe,
        files={"image": ("query.png", png(ink=False), "image/png")},
    )
    assert mismatch.status_code == 409
    assert mismatch.json()["error"]["code"] == "preprocessing_version_mismatch"


def test_search_rejects_oversize_body_before_parsing_multipart(tmp_path: Path) -> None:
    response = client(tmp_path).post(
        "/api/v1/search",
        content=b"x" * (MAX_BODY_BYTES + 1),
        headers={"Content-Type": "multipart/form-data; boundary=test-boundary"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "search_body_too_large"


def test_search_bounds_untrusted_field_names_and_numeric_parsing(tmp_path: Path) -> None:
    api = client(tmp_path)
    fields = search_data()
    fields["canvas_revision"] = "9" * 5000
    oversized_integer = api.post(
        "/api/v1/search",
        data=fields,
        files={"image": ("query.png", png(ink=False), "image/png")},
    )
    assert oversized_integer.status_code == 422
    assert oversized_integer.json()["error"]["code"] == "invalid_search_schema"

    private_marker = "private-path-" + "x" * 300
    unknown_field = api.post(
        "/api/v1/search",
        data={**search_data(), private_marker: "value"},
        files={"image": ("query.png", png(ink=False), "image/png")},
    )
    assert unknown_field.status_code == 422
    assert private_marker not in unknown_field.text


def test_search_rejects_non_png_and_malformed_stroke_gzip(tmp_path: Path) -> None:
    api = client(tmp_path)
    invalid_image = api.post(
        "/api/v1/search",
        data=search_data(),
        files={"image": ("query.png", b"not a png", "image/png")},
    )
    assert invalid_image.status_code == 400
    assert invalid_image.json()["error"]["code"] == "invalid_search_image"

    stroke_fields = search_data()
    stroke_fields["stroke_count"] = "1"
    stroke_fields["point_count"] = "20"
    stroke_fields["stroke_status"] = "available"
    malformed_strokes = api.post(
        "/api/v1/search",
        data=stroke_fields,
        files={
            "image": ("query.png", png(ink=True), "image/png"),
            "strokes": ("strokes.json.gz", b"not gzip", "application/gzip"),
        },
    )
    assert malformed_strokes.status_code == 400
    assert malformed_strokes.json()["error"]["code"] == "invalid_gzip"


def test_valid_bounded_stroke_sequence_reaches_not_configured_search(tmp_path: Path) -> None:
    tokens: list[dict[str, object]] = []
    for index in range(20):
        x = index / 19
        tokens.append(
            {
                "x": x,
                "y": 0.5,
                "dx": 0.0 if index == 0 else 1 / 19,
                "dy": 0.0,
                "pressure": 0.5,
                "pressure_available": False,
                "dt": 0.0,
                "time_available": False,
                "opacity": 1.0,
                "marker": "start" if index == 0 else "end" if index == 19 else "continue",
            }
        )
    encoded = gzip.compress(
        json.dumps({"schema_version": 1, "token_count": 20, "tokens": tokens}).encode("utf-8")
    )
    fields = search_data()
    fields["stroke_count"] = "1"
    fields["point_count"] = "20"
    fields["stroke_status"] = "available"
    response = client(tmp_path).post(
        "/api/v1/search",
        data=fields,
        files={
            "image": ("query.png", png(ink=True), "image/png"),
            "strokes": ("strokes.json.gz", encoded, "application/gzip"),
        },
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "search_not_configured"
