"""Security bounds: receive-layer body caps, gzip decompression, image headers.

The acceptance focus is *where* work stops, not only the final status code:
the byte counter is asserted to stop pulling, the multipart parser and
decompressor are asserted to stop at the limit, and forged metadata
(Content-Length, PNG IHDR) is asserted to fail before trusting it.
"""

from __future__ import annotations

import gzip
import io
import json
import random
import struct
import zlib
from collections.abc import Awaitable, Callable, Iterator
from typing import Any
from uuid import UUID
from zlib import crc32

import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from linescout_api.config import Settings
from linescout_api.limits import RequestBodyLimitMiddleware
from linescout_api.preprocessing import (
    MAX_COMPRESSED_BYTES,
    MAX_DECOMPRESSED_BYTES,
    MAX_SNAPSHOT_EDGE,
    GzipLimitError,
    SnapshotError,
    decode_snapshot,
    decode_strokes,
    safe_decompress_gzip,
)
from tests.conftest import draw_figure, png_bytes, post_search, search_form

# ----------------------------------------------------------------- ASGI harness

_SCOPE: Scope = {
    "type": "http",
    "http_version": "1.1",
    "asgi": {"version": "3.0"},
    "method": "POST",
    "scheme": "http",
    "path": "/api/v1/search",
    "raw_path": b"/api/v1/search",
    "query_string": b"",
    "root_path": "",
    "headers": [],
    "client": ("127.0.0.1", 1234),
    "server": ("127.0.0.1", 8000),
}


class ReceiveProbe:
    """Feeds scripted ``http.request`` chunks and records how many were pulled."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.pulled = 0
        self.bytes_pulled = 0

    async def __call__(self) -> Message:
        if self.pulled < len(self.chunks):
            chunk = self.chunks[self.pulled]
            self.pulled += 1
            self.bytes_pulled += len(chunk)
            more = self.pulled < len(self.chunks)
            return {"type": "http.request", "body": chunk, "more_body": more}
        return {"type": "http.disconnect"}


def recording_app(received: dict[str, Any]) -> Callable[[Scope, Receive, Send], Awaitable[None]]:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        received["called"] = True
        received_body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            received_body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        received["body"] = bytes(received_body)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    return app


async def run_middleware(
    chunks: list[bytes],
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    max_body_bytes: int = 1024,
    max_parts: int = 4,
) -> tuple[list[Message], ReceiveProbe, dict[str, Any]]:
    scope = dict(_SCOPE)
    scope["headers"] = list(headers or [])
    inner: dict[str, Any] = {"called": False, "body": b""}
    probe = ReceiveProbe(chunks)
    app = RequestBodyLimitMiddleware(
        recording_app(inner), max_body_bytes=max_body_bytes, max_multipart_parts=max_parts
    )
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, probe, send)
    return sent, probe, inner


def response_of(sent: list[Message]) -> tuple[int, dict[str, str], dict[str, Any]]:
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    headers = {k.decode(): v.decode() for k, v in start.get("headers", [])}
    parsed = json.loads(body) if body[:1] == b"{" else {}
    return start["status"], headers, parsed


def assert_error_envelope(
    status: int, headers: dict[str, str], body: dict[str, Any], code: str, expect_status: int
) -> None:
    assert status == expect_status
    assert body["schema_version"] == 1
    assert body["retryable"] is False
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    UUID(body["request_id"])
    assert headers.get("x-request-id") == body["request_id"]


# ----------------------------------------------------------------- receive-layer cap


@pytest.mark.parametrize("content_length_present", [True, False])
async def test_exact_body_boundary_passes_and_reaches_app(
    content_length_present: bool,
) -> None:
    chunks = [b"x" * 200] * 5  # exactly 1000 bytes
    headers = [(b"content-length", b"1000")] if content_length_present else []
    sent, probe, inner = await run_middleware(chunks, headers=headers, max_body_bytes=1000)
    status, _, _ = response_of(sent)
    assert status == 200
    assert inner["called"] and inner["body"] == b"".join(chunks)
    assert probe.bytes_pulled == 1000


async def test_boundary_plus_one_is_413_and_reception_stops() -> None:
    chunks = [b"x" * 200] * 20  # 4000 bytes available; cap is 1000
    sent, probe, inner = await run_middleware(chunks, max_body_bytes=1000)
    status, headers, body = response_of(sent)
    assert_error_envelope(status, headers, body, "request_too_large", 413)
    # The app (and therefore the multipart parser) never ran, and the
    # counter stopped pulling the moment it crossed the cap: 5 chunks fit,
    # the 6th crossed, chunks 7..20 were never consumed.
    assert inner["called"] is False
    assert probe.pulled == 6
    assert probe.bytes_pulled == 1200
    assert body["error"]["details"]["max_bytes"] == 1000
    assert body["error"]["details"]["received_bytes"] == 1200


async def test_forged_small_content_length_loses_to_the_byte_counter() -> None:
    # Declares 10 bytes, streams 4000: the header must only be an
    # early-rejection optimization, never the authority.
    headers = [(b"content-length", b"10")]
    sent, probe, inner = await run_middleware(
        [b"x" * 200] * 20, headers=headers, max_body_bytes=1000
    )
    status, headers, body = response_of(sent)
    assert_error_envelope(status, headers, body, "request_too_large", 413)
    assert inner["called"] is False
    assert probe.pulled == 6


async def test_oversized_content_length_is_rejected_without_reading() -> None:
    headers = [(b"content-length", b"99999"), (b"content-type", b"application/octet-stream")]
    sent, probe, inner = await run_middleware(
        [b"x" * 200] * 20, headers=headers, max_body_bytes=1000
    )
    status, headers, body = response_of(sent)
    assert_error_envelope(status, headers, body, "request_too_large", 413)
    assert inner["called"] is False
    assert probe.pulled == 0  # nothing was read at all
    # Early Content-Length rejection still reports the declared byte count.
    assert body["error"]["details"]["max_bytes"] == 1000
    assert body["error"]["details"]["received_bytes"] == 99999


@pytest.mark.parametrize(
    "header_values",
    [[b"banana"], [b"-7"], [b"1.5"], [b"12", b"34"]],
)
async def test_invalid_content_length_is_400(header_values: list[bytes]) -> None:
    headers = [(b"content-length", value) for value in header_values]
    sent, probe, inner = await run_middleware([b"hi"], headers=headers, max_body_bytes=1000)
    status, headers, body = response_of(sent)
    assert_error_envelope(status, headers, body, "invalid_content_length", 400)
    assert inner["called"] is False
    assert probe.pulled == 0


async def test_disconnected_client_needs_no_response() -> None:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        pytest.fail("inner app must not run")

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await RequestBodyLimitMiddleware(app, max_body_bytes=1000)(dict(_SCOPE), receive, send)
    assert sent == []


# ----------------------------------------------------------------- multipart part bound


def _multipart_body(boundary: bytes, parts: list[tuple[str, bytes]]) -> bytes:
    out = bytearray()
    for name, value in parts:
        out += b"--" + boundary + b"\r\n"
        out += f'Content-Disposition: form-data; name="{name}"'.encode() + b"\r\n\r\n"
        out += value + b"\r\n"
    out += b"--" + boundary + b"--\r\n"
    return bytes(out)


async def test_multipart_parts_are_bounded() -> None:
    boundary = b"pytestboundary0123"
    headers = [
        (b"content-type", b'multipart/form-data; boundary="' + boundary + b'"'),
        (b"content-length", b"100000"),
    ]
    parts = [(f"field_{i}", b"v") for i in range(5)]  # one over the 4-part cap
    body = _multipart_body(boundary, parts)
    sent, _, inner = await run_middleware(
        [body], headers=headers, max_body_bytes=100000, max_parts=4
    )
    status, headers, parsed = response_of(sent)
    assert_error_envelope(status, headers, parsed, "too_many_parts", 413)
    assert inner["called"] is False


async def test_multipart_at_exact_part_count_passes() -> None:
    boundary = b"pytestboundary0123"
    headers = [(b"content-type", b"multipart/form-data; boundary=" + boundary)]
    body = _multipart_body(boundary, [(f"field_{i}", b"v") for i in range(4)])
    sent, _, inner = await run_middleware([body], headers=headers, max_body_bytes=100000)
    assert response_of(sent)[0] == 200
    assert inner["called"] and inner["body"] == body


# ----------------------------------------------------------------- gzip hard caps


def _randbytes(seed: int, n: int) -> bytes:
    return random.Random(seed).randbytes(n)


def gzip_of_exact_size(payload: bytes, target_size: int) -> bytes:
    """A valid gzip member of exactly ``target_size`` bytes wrapping ``payload``.

    The length is matched by padding the gzip FEXTRA header field, never by
    hunting for an input whose deflate stream happens to land on the target —
    the latter depends on the linked zlib version. Decoders skip the extra
    field, so every zlib release inflates this identically. One FEXTRA field
    holds up to 65535 bytes of padding.
    """
    base = gzip.compress(payload, mtime=0)
    extra_len = target_size - len(base) - 2  # 2 bytes for the XLEN field itself
    assert 0 <= extra_len <= 0xFFFF, (len(base), extra_len)
    padded = (
        base[:3]
        + bytes([base[3] | 0x04])  # FLG bit 2: FEXTRA present
        + base[4:10]
        + struct.pack("<H", extra_len)
        + b"x" * extra_len
        + base[10:]
    )
    assert len(padded) == target_size
    assert gzip.decompress(padded) == payload  # sanity: decoders skip FEXTRA
    return padded


def test_stroke_limits_are_consistent_between_config_and_preprocessing() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.max_strokes_bytes == 256 * 1024 == MAX_COMPRESSED_BYTES
    assert settings.max_strokes_decompressed_bytes == 1024 * 1024 == MAX_DECOMPRESSED_BYTES


def test_compressed_boundary_exact_and_plus_one() -> None:
    payload = _randbytes(1, 210_000)  # incompressible; FEXTRA does the sizing
    exact = gzip_of_exact_size(payload, MAX_COMPRESSED_BYTES)
    assert len(exact) == 256 * 1024
    assert safe_decompress_gzip(exact) == payload  # boundary itself is accepted
    with pytest.raises(GzipLimitError, match="compressed_payload_too_large"):
        safe_decompress_gzip(gzip_of_exact_size(payload, MAX_COMPRESSED_BYTES + 1))


def test_expanded_boundary_exact_then_plus_one() -> None:
    block = _randbytes(7, 4096)  # repeats -> compresses far under 256 KiB
    payload = block * (MAX_DECOMPRESSED_BYTES // len(block))
    assert len(payload) == MAX_DECOMPRESSED_BYTES
    gz = gzip.compress(payload, mtime=0)
    assert len(gz) < MAX_COMPRESSED_BYTES
    assert safe_decompress_gzip(gz) == payload

    with pytest.raises(GzipLimitError, match="decompressed_payload_too_large") as error:
        safe_decompress_gzip(gzip.compress(payload + b"x", mtime=0))
    # The decompressor stopped at the cap + the single overflow byte, not later.
    assert error.value.output_bytes_produced == MAX_DECOMPRESSED_BYTES + 1


def test_highly_compressible_bomb_stops_mid_stream() -> None:
    bomb = gzip.compress(b"\x00" * (8 * 1024 * 1024), mtime=0)  # ~8 KiB on the wire
    with pytest.raises(GzipLimitError, match="decompressed_payload_too_large") as error:
        safe_decompress_gzip(bomb)
    assert error.value.output_bytes_produced == MAX_DECOMPRESSED_BYTES + 1
    # Work stops at the limit: only the prefix of the compressed stream that
    # produces 1 MiB + 1 was consumed, not the whole bomb.
    assert error.value.input_bytes_used < len(bomb)


def test_custom_caps_expand_exact_max_then_overflow_byte() -> None:
    gz = gzip.compress(b"a" * 4097, mtime=0)
    with pytest.raises(GzipLimitError) as error:
        safe_decompress_gzip(gz, max_expanded_bytes=4096)
    assert error.value.output_bytes_produced == 4097
    assert safe_decompress_gzip(gzip.compress(b"a" * 4096), max_expanded_bytes=4096) == b"a" * 4096


# ----------------------------------------------------------------- gzip container strictness


@pytest.fixture
def valid_member() -> bytes:
    sequence = {"version": 1, "canvas_width": 2048, "canvas_height": 2048, "strokes": []}
    return gzip.compress(json.dumps(sequence).encode(), mtime=0)


@pytest.mark.parametrize("cut", [1, 4, 8])
def test_truncated_gzip_is_rejected(valid_member: bytes, cut: int) -> None:
    with pytest.raises(ValueError, match="truncated"):
        safe_decompress_gzip(valid_member[:-cut])


def test_empty_gzip_is_rejected() -> None:
    with pytest.raises(ValueError, match="truncated"):
        safe_decompress_gzip(b"")


def test_concatenated_members_are_rejected(valid_member: bytes) -> None:
    second = gzip.compress(b'{"version": 1, "ok": false}', mtime=0)
    with pytest.raises(ValueError, match="trailing"):
        safe_decompress_gzip(valid_member + second)


def test_trailing_bytes_are_rejected(valid_member: bytes) -> None:
    with pytest.raises(ValueError, match="trailing"):
        safe_decompress_gzip(valid_member + b"\x00")


def test_invalid_crc_is_rejected(valid_member: bytes) -> None:
    corrupted = bytearray(valid_member)
    corrupted[-5] ^= 0xFF  # inside the CRC32 trailer
    with pytest.raises(zlib.error):
        safe_decompress_gzip(bytes(corrupted))


@pytest.mark.parametrize(
    "bad",
    [
        b"\x1f\x8c",  # wrong magic
        b"\x1f\x8b\x07",  # unknown compression method byte
        b"not a gzip stream at all",
    ],
)
def test_malformed_gzip_headers_are_rejected(bad: bytes) -> None:
    # zlib rejects garbage headers outright, or the (non-)stream dies as
    # truncated; either way it must never be accepted.
    with pytest.raises((zlib.error, ValueError)):
        safe_decompress_gzip(bad)


def test_decode_strokes_maps_structure_failures_to_malformed(valid_member: bytes) -> None:
    assert decode_strokes(valid_member, MAX_COMPRESSED_BYTES) is not None
    for broken in (valid_member[:-4], valid_member + valid_member, valid_member + b"junk"):
        with pytest.raises(SnapshotError, match="not valid gzip") as error:
            decode_strokes(broken, MAX_COMPRESSED_BYTES)
        assert error.value.code == "strokes_malformed"


# ----------------------------------------------------------------- image header inspection


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def forged_png(width: int, height: int) -> bytes:
    """Signature + IHDR + empty IDAT/IEND: a ``PIL``-openable header with no raster."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", b"")
        + _png_chunk(b"IEND", b"")
    )


@pytest.mark.parametrize(
    "width,height",
    [
        (100_000, 100_000),  # decompression-bomb shape: would be 40 GB decoded
        (MAX_SNAPSHOT_EDGE + 1, 64),  # one pixel past the edge cap
        (64, MAX_SNAPSHOT_EDGE + 1),
        (15, 512),  # below the minimum
    ],
)
def test_oversized_or_tiny_headers_rejected_without_decoding(width: int, height: int) -> None:
    data = forged_png(width, height)
    assert len(data) < 100  # no raster is even present to materialize
    with pytest.raises(SnapshotError) as error:
        decode_snapshot(data, max_bytes=4 * 1024 * 1024)
    assert error.value.code == "image_dimensions"


def test_header_with_valid_dims_but_no_pixels_is_malformed_not_oversized() -> None:
    # Control: a 512x512 header passes the dimension gate and only then fails
    # during the (attempted) pixel decode.
    with pytest.raises(SnapshotError, match="decoded") as error:
        decode_snapshot(forged_png(512, 512), max_bytes=4 * 1024 * 1024)
    assert error.value.code == "image_malformed"


def test_real_max_edge_png_still_decodes() -> None:
    gray = decode_snapshot(png_bytes(None, size=MAX_SNAPSHOT_EDGE, mode="L"), 4 * 1024 * 1024)
    assert gray.size == (MAX_SNAPSHOT_EDGE, MAX_SNAPSHOT_EDGE)
    assert gray.mode == "L"


def test_non_png_format_is_rejected_before_pixel_decode() -> None:
    from PIL import Image

    buffer = io.BytesIO()
    with Image.open(io.BytesIO(png_bytes(draw_figure))) as image:
        image.convert("RGB").save(buffer, "JPEG")
    with pytest.raises(SnapshotError, match="PNG") as error:
        decode_snapshot(buffer.getvalue(), max_bytes=4 * 1024 * 1024)
    assert error.value.code == "image_format"


# ----------------------------------------------------------------- end-to-end integration


def _search_parts(image: bytes) -> list[tuple[str, bytes]]:
    form = search_form("5e93f09a-3d34-4a11-8af0-2c1f0b28a980", stroke_count=0, point_count=0)
    parts = [(name, value.encode()) for name, value in form.items()]
    parts.append(("image", image))
    return parts


def _multipart_with_files(boundary: bytes, parts: list[tuple[str, bytes]]) -> bytes:
    out = bytearray()
    for name, value in parts:
        out += b"--" + boundary + b"\r\n"
        if name in ("image", "strokes"):
            out += (
                f'Content-Disposition: form-data; name="{name}"; filename="{name}.bin"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
        else:
            out += f'Content-Disposition: form-data; name="{name}"'.encode() + b"\r\n\r\n"
        out += value + b"\r\n"
    out += b"--" + boundary + b"--\r\n"
    return bytes(out)


def _post_raw(client: TestClient, body: bytes, boundary: bytes) -> Any:
    return client.post(
        "/api/v1/search",
        content=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary.decode()}"},
    )


def test_exact_envelope_boundary_json_then_plus_one(client: TestClient) -> None:
    settings: Settings = client.app.state.linescout.settings
    cap = settings.max_request_bytes
    # The envelope is documented separately from the 4 MiB image limit.
    assert cap == 4 * 1024 * 1024 + 256 * 1024 + settings.multipart_overhead_bytes

    event = {
        "session_id": "5e93f09a-3d34-4a11-8af0-2c1f0b28a980",
        "asset_id": "ls_synthetic_f1becf0b9d67dcc3",
        "event": "open",
        "style": "cartoon",
        "query_revision": 1,
    }
    body = json.dumps(event).encode()
    # Pad an (unknown, hence 422) key until the body lands exactly on the envelope.
    opener = b',"pad":"'
    pad_len = cap - (len(body) - 1 + len(opener) + 2)  # strip "}", add opener + '"}'
    padded = body[:-1] + opener + b"p" * pad_len + b'"}'
    assert len(padded) == cap

    exact = client.post(
        "/api/v1/events", content=padded, headers={"content-type": "application/json"}
    )
    # 422 — the whole body got through the receive-layer cap and was parsed.
    assert exact.status_code == 422

    over = client.post(
        "/api/v1/events", content=padded + b" ", headers={"content-type": "application/json"}
    )
    assert over.status_code == 413
    parsed = over.json()
    assert parsed["error"]["code"] == "request_too_large"
    assert parsed["schema_version"] == 1 and parsed["retryable"] is False
    assert over.headers["x-request-id"] == parsed["request_id"]
    UUID(parsed["request_id"])


def test_exact_envelope_multipart_reaches_per_part_caps(client: TestClient) -> None:
    """At exactly the envelope the middleware defers to the tighter per-part caps."""
    boundary = b"precise-boundary-7f3a"
    settings: Settings = client.app.state.linescout.settings
    cap = settings.max_request_bytes

    parts = _search_parts(png_bytes(draw_figure))
    base = _multipart_with_files(boundary, parts + [("strokes", b"")])
    parts.append(("strokes", b"z" * (cap - len(base))))  # fill to exactly the envelope
    body = _multipart_with_files(boundary, parts)
    assert len(body) == cap

    exact = _post_raw(client, body, boundary)
    assert exact.status_code == 413
    assert exact.json()["error"]["code"] == "strokes_too_large"  # router cap, not the envelope

    over = _post_raw(client, body + b"x", boundary)
    assert over.status_code == 413
    assert over.json()["error"]["code"] == "request_too_large"  # middleware cap


def _stream(body: bytes, chunk_size: int = 65_537) -> Iterator[bytes]:
    for offset in range(0, len(body), chunk_size):
        yield body[offset : offset + chunk_size]


def test_streamed_multipart_over_envelope_is_counted_chunked(client: TestClient) -> None:
    """No Content-Length to trust: the counter must catch a streamed over-cap body."""
    boundary = b"chunked-boundary-9c1e"
    settings: Settings = client.app.state.linescout.settings
    parts = _search_parts(png_bytes(draw_figure))
    parts.append(("pad", b"p" * (settings.max_request_bytes + 1)))
    body = _multipart_with_files(boundary, parts)
    response = client.post(
        "/api/v1/search",
        content=_stream(body),
        headers={"content-type": f"multipart/form-data; boundary={boundary.decode()}"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_oversized_form_field_is_structured_400(client: TestClient, session_id: str) -> None:
    # One non-file field over the parser's 1 MiB part cap (but under the total
    # envelope): the parser's 400 must use the standard error envelope too.
    files = {"image": ("snapshot.png", png_bytes(draw_figure), "image/png")}
    response = client.post(
        "/api/v1/search",
        data=search_form(session_id, text_hint="x" * (1024 * 1024 + 1)),
        files=files,
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "http_400"
    assert "maximum size" in body["error"]["message"]
    assert body["schema_version"] == 1 and body["retryable"] is False
    assert response.headers["x-request-id"] == body["request_id"]
    UUID(body["request_id"])


def test_oversized_header_image_is_422_not_memory_exhaustion(
    client: TestClient, session_id: str
) -> None:
    # 100k x 100k declared pixels in a sub-100-byte upload: rejected from the
    # header inspection, long before ~40 GB of pixels could materialize.
    forged = forged_png(100_000, 100_000)
    status, body = post_search(client, session_id, forged, stroke_count=0, point_count=0)
    assert status == 422
    assert body["error"]["code"] == "image_dimensions"
    assert body["schema_version"] == 1


def test_truncated_gzip_strokes_are_now_400(client: TestClient, session_id: str) -> None:
    valid = gzip.compress(
        b'{"version": 1, "canvas_width": 2048, "canvas_height": 2048, "strokes": []}'
    )
    status, body = post_search(
        client, session_id, png_bytes(draw_figure), strokes=valid[:-6], stroke_count=0
    )
    assert status == 400
    assert body["error"]["code"] == "strokes_malformed"


def test_concatenated_gzip_strokes_are_400(client: TestClient, session_id: str) -> None:
    valid = gzip.compress(b"{}")
    status, body = post_search(
        client, session_id, png_bytes(draw_figure), strokes=valid + valid, stroke_count=0
    )
    assert status == 400
    assert body["error"]["code"] == "strokes_malformed"


def test_strokes_exactly_256kib_compressed_pass_the_size_gate(
    client: TestClient, session_id: str
) -> None:
    # Exactly 256 KiB on the wire (sized via the gzip FEXTRA field): the
    # compressed gate does not fire, decoding proceeds, and the random payload
    # — not being JSON — fails later as malformed, never as too_large.
    exact = gzip_of_exact_size(random.Random(11).randbytes(210_000), MAX_COMPRESSED_BYTES)
    assert len(exact) == 256 * 1024
    status, body = post_search(client, session_id, png_bytes(draw_figure), strokes=exact)
    assert status == 400 and body["error"]["code"] == "strokes_malformed"


def test_openapi_error_responses_still_documented(client: TestClient) -> None:
    responses = client.get("/api/v1/openapi.json").json()["paths"]["/api/v1/search"]["post"][
        "responses"
    ]
    assert {"400", "413", "422", "503"} <= set(responses)
