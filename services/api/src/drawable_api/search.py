from __future__ import annotations

import json
import math
import re
import time
import unicodedata
import zlib
from dataclasses import dataclass
from email import policy
from email.message import Message
from email.parser import BytesParser
from io import BytesIO
from typing import Any, Literal
from uuid import UUID

from fastapi import Request
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

MAX_BODY_BYTES = 5 * 1024 * 1024
MAX_MULTIPART_PARTS = 16
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_STROKES_COMPRESSED_BYTES = 256 * 1024
MAX_STROKES_DECOMPRESSED_BYTES = 1024 * 1024
MAX_SAFE_INTEGER = 9_007_199_254_740_991
QUERY_PREPROCESSING_VERSION = "drawable-query-v1"
MAX_GENERATION_SESSIONS = 1024
GENERATION_SESSION_TTL_SECONDS = 30 * 60

StrokeStatus = Literal[
    "available",
    "too_short",
    "erased_layers",
    "raster_content",
    "unsupported_transform",
    "out_of_bounds",
    "too_many_boundaries",
    "disabled",
]


class StrictSearchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchFieldsV1(StrictSearchModel):
    session_id: UUID
    document_id: UUID
    canvas_revision: int = Field(ge=0, le=MAX_SAFE_INTEGER)
    query_generation: int = Field(ge=0, le=MAX_SAFE_INTEGER)
    canvas_width: Literal[2048]
    canvas_height: Literal[2048]
    stroke_count: int = Field(ge=0, le=10_000)
    point_count: int = Field(ge=0, le=1_000_000)
    counts_approximate: bool
    stroke_status: StrokeStatus
    preprocessing_version: str = Field(min_length=1, max_length=100)
    text_hint: str = Field(default="", max_length=120)

    @model_validator(mode="after")
    def normalize_hint(self) -> SearchFieldsV1:
        normalized = unicodedata.normalize("NFC", " ".join(self.text_hint.split()))
        if len(normalized) > 120:
            raise ValueError("text_hint exceeds 120 Unicode code points after normalization")
        self.text_hint = normalized
        if self.stroke_status == "available" and (self.stroke_count == 0 or self.point_count == 0):
            raise ValueError("available stroke status requires nonzero source stroke and point counts")
        return self


class StrokeTokenV1(StrictSearchModel):
    x: float = Field(ge=0, le=1, allow_inf_nan=False)
    y: float = Field(ge=0, le=1, allow_inf_nan=False)
    dx: float = Field(ge=-1, le=1, allow_inf_nan=False)
    dy: float = Field(ge=-1, le=1, allow_inf_nan=False)
    pressure: float = Field(ge=0, le=1, allow_inf_nan=False)
    pressure_available: bool
    dt: float = Field(ge=0, le=1, allow_inf_nan=False)
    time_available: bool
    opacity: float = Field(ge=0, le=1, allow_inf_nan=False)
    marker: Literal["start", "continue", "end", "tap"]


class StrokeSequenceV1(StrictSearchModel):
    schema_version: Literal[1]
    token_count: int = Field(ge=20, le=1024)
    tokens: list[StrokeTokenV1] = Field(min_length=20, max_length=1024)

    @model_validator(mode="after")
    def token_stream_is_coherent(self) -> StrokeSequenceV1:
        if self.token_count != len(self.tokens):
            raise ValueError("token_count must match the token list")
        active = False
        previous: StrokeTokenV1 | None = None
        for token in self.tokens:
            if not token.pressure_available and token.pressure != 0.5:
                raise ValueError("Missing pressure must use 0.5 with pressure_available=false")
            if not token.time_available and token.dt != 0:
                raise ValueError("Missing time must use zero with time_available=false")
            if token.marker == "tap":
                if active or token.dx != 0 or token.dy != 0 or token.dt != 0:
                    raise ValueError("A tap must be a standalone zero-delta token")
                previous = None
                continue
            if token.marker == "start":
                if active or token.dx != 0 or token.dy != 0 or token.dt != 0:
                    raise ValueError("Stroke starts must reset deltas and time")
                active = True
            else:
                if not active or previous is None:
                    raise ValueError("Continuation/end token must follow a stroke start")
                if abs(token.dx - (token.x - previous.x)) > 1e-5 or abs(token.dy - (token.y - previous.y)) > 1e-5:
                    raise ValueError("Token deltas must match successive normalized positions")
                if token.marker == "end":
                    active = False
            previous = token
        if active:
            raise ValueError("The final stroke must have an end marker")
        return self


@dataclass(frozen=True, slots=True)
class ParsedSearchRequest:
    fields: SearchFieldsV1
    image_png: bytes
    strokes: StrokeSequenceV1 | None
    insufficient: bool


class SearchInputError(ValueError):
    def __init__(self, status_code: int, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retryable = retryable


def _error(status: int, code: str, message: str) -> SearchInputError:
    return SearchInputError(status, code, message)


async def parse_search_request(request: Request) -> ParsedSearchRequest:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        if not content_length.isascii() or not content_length.isdecimal():
            raise _error(400, "malformed_multipart", "Content-Length is invalid.")
        significant_length = content_length.lstrip("0") or "0"
        if len(significant_length) > len(str(MAX_BODY_BYTES)):
            raise _error(413, "search_body_too_large", "Search request exceeds the 5 MiB body limit.")
        declared_length = int(significant_length)
        if declared_length > MAX_BODY_BYTES:
            raise _error(413, "search_body_too_large", "Search request exceeds the 5 MiB body limit.")

    content_type = request.headers.get("content-type", "")
    if not content_type.casefold().startswith("multipart/form-data;"):
        raise _error(400, "malformed_multipart", "Search requires multipart/form-data.")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise _error(413, "search_body_too_large", "Search request exceeds the 5 MiB body limit.")
        chunks.append(chunk)
    raw_body = b"".join(chunks)
    try:
        content_header = Message(policy=policy.default)
        content_header["Content-Type"] = content_type
        boundary = content_header.get_boundary()
    except (TypeError, ValueError) as error:
        raise _error(400, "malformed_multipart", "Multipart Content-Type is malformed.") from error
    if (
        not boundary
        or len(boundary) > 70
        or not boundary.isascii()
        or any(char < "!" or char > "~" for char in boundary)
    ):
        raise _error(400, "malformed_multipart", "Multipart boundary is missing or invalid.")
    boundary_token = re.escape(boundary.encode("ascii"))
    delimiter_pattern = re.compile(
        rb"(?:\A|\r\n)--" + boundary_token + rb"(?P<closing>--)?(?:\r\n|\Z)"
    )
    delimiters = list(delimiter_pattern.finditer(raw_body))
    if (
        len(delimiters) < 2
        or len(delimiters) - 1 > MAX_MULTIPART_PARTS
        or delimiters[0].start() != 0
        or delimiters[-1].end() != len(raw_body)
        or delimiters[-1].group("closing") is None
        or any(match.group("closing") is not None for match in delimiters[:-1])
    ):
        raise _error(400, "malformed_multipart", "Multipart boundaries or part count are invalid.")
    try:
        envelope = (
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
            + raw_body
        )
        message = BytesParser(policy=policy.default).parsebytes(envelope)
    except (UnicodeEncodeError, ValueError) as error:
        raise _error(400, "malformed_multipart", "Multipart headers are malformed.") from error
    if not message.is_multipart() or message.defects:
        raise _error(400, "malformed_multipart", "Multipart body is malformed.")
    if message.get_boundary() != boundary:
        raise _error(400, "malformed_multipart", "Multipart boundary is invalid.")

    values: dict[str, bytes] = {}
    content_types: dict[str, str] = {}
    filenames: dict[str, str | None] = {}
    scalar_names = {
        "session_id", "document_id", "canvas_revision", "query_generation", "canvas_width",
        "canvas_height", "stroke_count", "point_count", "counts_approximate", "stroke_status",
        "preprocessing_version", "text_hint",
    }
    allowed_names = scalar_names | {"image", "strokes"}
    parts = list(message.iter_parts())
    if len(parts) > MAX_MULTIPART_PARTS:
        raise _error(400, "too_many_multipart_parts", "Search supports at most 16 multipart fields.")
    for part in parts:
        if part.defects:
            raise _error(400, "malformed_multipart", "A multipart field is malformed.")
        disposition = part.get("Content-Disposition")
        if not disposition or part.get_content_disposition() != "form-data":
            raise _error(400, "malformed_multipart", "Each multipart field must use form-data disposition.")
        name = part.get_param("name", header="content-disposition")
        if not isinstance(name, str) or not name or name in values:
            raise _error(400, "malformed_multipart", "Multipart field names must be present and unique.")
        if name not in allowed_names:
            raise _error(422, "unknown_search_field", "Search includes an unsupported field.")
        if name in scalar_names and (part.get_filename() is not None or part.get_content_type().casefold() != "text/plain"):
            raise _error(422, "invalid_search_schema", f"{name} must be a text field.")
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            raise _error(400, "malformed_multipart", "Multipart field payload is invalid.")
        values[name] = payload
        content_types[name] = part.get_content_type().casefold()
        filenames[name] = part.get_filename()

    required = scalar_names - {"text_hint"}
    if not required.issubset(values) or "image" not in values:
        raise _error(422, "missing_search_field", "Required search fields or the PNG image are missing.")
    if not set(values).issubset(scalar_names | {"image", "strokes"}):
        raise _error(422, "unknown_search_field", "Search includes unsupported fields.")
    if filenames.get("image") is None or content_types.get("image") != "image/png":
        raise _error(422, "invalid_search_image", "image must be a file part with image/png content type.")
    image_bytes = values["image"]
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise _error(413, "search_image_too_large", "PNG image must be no larger than 4 MiB.")
    if not image_bytes:
        raise _error(400, "invalid_search_image", "PNG image must not be empty.")
    if "strokes" in values and (
        filenames.get("strokes") is None
        or content_types.get("strokes") not in {"application/gzip", "application/octet-stream"}
    ):
        raise _error(422, "invalid_stroke_file", "strokes must be a gzip-compressed JSON file.")
    if "strokes" in values and len(values["strokes"]) > MAX_STROKES_COMPRESSED_BYTES:
        raise _error(413, "compressed_strokes_too_large", "Compressed strokes exceed 256 KiB.")

    scalar_fields: dict[str, Any] = {}
    for name in scalar_names:
        if name not in values:
            continue
        try:
            scalar_fields[name] = values[name].decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise _error(400, "invalid_search_text", f"{name} must be valid UTF-8.") from error
    if scalar_fields.get("counts_approximate") in {"true", "false"}:
        scalar_fields["counts_approximate"] = scalar_fields["counts_approximate"] == "true"
    else:
        raise _error(422, "invalid_search_schema", "counts_approximate must be true or false.")
    for name in (
        "canvas_revision",
        "query_generation",
        "canvas_width",
        "canvas_height",
        "stroke_count",
        "point_count",
    ):
        value = scalar_fields.get(name)
        if (
            not isinstance(value, str)
            or not value.isascii()
            or not value.isdecimal()
            or len(value) > len(str(MAX_SAFE_INTEGER))
        ):
            raise _error(422, "invalid_search_schema", f"{name} must be an unsigned integer.")
        scalar_fields[name] = int(value)
    try:
        scalar_fields["session_id"] = UUID(scalar_fields["session_id"])
        scalar_fields["document_id"] = UUID(scalar_fields["document_id"])
    except (KeyError, ValueError, TypeError) as error:
        raise _error(422, "invalid_search_schema", "session_id and document_id must be UUIDs.") from error
    try:
        fields = SearchFieldsV1.model_validate(scalar_fields)
    except ValidationError as error:
        first = error.errors()[0]
        field_name = str(first["loc"][0]) if first.get("loc") else "search"
        raise _error(422, "invalid_search_schema", f"Search field validation failed: {field_name}.") from error
    if fields.preprocessing_version != QUERY_PREPROCESSING_VERSION:
        raise _error(
            409,
            "preprocessing_version_mismatch",
            "Reload the drawing client to use the active query recipe.",
        )

    if fields.stroke_status == "available" and "strokes" not in values:
        raise _error(422, "invalid_stroke_status", "available stroke status requires a strokes file.")
    if fields.stroke_status != "available" and "strokes" in values:
        raise _error(422, "invalid_stroke_status", "Unavailable stroke status requires the strokes field to be omitted.")
    strokes = _decode_strokes(values["strokes"]) if "strokes" in values else None
    image_sufficient = _image_is_sufficient(image_bytes)
    return ParsedSearchRequest(fields, image_bytes, strokes, not image_sufficient)


def _decode_strokes(compressed: bytes) -> StrokeSequenceV1:
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        decoded = inflater.decompress(compressed, MAX_STROKES_DECOMPRESSED_BYTES + 1)
        if len(decoded) > MAX_STROKES_DECOMPRESSED_BYTES or inflater.unconsumed_tail:
            raise _error(413, "decompressed_strokes_too_large", "Strokes exceed 1 MiB decompressed.")
        decoded += inflater.flush(MAX_STROKES_DECOMPRESSED_BYTES + 1 - len(decoded))
    except SearchInputError:
        raise
    except zlib.error as error:
        raise _error(400, "invalid_gzip", "Strokes are not a valid gzip stream.") from error
    if len(decoded) > MAX_STROKES_DECOMPRESSED_BYTES:
        raise _error(413, "decompressed_strokes_too_large", "Strokes exceed 1 MiB decompressed.")
    if not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
        raise _error(400, "invalid_gzip", "Strokes must be one complete gzip member with no trailing data.")
    try:
        raw = json.loads(decoded.decode("utf-8", errors="strict"), parse_constant=_reject_json_constant)
        strokes = StrokeSequenceV1.model_validate(raw)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
        RecursionError,
    ) as error:
        raise _error(422, "invalid_stroke_schema", "Stroke JSON does not match the strict token schema.") from error
    return strokes


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Unsupported JSON numeric constant: {value}")


def _image_is_sufficient(raw: bytes) -> bool:
    try:
        with Image.open(BytesIO(raw)) as source:
            if source.format != "PNG" or source.size != (512, 512) or source.mode not in {"L", "LA", "RGB", "RGBA"}:
                raise _error(422, "invalid_search_image", "image must be a supported 512 × 512 PNG.")
            source.load()
            gray: Image.Image
            if source.mode == "L":
                gray = source
            else:
                rgba = source.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                gray = Image.alpha_composite(white, rgba).convert("L")
            ink = gray.point(lambda value: 255 if value < 245 else 0)
            count = ink.histogram()[255]
            bounds = ink.getbbox()
    except SearchInputError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise _error(400, "invalid_search_image", "image is not a decodable PNG.") from error
    if count < 8 or bounds is None:
        return False
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return math.hypot(width, height) >= 3


def accept_generation(generations: dict[str, tuple[int, float]], session_id: UUID, generation: int) -> None:
    now = time.monotonic()
    expired = [key for key, (_, updated) in generations.items() if now - updated > GENERATION_SESSION_TTL_SECONDS]
    for key in expired:
        generations.pop(key, None)
    key = str(session_id)
    previous = generations.get(key)
    if previous is not None and generation <= previous[0]:
        raise _error(409, "stale_generation", "query_generation must be greater than the latest accepted generation.")
    if previous is None and len(generations) >= MAX_GENERATION_SESSIONS:
        raise SearchInputError(429, "search_capacity", "Too many active search sessions.", retryable=True)
    generations[key] = (generation, now)
