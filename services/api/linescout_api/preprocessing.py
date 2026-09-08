"""Snapshot decoding, normalization, and the insufficient-input rule.

This is the part of the search pipeline that exists before any model does:
validate the PNG, normalize it to a white-background grayscale view, measure
the ink bounding box, and decide whether there is enough to search on.
"""

from __future__ import annotations

import io
import json
import math
import zlib
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError

from linescout_api.schemas import StrokeSequence

INK_THRESHOLD = 200  # grayscale values below this count as ink
MAX_SNAPSHOT_EDGE = 4096
MAX_COMPRESSED_BYTES = 256 * 1024  # 256 KiB
MAX_DECOMPRESSED_BYTES = 1024 * 1024  # 1 MiB


class SnapshotError(ValueError):
    def __init__(self, code: str, message: str, received_bytes: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        #: Payload size when the failure is size-related, for error ``details``.
        self.received_bytes = received_bytes


@dataclass(frozen=True)
class InkStats:
    width: int
    height: int
    ink_pixels: int
    bbox: tuple[int, int, int, int] | None  # left, top, right, bottom (exclusive)

    @property
    def coverage(self) -> float:
        return self.ink_pixels / (self.width * self.height)

    @property
    def bbox_diagonal_ratio(self) -> float:
        """Ink bounding-box diagonal as a fraction of the image diagonal."""
        if self.bbox is None:
            return 0.0
        left, top, right, bottom = self.bbox
        return math.hypot(right - left, bottom - top) / math.hypot(self.width, self.height)


def decode_snapshot(data: bytes, max_bytes: int) -> Image.Image:
    """Decode a PNG snapshot into a white-background 8-bit grayscale image."""
    if len(data) > max_bytes:
        raise SnapshotError(
            "image_too_large", f"image exceeds {max_bytes} bytes", received_bytes=len(data)
        )
    if not data:
        raise SnapshotError("image_missing", "image field is empty")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise SnapshotError("image_malformed", f"image could not be decoded: {error}") from error
    if image.format != "PNG":
        raise SnapshotError("image_format", f"image must be PNG, got {image.format}")
    if image.width < 16 or image.height < 16 or max(image.size) > MAX_SNAPSHOT_EDGE:
        raise SnapshotError("image_dimensions", f"unsupported image dimensions {image.size}")
    return normalize_to_gray(image)


def normalize_to_gray(image: Image.Image) -> Image.Image:
    """Flatten transparency onto white and return mode ``L``. Never invert or edge-detect."""
    image = ImageOps.exif_transpose(image) or image
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("L")
    return image.convert("L")


def ink_stats(gray: Image.Image) -> InkStats:
    mask = gray.point(lambda value: 255 if value < INK_THRESHOLD else 0, mode="L")
    bbox = mask.getbbox()
    histogram = mask.histogram()
    return InkStats(gray.width, gray.height, ink_pixels=histogram[255], bbox=bbox)


def is_insufficient(
    stats: InkStats, point_count: int, min_points: int, min_diagonal_ratio: float
) -> bool:
    """Return True when the query has too little ink (or too few vector points).

    Imported raster drawings may have ``point_count == 0``; those are judged by
    ink coverage and bounding-box diagonal instead of vector sampling.
    """
    too_little_ink = stats.coverage <= 0.0 or stats.bbox_diagonal_ratio < min_diagonal_ratio
    if point_count == 0:
        return too_little_ink
    return point_count < min_points or stats.bbox_diagonal_ratio < min_diagonal_ratio


def tight_crop(gray: Image.Image, stats: InkStats, padding: float = 0.10) -> Image.Image:
    """Ink bounding box with 10% padding, letterboxed to a square on white."""
    if stats.bbox is None:
        return gray
    left, top, right, bottom = stats.bbox
    pad = int(round(max(right - left, bottom - top) * padding))
    box = (
        max(0, left - pad),
        max(0, top - pad),
        min(gray.width, right + pad),
        min(gray.height, bottom + pad),
    )
    crop = gray.crop(box)
    side = max(crop.size)
    square = Image.new("L", (side, side), 255)
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square


def safe_decompress_gzip(data: bytes) -> bytes:
    """Stream gzip decompression with hard caps (zip-bomb proof)."""
    if len(data) > MAX_COMPRESSED_BYTES:
        raise ValueError("compressed_payload_too_large")

    # 16 + zlib.MAX_WBITS tells zlib to decode standard gzip headers.
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    uncompressed = bytearray()
    chunk_size = 16 * 1024

    for offset in range(0, len(data), chunk_size):
        chunk = data[offset : offset + chunk_size]
        decompressed_chunk = decompressor.decompress(chunk)
        uncompressed.extend(decompressed_chunk)
        if len(uncompressed) > MAX_DECOMPRESSED_BYTES:
            raise ValueError("decompressed_payload_too_large")

    uncompressed.extend(decompressor.flush())
    if len(uncompressed) > MAX_DECOMPRESSED_BYTES:
        raise ValueError("decompressed_payload_too_large")

    if decompressor.unused_data:
        raise ValueError("trailing_garbage_rejected")

    return bytes(uncompressed)


def decode_strokes(data: bytes | None, max_bytes: int) -> StrokeSequence | None:
    """Decode the optional gzip-compressed JSON stroke sequence."""
    if data is None or len(data) == 0:
        return None
    if len(data) > max_bytes:
        raise SnapshotError("strokes_too_large", f"strokes exceed {max_bytes} bytes compressed")
    try:
        raw = safe_decompress_gzip(data)
    except ValueError as error:
        reason = str(error)
        if "too_large" in reason:
            raise SnapshotError(
                "strokes_too_large", "decompressed strokes exceed the size limit"
            ) from error
        raise SnapshotError("strokes_malformed", f"strokes are not valid gzip: {error}") from error
    except zlib.error as error:
        raise SnapshotError("strokes_malformed", f"strokes are not valid gzip: {error}") from error
    try:
        return StrokeSequence.model_validate(json.loads(raw))
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as error:
        raise SnapshotError("strokes_malformed", f"strokes JSON is invalid: {error}") from error
