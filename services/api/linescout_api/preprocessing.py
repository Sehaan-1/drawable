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
MAX_COMPRESSED_BYTES = 256 * 1024  # 256 KiB — gzipped strokes, as uploaded
MAX_DECOMPRESSED_BYTES = 1024 * 1024  # 1 MiB — stroke JSON after decompression

# Identity of the snapshot-preprocessing pipeline (PNG decode, grayscale
# normalization, ink measurement, insufficiency rule). Bumped only when a
# change would make two calls with identical inputs produce meaningfully
# different stats; echoed on every search response and in /health so clients
# can tell whether two queries were prepared by the same code.
PREPROCESSING_VERSION = "1.0.0"


class SnapshotError(ValueError):
    def __init__(self, code: str, message: str, received_bytes: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        #: Payload size when the failure is size-related, for error ``details``.
        self.received_bytes = received_bytes


class GzipLimitError(ValueError):
    """A stroke payload crossed a hard size cap during decompression.

    Carries the counters so logs and tests can confirm the decompressor
    stopped *at* the limit instead of buffering the whole bomb first.
    """

    def __init__(self, message: str, *, input_bytes_used: int, output_bytes_produced: int) -> None:
        super().__init__(message)
        self.input_bytes_used = input_bytes_used
        self.output_bytes_produced = output_bytes_produced


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
    except Image.DecompressionBombError as error:
        # PIL's own header-stage guard fired: the declared raster is enormous.
        raise SnapshotError("image_dimensions", f"unsupported image size: {error}") from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise SnapshotError("image_malformed", f"image could not be decoded: {error}") from error
    # Inspect the container header BEFORE decoding a single pixel, so a
    # hostile raster (huge IHDR dimensions, decompression-bomb shapes) is
    # rejected without materializing megabytes/gigabytes of pixels.
    if image.format != "PNG":
        raise SnapshotError("image_format", f"image must be PNG, got {image.format}")
    if image.width < 16 or image.height < 16 or max(image.size) > MAX_SNAPSHOT_EDGE:
        raise SnapshotError("image_dimensions", f"unsupported image dimensions {image.size}")
    try:
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise SnapshotError("image_malformed", f"image could not be decoded: {error}") from error
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


def safe_decompress_gzip(
    data: bytes,
    *,
    max_compressed_bytes: int = MAX_COMPRESSED_BYTES,
    max_expanded_bytes: int = MAX_DECOMPRESSED_BYTES,
) -> bytes:
    """Gunzip a stroke payload under hard caps, with strict container checks.

    * ``zlib.decompressobj`` is called with a ``max_length`` of
      ``remaining + 1`` output bytes — the single overflow-detection byte —
      and re-fed its own ``unconsumed_tail``, so expansion stops at the cap
      instead of flushing a zip bomb into memory.
    * The gzip member must end exactly at the end of the input: a truncated
      stream (no EOF marker), trailing bytes, or a concatenated second member
      are all rejected; zlib itself rejects bad CRCs and malformed headers.
    """
    if len(data) > max_compressed_bytes:
        raise GzipLimitError(
            "compressed_payload_too_large", input_bytes_used=0, output_bytes_produced=0
        )

    # 16 + zlib.MAX_WBITS tells zlib to decode standard gzip headers and
    # verify the CRC32/ISIZE trailer itself.
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    output = bytearray()
    pending: bytes = data
    while pending:
        # Remaining allowance plus exactly one overflow-detection byte. A
        # single zlib call can therefore never produce more than the cap + 1.
        allowance = max_expanded_bytes + 1 - len(output)
        output += decompressor.decompress(pending, allowance)
        if len(output) > max_expanded_bytes:
            raise GzipLimitError(
                "decompressed_payload_too_large",
                input_bytes_used=len(data) - len(decompressor.unconsumed_tail),
                output_bytes_produced=len(output),
            )
        # Non-empty only when the allowance bound stopped consumption; loop
        # with a fresh allowance instead of an unbounded flush().
        pending = decompressor.unconsumed_tail

    if not decompressor.eof:
        raise ValueError("gzip stream is truncated (missing end-of-stream marker)")
    if decompressor.unused_data:
        # Everything past the member's end: trailing garbage or a second
        # concatenated member — both rejected outright.
        raise ValueError("trailing data after the gzip member is not allowed")
    return bytes(output)


def decode_strokes(
    data: bytes | None,
    max_bytes: int,
    *,
    max_expanded_bytes: int = MAX_DECOMPRESSED_BYTES,
) -> StrokeSequence | None:
    """Decode the optional gzip-compressed JSON stroke sequence."""
    if data is None or len(data) == 0:
        return None
    if len(data) > max_bytes:
        raise SnapshotError("strokes_too_large", f"strokes exceed {max_bytes} bytes compressed")
    try:
        raw = safe_decompress_gzip(
            data, max_compressed_bytes=max_bytes, max_expanded_bytes=max_expanded_bytes
        )
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
