"""Pixel measurements: ink, text, quality, pHash, near-duplicate grouping.

Every number the manifest requires but a human should not have to type is
computed here, on a downscaled *analysis view* so cost stays flat regardless of
source resolution. The definitions deliberately mirror the API's own
preprocessing (``services/api/linescout_api/preprocessing.py``): ink is anything
darker than 200 on a white-flattened grayscale image, so an asset's
``ink_coverage`` in the manifest means the same thing as the query-side ink
ratio that drives the "insufficient input" rule.

Two of these are honest heuristics rather than measurements, and are labelled
as such wherever they surface:

* ``text_coverage`` — the share of the frame taken up by glyph-like runs
  (speech bubbles, captions, watermarks, signatures). Connected components are
  grouped into text lines by height and spacing; there is no OCR.
* ``quality_score`` — a fixed weighted blend of resolution, ink density, paper
  cleanliness, and text contamination. It exists to *order* the curation queue,
  not to judge art, and curation overrides it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from linescout_ml.manifest import CropBox

#: Grayscale values below this count as ink (matches the API's threshold).
INK_THRESHOLD = 200
#: Grayscale values at or above this count as clean paper.
BACKGROUND_THRESHOLD = 245

# --- text heuristic bounds, in analysis-view pixels -------------------------
TEXT_MIN_HEIGHT = 5
TEXT_MAX_HEIGHT = 40
TEXT_MIN_ASPECT = 0.5
TEXT_MAX_ASPECT = 10.0
TEXT_MIN_FILL = 0.12
TEXT_MIN_GLYPHS_PER_LINE = 3

# --- quality_score weights (sum to 1.0) -------------------------------------
QUALITY_WEIGHT_RESOLUTION = 0.25
QUALITY_WEIGHT_INK = 0.35
QUALITY_WEIGHT_PAPER = 0.20
QUALITY_WEIGHT_TEXT = 0.20
QUALITY_RESOLUTION_FLOOR = 256.0
QUALITY_RESOLUTION_TARGET = 1024.0
QUALITY_INK_MIN = 0.02
QUALITY_INK_MAX = 0.60
QUALITY_PAPER_MIN = 0.30
QUALITY_PAPER_TARGET = 0.80
QUALITY_TEXT_TOLERANCE = 0.15

#: ``(x, y, width, height, area)`` of one connected component, in analysis-view pixels.
ComponentBox = tuple[int, int, int, int, int]


class ImageReadError(RuntimeError):
    """A source file could not be decoded as an image."""


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def to_rgb(image: Image.Image) -> Image.Image:
    """Flatten transparency onto white, correct orientation, return mode ``RGB``.

    Same contract as the API's ``normalize_to_gray``: never invert, never
    edge-detect, never touch orientation beyond the EXIF transpose. Grayscale is
    derived from this so an asset's original, line art, and thumbnail all share
    one geometry.
    """
    transposed = ImageOps.exif_transpose(image)
    if transposed is not None:
        image = transposed
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def to_gray(image: Image.Image) -> Image.Image:
    """White-background grayscale view of an image (``to_rgb`` then ``L``)."""
    return to_rgb(image).convert("L")


def _decode(path: Path, mode: str) -> Image.Image:
    try:
        with Image.open(path) as handle:
            handle.load()
            return to_rgb(handle).convert(mode)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        msg = f"could not decode image {path}: {error}"
        raise ImageReadError(msg) from error


def load_gray(path: Path) -> Image.Image:
    """Decode ``path`` into a white-background grayscale image."""
    return _decode(path, "L")


def load_rgb(path: Path) -> Image.Image:
    """Decode ``path`` into a white-background RGB image."""
    return _decode(path, "RGB")


def analysis_view(gray: Image.Image, edge: int) -> np.ndarray:
    """Downscale so the *long* edge is ``edge`` px; never upscale."""
    width, height = gray.size
    longest = max(width, height)
    if longest > edge:
        scale = edge / longest
        gray = gray.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(gray, dtype=np.uint8)


def ink_coverage(arr: np.ndarray) -> float:
    """Fraction of pixels darker than :data:`INK_THRESHOLD`."""
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(arr < INK_THRESHOLD)) / float(arr.size)


def background_coverage(arr: np.ndarray) -> float:
    """Fraction of clean paper — low values mean noise, scans, or JPEG mush."""
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(arr >= BACKGROUND_THRESHOLD)) / float(arr.size)


def ink_bbox(arr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Ink bounding box ``(x, y, width, height)`` in array coordinates."""
    mask = arr < INK_THRESHOLD
    if not mask.any():
        return None
    rows = np.nonzero(mask.any(axis=1))[0]
    columns = np.nonzero(mask.any(axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    return left, top, right - left, bottom - top


def suggest_crop(
    bbox: tuple[int, int, int, int] | None,
    analysis_size: tuple[int, int],
    original_size: tuple[int, int],
    padding: float = 0.10,
) -> CropBox | None:
    """Scale an analysis-view ink box back to source pixels as a crop.

    Returns ``None`` when the ink already fills the frame (a crop would add
    noise to the manifest) or when there is no ink at all.
    """
    if bbox is None:
        return None
    analysis_width, analysis_height = analysis_size
    original_width, original_height = original_size
    if analysis_width <= 0 or analysis_height <= 0:
        return None

    x, y, width, height = bbox
    pad = int(round(max(width, height) * padding))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(analysis_width, x + width + pad)
    y1 = min(analysis_height, y + height + pad)
    if (x1 - x0) / analysis_width > 0.95 and (y1 - y0) / analysis_height > 0.95:
        return None

    scale_x = original_width / analysis_width
    scale_y = original_height / analysis_height
    crop_x = min(original_width - 1, int(round(x0 * scale_x)))
    crop_y = min(original_height - 1, int(round(y0 * scale_y)))
    crop_width = max(1, min(original_width - crop_x, int(round((x1 - x0) * scale_x))))
    crop_height = max(1, min(original_height - crop_y, int(round((y1 - y0) * scale_y))))
    if crop_width < 2 or crop_height < 2:
        return None
    return CropBox(x=crop_x, y=crop_y, width=crop_width, height=crop_height)


def _components(mask: np.ndarray) -> list[ComponentBox]:
    """8-connected components of ``mask`` as ``(x, y, width, height, area)``.

    Union-find over foreground pixels only, so cost tracks ink density rather
    than frame size. Runs in a few milliseconds at the 512 px analysis edge.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []

    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    def union(left: tuple[int, int], right: tuple[int, int]) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pixels = set(zip(ys.tolist(), xs.tolist(), strict=True))
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        node = (y, x)
        parent.setdefault(node, node)
        for neighbour in ((y - 1, x - 1), (y - 1, x), (y - 1, x + 1), (y, x - 1)):
            if neighbour in pixels:
                parent.setdefault(neighbour, neighbour)
                union(node, neighbour)

    boxes: dict[tuple[int, int], list[int]] = {}
    for y, x in pixels:
        root = find((y, x))
        box = boxes.setdefault(root, [x, y, x, y, 0])
        box[0] = min(box[0], x)
        box[1] = min(box[1], y)
        box[2] = max(box[2], x)
        box[3] = max(box[3], y)
        box[4] += 1
    return [
        (box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1, box[4]) for box in boxes.values()
    ]


def _glyph_candidates(components: Sequence[ComponentBox]) -> list[ComponentBox]:
    """Components small and dense enough to be a character."""
    glyphs = []
    for x, y, width, height, area in components:
        if not TEXT_MIN_HEIGHT <= height <= TEXT_MAX_HEIGHT:
            continue
        aspect = width / height
        if not TEXT_MIN_ASPECT <= aspect <= TEXT_MAX_ASPECT:
            continue
        if area / float(width * height) < TEXT_MIN_FILL:
            continue
        glyphs.append((x, y, width, height, area))
    return glyphs


def text_coverage(arr: np.ndarray) -> float:
    """Fraction of the frame covered by text-like glyph runs.

    Glyph candidates are clustered into lines by vertical centre and horizontal
    gap; a run of at least :data:`TEXT_MIN_GLYPHS_PER_LINE` similar-height
    glyphs counts as text. A single isolated blob never does, which keeps this
    from flagging eyes, hands, or hatching.
    """
    components = _components(arr < INK_THRESHOLD)
    glyphs = _glyph_candidates(components)
    if len(glyphs) < TEXT_MIN_GLYPHS_PER_LINE:
        return 0.0

    heights = np.array([glyph[3] for glyph in glyphs], dtype=np.float64)
    median_height = float(np.median(heights))
    glyphs.sort(key=lambda glyph: (glyph[1] + glyph[3] / 2.0, glyph[0]))

    lines: list[list[ComponentBox]] = []
    for glyph in glyphs:
        centre_y = glyph[1] + glyph[3] / 2.0
        placed = False
        for line in reversed(lines):
            last = line[-1]
            line_y = last[1] + last[3] / 2.0
            if abs(centre_y - line_y) > 0.75 * median_height:
                continue
            if glyph[0] - (last[0] + last[2]) > 1.5 * median_height:
                continue
            line.append(glyph)
            placed = True
            break
        if not placed:
            lines.append([glyph])

    text_pixels = 0
    for line in lines:
        if len(line) < TEXT_MIN_GLYPHS_PER_LINE:
            continue
        x0 = min(glyph[0] for glyph in line)
        y0 = min(glyph[1] for glyph in line)
        x1 = max(glyph[0] + glyph[2] for glyph in line)
        y1 = max(glyph[1] + glyph[3] for glyph in line)
        text_pixels += (x1 - x0) * (y1 - y0)
    if arr.size == 0:
        return 0.0
    return _clamp(text_pixels / float(arr.size))


def quality_score_from(
    *,
    ink: float,
    text: float,
    paper: float,
    short_edge: int,
) -> float:
    """Blend the measurements into a 0–1 ordering hint for the curation queue.

    * resolution: 0 at 256 px, 1 at 1024 px on the short edge
    * ink: ramps up to :data:`QUALITY_INK_MIN`, down to zero past
      :data:`QUALITY_INK_MAX` (an all-black frame is a failed extraction)
    * paper: rewards a clean white background
    * text: penalises captions and watermarks
    """
    resolution = _clamp(
        (short_edge - QUALITY_RESOLUTION_FLOOR)
        / (QUALITY_RESOLUTION_TARGET - QUALITY_RESOLUTION_FLOOR)
    )
    ink_term = _clamp(min(ink / QUALITY_INK_MIN, (QUALITY_INK_MAX - ink) / QUALITY_INK_MAX))
    paper_term = _clamp((paper - QUALITY_PAPER_MIN) / (QUALITY_PAPER_TARGET - QUALITY_PAPER_MIN))
    text_term = 1.0 - _clamp(text / QUALITY_TEXT_TOLERANCE)
    return round(
        _clamp(
            QUALITY_WEIGHT_RESOLUTION * resolution
            + QUALITY_WEIGHT_INK * ink_term
            + QUALITY_WEIGHT_PAPER * paper_term
            + QUALITY_WEIGHT_TEXT * text_term
        ),
        4,
    )


def _dct_matrix(size: int) -> np.ndarray:
    """Unnormalised DCT-II matrix, matching ``scipy.fftpack.dct(x, axis=...)``.

    ``norm=None`` (scipy's default) is what ``imagehash.phash`` uses, and it is
    *not* interchangeable with ``norm='ortho'``: the two differ by a per-row
    scale factor, which changes which coefficients clear the median threshold.
    """
    indices = np.arange(size, dtype=np.float64)
    frequencies = indices.reshape(-1, 1)
    return 2.0 * np.cos(np.pi * (2.0 * indices + 1.0) * frequencies / (2.0 * size))


def phash(gray: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """64-bit DCT perceptual hash as 16 hex characters.

    Bit-for-bit compatible with ``imagehash.phash``: same 32×32 LANCZOS
    reduction, same unnormalised 2-D DCT-II, same median threshold over the
    top-left ``hash_size``² block. scipy is not required, so ``ml`` stays light.
    """
    if hash_size < 2:
        msg = "hash_size must be at least 2"
        raise ValueError(msg)
    side = hash_size * highfreq_factor
    small = gray.convert("L").resize((side, side), Image.Resampling.LANCZOS)
    array = np.asarray(small, dtype=np.float64)
    basis = _dct_matrix(side)
    coefficients = basis @ array @ basis.T
    low = coefficients[:hash_size, :hash_size]
    threshold = float(np.median(low))
    bits = (low > threshold).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming(left: str, right: str) -> int:
    """Hamming distance between two equal-length hex hashes."""
    if len(left) != len(right):
        msg = f"hash length mismatch: {left!r} vs {right!r}"
        raise ValueError(msg)
    return int.bit_count(int(left, 16) ^ int(right, 16))


def _popcount64(values: np.ndarray) -> np.ndarray:
    """Bit population count, using ``numpy.bitwise_count`` when available."""
    counter = getattr(np, "bitwise_count", None)
    if counter is not None:
        counted: np.ndarray = counter(values).astype(np.int32)
        return counted
    table = np.array([int.bit_count(byte) for byte in range(256)], dtype=np.int32)
    octets = np.ascontiguousarray(values).view(np.uint8).reshape(*values.shape, 8)
    lookup: np.ndarray = table[octets].sum(axis=-1, dtype=np.int32)
    return lookup


def duplicate_groups(hashes: Sequence[str], threshold: int) -> list[list[int]]:
    """Indices of near-duplicate candidates, grouped, in ascending order.

    Chunked pairwise Hamming distance over packed 64-bit hashes plus union-find:
    quadratic in candidate count but vectorised, so 10k candidates take seconds
    rather than the minutes a pure-Python loop would need. Each group is sorted
    so the caller can keep ``group[0]`` (the lowest sort key) deterministically.
    """
    if not hashes or threshold < 0:
        return []
    packed = np.array([int(value, 16) for value in hashes], dtype=np.uint64)
    count = packed.shape[0]

    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    chunk = max(1, min(count, 512))
    for start in range(0, count, chunk):
        left = packed[start : start + chunk]
        distances = _popcount64(left[:, None] ^ packed[None, :])
        rows, columns = np.nonzero(distances <= threshold)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            first, second = find(start + int(row)), find(int(column))
            if first != second:
                parent[max(first, second)] = min(first, second)

    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(find(index), []).append(index)
    return sorted(
        (sorted(members) for members in grouped.values() if len(members) > 1),
        key=lambda group: group[0],
    )


def make_thumbnail(image: Image.Image, size: int) -> Image.Image:
    """Aspect-fit the extracted artifact into a ``size``×``size`` white square."""
    width, height = image.size
    scale = size / max(width, height)
    fitted = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    tile = Image.new("L", (size, size), 255)
    tile.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return tile


@dataclass(frozen=True)
class MeasurementResult:
    """Everything the manifest needs from one image, plus its geometry."""

    width: int
    height: int
    ink_coverage: float
    text_coverage: float
    background_coverage: float
    quality_score: float
    phash: str
    crop: CropBox | None


def measure_image(gray: Image.Image, *, analysis_edge: int = 512) -> MeasurementResult:
    """Measure a decoded grayscale image.

    Geometry (``width``/``height``/``crop``) is reported in *source* pixels; the
    coverage numbers and the pHash come from the downscaled analysis view.
    """
    view = analysis_view(gray, analysis_edge)
    ink = round(ink_coverage(view), 4)
    text = round(text_coverage(view), 4)
    paper = round(background_coverage(view), 4)
    return MeasurementResult(
        width=gray.width,
        height=gray.height,
        ink_coverage=ink,
        text_coverage=text,
        background_coverage=paper,
        quality_score=quality_score_from(
            ink=ink, text=text, paper=paper, short_edge=min(gray.width, gray.height)
        ),
        phash=phash(gray),
        crop=suggest_crop(ink_bbox(view), (view.shape[1], view.shape[0]), gray.size),
    )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, used for source and derivative checksums."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()
