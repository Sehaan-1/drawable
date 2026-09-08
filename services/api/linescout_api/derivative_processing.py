"""Measurements and artifact generation for curator-created crop derivatives.

A crop child is an **immutable derivative**: its bytes are cut exactly once
from the parent's verified artifacts, and every derived number (ink, text,
paper, quality, pHash) must be computed from *its own* bytes — never copied
from the parent. This module is that measurement step, plus the thumbnail
generator and the file digest helper the crop lifecycle needs.

The definitions deliberately mirror ``ml/linescout_ml/colab/measure.py``
(the dataset pipeline's source of truth): ink is anything darker than 200 on
a white-flattened grayscale view, the text heuristic looks for runs of
small, dense, similarly-sized components, the quality score is the same
fixed weighted blend, and the pHash is the same 64-bit unnormalised DCT-II
construction ``imagehash.phash`` uses. The ml package implements them over
numpy; this module re-implements the same formulas with PIL + pure Python so
the API keeps its light dependency set (the ``ml`` import stays
pydantic-only). When the definitions change in ``ml``, they must change here
in the same commit — the module docstrings cross-reference each other.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from linescout_api.preprocessing import normalize_to_gray

#: Grayscale values below this count as ink (ml/colab/measure.py parity).
INK_THRESHOLD = 200
#: Grayscale values at or above this count as clean paper.
BACKGROUND_THRESHOLD = 245

# --- text heuristic bounds, in analysis-view pixels (ml parity) -------------
TEXT_MIN_HEIGHT = 5
TEXT_MAX_HEIGHT = 40
TEXT_MIN_ASPECT = 0.5
TEXT_MAX_ASPECT = 10.0
TEXT_MIN_FILL = 0.12
TEXT_MIN_GLYPHS_PER_LINE = 3

# --- quality_score weights (ml parity; they sum to 1.0) ---------------------
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

#: Analysis view long edge. The ml pipeline measures at 512; crops are small,
#: so 256 keeps the pure-Python component pass cheap without moving the
#: thresholds' meaning for gallery-sized crops.
ANALYSIS_EDGE = 256


class DerivativeImageError(RuntimeError):
    """A derivative file could not be decoded or is structurally invalid."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Measurements:
    """Everything processing computes from one derivative's own bytes."""

    width: int
    height: int
    ink_coverage: float
    text_coverage: float
    background_coverage: float
    quality_score: float
    phash: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "width": self.width,
            "height": self.height,
            "ink_coverage": self.ink_coverage,
            "text_coverage": self.text_coverage,
            "background_coverage": self.background_coverage,
            "quality_score": self.quality_score,
            "phash": self.phash,
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def decode_gray(path: Path) -> Image.Image:
    """Decode a file into a white-flattened grayscale image.

    Raises :class:`DerivativeImageError` (never a bare PIL error) so callers
    can map the failure to a structured, retryable response.
    """
    try:
        with Image.open(path) as handle:
            handle.load()
            return normalize_to_gray(handle)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise DerivativeImageError(
            "derivative_decode_failed", f"could not decode image: {error}"
        ) from error


def analysis_view(gray: Image.Image, edge: int = ANALYSIS_EDGE) -> Image.Image:
    """Downscale so the *long* edge is ``edge`` px; never upscale (ml parity)."""
    longest = max(gray.size)
    if longest <= edge:
        return gray
    scale = edge / longest
    return gray.resize(
        (max(1, round(gray.width * scale)), max(1, round(gray.height * scale))),
        Image.Resampling.LANCZOS,
    )


def _ink_pixels(view: Image.Image) -> list[int]:
    """Indices of pixels darker than :data:`INK_THRESHOLD`, row-major."""
    # ``tobytes()`` on an "L"-mode view is one byte per pixel — no iterator
    # typing gymnastics and no deprecated ``getdata()`` call.
    return [index for index, value in enumerate(view.tobytes()) if value < INK_THRESHOLD]


def ink_coverage(view: Image.Image) -> float:
    total = view.width * view.height
    if total == 0:
        return 0.0
    return len(_ink_pixels(view)) / float(total)


def background_coverage(view: Image.Image) -> float:
    total = view.width * view.height
    if total == 0:
        return 0.0
    clean = sum(1 for value in view.tobytes() if value >= BACKGROUND_THRESHOLD)
    return clean / float(total)


#: ``(x, y, width, height, area)`` of one connected component, in analysis-view pixels.
ComponentBox = tuple[int, int, int, int, int]


def _components(ink: list[int], width: int) -> list[ComponentBox]:
    """8-connected components of the ink mask as ``(x, y, w, h, area)``.

    Union-find over foreground pixels only (ml parity): cost tracks ink
    density, not frame size.
    """
    pixels = {(index // width, index % width) for index in ink}
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    for node in pixels:
        parent.setdefault(node, node)
        y, x = node
        for neighbour in ((y - 1, x - 1), (y - 1, x), (y - 1, x + 1), (y, x - 1)):
            if neighbour in pixels:
                parent.setdefault(neighbour, neighbour)
                left_root, right_root = find(node), find(neighbour)
                if left_root != right_root:
                    parent[right_root] = left_root

    boxes: dict[tuple[int, int], list[int]] = {}
    for y, x in pixels:
        root = find((y, x))
        box = boxes.setdefault(root, [x, y, x, y, 0])
        box[0] = min(box[0], x)
        box[1] = min(box[1], y)
        box[2] = max(box[2], x)
        box[3] = max(box[3], y)
        box[4] += 1
    return [(b[0], b[1], b[2] - b[0] + 1, b[3] - b[1] + 1, b[4]) for b in boxes.values()]


def _glyph_candidates(components: list[ComponentBox]) -> list[ComponentBox]:
    """Components small and dense enough to be a character (ml parity)."""
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


def text_coverage(view: Image.Image) -> float:
    """Fraction of the frame covered by text-like glyph runs (ml parity).

    Glyph candidates are clustered into lines by vertical centre and
    horizontal gap; a run of at least :data:`TEXT_MIN_GLYPHS_PER_LINE`
    similar-height glyphs counts as text. A single isolated blob never does,
    which keeps this from flagging eyes, hands, or hatching.
    """
    ink = _ink_pixels(view)
    glyphs = _glyph_candidates(_components(ink, view.width))
    if len(glyphs) < TEXT_MIN_GLYPHS_PER_LINE:
        return 0.0

    median_height = statistics.median(glyph[3] for glyph in glyphs)
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
    total = view.width * view.height
    if total == 0:
        return 0.0
    return _clamp(text_pixels / float(total))


def quality_score_from(*, ink: float, text: float, paper: float, short_edge: int) -> float:
    """Blend the measurements into a 0–1 ordering hint (ml parity).

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


def _dct_matrix(size: int) -> list[list[float]]:
    """Unnormalised DCT-II matrix — the ``norm=None`` form ``imagehash`` uses."""
    indices = list(range(size))
    return [
        [2.0 * math.cos(math.pi * (2 * n + 1) * k / (2 * size)) for n in indices] for k in indices
    ]


def phash(gray: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """64-bit DCT perceptual hash as 16 hex characters (``imagehash`` parity).

    Same 32×32 LANCZOS reduction, same unnormalised 2-D DCT-II, same median
    threshold over the top-left ``hash_size``² block as
    ``ml/linescout_ml/colab/measure.py::phash`` — implemented with nested
    lists instead of numpy so the API stays light.
    """
    if hash_size < 2:
        msg = "hash_size must be at least 2"
        raise ValueError(msg)
    side = hash_size * highfreq_factor
    small = gray.convert("L").resize((side, side), Image.Resampling.LANCZOS)
    pixels = list(small.tobytes())
    basis = _dct_matrix(side)

    # coefficients = basis @ image @ basis.T, computed row by row;
    # image[i][j] = pixels[i * side + j].
    temp: list[list[float]] = [[0.0] * side for _ in range(side)]
    for i in range(side):
        source_row = pixels[i * side : (i + 1) * side]
        for k in range(side):
            factor = basis[k][i]
            if factor == 0.0:
                continue
            target = temp[k]
            for j in range(side):
                target[j] += factor * source_row[j]
    low: list[float] = []
    for k in range(hash_size):
        for col in range(hash_size):
            low.append(sum(temp[k][j] * basis[col][j] for j in range(side)))
    threshold = statistics.median(low)
    value = 0
    for coefficient in low:
        value = (value << 1) | (1 if coefficient > threshold else 0)
    return f"{value:0{hash_size * hash_size // 4}x}"


def measure_line_art(gray: Image.Image) -> Measurements:
    """Measure a decoded, white-flattened grayscale line-art image.

    Geometry is reported in *source* pixels; the coverage numbers and the
    pHash come from the downscaled analysis view (ml parity).
    """
    view = analysis_view(gray)
    ink = round(ink_coverage(view), 4)
    text = round(text_coverage(view), 4)
    paper = round(background_coverage(view), 4)
    return Measurements(
        width=gray.width,
        height=gray.height,
        ink_coverage=ink,
        text_coverage=text,
        background_coverage=paper,
        quality_score=quality_score_from(
            ink=ink, text=text, paper=paper, short_edge=min(gray.width, gray.height)
        ),
        phash=phash(gray),
    )


def make_thumbnail(image: Image.Image, size: int) -> Image.Image:
    """Aspect-fit the artifact into a ``size``×``size`` white square (ml parity)."""
    width, height = image.size
    scale = size / max(width, height)
    fitted = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    tile = Image.new("L", (size, size), 255)
    tile.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return tile


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, used for derivative checksums."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()
