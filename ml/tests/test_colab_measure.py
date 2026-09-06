"""Measurements: pHash, dedupe, the text heuristic, quality, crops, thumbnails.

The pHash tests cross-check the fast matrix form against a textbook double-sum
DCT rather than pinning golden hashes: golden values would silently break when
Pillow changes a resampling filter, while the reference implementation pins the
*algorithm*, which is what ``imagehash.phash`` compatibility actually depends on.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pytest
from colab_images import glyph_rows, line_art, solid, write_png
from PIL import Image, ImageDraw

from linescout_ml.colab.measure import (
    INK_THRESHOLD,
    ImageReadError,
    analysis_view,
    background_coverage,
    duplicate_groups,
    hamming,
    ink_bbox,
    ink_coverage,
    load_gray,
    load_rgb,
    make_thumbnail,
    measure_image,
    phash,
    quality_score_from,
    sha256_file,
    suggest_crop,
    text_coverage,
    to_gray,
)


def _reference_phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Textbook 2-D unnormalised DCT-II pHash — the definition, not the shortcut."""
    side = hash_size * highfreq_factor
    small = image.convert("L").resize((side, side), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float64)
    coefficients = np.zeros((side, side), dtype=np.float64)
    for k in range(side):
        for n in range(side):
            coefficients[k, n] = 2.0 * np.cos(np.pi * (2 * n + 1) * k / (2 * side))
    dct = coefficients @ pixels @ coefficients.T
    low = dct[:hash_size, :hash_size]
    threshold = float(np.median(low))
    value = 0
    for bit in (low > threshold).reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


# --------------------------------------------------------------------- pHash


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_phash_matches_the_dct_definition(seed: int) -> None:
    image = line_art(320, seed=seed)
    assert phash(image) == _reference_phash(image)


def test_phash_is_deterministic_and_well_formed() -> None:
    image = line_art(512, seed=3)
    first, second = phash(image), phash(image)
    assert first == second
    assert len(first) == 16
    int(first, 16)  # must be valid hex


def test_phash_is_close_for_a_rescaled_duplicate() -> None:
    image = line_art(512, seed=4)
    resized = image.resize((480, 480), Image.Resampling.LANCZOS).resize(
        (512, 512), Image.Resampling.LANCZOS
    )
    assert hamming(phash(image), phash(resized)) <= 6


def test_phash_separates_different_content() -> None:
    drawing = line_art(512, seed=5)
    filled = solid(512, 20)
    assert hamming(phash(drawing), phash(filled)) > 12


def test_phash_rejects_a_hash_size_that_cannot_work() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        phash(line_art(64), hash_size=1)


def test_phash_supports_other_sizes() -> None:
    digest = phash(line_art(256, seed=6), hash_size=4, highfreq_factor=4)
    assert len(digest) == 4  # 16 bits -> 4 hex characters


def test_hamming_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        hamming("00ff", "00ff00ff")


def test_hamming_counts_differing_bits() -> None:
    assert hamming("0000000000000000", "0000000000000001") == 1
    assert hamming("ffffffffffffffff", "0000000000000000") == 64


# --------------------------------------------------------------------- dedupe


def _brute_force_groups(hashes: list[str], threshold: int) -> list[list[int]]:
    parent = list(range(len(hashes)))

    def find(index: int) -> int:
        while parent[index] != index:
            index = parent[index]
        return index

    pairs = ((left, right) for left in range(len(hashes)) for right in range(left + 1, len(hashes)))
    for left, right in pairs:
        if hamming(hashes[left], hashes[right]) <= threshold:
            parent[max(find(left), find(right))] = min(find(left), find(right))
    grouped: dict[int, list[int]] = {}
    for index in range(len(hashes)):
        grouped.setdefault(find(index), []).append(index)
    return sorted(sorted(group) for group in grouped.values() if len(group) > 1)


def test_duplicate_groups_finds_exact_copies() -> None:
    image = line_art(512, seed=8)
    hashes = [phash(image), phash(image), phash(solid(512, 20))]
    assert duplicate_groups(hashes, 6) == [[0, 1]]


def test_duplicate_groups_is_transitive() -> None:
    # a == b, b == c, but a and c are far apart: union-find must still merge them.
    hashes = ["0000000000000000", "0000000000000003", "000000000000000f", "ffffffffffffffff"]
    groups = duplicate_groups(hashes, 2)
    assert groups == [[0, 1, 2]]


def test_duplicate_groups_matches_brute_force() -> None:
    rng = random.Random(11)
    hashes = [f"{rng.getrandbits(64):016x}" for _ in range(120)]
    for threshold in (0, 4, 10):
        assert duplicate_groups(hashes, threshold) == _brute_force_groups(hashes, threshold)


def test_duplicate_groups_handles_empty_input() -> None:
    assert duplicate_groups([], 6) == []


# --------------------------------------------------------------------- coverage


def test_ink_and_background_are_complementary() -> None:
    view = analysis_view(line_art(512, seed=1), 512)
    ink, paper = ink_coverage(view), background_coverage(view)
    assert 0.0 < ink < 0.5
    assert paper > 0.5
    assert ink + paper <= 1.0


def test_ink_bbox_matches_the_drawing() -> None:
    view = analysis_view(line_art(512, seed=2), 512)
    x, y, width, height = ink_bbox(view) or (0, 0, 0, 0)
    assert width > 0 and height > 0
    assert x + width <= view.shape[1] and y + height <= view.shape[0]


def test_ink_bbox_is_none_for_a_blank_frame() -> None:
    assert ink_bbox(analysis_view(solid(256, 255), 256)) is None


def test_text_coverage_flags_glyph_rows_but_not_line_art() -> None:
    plain = text_coverage(analysis_view(line_art(512, seed=9), 512))
    captioned = text_coverage(analysis_view(glyph_rows(512, seed=9), 512))
    assert plain == 0.0
    assert captioned > plain


def test_text_coverage_needs_a_run_of_glyphs() -> None:
    # A single small blob is an eye or a freckle, not a caption.
    image = solid(512, 255)
    ImageDraw.Draw(image).rectangle((100, 100, 112, 116), fill=20)
    assert text_coverage(analysis_view(image, 512)) == 0.0


def test_quality_score_penalises_text() -> None:
    plain_view = analysis_view(line_art(512, seed=12), 512)
    captioned_view = analysis_view(glyph_rows(512, seed=12), 512)
    plain = quality_score_from(
        ink=ink_coverage(plain_view),
        text=text_coverage(plain_view),
        paper=background_coverage(plain_view),
        short_edge=512,
    )
    captioned = quality_score_from(
        ink=ink_coverage(captioned_view),
        text=0.20,  # force the penalty: the heuristic only sees a few glyph rows
        paper=background_coverage(captioned_view),
        short_edge=512,
    )
    assert captioned < plain


def test_quality_score_rewards_resolution_and_clean_paper() -> None:
    small = quality_score_from(ink=0.08, text=0.0, paper=0.9, short_edge=256)
    large = quality_score_from(ink=0.08, text=0.0, paper=0.9, short_edge=1024)
    assert large > small
    dirty = quality_score_from(ink=0.08, text=0.0, paper=0.35, short_edge=1024)
    assert dirty < large


def test_quality_score_punishes_a_failed_extraction() -> None:
    inked = quality_score_from(ink=0.08, text=0.0, paper=0.9, short_edge=1024)
    all_black = quality_score_from(ink=0.95, text=0.0, paper=0.02, short_edge=1024)
    assert all_black < inked
    assert 0.0 <= all_black <= 1.0


# --------------------------------------------------------------------- crops


def test_suggest_crop_scales_back_to_source_pixels() -> None:
    image = line_art(1024, seed=13)
    view = analysis_view(image, 512)
    crop = suggest_crop(ink_bbox(view), (view.shape[1], view.shape[0]), image.size)
    assert crop is not None
    assert crop.x + crop.width <= 1024
    assert crop.y + crop.height <= 1024
    # The ink occupies roughly the middle of the frame, so the crop must shrink it.
    assert crop.width < 1024


def test_suggest_crop_is_none_when_ink_fills_the_frame() -> None:
    image = solid(512, 20)
    view = analysis_view(image, 512)
    assert suggest_crop(ink_bbox(view), (512, 512), image.size) is None
    assert suggest_crop(None, (512, 512), image.size) is None


# --------------------------------------------------------------------- images


def test_measure_image_reports_source_geometry() -> None:
    image = line_art(768, seed=14)
    result = measure_image(image, analysis_edge=512)
    assert (result.width, result.height) == (768, 768)
    assert 0.0 <= result.ink_coverage <= 1.0
    assert len(result.phash) == 16
    assert 0.0 <= result.quality_score <= 1.0


def test_analysis_view_never_upscales() -> None:
    small = analysis_view(solid(128, 255), 512)
    assert small.shape == (128, 128)
    large = analysis_view(solid(1024, 255), 512)
    assert max(large.shape) == 512


def test_to_gray_flattens_transparency_onto_white(tmp_path: Path) -> None:
    rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    gray = to_gray(rgba)
    assert gray.mode == "L"
    assert int(gray.getpixel((32, 32))) == 255
    assert int(to_gray(rgba).getpixel((0, 0))) >= INK_THRESHOLD


def test_make_thumbnail_letterboxes_onto_white() -> None:
    wide = Image.new("L", (400, 200), 20)
    tile = make_thumbnail(wide, 128)
    assert tile.size == (128, 128)
    assert tile.mode == "L"
    # Top-left corner is padding, so it stays paper white.
    assert int(tile.getpixel((0, 0))) == 255


def test_load_gray_and_load_rgb_reject_non_images(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.png"
    bogus.write_bytes(b"this is not a png at all")
    with pytest.raises(ImageReadError):
        load_gray(bogus)
    with pytest.raises(ImageReadError):
        load_rgb(bogus)


def test_load_gray_round_trips_a_written_png(tmp_path: Path) -> None:
    path = write_png(tmp_path / "art.png", line_art(300, seed=15))
    gray = load_gray(path)
    assert gray.size == (300, 300)
    assert gray.mode == "L"


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = write_png(tmp_path / "art.png", line_art(128, seed=16))
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
