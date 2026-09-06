"""Deterministic CPU-only test images for the Colab pipeline tests.

Nothing here needs torch, scipy, or a GPU: the pipeline's CPU stages
(discovery, measurement, dedupe, manifest assembly, export) are fully
exercised with Pillow-drawn fixtures.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw


def line_art(size: int = 512, *, seed: int = 0, strokes: int = 10) -> Image.Image:
    """A white canvas with a few dark strokes: the shape of a real reference."""
    rng = random.Random(seed)
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (size * 0.2, size * 0.15, size * 0.8, size * 0.85), outline=20, width=max(1, size // 200)
    )
    for index in range(strokes):
        start = (rng.uniform(0.15, 0.85) * size, rng.uniform(0.15, 0.85) * size)
        end = (rng.uniform(0.15, 0.85) * size, rng.uniform(0.15, 0.85) * size)
        draw.line((*start, *end), fill=rng.choice((20, 40, 60)), width=1 + index % 3)
    return image


def glyph_rows(size: int = 512, *, rows: int = 3, glyphs: int = 8, seed: int = 0) -> Image.Image:
    """Line art plus rectangular glyph runs, to exercise the text heuristic."""
    image = line_art(size, seed=seed, strokes=4)
    draw = ImageDraw.Draw(image)
    height = max(6, size // 32)
    for row in range(rows):
        top = size // 16 + row * (height * 2)
        for column in range(glyphs):
            left = size // 8 + column * (height * 2)
            draw.rectangle((left, top, left + height, top + height), fill=20)
    return image


def solid(size: int = 512, value: int = 255) -> Image.Image:
    return Image.new("L", (size, size), value)


def write_png(path: Path, image: Image.Image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def source_tree(
    root: Path,
    *,
    count: int = 6,
    size: int = 512,
    subdirs: bool = False,
    duplicate_pairs: int = 0,
) -> list[Path]:
    """A fake raw dataset: ``count`` distinct drawings plus optional duplicates."""
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(count):
        image = line_art(size, seed=index, strokes=6 + index)
        relative = (
            Path(f"book-{index // 3:02d}") / f"page-{index:03d}.png"
            if subdirs
            else Path(f"drawing-{index:03d}.png")
        )
        written.append(write_png(root / relative, image))
    for index in range(duplicate_pairs):
        # A byte-identical copy under another name: pHash must catch it.
        original = written[index]
        copy = root / f"copy-of-{original.name}"
        copy.write_bytes(original.read_bytes())
        written.append(copy)
    return sorted(written)
