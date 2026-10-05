from __future__ import annotations

from PIL import Image, ImageDraw

from drawable_ml.inference import prepare_model_image, preprocessing_profile_sha256


def _dark_bounds(image: Image.Image) -> tuple[int, int, int, int] | None:
    mask = image.convert("L").point(lambda value: 255 if value < 32 else 0)
    return mask.getbbox()


def test_transparency_is_composited_over_white_before_framing() -> None:
    source = Image.new("RGBA", (20, 10), (0, 0, 0, 128))

    framed = prepare_model_image(source, frame_size=40)

    assert framed.mode == "RGB"
    assert framed.size == (40, 40)
    assert framed.getpixel((2, 2)) == (255, 255, 255)
    assert framed.getpixel((20, 20)) == (127, 127, 127)


def test_full_view_letterboxes_without_cropping_portrait_or_landscape_content() -> None:
    landscape = Image.new("RGB", (80, 40), "white")
    ImageDraw.Draw(landscape).rectangle((0, 0, 7, 7), fill="black")
    portrait = Image.new("RGB", (40, 80), "white")
    ImageDraw.Draw(portrait).rectangle((0, 0, 7, 7), fill="black")

    landscape_framed = prepare_model_image(landscape, frame_size=80)
    portrait_framed = prepare_model_image(portrait, frame_size=80)

    assert _dark_bounds(landscape_framed) == (0, 20, 8, 28)
    assert _dark_bounds(portrait_framed) == (20, 0, 28, 8)
    assert landscape_framed.getpixel((40, 2)) == (255, 255, 255)
    assert portrait_framed.getpixel((2, 40)) == (255, 255, 255)


def test_tight_view_uses_ink_bounds_and_exif_orientation_is_applied() -> None:
    sparse = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(sparse).rectangle((40, 40, 59, 59), fill="black")
    full = prepare_model_image(sparse, frame_size=64)
    tight = prepare_model_image(sparse, frame_size=64, view="tight")

    full_bounds = _dark_bounds(full)
    tight_bounds = _dark_bounds(tight)
    assert full_bounds is not None and tight_bounds is not None
    assert (full_bounds[2] - full_bounds[0]) < (tight_bounds[2] - tight_bounds[0])
    assert (full_bounds[3] - full_bounds[1]) < (tight_bounds[3] - tight_bounds[1])

    oriented = Image.new("RGB", (20, 10), "white")
    ImageDraw.Draw(oriented).rectangle((0, 4, 2, 6), fill="black")
    exif = oriented.getexif()
    exif[274] = 6
    oriented.info["exif"] = exif.tobytes()
    transposed = prepare_model_image(oriented, frame_size=64)
    bounds = _dark_bounds(transposed)
    assert bounds is not None
    assert bounds[1] <= 3
    assert 27 <= (bounds[0] + bounds[2]) // 2 <= 37


def test_preprocessing_profile_identity_uses_canonical_json() -> None:
    first = {"frame_size": 256, "mean": [0.48, 0.46, 0.41]}
    reordered = {"mean": [0.48, 0.46, 0.41], "frame_size": 256}

    assert preprocessing_profile_sha256(first) == preprocessing_profile_sha256(reordered)
