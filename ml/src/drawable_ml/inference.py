from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps

from .io import MLError, load_json, resolve_private_path

MODEL_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "config" / "models.json"


def preprocessing_profile_sha256(profile: dict[str, Any]) -> str:
    canonical = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_model_image(
    image: Image.Image,
    *,
    frame_size: int,
    view: Literal["full", "tight"] = "full",
) -> Image.Image:
    """Apply EXIF orientation, white alpha compositing, and a non-cropping square frame."""
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")

    if view == "tight":
        ink = image.convert("L").point(lambda value: 255 if value < 250 else 0)
        bounds = ink.getbbox()
        if bounds is not None:
            left, top, right, bottom = bounds
            margin_x = max(1, round((right - left) * 0.02))
            margin_y = max(1, round((bottom - top) * 0.02))
            image = image.crop(
                (
                    max(0, left - margin_x),
                    max(0, top - margin_y),
                    min(image.width, right + margin_x),
                    min(image.height, bottom + margin_y),
                )
            )
    elif view != "full":
        raise MLError(f"Unsupported preprocessing view: {view}")

    fitted = ImageOps.contain(image, (frame_size, frame_size), method=Image.Resampling.BILINEAR)
    frame = Image.new("RGB", (frame_size, frame_size), "white")
    frame.paste(fitted, ((frame_size - fitted.width) // 2, (frame_size - fitted.height) // 2))
    return frame


def _registry_profile(model_id: str, registry_path: Path | None) -> dict[str, Any]:
    path = (registry_path or MODEL_REGISTRY_PATH).resolve(strict=True)
    registry = load_json(path)
    rows = registry.get("models") if isinstance(registry, dict) else None
    row = next(
        (item for item in rows or [] if isinstance(item, dict) and item.get("id") == model_id),
        None,
    )
    profile = row.get("preprocessing_profile") if isinstance(row, dict) else None
    if not isinstance(profile, dict):
        raise MLError(f"No pinned preprocessing profile is registered for {model_id}")
    return profile


def _validate_tensor(tensor: Any, expected_size: int, model_id: str) -> None:
    if getattr(tensor, "ndim", None) != 3 or tuple(tensor.shape) != (
        3,
        expected_size,
        expected_size,
    ):
        raise MLError(
            f"{model_id} preprocessing produced {tuple(getattr(tensor, 'shape', ()))}, "
            f"expected (3, {expected_size}, {expected_size})"
        )
    import torch

    if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
        raise MLError(f"{model_id} preprocessing produced non-finite or non-floating values")


@dataclass(slots=True)
class FrozenImageEncoder:
    model_id: str
    model: Any
    preprocess: Callable[[Image.Image], Any]
    device: Any
    torch: Any
    expected_dimension: int
    profile: dict[str, Any]
    preprocessing_sha256: str

    def encode(self, image: Image.Image, *, view: Literal["full", "tight"] = "full") -> Any:
        from torch.nn import functional

        framed = prepare_model_image(image, frame_size=int(self.profile["frame_size"]), view=view)
        tensor = self.preprocess(framed).unsqueeze(0).to(self.device)
        expected_size = int(self.profile.get("output_size", self.profile["frame_size"]))
        _validate_tensor(tensor[0], expected_size, self.model_id)
        with self.torch.inference_mode():
            if self.model_id == "dinov2-small":
                output = self.model(pixel_values=tensor)
                feature = output.last_hidden_state[:, 0].float()
            else:
                feature = self.model.encode_image(tensor, normalize=False).float()
        if feature.ndim != 2 or feature.shape[1] != self.expected_dimension:
            raise MLError(
                f"{self.model_id} returned an unexpected embedding shape: {tuple(feature.shape)}"
            )
        feature = functional.normalize(feature, dim=-1)
        if not self.torch.isfinite(feature).all().item():
            raise MLError(f"{self.model_id} returned non-finite feature values")
        return feature[0].detach().cpu()


def load_frozen_encoder(
    data_root: Path,
    model_id: str,
    model_registry_path: Path | None = None,
) -> FrozenImageEncoder:
    """Load one reviewed frozen encoder using only its verified local files on CPU."""
    try:
        import torch
    except ImportError as error:
        raise MLError("Install the locked ML CPU profile before loading frozen encoders") from error
    device = torch.device("cpu")
    profile = _registry_profile(model_id, model_registry_path)
    profile_hash = preprocessing_profile_sha256(profile)
    if profile.get("exif_orientation") != "transpose" or profile.get("alpha_background") != "white":
        raise MLError(
            f"{model_id} preprocessing does not preserve orientation and white alpha composition"
        )
    if profile.get("framing") != "orientation-preserving-square-letterbox":
        raise MLError(f"{model_id} preprocessing profile has an unsupported framing operation")
    files_root = resolve_private_path(
        data_root.resolve(), f"models/{model_id}/files", must_exist=False
    )
    if not files_root.is_dir() or files_root.is_symlink():
        raise MLError(f"Verified local artifact directory is missing for {model_id}")
    try:
        if model_id == "mobileclip2-s2-dfndr2b":
            import open_clip

            weight_path = resolve_private_path(
                data_root.resolve(),
                f"models/{model_id}/files/open_clip_model.safetensors",
            )
            model, _, transform = open_clip.create_model_and_transforms(
                "MobileCLIP2-S2",
                pretrained=str(weight_path),
                device="cpu",
                force_image_size=int(profile["frame_size"]),
                image_mean=tuple(profile["mean"]),
                image_std=tuple(profile["std"]),
                image_interpolation=str(profile["interpolation"]),
                image_resize_mode=str(profile["resize"]),
            )
            actual_profile = getattr(getattr(model, "visual", None), "preprocess_cfg", {})
            for key, expected in {
                "size": int(profile["frame_size"]),
                "mean": tuple(profile["mean"]),
                "std": tuple(profile["std"]),
                "interpolation": profile["interpolation"],
                "resize_mode": profile["resize"],
            }.items():
                actual = actual_profile.get(key)
                if (
                    key == "size"
                    and isinstance(actual, (tuple, list))
                    and tuple(actual) == (expected, expected)
                ):
                    actual = expected
                if key in {"mean", "std"} and actual is not None:
                    actual = tuple(actual)
                if actual != expected:
                    raise MLError(
                        f"MobileCLIP preprocessing mismatch for {key}: {actual!r} != {expected!r}"
                    )
            model.to(device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            sample = transform(
                prepare_model_image(Image.new("RGBA", (31, 67), (0, 0, 0, 0)), frame_size=256)
            )
            _validate_tensor(sample, 256, model_id)
            return FrozenImageEncoder(
                model_id, model, transform, device, torch, 512, profile, profile_hash
            )
        if model_id == "dinov2-small":
            from transformers import AutoImageProcessor, AutoModel

            processor_path = resolve_private_path(
                data_root.resolve(), "models/dinov2-small/files/preprocessor_config.json"
            )
            processor_config = load_json(processor_path)
            expected_processor = {
                "do_center_crop": True,
                "do_normalize": True,
                "do_rescale": True,
                "do_resize": True,
                "image_mean": profile["mean"],
                "image_std": profile["std"],
                "resample": 3,
                "rescale_factor": 1 / 255,
                "size": {"shortest_edge": int(profile["resize"].rsplit("-", 1)[-1])},
                "crop_size": {
                    "height": int(profile["output_size"]),
                    "width": int(profile["output_size"]),
                },
            }
            if any(
                processor_config.get(key) != expected
                for key, expected in expected_processor.items()
            ):
                raise MLError(
                    "Pinned DINOv2 preprocessor_config.json differs from the 224-pixel baseline"
                )

            model = AutoModel.from_pretrained(
                files_root,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            processor = AutoImageProcessor.from_pretrained(
                files_root,
                use_fast=False,
                local_files_only=True,
                trust_remote_code=False,
                size={"shortest_edge": 256},
                crop_size={"height": 224, "width": 224},
                image_mean=tuple(profile["mean"]),
                image_std=tuple(profile["std"]),
                resample=3,
            )
            if (
                processor.size != {"shortest_edge": 256}
                or processor.crop_size != {"height": 224, "width": 224}
                or tuple(processor.image_mean) != tuple(profile["mean"])
                or tuple(processor.image_std) != tuple(profile["std"])
            ):
                raise MLError(
                    "Loaded DINOv2 image processor differs from the pinned 224-pixel baseline"
                )
            if getattr(model.config, "hidden_size", None) != 384:
                raise MLError("Pinned DINOv2-small does not expose the required 384-d CLS profile")
            model.to(device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

            def preprocess(image: Image.Image) -> Any:
                return processor(images=image, return_tensors="pt")["pixel_values"][0]

            sample = preprocess(
                prepare_model_image(Image.new("RGBA", (31, 67), (0, 0, 0, 0)), frame_size=256)
            )
            _validate_tensor(sample, 224, model_id)
            return FrozenImageEncoder(
                model_id, model, preprocess, device, torch, 384, profile, profile_hash
            )
    except MLError:
        raise
    except Exception as error:
        raise MLError(
            f"Could not load {model_id} from verified local files with network access disabled: {error}"
        ) from error
    raise MLError(f"No frozen inference adapter is implemented for {model_id}")
