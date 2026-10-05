from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import MLError


class SemanticTask:
    """A MobileCLIP image encoder plus the project's visible-scope head."""

    def __init__(self, clip: Any, scope_head: Any, stage: Any, projection: Any, projection_name: str) -> None:
        self.clip = clip
        self.scope_head = scope_head
        self.stage = stage
        self.projection = projection
        self.projection_name = projection_name

    def trainables(self) -> dict[str, Any]:
        expected: dict[str, Any] = {}
        for name, parameter in self.clip.named_parameters():
            if name.startswith("visual.trunk.stages.") and name.split(".")[3] == str(len(self.clip.visual.trunk.stages) - 1) or name == self.projection_name:
                expected[name] = parameter
        expected.update({f"scope_head.{name}": parameter for name, parameter in self.scope_head.named_parameters()})
        if not any(name.startswith("visual.trunk.stages.") for name in expected) or self.projection_name not in expected:
            raise MLError("Pinned MobileCLIP2 architecture did not expose the approved final-stage/projection trainables")
        return expected

    def image_features(self, images: Any) -> tuple[Any, Any]:
        features = self.clip.encode_image(images, normalize=False)
        if features is None:
            raise MLError("MobileCLIP2 returned no image embeddings")
        if features.ndim != 2 or features.shape[-1] != 512:
            raise MLError("MobileCLIP2 embedding shape differs from the pinned 512-dimensional profile")
        normalized = features.float()
        normalized = normalized / normalized.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return normalized, self.scope_head(normalized)


def build_semantic_task(weight_path: Path, *, device: str) -> tuple[SemanticTask, Any]:
    try:
        import open_clip
        import torch
    except ImportError as error:
        raise MLError("Install the pinned ML `cuda` profile before loading a training model") from error
    if not weight_path.is_file() or weight_path.is_symlink():
        raise MLError("Pinned MobileCLIP2 weights are missing or unsafe")
    try:
        model, _, transform = open_clip.create_model_and_transforms(
            "MobileCLIP2-S2", pretrained=str(weight_path), image_mean=(0, 0, 0), image_std=(1, 1, 1)
        )
    except Exception as error:
        raise MLError(f"Pinned MobileCLIP2 checkpoint could not load in the locked OpenCLIP environment: {error}") from error
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trunk = getattr(model.visual, "trunk", None)
    stages = getattr(trunk, "stages", None)
    if stages is None or len(stages) < 1:
        raise MLError("Pinned MobileCLIP2 image trunk has no recognized ordered stages")
    stage = stages[-1]
    for parameter in stage.parameters():
        parameter.requires_grad_(True)
    projection_name = "visual.proj"
    projection = getattr(model.visual, "proj", None)
    if projection is None:
        raise MLError("Pinned MobileCLIP2 profile did not expose its approved visual.proj parameter")
    if isinstance(projection, torch.nn.Parameter):
        projection_parameter = projection
    else:
        projection_parameter = next(model.visual.proj.parameters(), None)
    if projection_parameter is None:
        raise MLError("Pinned MobileCLIP2 projection has no trainable parameters")
    projection_parameter.requires_grad_(True)
    scope_head = torch.nn.Linear(512, 11).to(device)
    task = SemanticTask(model, scope_head, stage, projection, projection_name)
    task.trainables()
    return task, transform


class StructuralTask:
    """Frozen DINOv2 CLS features with the specified learned projection."""

    def __init__(self, encoder: Any, projection: Any) -> None:
        self.encoder = encoder
        self.projection = projection

    def __call__(self, pixels: Any) -> Any:
        from torch.nn import functional

        with __import__("torch").no_grad():
            output = self.encoder(pixel_values=pixels)
            cls = output.last_hidden_state[:, 0]
        return functional.normalize(self.projection(cls.float()), dim=-1)


def build_structural_task(model_directory: Path, *, device: str) -> tuple[StructuralTask, Any]:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise MLError("Install the pinned ML `cuda` profile before loading a training model") from error
    if not model_directory.is_dir() or model_directory.is_symlink():
        raise MLError("Pinned local DINOv2 model directory is missing or unsafe")
    try:
        encoder = AutoModel.from_pretrained(model_directory, local_files_only=True, trust_remote_code=False)
        processor = AutoImageProcessor.from_pretrained(model_directory, local_files_only=True, trust_remote_code=False)
    except Exception as error:
        raise MLError(f"Pinned local DINOv2 artifacts could not load offline: {error}") from error
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if getattr(encoder.config, "hidden_size", None) != 384:
        raise MLError("DINOv2 profile must return 384-dimensional CLS features")
    projection = torch.nn.Sequential(torch.nn.Linear(384, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256)).to(device)
    return StructuralTask(encoder, projection), processor


def freeze_batch_norm(model: Any) -> None:
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
