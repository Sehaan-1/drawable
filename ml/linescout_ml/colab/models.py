"""Frozen feature-extractor wrappers: MobileCLIP2 (open_clip) and DINOv2 (torch.hub).

Two families, both used read-only:

* **MobileCLIP2** — a fast CLIP variant. The image tower supplies the
  text-aligned retrieval embedding; the text tower supplies the zero-shot
  style/scope labels, so one model serves two stages and is loaded once.
* **DINOv2** — self-supervised shape features. Line art has almost no colour or
  texture, which is exactly where a text-aligned encoder is weakest, so the two
  embeddings are complementary and Milestone 4 concatenates them.

Weights are public and un-gated: MobileCLIP2 comes from ``timm/*-OpenCLIP`` on
Hugging Face via open_clip's ``dfndr2b`` tag, DINOv2 from
``dl.fbaipublicfiles.com`` via torch.hub. No token, no account, no paid tier.

Licence note, because the manifest is a provenance document and so is this:
DINOv2 code *and* weights are Apache-2.0, but **MobileCLIP2 weights are the
Apple ML Research Model License — research use only, no commercial use**. The
:attr:`ModelCard.license` field carries that text so it lands in the run report
next to the features it produced.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from PIL import Image

from linescout_ml.colab._optional import optional_module
from linescout_ml.colab.runtime import clear_gpu_cache

OPEN_CLIP_HINT = "pip install open-clip-torch timm   # MobileCLIP2 needs open_clip >= 3.1"
TORCH_HINT = "pip install torch torchvision   # Colab already ships a CUDA build"

#: DINOv2's published eval transform: resize shortest edge, centre-crop, ImageNet norm.
DINO_RESIZE_EDGE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

APPLE_RESEARCH_LICENSE = (
    "Apple ML Research Model License (research purposes only, no commercial use)"
)


class EncoderError(RuntimeError):
    """A model could not be loaded, or produced features of the wrong shape."""


@dataclass(frozen=True)
class ModelCard:
    """Identity, shape, and licence of one frozen encoder."""

    key: str
    family: Literal["open_clip", "torch_hub"]
    name: str
    pretrained: str
    dim: int
    image_size: int
    license: str
    upstream: str
    citation: str


MOBILECLIP2_S2 = ModelCard(
    key="mobileclip2_s2",
    family="open_clip",
    name="MobileCLIP2-S2",
    pretrained="dfndr2b",
    dim=512,
    image_size=256,
    license=APPLE_RESEARCH_LICENSE,
    upstream="https://huggingface.co/timm/MobileCLIP2-S2-OpenCLIP",
    citation="Faghri et al., MobileCLIP2: Improving Multi-Modal Reinforced Training, TMLR 2025",
)
MOBILECLIP2_S0 = ModelCard(
    key="mobileclip2_s0",
    family="open_clip",
    name="MobileCLIP2-S0",
    pretrained="dfndr2b",
    dim=512,
    image_size=256,
    license=APPLE_RESEARCH_LICENSE,
    upstream="https://huggingface.co/timm/MobileCLIP2-S0-OpenCLIP",
    citation="Faghri et al., MobileCLIP2: Improving Multi-Modal Reinforced Training, TMLR 2025",
)
DINOV2_VITS14 = ModelCard(
    key="dinov2_vits14",
    family="torch_hub",
    name="dinov2_vits14",
    pretrained="",
    dim=384,
    image_size=224,
    license="Apache-2.0 (code and weights)",
    upstream="https://github.com/facebookresearch/dinov2",
    citation="Oquab et al., DINOv2: Learning Robust Visual Features without Supervision, 2023",
)
DINOV2_VITB14_REG = ModelCard(
    key="dinov2_vitb14_reg",
    family="torch_hub",
    name="dinov2_vitb14_reg",
    pretrained="",
    dim=768,
    image_size=224,
    license="Apache-2.0 (code and weights)",
    upstream="https://github.com/facebookresearch/dinov2",
    citation="Oquab et al., DINOv2; Darcet et al., Vision Transformers Need Registers, 2023",
)

MODEL_CARDS: dict[str, ModelCard] = {
    card.key: card for card in (MOBILECLIP2_S2, MOBILECLIP2_S0, DINOV2_VITS14, DINOV2_VITB14_REG)
}


def card_for(key: str) -> ModelCard:
    try:
        return MODEL_CARDS[key]
    except KeyError:
        msg = f"unknown encoder {key!r}; available: {sorted(MODEL_CARDS)}"
        raise EncoderError(msg) from None


def open_clip_card(model: str, pretrained: str | None = None) -> ModelCard:
    """Resolve an open_clip model by registry key, architecture name, or neither.

    An unregistered architecture still works — it just gets ``dim=0`` (the
    dimension is learned from the first batch) and an ``unverified`` licence
    string, because we will not guess at the terms of a model we do not know.
    """
    if model in MODEL_CARDS and MODEL_CARDS[model].family == "open_clip":
        card = MODEL_CARDS[model]
        return card if pretrained is None else replace(card, pretrained=pretrained)
    for card in MODEL_CARDS.values():
        if card.family == "open_clip" and card.name == model:
            return card if pretrained is None else replace(card, pretrained=pretrained)
    slug = "".join(ch if ch.isalnum() else "_" for ch in model).strip("_").lower()[:32]
    return ModelCard(
        key=slug or "open_clip",
        family="open_clip",
        name=model,
        pretrained=pretrained or "",
        dim=0,
        image_size=0,
        license="unverified — check the upstream model card before redistributing features",
        upstream=f"open_clip:{model}",
        citation="",
    )


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized: np.ndarray = features / np.maximum(norms, 1e-12)
    return normalized


def _torch() -> Any:
    return optional_module("torch", TORCH_HINT)


class ImageEncoder:
    """Common surface for both encoder families."""

    def __init__(self, card: ModelCard, device: str) -> None:
        self.card = card
        self.device = device
        #: Learned from the first batch when the card does not pin a dimension.
        self.observed_dim = card.dim

    @property
    def key(self) -> str:
        return self.card.key

    @property
    def dim(self) -> int:
        return self.observed_dim or self.card.dim

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:  # pragma: no cover
        msg = "subclasses implement encode_images"
        raise NotImplementedError(msg)

    def release(self) -> None:
        """Drop references and free GPU blocks so the next model fits."""
        clear_gpu_cache()

    def _check_shape(self, features: np.ndarray, count: int) -> np.ndarray:
        if features.ndim != 2 or features.shape[0] != count:
            msg = f"{self.card.key} produced {features.shape} for {count} images"
            raise EncoderError(msg)
        if not self.card.dim:
            self.observed_dim = int(features.shape[1])
            return features
        if features.shape != (count, self.card.dim):
            msg = (
                f"{self.card.key} produced {features.shape}, expected "
                f"{(count, self.card.dim)}; pin a different encoder or dim"
            )
            raise EncoderError(msg)
        return features


class OpenClipEncoder(ImageEncoder):
    """MobileCLIP2 (and any open_clip architecture) image + text encoder."""

    def __init__(
        self,
        card: ModelCard,
        device: str,
        model: Any,
        preprocess: Any,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__(card, device)
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    @classmethod
    def load(
        cls,
        model: str,
        device: str = "cpu",
        *,
        pretrained: str | None = None,
    ) -> OpenClipEncoder:
        """``model`` may be a registry key (``mobileclip2_s2``) or an open_clip
        architecture name (``MobileCLIP2-S2``)."""
        card = open_clip_card(model, pretrained)
        _torch()  # fail early with the torch hint, not open_clip's vaguer ImportError
        open_clip = optional_module("open_clip", OPEN_CLIP_HINT)
        try:
            # ``network`` not ``model``: the parameter is the model *name*.
            network, _, preprocess = open_clip.create_model_and_transforms(
                card.name, pretrained=pretrained or card.pretrained, device=device
            )
            tokenizer = open_clip.get_tokenizer(card.name)
        except Exception as error:  # any load failure gets the same actionable message
            msg = f"could not load {card.name}: {error}. Try: {OPEN_CLIP_HINT}"
            raise EncoderError(msg) from error
        network.eval()
        return cls(card, device, network, preprocess, tokenizer)

    def _autocast(self, torch: Any) -> Any:
        """Half precision on the GPU; no autocast on CPU (bfloat16 is slower here)."""
        if not self.device.startswith("cuda"):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = _torch()
        batch = torch.stack([self.preprocess(image.convert("RGB")) for image in images])
        batch = batch.to(self.device)
        with torch.inference_mode(), self._autocast(torch):
            encoded = self.model.encode_image(batch)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        array: np.ndarray = np.asarray(encoded.float().cpu().numpy(), dtype=np.float32)
        return _l2_normalize(self._check_shape(array, len(images)))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalised text embeddings, one row per prompt."""
        if self.tokenizer is None:
            msg = f"{self.card.key} was loaded without a tokenizer"
            raise EncoderError(msg)
        torch = _torch()
        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.inference_mode(), self._autocast(torch):
            encoded = self.model.encode_text(tokens)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        texts_array: np.ndarray = np.asarray(encoded.float().cpu().numpy(), dtype=np.float32)
        return texts_array

    def release(self) -> None:
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        super().release()


class DinoEncoder(ImageEncoder):
    """DINOv2 through torch.hub, preprocessed with PIL + numpy only.

    The canonical eval transform (resize shortest edge to 256 with bicubic,
    centre-crop 224, ImageNet normalisation) is reproduced here so the stage
    needs no torchvision transforms and stays deterministic across versions.
    """

    def __init__(self, card: ModelCard, device: str, model: Any) -> None:
        super().__init__(card, device)
        self.model = model

    @classmethod
    def load(cls, key: str, device: str = "cpu") -> DinoEncoder:
        card = card_for(key)
        if card.family != "torch_hub":
            msg = f"{key} is not a torch.hub model"
            raise EncoderError(msg)
        torch = _torch()
        try:
            # trust_repo=True keeps a non-interactive Colab cell from blocking on
            # torch.hub's "do you trust this repo?" prompt.
            model = torch.hub.load("facebookresearch/dinov2", card.name, trust_repo=True).to(device)
        except Exception as error:  # hub failures are network or permission shaped
            msg = f"could not load DINOv2 {card.name} from torch.hub: {error}"
            raise EncoderError(msg) from error
        model.eval()
        return cls(card, device, model)

    def _autocast(self, torch: Any) -> Any:
        if not self.device.startswith("cuda"):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = DINO_RESIZE_EDGE / min(width, height)
        if scale != 1.0:
            rgb = rgb.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.BICUBIC,
            )
        left = (rgb.width - self.card.image_size) // 2
        top = (rgb.height - self.card.image_size) // 2
        rgb = rgb.crop((left, top, left + self.card.image_size, top + self.card.image_size))
        scaled = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(IMAGENET_STD, dtype=np.float32)
        normalised = (scaled - mean) / std
        channels: np.ndarray = np.transpose(normalised, (2, 0, 1))
        return channels

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = _torch()
        stacked = np.stack([self._preprocess(image) for image in images])
        batch = torch.from_numpy(stacked).to(self.device)
        with torch.inference_mode(), self._autocast(torch):
            encoded = self.model(batch)
        array: np.ndarray = np.asarray(encoded.float().cpu().numpy(), dtype=np.float32)
        return _l2_normalize(self._check_shape(array, len(images)))

    def release(self) -> None:
        self.model = None
        super().release()


def load_encoder(key: str, device: str = "cpu") -> ImageEncoder:
    """Load any registered encoder by key."""
    if card_for(key).family == "open_clip":
        return OpenClipEncoder.load(key, device)
    return DinoEncoder.load(key, device)
