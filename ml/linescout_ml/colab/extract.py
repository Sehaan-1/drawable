"""Line-art extraction.

Both extractors named in the Milestone 2 plan are reachable through one
dependency, ``controlnet_aux``, which mirrors the upstream research weights on
Hugging Face (``lllyasviel/Annotators``) instead of the original Google Drive
folders that routinely hit their quota:

========================  ======================  ==========================================
``SourceSpec.extractor``  controlnet_aux class    upstream model (licence)
========================  ======================  ==========================================
``anime2sketch``          ``LineartAnimeDetector`` Mukosame/Anime2Sketch ``netG.pth`` (MIT)
``informative_drawings``  ``LineartDetector``      carolineec/informative-drawings (MIT)
``none``                  —                        the source is already line art
========================  ======================  ==========================================

Rule of thumb: ``anime2sketch`` for manga, anime, and cel-shaded illustration;
``informative_drawings`` for photographic, painterly, and academic-drawing
sources. Whichever ran is recorded on the asset as ``extraction_model`` plus
``extraction_version`` (the ``controlnet_aux`` release), because the manifest
refuses to call an asset "extracted" without naming the extractor.

Geometry contract: ``controlnet_aux`` rounds its working size to a multiple of
64, so the map it returns is *not* the size of the input. We resize back to the
source dimensions before saving, which keeps ``width``/``height``, the crop box,
and the three files per asset in one coordinate system — the API serves
``line_art`` and ``thumbnail`` side by side and the crop overlay assumes they
line up pixel for pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from linescout_ml.colab._optional import optional_module
from linescout_ml.colab.checkpoints import CheckpointError, CheckpointVerifier
from linescout_ml.colab.measure import to_gray
from linescout_ml.colab.runtime import clear_gpu_cache

CONTROLNET_AUX_HINT = "pip install controlnet-aux   # pulls torch/timm/einops/opencv-headless"

ANNOTATORS_REPO = "lllyasviel/Annotators"


@dataclass(frozen=True)
class ExtractorSpec:
    """Which detector to build and what to record in the manifest."""

    key: str
    model: str
    class_name: str
    weights_file: str
    upstream: str
    license: str
    best_for: str
    #: Artifacts in ``models.lock.json`` that this detector may load. The files
    #: are handed to ``controlnet_aux`` as a *directory*, so the library's own
    #: download path is bypassed rather than trusted.
    checkpoint_group: str = ""


ANIME2SKETCH = ExtractorSpec(
    key="anime2sketch",
    model="anime2sketch",
    class_name="LineartAnimeDetector",
    weights_file="netG.pth",
    upstream="https://github.com/Mukosame/Anime2Sketch",
    license="MIT",
    best_for="manga, anime, cel-shaded illustration",
    checkpoint_group="anime2sketch",
)
INFORMATIVE_DRAWINGS = ExtractorSpec(
    key="informative_drawings",
    model="informative_drawings",
    class_name="LineartDetector",
    weights_file="sk_model.pth",
    upstream="https://github.com/carolineec/informative-drawings",
    license="MIT",
    best_for="photographs, paintings, academic figure drawing",
    checkpoint_group="informative_drawings",
)

EXTRACTOR_SPECS: dict[str, ExtractorSpec] = {
    spec.key: spec for spec in (ANIME2SKETCH, INFORMATIVE_DRAWINGS)
}


class ExtractorError(RuntimeError):
    """The extractor could not be loaded or could not process an image."""


def spec_for(key: str) -> ExtractorSpec:
    try:
        return EXTRACTOR_SPECS[key]
    except KeyError:
        msg = f"unknown extractor {key!r}; available: {sorted(EXTRACTOR_SPECS)}"
        raise ExtractorError(msg) from None


def library_version() -> str:
    """``controlnet_aux`` release, recorded as ``extraction_version``."""
    try:
        module = optional_module("controlnet_aux", CONTROLNET_AUX_HINT)
    except RuntimeError:
        return "unknown"
    return str(getattr(module, "__version__", "unknown"))


class LineArtExtractor:
    """A loaded detector plus the provenance to record alongside its output."""

    def __init__(
        self,
        spec: ExtractorSpec,
        detector: Any,
        version: str,
        device: str,
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.detector = detector
        self.version = version
        self.device = device
        #: What the checkpoint verifier concluded about the weights in use.
        self.checkpoint = checkpoint or {"pinned": False, "artifacts": []}

    @classmethod
    def load(
        cls,
        key: str,
        device: str = "cpu",
        *,
        verifier: CheckpointVerifier | None = None,
    ) -> LineArtExtractor:
        """Build a detector.

        With a :class:`~linescout_ml.colab.checkpoints.CheckpointVerifier` the
        weights come from the pinned, hash-checked directory instead of whatever
        ``lllyasviel/Annotators`` happens to serve today; without one the
        behaviour is the historical ``from_pretrained(repo)``.
        """
        spec = spec_for(key)
        controlnet_aux = optional_module("controlnet_aux", CONTROLNET_AUX_HINT)
        detector_class = getattr(controlnet_aux, spec.class_name, None)
        if detector_class is None:
            msg = f"controlnet_aux has no {spec.class_name}; upgrade it: {CONTROLNET_AUX_HINT}"
            raise ExtractorError(msg)
        source: Any = ANNOTATORS_REPO
        checkpoint: dict[str, Any] | None = None
        if verifier is not None and spec.checkpoint_group and verifier.knows(spec.checkpoint_group):
            try:
                source = verifier.directory(spec.checkpoint_group)
            except CheckpointError as error:
                msg = f"{spec.key}: {error}"
                raise ExtractorError(msg) from error
            checkpoint = verifier.as_dict()
        try:
            detector = detector_class.from_pretrained(source)
        except Exception as error:  # weight download or checkpoint-shape failures
            msg = f"could not load {spec.key} weights from {source}: {error}"
            raise ExtractorError(msg) from error
        detector.to(device)
        return cls(
            spec,
            detector,
            str(getattr(controlnet_aux, "__version__", "unknown")),
            device,
            checkpoint,
        )

    def extract(
        self,
        image: Image.Image,
        *,
        detect_resolution: int = 1024,
    ) -> Image.Image:
        """Run the detector and restore the source geometry.

        Returns a mode-``L`` image with dark ink on white, the polarity the
        manifest and the canvas both assume.
        """
        rgb = image.convert("RGB")
        try:
            produced = self.detector(
                rgb,
                detect_resolution=detect_resolution,
                image_resolution=detect_resolution,
                output_type="pil",
            )
        except RuntimeError as error:
            msg = f"{self.spec.key} failed on a {rgb.size[0]}x{rgb.size[1]} image: {error}"
            raise ExtractorError(msg) from error
        gray = to_gray(produced)
        if gray.size != rgb.size:
            gray = gray.resize(rgb.size, Image.Resampling.LANCZOS)
        return gray

    @property
    def checkpoint_sha256(self) -> str | None:
        """Digest of the weights file this detector actually loaded, if pinned.

        ``None`` means the run used unpinned bytes and says so in the manifest,
        rather than leaving the field empty for someone to misread as "verified".
        """
        for entry in self.checkpoint.get("artifacts", []):
            if not isinstance(entry, dict):
                continue
            if str(entry.get("path", "")).endswith(self.spec.weights_file):
                digest = entry.get("sha256")
                return str(digest) if digest else None
        return None

    def release(self) -> None:
        self.detector = None
        clear_gpu_cache()


def native_line_art(image: Image.Image) -> Image.Image:
    """For sources that are already line art: normalise, never re-draw.

    Alpha is flattened onto white and orientation is corrected, matching what
    the API does to a query snapshot, so a native asset and an extracted asset
    are measured the same way.
    """
    return to_gray(image)
