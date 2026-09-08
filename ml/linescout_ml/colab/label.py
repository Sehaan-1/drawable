"""Provisional labels: zero-shot style and scope, plus the SFW gate.

Ingestion cannot leave ``primary_style`` and ``scopes`` empty — the manifest
requires exactly one style and at least one *gallery* scope, and the curation
queue stratifies over the 5×10 style×scope grid those produce. Rather than inventing a
placeholder, the pipeline asks the MobileCLIP2 text encoder it already loaded
for embeddings: a handful of prompts per label, softmax over the averaged prompt
embeddings, and the winner is written down as *provisional*.

Provisional is the operative word. ``AssetLabels.labelled_by`` records whether
a CLIP encoder ranked the label or the source default was used, the raw
probabilities are kept beside it, and the curation UI exists to correct all of
it. Nothing here is a ground truth claim.

The SFW gate is deliberately separate from the style/scope gate, and the rule for
which one runs is *who made the guarantee*, not how pretty the dataset is: a
publisher's own terms (a museum open-access programme, an application-gated research
corpus) are recorded as ``method="source_rating"`` and no classifier runs; anything
scraped from a community pays for ``opennsfw2``, including when the site ships rating
tags, because those tags are the claim under test. An asset that fails the gate is
quarantined —
``review.state="quarantined"`` and ``enabled=false`` — which the manifest
enforces as an invariant, so an unsafe asset cannot be served by accident.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from linescout_ml.colab._optional import optional_module
from linescout_ml.colab.checkpoints import CheckpointError, CheckpointVerifier
from linescout_ml.colab.config import SfwMethod, SourceSpec
from linescout_ml.colab.models import OpenClipEncoder
from linescout_ml.colab.sources import AssetLabels
from linescout_ml.manifest import SfwDecision
from linescout_ml.taxonomy import GALLERY_SCOPES, PrimaryStyle, ReviewState, ScopeLabel

OPENNSFW2_HINT = "pip install opennsfw2   # needs TensorFlow, which Colab ships"


class OpenNsfw2Error(RuntimeError):
    """The NSFW classifier could not be loaded or verified."""


#: Prompt sets. Several phrasings per label, averaged, because a single prompt
#: makes CLIP-family models brittle on line art (no colour, no shading cues).
STYLE_PROMPTS: dict[PrimaryStyle, tuple[str, ...]] = {
    PrimaryStyle.MANGA_ANIME: (
        "a manga panel with clean ink lines",
        "an anime illustration of a character",
        "japanese comic book line art",
        "a cel-shaded anime character drawing",
    ),
    PrimaryStyle.WESTERN_INK: (
        "an american comic book inked page",
        "a western ink drawing with bold brush lines",
        "franco-belgian bande dessinée line art",
        "an inked superhero comic illustration",
    ),
    PrimaryStyle.REALISTIC_ACADEMIC: (
        "an academic figure drawing from life",
        "a realistic anatomical study",
        "a classical portrait study in charcoal",
        "an observational life-drawing sketch",
    ),
    PrimaryStyle.CARTOON: (
        "a cartoon character drawing",
        "a children's book illustration outline",
        "a simple cartoon mascot with bold outlines",
        "an animation cel of a cartoon character",
    ),
    PrimaryStyle.GESTURE_SKETCH: (
        "a loose gesture sketch of a pose",
        "a quick thumbnail pose sketch",
        "a rough construction drawing with visible sketch lines",
        "a dynamic action scribble sketch",
    ),
}

SCOPE_PROMPTS: dict[ScopeLabel, tuple[str, ...]] = {
    ScopeLabel.EYE: (
        "a close-up drawing of a single eye",
        "an eye study with eyelashes and iris",
    ),
    ScopeLabel.EYEBROW: (
        "a drawing of an eyebrow",
        "an eyebrow study with individual hairs",
    ),
    ScopeLabel.MOUTH: (
        "a drawing of a mouth",
        "a close-up study of lips",
    ),
    ScopeLabel.FACE_HEAD: (
        "a drawing of a face",
        "a portrait head and shoulders study",
        "a drawing of a head from the front",
    ),
    ScopeLabel.HAIR: (
        "a drawing of hair only",
        "a hairstyle study showing strands of hair",
    ),
    ScopeLabel.HAND: (
        "a drawing of a hand",
        "a study of fingers and a wrist",
        "a hand pose study",
    ),
    ScopeLabel.FOOT: (
        "a drawing of a foot",
        "a study of feet and ankles",
        "a drawing of a leg from the knee down",
    ),
    ScopeLabel.UPPER_BODY_CLOTHING: (
        "a drawing of a torso wearing clothing",
        "a costume design of the upper body",
        "a drawing of a chest and shoulders with a jacket",
    ),
    ScopeLabel.FULL_BODY: (
        "a drawing of a full body character",
        "a full-length figure drawing head to toe",
        "a character design sheet showing the whole body",
    ),
    ScopeLabel.MULTI_CHARACTER: (
        "a drawing with two or more characters",
        "a group of people drawn together in one frame",
        "a scene with several characters",
    ),
}

#: Labels a zero-shot pass may produce. ``unknown`` stays query-only.
LABEL_SCOPES: tuple[ScopeLabel, ...] = tuple(
    scope for scope in ScopeLabel if scope in GALLERY_SCOPES
)


@dataclass(frozen=True)
class LabelScores:
    """Softmax probabilities per label, kept for the curation UI's benefit."""

    styles: dict[PrimaryStyle, float]
    scopes: dict[ScopeLabel, float]

    @property
    def top_style(self) -> PrimaryStyle:
        return max(self.styles, key=lambda style: self.styles[style])

    @property
    def top_scope(self) -> ScopeLabel:
        return max(self.scopes, key=lambda scope: self.scopes[scope])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities: np.ndarray = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    return probabilities


def _label_matrix(encoder: OpenClipEncoder, prompts: dict[Any, tuple[str, ...]]) -> np.ndarray:
    """One row per label: the L2-normalised mean of that label's prompt embeddings."""
    labels = list(prompts)
    flat = [prompt for label in labels for prompt in prompts[label]]
    embeddings = encoder.encode_texts(flat)
    rows: list[np.ndarray] = []
    cursor = 0
    for label in labels:
        count = len(prompts[label])
        block = embeddings[cursor : cursor + count]
        cursor += count
        mean = block.mean(axis=0)
        rows.append(mean / max(float(np.linalg.norm(mean)), 1e-12))
    stacked: np.ndarray = np.stack(rows).astype(np.float32)
    return stacked


class ZeroShotLabeler:
    """Ranks the five styles and gallery scopes for one image at a time."""

    def __init__(self, encoder: OpenClipEncoder, logit_scale: float = 100.0) -> None:
        self.encoder = encoder
        self.logit_scale = logit_scale
        self._styles: list[PrimaryStyle] = list(STYLE_PROMPTS)
        self._scopes: list[ScopeLabel] = list(SCOPE_PROMPTS)
        self._style_matrix: np.ndarray | None = None
        self._scope_matrix: np.ndarray | None = None

    @classmethod
    def load(
        cls,
        model: str = "MobileCLIP2-S2",
        pretrained: str = "dfndr2b",
        device: str = "cpu",
        *,
        verifier: CheckpointVerifier | None = None,
    ) -> ZeroShotLabeler:
        """Load the labeler.

        ``verifier`` pins the CLIP weights and tokenizer to a repository
        revision and a SHA-256; without it the encoder still loads, but the run
        report says the labels came from unpinned bytes.
        """
        encoder = OpenClipEncoder.load(model, device, pretrained=pretrained, verifier=verifier)
        return cls(encoder, logit_scale=_read_logit_scale(encoder))

    @classmethod
    def from_encoder(cls, encoder: OpenClipEncoder) -> ZeroShotLabeler:
        """Reuse an encoder the embedding stage already loaded."""
        return cls(encoder, logit_scale=_read_logit_scale(encoder))

    def _matrices(self) -> tuple[np.ndarray, np.ndarray]:
        if self._style_matrix is None or self._scope_matrix is None:
            self._style_matrix = _label_matrix(self.encoder, STYLE_PROMPTS)
            self._scope_matrix = _label_matrix(self.encoder, SCOPE_PROMPTS)
        return self._style_matrix, self._scope_matrix

    def score(self, image: Image.Image) -> LabelScores:
        """Probabilities for a single image."""
        return self.score_batch([image])[0]

    def score_batch(self, images: Sequence[Image.Image]) -> list[LabelScores]:
        """Probabilities for a batch — one GPU round trip instead of ``len(images)``."""
        if not images:
            return []
        style_matrix, scope_matrix = self._matrices()
        features = self.encoder.encode_images(images)
        style_probs = _softmax(((features @ style_matrix.T) * self.logit_scale).astype(np.float64))
        scope_probs = _softmax(((features @ scope_matrix.T) * self.logit_scale).astype(np.float64))
        return [
            LabelScores(
                styles={
                    label: round(float(value), 6)
                    for label, value in zip(self._styles, style_row, strict=True)
                },
                scopes={
                    label: round(float(value), 6)
                    for label, value in zip(self._scopes, scope_row, strict=True)
                },
            )
            for style_row, scope_row in zip(style_probs, scope_probs, strict=True)
        ]

    def release(self) -> None:
        self._style_matrix = None
        self._scope_matrix = None
        self.encoder.release()


def _read_logit_scale(encoder: OpenClipEncoder) -> float:
    """open_clip keeps the temperature in log space; fall back to CLIP's 100."""
    try:
        return float(np.exp(float(encoder.model.logit_scale.item())))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 100.0


def select_scopes(
    scores: dict[ScopeLabel, float],
    *,
    top_k: int = 2,
    min_score: float = 0.15,
    fallback: ScopeLabel = ScopeLabel.FULL_BODY,
) -> list[ScopeLabel]:
    """Keep the strongest scope, then any others clearing ``min_score``.

    The result is never empty and never contains ``unknown``: both would fail
    manifest validation. Ordering follows the score, so the first entry is the
    reviewer's most likely correction target.
    """
    ranked = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
    chosen = [label for label, probability in ranked[:top_k] if probability >= min_score]
    if not chosen:
        chosen = [ranked[0][0]] if ranked else [fallback]
    if chosen == [ScopeLabel.MULTI_CHARACTER]:
        # "Two or more characters" alone is useless to search: add the best
        # content scope so the asset lands in a real bucket as well.
        extras = [label for label, _ in ranked if label is not ScopeLabel.MULTI_CHARACTER]
        if extras:
            chosen.append(extras[0])
    return chosen


def person_count_for(scopes: Sequence[ScopeLabel]) -> int | None:
    """Provisional person count. ``None`` when the sketch is not of a person.

    The manifest ties ``multi_character`` to ``>= 2``. Human anatomy scopes
    imply one person; anything else (still-life doodles, objects) stays unset.
    """
    if ScopeLabel.MULTI_CHARACTER in scopes:
        return 2
    human = {
        ScopeLabel.EYE,
        ScopeLabel.EYEBROW,
        ScopeLabel.MOUTH,
        ScopeLabel.FACE_HEAD,
        ScopeLabel.HAIR,
        ScopeLabel.HAND,
        ScopeLabel.FOOT,
        ScopeLabel.UPPER_BODY_CLOTHING,
        ScopeLabel.FULL_BODY,
    }
    if any(scope in human for scope in scopes):
        return 1
    return None


def source_rating_sfw(confidence: float = 1.0) -> SfwDecision:
    """Record that the *source* vouched for this content, and that nothing checked.

    No image is opened here. That is the point of the method existing — a museum's
    open-access programme is a real guarantee and paying a classifier to rediscover
    it would only add false positives over public-domain nudes — but the recorded
    ``method="source_rating"`` is a claim about provenance, not an inspection, which
    is why a preset that scraped a community site does not get to use it alone.
    """
    return SfwDecision(
        safe=True, confidence=round(min(max(confidence, 0.0), 1.0), 4), method="source_rating"
    )


#: Where opennsfw2 looks for its weights, and what its own download helper uses.
WEIGHTS_FILENAME = "open_nsfw_weights.h5"
#: The lock id of the classifier's checkpoint (``opennsfw2`` group).
NSFW_ARTIFACT_ID = "opennsfw2/open_nsfw_weights.h5"
NSFW_GROUP = "opennsfw2"


def default_weights_path() -> Path:
    """opennsfw2's own default location, honouring ``OPENNSFW2_HOME``."""
    home = os.environ.get("OPENNSFW2_HOME") or str(Path.home())
    return Path(home) / ".opennsfw2" / "weights" / WEIGHTS_FILENAME


class OpenNsfw2Classifier:
    """Optional NSFW screen. Needs TensorFlow, which Colab ships preinstalled.

    The gate decides whether an asset is quarantined, so the classifier is pinned
    like everything else: with a verifier, the weights are downloaded through it
    and hash-checked before Keras ever reads them.
    """

    def __init__(
        self,
        module: Any,
        batch_size: int = 8,
        weights_path: Path | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        self.module = module
        self.batch_size = batch_size
        self.weights_path = weights_path
        self.checkpoint = checkpoint or {"pinned": False, "artifacts": []}

    @classmethod
    def load(
        cls, batch_size: int = 8, *, verifier: CheckpointVerifier | None = None
    ) -> OpenNsfw2Classifier:
        module = optional_module("opennsfw2", OPENNSFW2_HINT)
        weights: Path | None = None
        checkpoint: dict[str, Any] | None = None
        if verifier is not None and verifier.knows(NSFW_GROUP):
            destination = default_weights_path()
            try:
                verifier.ensure_at(NSFW_ARTIFACT_ID, destination)
            except CheckpointError as error:
                msg = f"opennsfw2: {error}"
                raise OpenNsfw2Error(msg) from error
            weights, checkpoint = destination, verifier.as_dict()
        return cls(module, batch_size=batch_size, weights_path=weights, checkpoint=checkpoint)

    def nsfw_probabilities(self, images: Sequence[Image.Image]) -> list[float]:
        """P(NSFW) per image, in input order."""
        if not images:
            return []
        kwargs: dict[str, Any] = {"batch_size": self.batch_size}
        if self.weights_path is not None:
            kwargs["weights_path"] = str(self.weights_path)
        produced = self.module.predict_images([image.convert("RGB") for image in images], **kwargs)
        return [float(probability) for probability in produced]

    def decision(self, image: Image.Image, *, min_confidence: float) -> SfwDecision:
        probability = self.nsfw_probabilities([image])[0]
        safe_probability = 1.0 - probability
        return SfwDecision(
            safe=safe_probability >= min_confidence,
            confidence=round(safe_probability, 4),
            method="opennsfw2",
        )


def combine_sfw(
    source: SfwDecision,
    classifier: SfwDecision,
    *,
    method: SfwMethod = "source_rating+opennsfw2",
) -> SfwDecision:
    """Keep the stricter of two verdicts — unsafe wins, lower confidence wins."""
    return SfwDecision(
        safe=bool(source.safe and classifier.safe),
        confidence=round(min(source.confidence, classifier.confidence), 4),
        method=method,
    )


def labels_from_scores(
    scores: LabelScores,
    source: SourceSpec,
    sfw: SfwDecision,
    *,
    scope_top_k: int = 2,
    scope_min_score: float = 0.15,
) -> AssetLabels:
    """Turn zero-shot probabilities into the labels stored on a candidate."""
    scopes = select_scopes(
        scores.scopes,
        top_k=scope_top_k,
        min_score=scope_min_score,
        fallback=source.default_scopes[0],
    )
    return AssetLabels(
        primary_style=scores.top_style,
        scopes=scopes,
        person_count=person_count_for(scopes),
        sfw=sfw,
        labelled_by="zero_shot",
        style_scores={style.value: value for style, value in scores.styles.items()},
        scope_scores={scope.value: value for scope, value in scores.scopes.items()},
    )


def source_default_labels(source: SourceSpec, sfw: SfwDecision) -> AssetLabels:
    """Fallback when labelling is off: the source's declared style and scopes."""
    return AssetLabels(
        primary_style=source.default_style,
        scopes=list(source.default_scopes),
        person_count=person_count_for(source.default_scopes),
        sfw=sfw,
        labelled_by="source_default",
    )


def review_state_for(sfw: SfwDecision) -> ReviewState:
    """Unsafe assets are quarantined at ingestion; everything else awaits review."""
    return ReviewState.UNREVIEWED if sfw.safe else ReviewState.QUARANTINED
