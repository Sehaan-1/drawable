"""Provisional labels: zero-shot style and scope, plus the SFW gate.

Ingestion must fill ``primary_style`` and ``primary_scope`` — and the curation
queue stratifies over the 5×10 style×scope grid those produce. Rather than
inventing a placeholder, the pipeline asks the MobileCLIP2 text encoder it
already loaded for embeddings: a handful of prompts per label, softmax over
the averaged prompt embeddings, and the winner is written down as
*provisional*. When no scope clears the confidence floor, ``primary_scope`` is
``unknown`` — an explicit "not determined" the curation UI must resolve before
acceptance, never a guessed label.

Provisional is the operative word. ``AssetLabels.labelled_by`` records whether
a CLIP encoder ranked the label or the source default was used, the raw
probabilities are kept beside it, and the curation UI exists to correct all of
it. Nothing here is a ground truth claim.

The SFW gate is deliberately separate from the style/scope gate, and under the
v2 contract it is an *automated screen only*: a source whose terms already
guarantee SFW content (museum open-access scans, children's drawing datasets)
is recorded with ``method="source_rating"``, and only scraped sources pay for
``opennsfw2``. A screen that does not clear the confidence floor is
quarantined — ``review.state="quarantined"`` — so an unsafe or unsure asset
cannot be served by accident. A screen, however safe, is never a human
approval: only curation records that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from linescout_ml.colab._optional import optional_module
from linescout_ml.colab.config import SfwMethod, SourceSpec
from linescout_ml.colab.models import OpenClipEncoder
from linescout_ml.colab.sources import AssetLabels
from linescout_ml.manifest import SfwHumanDecision, SfwScreening
from linescout_ml.taxonomy import (
    GALLERY_SCOPES,
    PrimaryStyle,
    ReviewState,
    ScopeLabel,
    SfwScreeningMethod,
    SfwVerdict,
)

OPENNSFW2_HINT = "pip install opennsfw2   # needs TensorFlow, which Colab ships"

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
    ) -> ZeroShotLabeler:
        encoder = OpenClipEncoder.load(model, device, pretrained=pretrained)
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
) -> list[ScopeLabel]:
    """Keep the strongest scope, then any others clearing ``min_score``.

    The result is ordered by descending score, so the first entry becomes the
    ``primary_scope`` and the rest are secondary. When nothing clears
    ``min_score`` the result is ``[unknown]``: an explicit "not determined"
    that curation must resolve, never an invented label.
    """
    ranked = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
    chosen = [label for label, probability in ranked[:top_k] if probability >= min_score]
    if not chosen:
        return [ScopeLabel.UNKNOWN]
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


#: Below this SFW probability an ``opennsfw2`` screen is *unsafe*; between it
#: and ``sfw_min_confidence`` it is *unsure* (borderline — still quarantined).
UNSAFE_FLOOR = 0.5


def source_rating_sfw(confidence: float = 1.0) -> SfwScreening:
    """SFW verdict taken from the source's own rating or terms."""
    return SfwScreening(
        verdict=SfwVerdict.SAFE,
        confidence=round(min(max(confidence, 0.0), 1.0), 4),
        method=SfwScreeningMethod.SOURCE_RATING,
    )


class OpenNsfw2Classifier:
    """Optional NSFW screen. Needs TensorFlow, which Colab ships preinstalled."""

    def __init__(self, module: Any, batch_size: int = 8) -> None:
        self.module = module
        self.batch_size = batch_size

    @classmethod
    def load(cls, batch_size: int = 8) -> OpenNsfw2Classifier:
        return cls(optional_module("opennsfw2", OPENNSFW2_HINT), batch_size=batch_size)

    def nsfw_probabilities(self, images: Sequence[Image.Image]) -> list[float]:
        """P(NSFW) per image, in input order."""
        if not images:
            return []
        produced = self.module.predict_images(
            [image.convert("RGB") for image in images], batch_size=self.batch_size
        )
        return [float(probability) for probability in produced]

    def decision(self, image: Image.Image, *, min_confidence: float) -> SfwScreening:
        probability = self.nsfw_probabilities([image])[0]
        safe_probability = 1.0 - probability
        if safe_probability >= min_confidence:
            verdict = SfwVerdict.SAFE
        elif safe_probability >= UNSAFE_FLOOR:
            verdict = SfwVerdict.UNSURE
        else:
            verdict = SfwVerdict.UNSAFE
        return SfwScreening(
            verdict=verdict,
            confidence=round(safe_probability, 4),
            method=SfwScreeningMethod.OPENNSFW2,
        )


def combine_sfw(
    source: SfwScreening,
    classifier: SfwScreening,
    *,
    method: SfwMethod = "source_rating+opennsfw2",
) -> SfwScreening:
    """Keep the stricter of two verdicts — unsafe beats unsure beats safe."""
    severity = {SfwVerdict.SAFE: 0, SfwVerdict.UNSURE: 1, SfwVerdict.UNSAFE: 2}
    verdict = (
        source.verdict
        if severity[source.verdict] >= severity[classifier.verdict]
        else classifier.verdict
    )
    confidences = [
        value for value in (source.confidence, classifier.confidence) if value is not None
    ]
    return SfwScreening(
        verdict=verdict,
        confidence=round(min(confidences), 4) if confidences else None,
        method=SfwScreeningMethod(method),
    )


def labels_from_scores(
    scores: LabelScores,
    source: SourceSpec,
    sfw: SfwScreening | None,
    *,
    human: SfwHumanDecision | None = None,
    scope_top_k: int = 2,
    scope_min_score: float = 0.15,
) -> AssetLabels:
    """Turn zero-shot probabilities into the labels stored on a candidate."""
    scopes = select_scopes(
        scores.scopes,
        top_k=scope_top_k,
        min_score=scope_min_score,
    )
    person_count = person_count_for(scopes)
    return AssetLabels(
        primary_style=scores.top_style,
        primary_scope=scopes[0],
        secondary_scopes=scopes[1:],
        person_count=person_count,
        person_count_approximate=person_count is not None,
        sfw=sfw,
        sfw_human=human,
        labelled_by="zero_shot",
        style_scores={style.value: value for style, value in scores.styles.items()},
        scope_scores={scope.value: value for scope, value in scores.scopes.items()},
    )


def source_default_labels(
    source: SourceSpec, sfw: SfwScreening | None, human: SfwHumanDecision | None = None
) -> AssetLabels:
    """Fallback when labelling is off: the source's declared style and scopes."""
    scopes = list(source.default_scopes)
    person_count = person_count_for(scopes)
    return AssetLabels(
        primary_style=source.default_style,
        primary_scope=scopes[0],
        secondary_scopes=scopes[1:],
        person_count=person_count,
        person_count_approximate=person_count is not None,
        sfw=sfw,
        sfw_human=human,
        labelled_by="source_default",
    )


def review_state_for(sfw: SfwScreening) -> ReviewState:
    """Anything but a safe screen quarantines at ingestion; the rest awaits review."""
    return ReviewState.UNREVIEWED if sfw.verdict is SfwVerdict.SAFE else ReviewState.QUARANTINED
