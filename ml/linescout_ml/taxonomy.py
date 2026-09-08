"""Canonical LineScout taxonomy.

These enums are the single source of truth for scope, style, line-art origin,
learning assignment, review state, blockers, SFW decisions, and permission
basis. The API re-exports them for its OpenAPI schema, and
``packages/contracts`` mirrors them for the frontend, so any change here must
be followed by regenerating the TypeScript contracts.

Schema v2 (the frozen contract) made these splits explicit:

* :class:`LearningSplit` is the *learning* assignment only. Gallery and gold
  membership are separate booleans on the manifest record, so an asset can be
  ``train`` **and** served in the gallery, or ``none`` and gold.
* :class:`CurationBlocker` replaces the free-floating ``malformed_anatomy`` /
  ``poor_extraction`` booleans with named blockers whose use effects are
  defined by the contract (see ``docs/contracts/manifest-v2.md``).
* :class:`PermissionBasis` records *why* an asset's allowed uses are claimed.
  ``unknown`` is the default and grants nothing.
* SFW decisions split into an automated :class:`SfwVerdict` screen and a
  separate human approval recorded on the manifest record.
"""

from __future__ import annotations

from enum import StrEnum


class ScopeLabel(StrEnum):
    """What part of a character (or how many characters) an image depicts.

    An asset carries exactly one ``primary_scope`` plus any number of
    ``secondary_scopes``; queries return one probability per label plus
    ``unknown``.

    ``eyebrow`` and ``mouth`` are retained as first-class detail scopes, and
    ``person_count`` stays nullable on the record (null = not applicable or
    not assessed).
    """

    EYE = "eye"
    EYEBROW = "eyebrow"
    MOUTH = "mouth"
    FACE_HEAD = "face_head"
    HAIR = "hair"
    HAND = "hand"
    FOOT = "foot"
    UPPER_BODY_CLOTHING = "upper_body_clothing"
    FULL_BODY = "full_body"
    MULTI_CHARACTER = "multi_character"
    UNKNOWN = "unknown"


class PrimaryStyle(StrEnum):
    """Exactly one primary style per asset. Also the five reference-panel rows."""

    MANGA_ANIME = "manga_anime"
    WESTERN_INK = "western_ink"
    REALISTIC_ACADEMIC = "realistic_academic"
    CARTOON = "cartoon"
    GESTURE_SKETCH = "gesture_sketch"


class LineArtOrigin(StrEnum):
    """Whether the visible reference is native line art or machine-extracted."""

    NATIVE = "native_line_art"
    EXTRACTED = "extracted_line_art"


class LearningSplit(StrEnum):
    """Learning assignment for an asset. Gallery/gold membership is separate.

    ``none`` means *no training assignment*: the asset is excluded from every
    learning population. It is not a conflict in the one-group-one-split rule
    (only two different assigned splits on the same work or leakage group
    are). The default derivation policy keeps 70/15/15 over assigned works.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    NONE = "none"


class ReviewState(StrEnum):
    """Human curation state for an asset (definitions are part of the contract).

    * ``unreviewed`` — no human decision recorded. Not servable, not trainable.
    * ``accepted``   — a human approved the asset's labels. Servable iff the
      other gates pass (display permission, gallery membership, human SFW
      approval, no blockers); trainable iff training permission and split allow.
    * ``rejected``   — terminal human verdict that the asset is unsuitable.
      Never served, never trained on as a positive example; retained for audit
      and (only where training use is permitted) as a negative example for
      quality models.
    * ``quarantined``— reversible hold pending investigation (SFW concern,
      provenance concern, extraction rework). All uses blocked while held.
    """

    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class CurationBlocker(StrEnum):
    """A named defect that blocks specific uses regardless of review state.

    Blockers are not review states; they are assertions about the asset that
    outlive individual decisions. While any blocker is present the asset is
    neither servable nor trainable (see ``is_servable`` / ``is_trainable`` in
    :mod:`linescout_ml.manifest`). The curation API refuses a ``keep`` decision
    that carries blockers — blockers force ``reject`` or ``quarantine``.

    Adding a new blocker kind is a schema change: extend this enum, bump the
    manifest schema version, and migrate the ``blockers_json`` columns.
    """

    ANATOMY = "anatomy"
    EXTRACTION = "extraction"


class SfwVerdict(StrEnum):
    """Tri-state result of an automated SFW screen.

    ``unsure`` is distinct from ``unsafe``: borderline confidence is a concern
    to resolve, not a condemnation. Both ``unsafe`` and ``unsure`` keep an
    asset out of the unreviewed queue at ingestion (quarantined instead), and
    neither ever substitutes for human approval.
    """

    SAFE = "safe"
    UNSAFE = "unsafe"
    UNSURE = "unsure"


class SfwScreeningMethod(StrEnum):
    """How an automated SFW screen was produced.

    ``manual`` is deliberately absent: a human SFW decision is recorded
    separately on the manifest record (``sfw_human``), never as a screening
    method.
    """

    NONE = "none"
    SOURCE_RATING = "source_rating"
    OPENNSFW2 = "opennsfw2"
    SOURCE_RATING_OPENNSFW2 = "source_rating+opennsfw2"


class PermissionBasis(StrEnum):
    """Why the asset's allowed uses may be claimed.

    * ``license_terms``    — the recorded licence's terms grant the use.
    * ``public_domain``    — no rights reserved (verified dedication).
    * ``explicit_consent`` — the artist/source gave recorded permission.
    * ``first_party``      — content produced by this project itself.
    * ``unknown``          — the default. Grants *nothing*: unknown permission
      must not imply permission.
    """

    LICENSE_TERMS = "license_terms"
    PUBLIC_DOMAIN = "public_domain"
    EXPLICIT_CONSENT = "explicit_consent"
    FIRST_PARTY = "first_party"
    UNKNOWN = "unknown"


#: Fixed style-row order for new preference profiles (spec §3, "Reference panel").
DEFAULT_STYLE_ORDER: tuple[PrimaryStyle, ...] = (
    PrimaryStyle.MANGA_ANIME,
    PrimaryStyle.REALISTIC_ACADEMIC,
    PrimaryStyle.WESTERN_INK,
    PrimaryStyle.CARTOON,
    PrimaryStyle.GESTURE_SKETCH,
)

#: Scope labels that a *gallery* asset may carry as secondary scopes.
#: ``unknown`` is query-only and may never be a secondary scope.
GALLERY_SCOPES: frozenset[ScopeLabel] = frozenset(ScopeLabel) - {ScopeLabel.UNKNOWN}

#: Values a ``primary_scope`` may take. Unlike secondaries, ``unknown`` is a
#: legal *provisional* primary: it records "no scope confidently determined"
#: instead of inventing a label, and must be resolved before acceptance.
PRIMARY_SCOPES: frozenset[ScopeLabel] = frozenset(ScopeLabel)
