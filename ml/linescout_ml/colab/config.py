"""Configuration for the Milestone 2 GPU ingestion pipeline.

Everything the Colab notebook decides *before* it touches a GPU lives here:
which sources to read, where the gallery is written, which stages run, and
which models do the work. The runner takes a :class:`PipelineConfig` and never
reads environment variables or module-level globals, so a run is reproducible
from the config alone (it is serialised into the run report).

Two rules from the manifest contract shape this file:

* ``license_id`` has no default. Provenance is the point of the manifest, and a
  silent default would let an unverified licence claim ship in a dataset.
* Splits are derived per *source work* (see :mod:`linescout_ml.colab.sources`),
  never per image, so the "one work, one split" invariant holds by construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from linescout_ml.taxonomy import (
    LineArtOrigin,
    PermissionBasis,
    PrimaryStyle,
    ScopeLabel,
)

#: Line-art extractors we support. Both are served by ``controlnet_aux`` from
#: Hugging Face-hosted weights, so no Google Drive checkpoint hunting:
#:
#: ``anime2sketch``         -> ``LineartAnimeDetector`` (Mukosame/Anime2Sketch ``netG.pth``)
#: ``informative_drawings`` -> ``LineartDetector`` (Chan et al. ``sk_model.pth``)
#: ``none``                 -> the source is already line art (``origin=native_line_art``)
ExtractorKey = Literal["anime2sketch", "informative_drawings", "none"]

#: Feature extractors. ``mobileclip2_s2`` gives the text-aligned embedding used
#: for zero-shot labelling and cross-modal retrieval; ``dinov2_vits14`` gives
#: the self-supervised shape embedding. Milestone 4 concatenates the two.
EmbedderKey = Literal["mobileclip2_s2", "mobileclip2_s0", "dinov2_vits14", "dinov2_vitb14_reg"]

#: How ``sfw`` is decided for a source. ``opennsfw2`` needs TensorFlow, which
#: Colab ships; the others need nothing.
SfwMethod = Literal["source_rating", "opennsfw2", "source_rating+opennsfw2", "manual"]

DEFAULT_IMAGE_PATTERNS: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.gif")

DATASET_VERSION_PATTERN = r"^\d{4}\.\d{2}\.\d{2}(-[a-z0-9]+)?$"

#: A committed revision or a verified digest: 64 lowercase hex characters.
Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class SplitFractions(BaseModel):
    """Share of *source works* assigned to each learning split. Must total 1.0.

    ``unassigned`` works get ``learning_split=none`` (no training assignment)
    while remaining gallery candidates — the v2 replacement for v1's
    ``gallery_only`` split. The default keeps the 70/15/15 learning policy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    train: Annotated[float, Field(ge=0.0, le=1.0)] = 0.70
    validation: Annotated[float, Field(ge=0.0, le=1.0)] = 0.15
    test: Annotated[float, Field(ge=0.0, le=1.0)] = 0.15
    unassigned: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    @model_validator(mode="after")
    def _sums_to_one(self) -> Self:
        total = self.train + self.validation + self.test + self.unassigned
        if abs(total - 1.0) > 1e-6:
            msg = f"split fractions must sum to 1.0, got {total:.4f}"
            raise ValueError(msg)
        return self


class SourceSpec(BaseModel):
    """One raw dataset directory and the provenance claims attached to it.

    Permission fields follow the frozen v2 contract: the operator records what
    the source's terms actually permit. Every use defaults to **denied** and
    ``basis`` defaults to ``unknown`` — unknown permission grants nothing, so
    a source with unverified terms ingests assets that are not servable and
    not trainable until the permission decision is recorded.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    #: Becomes ``source_dataset`` and the slug inside every ``asset_id``.
    name: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_]{1,15}$")]
    #: Directory holding the raw images (a Drive folder in the Colab flow).
    root: Path
    #: Licence identifier recorded verbatim in the manifest. Never defaulted.
    license_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]

    # ------------------------------------------------------------------
    # Permission provenance (v2). Uses stay denied until proven otherwise.
    # ------------------------------------------------------------------
    permission_basis: PermissionBasis = PermissionBasis.UNKNOWN
    permission_url: Annotated[str, StringConstraints(max_length=2048)] | None = None
    attribution: Annotated[str, StringConstraints(max_length=256)] | None = None
    attribution_required: bool = False
    allowed_display: bool = False
    allowed_training: bool = False
    allowed_trace: bool = False

    #: Artist identity as declared by the source, when the whole source
    #: shares one artist. ``None`` = unknown — never invented.
    default_artist_id: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None = None

    #: How the leakage group is derived. ``work`` (default) keeps the
    #: one-work-one-split rule; ``artist`` groups by artist when known;
    #: ``source`` puts the whole source in one group (maximum protection).
    leakage_grouping: Literal["work", "artist", "source"] = "work"

    # Labels written before human curation. The curation UI is expected to
    # correct both; they only have to be valid to pass manifest validation.
    default_style: PrimaryStyle = PrimaryStyle.MANGA_ANIME
    default_scopes: list[ScopeLabel] = Field(
        default_factory=lambda: [ScopeLabel.FULL_BODY], min_length=1
    )

    # Line-art provenance
    origin: LineArtOrigin = LineArtOrigin.EXTRACTED
    extractor: ExtractorKey = "anime2sketch"

    #: What the SFW gate is allowed to trust. ``source_rating`` runs no classifier and
    #: is only defensible when a *publisher's* terms make the guarantee (a museum's
    #: open-access programme, an application-gated research corpus); it is not
    #: defensible for a community site whose rules are enforced by the people posting.
    #: ``opennsfw2`` is the classifier alone, and ``+`` runs both and keeps the
    #: stricter verdict — the right answer for anything scraped, including where the
    #: source claims a rating, because that claim is what is being checked.
    sfw_method: SfwMethod = "source_rating"

    #: Images that share a work must share a split. ``filename`` treats every
    #: file as its own work; ``parent_dir`` groups the files in one folder
    #: (a manga chapter, a sketchbook page, one artist's batch).
    work_grouping: Literal["filename", "parent_dir"] = "filename"

    patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_IMAGE_PATTERNS))
    #: Optional provenance URL template; ``{item_id}`` is substituted.
    url_template: Annotated[str, StringConstraints(max_length=2048)] | None = None

    @model_validator(mode="after")
    def _origin_matches_extractor(self) -> Self:
        if self.origin is LineArtOrigin.NATIVE and self.extractor != "none":
            msg = f"native line-art source {self.name!r} must use extractor='none'"
            raise ValueError(msg)
        if self.origin is LineArtOrigin.EXTRACTED and self.extractor == "none":
            msg = f"extracted source {self.name!r} must name an extractor"
            raise ValueError(msg)
        if ScopeLabel.UNKNOWN in self.default_scopes:
            msg = "'unknown' is a query-only scope and cannot be stored on an asset"
            raise ValueError(msg)
        # Uses require a known permission basis; trace requires display.
        if self.permission_basis is PermissionBasis.UNKNOWN and (
            self.allowed_display or self.allowed_training or self.allowed_trace
        ):
            msg = (
                f"source {self.name!r} claims allowed uses with an unknown permission "
                "basis; record the basis (and where it is documented) first"
            )
            raise ValueError(msg)
        if self.allowed_trace and not self.allowed_display:
            msg = f"source {self.name!r}: trace use requires display use"
            raise ValueError(msg)
        if self.attribution_required and not self.attribution:
            msg = f"source {self.name!r}: attribution_required needs an attribution line"
            raise ValueError(msg)
        if self.leakage_grouping == "artist" and self.default_artist_id is None:
            msg = (
                f"source {self.name!r}: leakage_grouping='artist' needs a "
                "default_artist_id; falling back to work grouping is not silent"
            )
            raise ValueError(msg)
        return self

    def source_url(self, item_id: str) -> str | None:
        if self.url_template is None:
            return None
        return self.url_template.format(item_id=item_id)

    @property
    def uses_extractor(self) -> bool:
        """Whether this source needs a GPU line-art detector loaded."""
        return self.origin is LineArtOrigin.EXTRACTED and self.extractor != "none"

    @property
    def requires_nsfw(self) -> bool:
        """Whether the SFW gate must run a classifier over this source.

        Read from the *resolved* spec on purpose: the Human-Art preset asks for
        ``opennsfw2`` by default, and a run that leaves the form field blank only
        knows that once the preset has been expanded. Dependency preflight and
        the label stage both ask this question, so they can never disagree.
        """
        return self.sfw_method in {"opennsfw2", "source_rating+opennsfw2"}


def _default_embedders() -> list[EmbedderKey]:
    """One text-aligned and one self-supervised encoder: complementary on line art."""
    return ["mobileclip2_s2", "dinov2_vits14"]


class PipelineConfig(BaseModel):
    """A complete, serialisable description of one ingestion run."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    #: Stamped into ``dataset_version``; must match the manifest's pattern.
    dataset_version: Annotated[str, StringConstraints(pattern=DATASET_VERSION_PATTERN)]
    #: Gallery root: ``originals/``, ``line_art/``, ``thumbnails/``, ``manifest.json``.
    output_root: Path
    sources: list[SourceSpec] = Field(min_length=1)

    # Stage switches. Turning one off never breaks the manifest: measurements
    # always run, and labelling/embedding fall back to source defaults.
    extract_line_art: bool = True
    label: bool = True
    embed: bool = True
    dedupe: bool = True

    # Runtime
    device: Literal["auto", "cuda", "cpu"] = "auto"
    batch_size: Annotated[int, Field(ge=1, le=256)] = 8
    #: Cap per source — the notebook uses this for smoke runs on a free T4.
    limit_per_source: Annotated[int, Field(ge=1)] | None = None
    seed: int = 7
    overwrite: bool = False
    splits: SplitFractions = Field(default_factory=SplitFractions)

    # Geometry and measurement
    #: Manifest rejects anything with a short edge under 256 px.
    min_short_edge: Annotated[int, Field(ge=16)] = 256
    thumbnail_size: Annotated[int, Field(ge=32, le=2048)] = 256
    #: Short edge the extractor works at (controlnet_aux rounds to /64).
    line_art_resolution: Annotated[int, Field(ge=256, le=4096)] = 1024
    #: Measurements (ink, text, pHash, quality) run on a view scaled to this edge.
    analysis_edge: Annotated[int, Field(ge=64, le=2048)] = 512

    # Near-duplicate removal
    #: pHash Hamming distance at or below which two candidates are duplicates.
    dedupe_threshold: Annotated[int, Field(ge=0, le=64)] = 6

    # Zero-shot labelling
    labeler_model: str = "MobileCLIP2-S2"
    labeler_pretrained: str = "dfndr2b"
    scope_top_k: Annotated[int, Field(ge=1, le=8)] = 2
    #: A scope is kept only above this softmax probability; the top score is
    #: always kept so ``scopes`` is never empty (the manifest forbids that).
    scope_min_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.15

    # SFW gate
    sfw_min_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9

    # Embeddings
    embedders: list[EmbedderKey] = Field(default_factory=_default_embedders)
    #: Assets per ``.npz`` shard. Small shards survive a Colab disconnect.
    embedding_shard_size: Annotated[int, Field(ge=1, le=100_000)] = 512
    #: Where feature shards land. ``None`` -> ``<output_root>/../indexes/<version>``.
    embeddings_root: Path | None = None

    #: Recorded on every asset so a rebuild is attributable.
    pipeline_version: Annotated[str, StringConstraints(min_length=1, max_length=32)] = "colab-m2-1"
    #: Version of the labeling contract (prompt sets, models, taxonomy) behind
    #: the automated labels. Bump it whenever the labeler changes; automated
    #: labels from a different version are stale, human decisions are not.
    label_version: Annotated[str, StringConstraints(min_length=1, max_length=32)] = "auto-1"

    # Reproducibility. Everything here is *recorded* from the runtime rather than
    # assumed: the notebook fills in the git facts after verifying HEAD, and the
    # digests name the environment and model pins the run actually used.
    #: Repository that supplied the code, e.g. ``junosapollo/drawable``.
    source_repo: Annotated[str, StringConstraints(max_length=128)] | None = None
    #: Full commit SHA the checkout was verified against (never a branch name).
    source_revision: Annotated[str | None, StringConstraints(pattern=r"^[0-9a-f]{40}$")] = None
    #: The tree carried local edits — recorded so a rerun knows it was not the pin.
    source_dirty: bool | None = None
    #: How the code got there, from :func:`linescout_ml.colab.repro.ensure_checkout`.
    source_action: Literal["cloned", "reused", "fetched", "failed"] | None = None
    #: SHA-256 of ``ml/colab/requirements-colab.txt`` as installed.
    environment_sha256: Sha256 | None = None
    #: SHA-256 of ``models.lock.json``, the checkpoint pins this run checked against.
    model_lock_sha256: Sha256 | None = None
    #: ``strict`` refuses any artifact without a pinned digest; ``record`` (the
    #: default) verifies what is pinned and records the rest; ``off`` is debug-only.
    checkpoint_policy: Literal["strict", "record", "off"] = "record"
    #: Where verified weights live. Defaults to ``~/.cache/linescout/checkpoints``.
    checkpoint_cache_dir: Path | None = None
    #: Set to verify against a different lock (tests, or a fork of the pins).
    model_lock_path: Path | None = None

    @model_validator(mode="after")
    def _unique_source_names(self) -> Self:
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            msg = f"source names must be unique, got {names}"
            raise ValueError(msg)
        return self

    @property
    def state_dir(self) -> Path:
        """Scratch dir for candidate state and the run report."""
        return self.output_root / "_pipeline"

    @property
    def candidates_path(self) -> Path:
        return self.state_dir / "candidates.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "manifest.json"

    def source_provenance(self) -> dict[str, Any] | None:
        """The checkout facts to record, or ``None`` when nothing was verified.

        ``None`` is deliberately distinguishable from "clean at an unknown
        revision": a report that says nothing and a report that says "we did not
        look" should not look the same to whoever audits a dataset later.
        """
        recorded = (
            self.source_repo,
            self.source_revision,
            self.source_dirty,
            self.source_action,
        )
        if all(value is None for value in recorded):
            return None
        return {
            "repo": self.source_repo,
            "revision": self.source_revision,
            "dirty": self.source_dirty,
            "action": self.source_action,
            #: HEAD was compared against the pin this run asked for.
            "verified": self.source_revision is not None,
        }

    def resolved_embeddings_root(self) -> Path:
        if self.embeddings_root is not None:
            return self.embeddings_root
        return self.output_root.parent / "indexes" / self.dataset_version


# --------------------------------------------------------------------------- presets


@dataclass(frozen=True)
class SourcePreset:
    """Sensible defaults for one of the datasets named in the project README.

    A preset deliberately does *not* carry a ``license_id``: the licence is a
    claim about someone else's data and has to be confirmed by the operator, so
    :func:`preset_source` requires it explicitly. ``license_note`` records what
    to check, and the notebook prints it before a run starts.
    """

    key: str
    description: str
    license_note: str
    defaults: dict[str, Any]
    #: Slug used in ``asset_id`` and ``source_dataset`` when the preset key is
    #: longer than :data:`SourceSpec.name` allows (16 characters, because the slug
    #: is embedded in every asset id).
    slug: str | None = None


SOURCE_PRESETS: dict[str, SourcePreset] = {
    preset.key: preset
    for preset in (
        SourcePreset(
            key="synthetic",
            description="the committed fixture in ml/fixtures/synthetic — CPU dry run",
            license_note="generated by this repository; no third-party terms",
            defaults={
                "default_style": PrimaryStyle.GESTURE_SKETCH,
                "default_scopes": [ScopeLabel.FULL_BODY, ScopeLabel.FACE_HEAD],
                "origin": LineArtOrigin.NATIVE,
                "extractor": "none",
                "sfw_method": "source_rating",
                "work_grouping": "filename",
                "patterns": ["*.png"],
            },
        ),
        SourcePreset(
            key="quickdraw",
            description="Quick, Draw! doodles (stroke .npy files must be rendered to PNG first)",
            license_note="Google terms for the Quick, Draw! dataset; confirm before redistributing",
            defaults={
                "default_style": PrimaryStyle.GESTURE_SKETCH,
                "default_scopes": [ScopeLabel.FULL_BODY],
                "origin": LineArtOrigin.NATIVE,
                "extractor": "none",
                "sfw_method": "source_rating",
                "work_grouping": "filename",
                "patterns": ["*.png"],
            },
        ),
        SourcePreset(
            key="amateur_drawings",
            description="Amateur Drawings (Informative Drawings) — 1,338 community sketch pages",
            license_note="informative-drawings: MIT code, dataset terms on the project page",
            defaults={
                "default_style": PrimaryStyle.GESTURE_SKETCH,
                "default_scopes": [ScopeLabel.FULL_BODY],
                "origin": LineArtOrigin.NATIVE,
                "extractor": "none",
                "sfw_method": "source_rating+opennsfw2",
                "work_grouping": "parent_dir",
            },
        ),
        SourcePreset(
            key="human_art",
            description="Human-Art — photographs and artwork of people across styles",
            license_note="research-only terms; apply for access and cite the paper",
            defaults={
                "default_style": PrimaryStyle.REALISTIC_ACADEMIC,
                "default_scopes": [ScopeLabel.FULL_BODY, ScopeLabel.UPPER_BODY_CLOTHING],
                "origin": LineArtOrigin.EXTRACTED,
                "extractor": "informative_drawings",
                "sfw_method": "opennsfw2",
                "work_grouping": "parent_dir",
            },
        ),
        SourcePreset(
            key="manga109",
            description="Manga109 — 109 professionally drawn manga (access application required)",
            license_note="Manga109 requires an application and forbids redistribution; start early",
            defaults={
                "default_style": PrimaryStyle.MANGA_ANIME,
                "default_scopes": [ScopeLabel.FACE_HEAD, ScopeLabel.FULL_BODY],
                "origin": LineArtOrigin.EXTRACTED,
                "extractor": "anime2sketch",
                "sfw_method": "source_rating",
                "work_grouping": "parent_dir",
            },
        ),
        SourcePreset(
            key="ebdtheque",
            description="eBDtheque — panels from Franco-Belgian bande dessinée",
            license_note="access application; per-album terms, no redistribution of scans",
            defaults={
                "default_style": PrimaryStyle.WESTERN_INK,
                "default_scopes": [ScopeLabel.FULL_BODY, ScopeLabel.UPPER_BODY_CLOTHING],
                "origin": LineArtOrigin.EXTRACTED,
                "extractor": "anime2sketch",
                "sfw_method": "source_rating",
                "work_grouping": "parent_dir",
            },
        ),
        SourcePreset(
            key="safebooru",
            description="Safebooru — a booru whose posts carry user-assigned rating tags",
            license_note="per-post artist licences vary; record the post licence, not the site's",
            defaults={
                "default_style": PrimaryStyle.MANGA_ANIME,
                "default_scopes": [ScopeLabel.FULL_BODY, ScopeLabel.FACE_HEAD],
                "origin": LineArtOrigin.EXTRACTED,
                "extractor": "anime2sketch",
                "sfw_method": "source_rating+opennsfw2",
                "work_grouping": "filename",
                "url_template": "https://safebooru.org/index.php?page=post&s=view&id={item_id}",
            },
        ),
        SourcePreset(
            key="met_openaccess",
            description="The Met Open Access — public-domain drawings, prints, and sculptures",
            license_note="CC0 for Open Access works; check the per-object rights statement",
            defaults={
                "default_style": PrimaryStyle.REALISTIC_ACADEMIC,
                "default_scopes": [ScopeLabel.FULL_BODY, ScopeLabel.FACE_HEAD],
                "origin": LineArtOrigin.EXTRACTED,
                "extractor": "informative_drawings",
                "sfw_method": "source_rating",
                "work_grouping": "filename",
            },
        ),
        SourcePreset(
            key="smithsonian_openaccess",
            slug="smithsonian",
            description="Smithsonian Open Access — public-domain scans across the institutions",
            license_note="CC0 for Open Access assets; verify per-record metadata",
            defaults={
                "default_style": PrimaryStyle.REALISTIC_ACADEMIC,
                "default_scopes": [ScopeLabel.FULL_BODY],
                "origin": LineArtOrigin.EXTRACTED,
                "extractor": "informative_drawings",
                "sfw_method": "source_rating",
                "work_grouping": "filename",
            },
        ),
    )
}


def preset_source(
    key: str,
    *,
    root: Path,
    license_id: str,
    name: str | None = None,
    **overrides: Any,
) -> SourceSpec:
    """Build a :class:`SourceSpec` from a preset, with the licence supplied by you.

    ``overrides`` are applied last, so any preset field can be tuned per run::

        preset_source("manga109", root=drive / "manga109", license_id="manga109-research",
                      extractor="informative_drawings")
    """
    try:
        preset = SOURCE_PRESETS[key]
    except KeyError:
        msg = f"unknown source preset {key!r}; available: {sorted(SOURCE_PRESETS)}"
        raise ValueError(msg) from None
    fields = dict(preset.defaults)
    fields.update(overrides)
    return SourceSpec(name=name or preset.slug or key, root=root, license_id=license_id, **fields)


#: Licence values that mean "nobody verified anything". The manifest is a
#: provenance document, so a placeholder stops the run instead of being written
#: into a dataset that outlives it.
PLACEHOLDER_LICENSES: frozenset[str] = frozenset(
    {
        "-",
        "changeme",
        "fixme",
        "n/a",
        "na",
        "none",
        "null",
        "placeholder",
        "proprietary",
        "replace",
        "replace-me",
        "replace_me",
        "tbd",
        "todo",
        "unknown",
        "unverified",
    }
)

#: Which ``SOURCES`` form fields a run may override on a preset. Anything left
#: blank keeps the preset's own value; that is why the preset has to be expanded
#: before the runtime can know what it needs.
FORM_OVERRIDE_FIELDS: tuple[str, ...] = ("extractor", "sfw_method", "work_grouping")


class SourceConfigurationError(ValueError):
    """A form entry cannot be turned into a valid :class:`SourceSpec`."""


def spec_from_entry(
    entry: Mapping[str, Any],
    *,
    sources_root: Path | None = None,
    root: Path | None = None,
) -> SourceSpec:
    """Expand one ``SOURCES`` entry (preset + licence + overrides) into a spec.

    Lives in the package rather than in a notebook cell because two things need
    the same answer *before* any heavy work: the dependency preflight, which has
    to know that the Human-Art preset asks for an NSFW classifier, and the
    runner, which has to agree about the extractor. Resolving twice, in two
    languages, is how those two drift apart.
    """
    preset = str(entry.get("preset", "")).strip()
    if not preset:
        msg = f"every SOURCES entry needs a 'preset' key; got {entry}"
        raise SourceConfigurationError(msg)
    licence = str(entry.get("license_id", "")).strip()
    if licence.lower() in PLACEHOLDER_LICENSES:
        msg = (
            f"source '{preset}' needs a real license_id. The manifest is a provenance\n"
            "document: record the licence you were actually granted (for example\n"
            "'CC0-1.0', 'cc-by-nc-4.0', 'manga109-research-only'), not a placeholder."
        )
        raise SourceConfigurationError(msg)
    overrides = {
        field: str(entry[field])
        for field in FORM_OVERRIDE_FIELDS
        if str(entry.get(field, "")).strip()
    }
    folder = root or _entry_root(entry, preset, sources_root=sources_root)
    try:
        return preset_source(
            preset, root=folder, license_id=licence, name=entry.get("name"), **overrides
        )
    except ValidationError as error:
        # For example: an extractor override on a native-line-art source.
        msg = f"source '{preset}' is not a valid combination:\n{error}"
        raise SourceConfigurationError(msg) from error
    except ValueError as error:
        raise SourceConfigurationError(str(error)) from error


def _entry_root(entry: Mapping[str, Any], preset: str, *, sources_root: Path | None) -> Path:
    """An absolute ``root`` beats ``folder``, which beats the preset's own name."""
    explicit = str(entry.get("root", "")).strip()
    if explicit:
        return Path(explicit)
    if sources_root is None:
        msg = (
            f"source '{preset}' needs either a 'root' or a sources_root to resolve 'folder' against"
        )
        raise SourceConfigurationError(msg)
    return Path(sources_root) / str(entry.get("folder") or preset)


def resolve_sources(
    entries: Sequence[Mapping[str, Any]],
    *,
    sources_root: Path | None = None,
) -> list[SourceSpec]:
    """Resolve every form entry into complete specs, all errors reported at once.

    Called before any package is installed: the plan for what to install is a
    function of these objects, so an unresolved preset cannot be a reason to
    skip a dependency.
    """
    problems: list[str] = []
    specs: list[SourceSpec] = []
    for entry in entries:
        try:
            specs.append(spec_from_entry(entry, sources_root=sources_root))
        except SourceConfigurationError as error:
            problems.append(str(error))
    if problems:
        raise SourceConfigurationError("\n\n".join(problems))
    if not specs:
        msg = "SOURCES is empty — add at least one dataset to process."
        raise SourceConfigurationError(msg)
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        msg = f"source names must be unique, got {names}"
        raise SourceConfigurationError(msg)
    return specs


def preset_table() -> list[dict[str, str]]:
    """Preset key, description, and licence note — the notebook prints this."""
    return [
        {"key": preset.key, "description": preset.description, "license_note": preset.license_note}
        for preset in SOURCE_PRESETS.values()
    ]
