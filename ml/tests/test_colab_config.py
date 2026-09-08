"""Config validation: the guardrails that keep a run's provenance honest."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from linescout_ml.colab import repro
from linescout_ml.colab.config import (
    SOURCE_PRESETS,
    PipelineConfig,
    SourceSpec,
    SplitFractions,
    preset_source,
    preset_table,
    resolve_sources,
)
from linescout_ml.taxonomy import LineArtOrigin, PrimaryStyle, ScopeLabel


def _source(**overrides: object) -> SourceSpec:
    fields: dict[str, object] = {
        "name": "testset",
        "root": Path("/tmp/sources/testset"),
        "license_id": "CC0-1.0",
    }
    fields.update(overrides)
    return SourceSpec(**fields)  # type: ignore[arg-type]


def _config(**overrides: object) -> PipelineConfig:
    fields: dict[str, object] = {
        "dataset_version": "2026.09.06-test",
        "output_root": Path("/tmp/gallery"),
        "sources": [_source()],
    }
    fields.update(overrides)
    return PipelineConfig(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------- sources


def test_license_id_is_required() -> None:
    with pytest.raises(ValidationError, match="license_id"):
        SourceSpec(name="testset", root=Path("/tmp/x"))  # type: ignore[call-arg]


def test_license_id_cannot_be_blank() -> None:
    with pytest.raises(ValidationError):
        _source(license_id="   ")


def test_native_source_must_not_name_an_extractor() -> None:
    with pytest.raises(ValidationError, match="extractor='none'"):
        _source(origin=LineArtOrigin.NATIVE, extractor="anime2sketch")


def test_extracted_source_must_name_an_extractor() -> None:
    with pytest.raises(ValidationError, match="must name an extractor"):
        _source(origin=LineArtOrigin.EXTRACTED, extractor="none")


def test_query_only_scope_cannot_be_a_default() -> None:
    with pytest.raises(ValidationError, match="query-only"):
        _source(default_scopes=[ScopeLabel.UNKNOWN])


def test_default_scopes_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        _source(default_scopes=[])


def test_source_url_template_is_optional() -> None:
    assert _source().source_url("item-1") is None
    templated = _source(url_template="https://example.test/post/{item_id}")
    assert templated.source_url("42") == "https://example.test/post/42"


def test_source_name_is_a_slug() -> None:
    with pytest.raises(ValidationError):
        _source(name="Not A Slug")


# --------------------------------------------------------------------- splits


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1.0"):
        SplitFractions(train=0.5, validation=0.1, test=0.1, gallery_only=0.1)


def test_default_split_fractions_are_valid() -> None:
    fractions = SplitFractions()
    total = fractions.train + fractions.validation + fractions.test + fractions.gallery_only
    assert total == pytest.approx(1.0)
    assert (fractions.train, fractions.validation, fractions.test, fractions.gallery_only) == (
        0.70,
        0.15,
        0.15,
        0.0,
    )


# --------------------------------------------------------------------- config


def test_dataset_version_must_match_the_manifest_pattern() -> None:
    with pytest.raises(ValidationError):
        _config(dataset_version="v1")


def test_duplicate_source_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        _config(sources=[_source(), _source()])


def test_at_least_one_source_is_required() -> None:
    with pytest.raises(ValidationError):
        _config(sources=[])


def test_paths_derive_from_the_output_root() -> None:
    config = _config(output_root=Path("/tmp/data/gallery/2026.09.06-test"))
    assert config.manifest_path == Path("/tmp/data/gallery/2026.09.06-test/manifest.json")
    assert config.state_dir == Path("/tmp/data/gallery/2026.09.06-test/_pipeline")
    assert config.candidates_path.name == "candidates.jsonl"
    assert config.resolved_embeddings_root() == Path("/tmp/data/gallery/indexes/2026.09.06-test")


def test_embeddings_root_can_be_overridden() -> None:
    config = _config(embeddings_root=Path("/tmp/elsewhere"))
    assert config.resolved_embeddings_root() == Path("/tmp/elsewhere")


def test_defaults_match_the_milestone_2_plan() -> None:
    config = _config()
    assert config.extract_line_art and config.label and config.embed and config.dedupe
    assert config.embedders == ["mobileclip2_s2", "dinov2_vits14"]
    assert config.labeler_model == "MobileCLIP2-S2"
    assert config.min_short_edge == 256  # the manifest's hard floor
    assert config.pipeline_version


def test_config_is_serialisable_for_the_run_report() -> None:
    payload = _config().model_dump(mode="json")
    assert payload["dataset_version"] == "2026.09.06-test"
    assert payload["sources"][0]["license_id"] == "CC0-1.0"
    assert PipelineConfig.model_validate(payload).dataset_version == "2026.09.06-test"


# --------------------------------------------------------------------- presets


def test_every_preset_documents_a_licence_note() -> None:
    """A preset may suggest defaults, but never a licence."""
    for key, preset in SOURCE_PRESETS.items():
        assert preset.license_note.strip(), key
        assert "license_id" not in preset.defaults, key


def test_every_preset_builds_a_valid_source(tmp_path: Path) -> None:
    """The preset table is documentation a notebook user acts on, so all of it must work.

    Regression: ``smithsonian_openaccess`` is longer than the 16-character slug
    pattern allows — the slug is embedded in every ``asset_id`` — so that preset
    needed an explicit short ``slug``.
    """
    for key, preset in SOURCE_PRESETS.items():
        source = preset_source(key, root=tmp_path / key, license_id="cc0-1.0")
        assert source.name == (preset.slug or key), key
        assert len(source.name) <= 16, key
        assert source.license_id == "cc0-1.0"
        assert source.origin.value in {"native_line_art", "extracted_line_art"}


def test_a_long_preset_key_gets_a_short_slug() -> None:
    source = preset_source("smithsonian_openaccess", root=Path("/tmp/si"), license_id="cc0-1.0")
    assert source.name == "smithsonian"
    # An explicit name still wins over the preset slug.
    renamed = preset_source(
        "smithsonian_openaccess", root=Path("/tmp/si"), license_id="cc0-1.0", name="si_oa"
    )
    assert renamed.name == "si_oa"


def test_preset_source_requires_an_explicit_licence() -> None:
    with pytest.raises(TypeError):
        preset_source("manga109", root=Path("/tmp/manga109"))  # type: ignore[call-arg]


def test_preset_source_applies_defaults_and_overrides() -> None:
    source = preset_source(
        "manga109",
        root=Path("/tmp/manga109"),
        license_id="manga109-research",
        extractor="informative_drawings",
    )
    assert source.name == "manga109"
    assert source.default_style is PrimaryStyle.MANGA_ANIME
    assert source.origin is LineArtOrigin.EXTRACTED
    assert source.extractor == "informative_drawings"  # override wins
    assert source.work_grouping == "parent_dir"
    assert source.license_id == "manga109-research"


def test_preset_source_can_rename_the_dataset() -> None:
    source = preset_source(
        "synthetic", root=Path("/tmp/fixture"), license_id="synthetic-fixture", name="fixture_dry"
    )
    assert source.name == "fixture_dry"
    assert source.origin is LineArtOrigin.NATIVE


def test_the_dry_run_preset_is_native_line_art() -> None:
    """The CPU smoke run must not need a GPU or a detector."""
    source = preset_source("synthetic", root=Path("/tmp/fixture"), license_id="synthetic-fixture")
    assert source.extractor == "none"
    assert source.origin is LineArtOrigin.NATIVE


def test_unknown_preset_names_the_alternatives() -> None:
    with pytest.raises(ValueError, match="manga109"):
        preset_source("nope", root=Path("/tmp/x"), license_id="CC0-1.0")


def test_preset_table_is_printable() -> None:
    rows = preset_table()
    assert rows and {"key", "description", "license_note"} <= set(rows[0])


# --------------------------------------------------- presets decide dependencies
#
# These read like dependency tests because the ordering is the point: presets are
# resolved into `SourceSpec` objects *before* anything asks what to install. See
# `test_colab_repro.py` for the installer side of the same contract.


def _resolved(**entry: object) -> SourceSpec:
    (spec,) = resolve_sources([dict(entry)], sources_root=Path("/tmp/linescout-sources"))
    return spec


def test_a_preset_opinion_survives_an_empty_form_field() -> None:
    """Human-Art asks for the classifier; nothing in the run has to remember to.

    The dataset ships no ratings, so `sfw_method` is the preset's call rather than
    the operator's. A plan built from the raw entry would see a blank field,
    install no gate, and then fail — or skip screening — on a T4, forty minutes in.
    """
    human_art = _resolved(preset="human_art", license_id="research-only")
    assert human_art.sfw_method == "opennsfw2"
    assert human_art.requires_nsfw and human_art.uses_extractor

    plan = repro.plan_environment([human_art], spec=repro.load_requirement_spec())
    assert "nsfw" in plan.groups, plan.groups
    assert {"opennsfw2", "gdown"} <= set(plan.spec.pins if plan.spec else ())
    # Everything handed to pip arrives pinned, so the gate is reproducible too.
    assert all("==" in spec for spec in plan.pip_specs), plan.pip_specs


def test_an_explicit_override_replaces_the_presets_opinion_completely() -> None:
    """Two directions, because a half-implemented override is the interesting bug.

    Saying "this source rates its own content" removes the dependency; saying
    "screen this one anyway" adds it to a preset that would not have asked.
    """
    trust_the_source = _resolved(
        preset="human_art", license_id="research-only", sfw_method="source_rating"
    )
    assert not trust_the_source.requires_nsfw

    screen_anyway = _resolved(preset="quickdraw", license_id="CC0-1.0", sfw_method="opennsfw2")
    assert screen_anyway.requires_nsfw

    spec = repro.load_requirement_spec()
    groups_without = set(repro.plan_environment([trust_the_source], spec=spec).groups)
    groups_with = set(repro.plan_environment([screen_anyway], spec=spec).groups)
    assert "nsfw" not in groups_without, groups_without
    assert "nsfw" in groups_with, groups_with


def test_overriding_one_field_does_not_strand_the_rest_of_the_preset() -> None:
    """A spec is resolved or it is not: half an override would lose the origin."""
    swapped = _resolved(preset="human_art", license_id="research-only", extractor="anime2sketch")
    assert swapped.extractor == "anime2sketch"
    assert swapped.origin is LineArtOrigin.EXTRACTED
    assert swapped.uses_extractor and swapped.requires_nsfw
    assert swapped.license_id == "research-only"


# ------------------------------------------------------------- the SFW trust rule
#
# `source_rating` runs no classifier at all, so the honest question about a preset is
# not "is this dataset nice" but "who said so, and can the gallery audit them". Those
# are policy decisions, which is exactly why they are spelled out here: changing one
# should be a reviewable edit to this table, not a quiet default in a dict nobody
# reads.

SFW_POLICY: dict[str, str] = {
    # A publisher's own programme, or a corpus gated behind an application.
    "synthetic": "source_rating",
    "quickdraw": "source_rating",
    "manga109": "source_rating",
    "ebdtheque": "source_rating",
    "met_openaccess": "source_rating",
    "smithsonian_openaccess": "source_rating",
    # Community uploads: the site's rules are enforced by the people posting.
    "amateur_drawings": "source_rating+opennsfw2",
    "safebooru": "source_rating+opennsfw2",
    # Research-only artwork of people, with no per-image rating to inherit.
    "human_art": "opennsfw2",
}


def test_each_presets_sfw_policy_is_a_reviewed_choice() -> None:
    assert set(SFW_POLICY) == set(SOURCE_PRESETS), "a preset appeared with no policy"
    for key, method in SFW_POLICY.items():
        spec = _resolved(preset=key, license_id="reviewed-in-a-test")
        assert spec.sfw_method == method, f"{key} changed its SFW policy to {spec.sfw_method}"
        assert spec.requires_nsfw is (method != "source_rating"), key


def test_a_community_site_rating_tag_is_not_treated_as_a_guarantee() -> None:
    """Safebooru's `rating` tags are user-assigned, and the gate exists to check them.

    Honouring them would make the gallery's SFW screen a copy of a stranger's
    moderation queue — which is fail-open twice over, once when a post is mis-tagged
    and once when the site's definition of "safe" is looser than a product's.
    """
    from linescout_ml.colab import repro

    for key in ("safebooru", "amateur_drawings"):
        assert "opennsfw2" in _resolved(preset=key, license_id="x").sfw_method, key

    spec = repro.load_requirement_spec()
    specs = resolve_sources(
        [{"preset": "safebooru", "license_id": "x"}], sources_root=Path("/tmp/linescout-sources")
    )
    plan = repro.plan_environment(specs, spec=spec)
    assert "nsfw" in plan.groups, plan.groups
    assert "extraction" in plan.groups, "booru art is not line art; it still needs a detector"


def test_turning_off_zero_shot_labels_does_not_turn_off_screening() -> None:
    """`label=False` is a quality knob; treating it as a safety knob is the bug.

    The stage still loads the classifier for sources that need it, so the only thing
    the toggle buys is skipping CLIP — and the config keeps recording which happened.
    """
    config = PipelineConfig(
        dataset_version="2026.09.08-test",
        output_root=Path("/tmp/linescout-label-off"),
        sources=[preset_source("safebooru", root=Path("/tmp/x"), license_id="x")],
        label=False,
    )
    assert config.label is False
    assert any(spec.requires_nsfw for spec in config.sources), "the source still needs screening"
    fields = set(type(config).model_fields)
    assert "label" in fields and "sfw_min_confidence" in fields
    # There is a knob for how strict the classifier is and none for whether the gate
    # runs: skipping screening is a per-source statement (`sfw_method="manual"`), not
    # a global "go faster" flag that a run forgets it set.
    assert not {"sfw", "skip_sfw", "sfw_enabled", "screen_sfw"} & fields
