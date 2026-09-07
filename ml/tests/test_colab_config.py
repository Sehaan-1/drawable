"""Config validation: the guardrails that keep a run's provenance honest."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from linescout_ml.colab.config import (
    SOURCE_PRESETS,
    PipelineConfig,
    SourceSpec,
    SplitFractions,
    preset_source,
    preset_table,
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
