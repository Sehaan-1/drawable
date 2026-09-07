"""End-to-end CPU dry runs of the Colab pipeline.

These are the tests that stand in for a GPU session: a native line-art source
needs no detector, and the labelling/embedding/SFW stages are exercised through
stubs, so the whole path — discovery, extraction, measurement, dedupe, labels,
embeddings, manifest, export — runs in CI in a second or two.

Everything a real run must guarantee is asserted here, most importantly that the
manifest the pipeline writes is the manifest the API will load: it validates,
its files exist, and ``linescout-manifest validate --require-files`` accepts it.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from colab_images import solid, source_tree, write_png
from PIL import Image

from linescout_ml.cli import main as cli_main
from linescout_ml.colab.assets import missing_files, original_suffix_for, read_manifest
from linescout_ml.colab.config import PipelineConfig, SourceSpec
from linescout_ml.colab.embed import EmbeddingStore, EmbeddingStoreError
from linescout_ml.colab.export import zip_gallery
from linescout_ml.colab.label import LabelScores, OpenNsfw2Classifier, ZeroShotLabeler
from linescout_ml.colab.models import MODEL_CARDS
from linescout_ml.colab.runner import PipelineError, PipelineRunner, candidate_summary
from linescout_ml.manifest import Manifest, ManifestRecord, SfwDecision
from linescout_ml.taxonomy import PrimaryStyle, ReviewState, ScopeLabel

# --------------------------------------------------------------------- fixtures


def _source(root: Path, **overrides: Any) -> SourceSpec:
    fields: dict[str, Any] = {
        "name": "sketches",
        "root": root,
        "license_id": "CC0-1.0",
        "origin": "native_line_art",
        "extractor": "none",
        "default_style": PrimaryStyle.GESTURE_SKETCH,
        "default_scopes": [ScopeLabel.FULL_BODY],
    }
    fields.update(overrides)
    return SourceSpec(**fields)


def _config(root: Path, **overrides: Any) -> PipelineConfig:
    """A CPU-only config: no detector, no CLIP, no embeddings unless a test asks."""
    sources_root = root / "sources"
    fields: dict[str, Any] = {
        "dataset_version": "2026.09.06-dryrun",
        "output_root": root / "gallery",
        "sources": [_source(sources_root)],
        "label": False,
        "embed": False,
        "device": "cpu",
        "line_art_resolution": 512,
        "analysis_edge": 256,
        "thumbnail_size": 128,
    }
    fields.update(overrides)
    return PipelineConfig(**fields)


@pytest.fixture
def dry_run(tmp_path: Path) -> tuple[PipelineConfig, PipelineRunner]:
    """A finished CPU run over six generated drawings."""
    source_tree(tmp_path / "sources", count=6, size=384)
    config = _config(tmp_path)
    runner = PipelineRunner(config)
    runner.run_all()
    return config, runner


# --------------------------------------------------------------------- dry run


def test_dry_run_writes_a_complete_gallery(dry_run: tuple[PipelineConfig, PipelineRunner]) -> None:
    config, runner = dry_run
    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    assert len(manifest.records) == 6
    assert missing_files(manifest, config.output_root) == []

    for record in manifest.records:
        original = config.output_root / record.original_path
        line_art_path = config.output_root / record.line_art_path
        thumbnail = config.output_root / record.thumbnail_path
        assert original.is_file() and line_art_path.is_file() and thumbnail.is_file()
        with Image.open(thumbnail) as tile:
            assert tile.size == (config.thumbnail_size, config.thumbnail_size)


def test_dry_run_records_are_uncurated_and_disabled(
    dry_run: tuple[PipelineConfig, PipelineRunner],
) -> None:
    """Assets start disabled; production search only serves accepted records."""
    config, _ = dry_run
    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    for record in manifest.records:
        assert record.enabled is False
        assert record.review.state is ReviewState.UNREVIEWED
        assert record.review.quality is None
        assert record.sfw.safe is True
        assert record.sfw.method == "source_rating"
        assert record.origin.value == "native_line_art"
        assert record.extraction_model is None  # native assets must not claim an extractor
        assert record.license_id == "CC0-1.0"
        assert record.pipeline_version == config.pipeline_version


def test_dry_run_manifest_passes_the_project_cli(
    dry_run: tuple[PipelineConfig, PipelineRunner], capsys: pytest.CaptureFixture[str]
) -> None:
    config, _ = dry_run
    assert cli_main(["validate", str(config.manifest_path), "--require-files"]) == 0
    assert "OK" in capsys.readouterr().out


def test_dry_run_reports_stages_and_summary(
    dry_run: tuple[PipelineConfig, PipelineRunner],
) -> None:
    config, runner = dry_run
    report = runner.report()
    assert report["summary"]["total"] == 6
    assert report["summary"]["candidates"]["active"] == 6
    assert {stage["name"] for stage in report["stages"]} >= {
        "discover",
        "extract",
        "measure",
        "dedupe",
        "label",
        "embed",
        "build",
        "export",
    }
    assert report["config"]["dataset_version"] == config.dataset_version
    assert "cuda_available" in report["gpu"]


def test_measurements_land_on_every_record(
    dry_run: tuple[PipelineConfig, PipelineRunner],
) -> None:
    config, _ = dry_run
    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    for record in manifest.records:
        assert len(record.phash) == 16
        assert 0.0 < record.ink_coverage < 1.0
        assert record.text_coverage == 0.0  # generated drawings have no glyph runs
        assert 0.0 <= record.quality_score <= 1.0
        assert record.width == record.height == 384
        assert len(record.source_checksum) == 64
        assert len(record.line_art_checksum) == 64
        assert len(record.thumbnail_checksum) == 64


def test_original_suffix_preserves_source_extension() -> None:
    assert original_suffix_for("scan.JPEG") == ".jpg"
    assert original_suffix_for("art.png") == ".png"
    assert original_suffix_for("photo.webp") == ".webp"
    assert original_suffix_for("noext") == ".png"


def test_jpeg_original_is_copied_byte_for_byte(tmp_path: Path) -> None:
    """Source bytes stay intact; JPEG is not re-encoded as PNG."""
    root = tmp_path / "sources"
    root.mkdir()
    jpeg_path = root / "scan.jpeg"
    Image.new("RGB", (320, 320), (240, 240, 240)).save(jpeg_path, format="JPEG", quality=85)
    original_bytes = jpeg_path.read_bytes()

    runner = PipelineRunner(_config(tmp_path))
    runner.run_all()
    manifest = read_manifest(runner.config.manifest_path)
    assert manifest is not None
    assert len(manifest.records) == 1
    record = manifest.records[0]
    assert record.original_path.endswith(".jpg")
    copied = (runner.config.output_root / record.original_path).read_bytes()
    assert copied == original_bytes
    assert record.source_checksum != record.line_art_checksum


def test_native_line_art_is_normalised_not_redrawn(
    dry_run: tuple[PipelineConfig, PipelineRunner],
) -> None:
    config, _ = dry_run
    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    record = manifest.records[0]
    with Image.open(config.output_root / record.line_art_path) as extracted:
        assert extracted.mode == "L"
        assert extracted.size == (record.width, record.height)


# --------------------------------------------------------------------- resume


def test_rerunning_a_finished_pipeline_changes_nothing(
    dry_run: tuple[PipelineConfig, PipelineRunner],
) -> None:
    config, runner = dry_run
    before = read_manifest(config.manifest_path)
    assert before is not None
    first_hash = before.content_hash()
    zip_before = config.output_root.parent / "first.zip"
    zip_gallery(config.output_root, before, zip_before)

    second = PipelineRunner(config)
    second.run_all()
    after = read_manifest(config.manifest_path)
    assert after is not None
    assert after.content_hash() == first_hash
    assert after.records == before.records

    extract = next(stage for stage in second.stages if stage.name == "extract")
    assert extract.processed == 0
    assert extract.skipped == 6


def test_resume_reloads_candidate_state(tmp_path: Path) -> None:
    source_tree(tmp_path / "sources", count=4, size=320)
    config = _config(tmp_path)
    first = PipelineRunner(config)
    first.discover()
    first.run_extract()
    first.run_measure()

    resumed = PipelineRunner(config)
    stage = resumed.discover()
    assert "resumed 4 candidates" in " ".join(stage.notes)
    assert all(candidate.measurements is not None for candidate in resumed.store)


def test_a_half_finished_run_picks_up_where_it_stopped(tmp_path: Path) -> None:
    source_tree(tmp_path / "sources", count=4, size=320)
    config = _config(tmp_path)
    runner = PipelineRunner(config)
    runner.discover()
    runner.run_extract()
    # Simulate a disconnect: measure never ran.
    partial = PipelineRunner(config)
    partial.discover()
    extract = partial.run_extract()
    assert extract.processed == 0 and extract.skipped == 4
    measure = partial.run_measure()
    assert measure.processed == 4
    partial.run_label()
    assert partial.run_build().records


# --------------------------------------------------------------------- filtering


def test_undersized_images_are_skipped_not_written(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    source_tree(root, count=2, size=384)
    write_png(root / "tiny.png", solid(64, 20))
    config = _config(tmp_path)
    runner = PipelineRunner(config)
    runner.run_all()

    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    assert len(manifest.records) == 2
    summary = candidate_summary(runner.store)
    assert summary["skipped"] == {"too_small": 1}
    skipped = [candidate for candidate in runner.store if candidate.skip_reason]
    assert skipped and "too_small" in str(skipped[0].skip_reason)
    assert len(list((config.output_root / "line_art").glob("*.png"))) == 2


def test_undecodable_files_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    source_tree(root, count=2, size=320)
    (root / "broken.png").write_bytes(b"definitely not a png")
    runner = PipelineRunner(_config(tmp_path))
    runner.run_all()
    manifest = read_manifest(runner.config.manifest_path)
    assert manifest is not None
    assert len(manifest.records) == 2
    assert any("could not decode" in str(c.skip_reason) for c in runner.store if c.skip_reason)


def test_near_duplicates_are_dropped_from_the_manifest(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    source_tree(root, count=4, size=384, duplicate_pairs=2)
    config = _config(tmp_path)
    runner = PipelineRunner(config)
    runner.run_all()

    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    assert len(manifest.records) == 4  # two of the six files were exact copies
    summary = candidate_summary(runner.store)
    assert summary["duplicates"] == 2
    dedupe = next(stage for stage in runner.stages if stage.name == "dedupe")
    assert dedupe.processed == 2


def test_dedupe_can_be_disabled(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    source_tree(root, count=2, size=384, duplicate_pairs=2)
    config = _config(tmp_path, dedupe=False)
    runner = PipelineRunner(config)
    runner.discover()
    runner.run_extract()
    runner.run_measure()
    stage = runner.run_dedupe()
    assert "disabled by config" in stage.notes
    runner.run_label()
    assert len(runner.run_build().records) == 4  # copies survive when dedupe is off


# --------------------------------------------------------------------- export


def test_zip_contains_exactly_what_the_manifest_vouches_for(
    dry_run: tuple[PipelineConfig, PipelineRunner],
) -> None:
    config, runner = dry_run
    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    archive = config.output_root.parent / "gallery.zip"
    report_path = config.state_dir / "run_report.json"
    zip_gallery(config.output_root, manifest, archive, extra=[report_path])
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    expected = {
        relative
        for record in manifest.records
        for relative in (record.original_path, record.line_art_path, record.thumbnail_path)
    }
    assert expected <= names
    assert names - expected == {"manifest.json", "run_report.json"}


def test_duplicates_and_skips_stay_out_of_the_zip(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    source_tree(root, count=3, size=384, duplicate_pairs=1)
    write_png(root / "tiny.png", solid(64, 20))
    config = _config(tmp_path)
    runner = PipelineRunner(config)
    archive = tmp_path / "out.zip"
    runner.run_all(zip_path=archive)

    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    kept = {record.asset_id for record in manifest.records}
    assert len(kept) == 3
    for name in names:
        if name.endswith(".png"):
            stem = Path(name).stem
            assert stem in kept, f"{name} should not be in the archive"


def test_run_export_writes_a_report_next_to_the_gallery(
    dry_run: tuple[PipelineConfig, PipelineRunner],
) -> None:
    config, runner = dry_run
    outputs = runner.run_export(zip_path=config.output_root.parent / "again.zip")
    report_path = Path(str(outputs["run_report"]))
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 6
    assert payload["outputs"]["zip"]["sha256"]
    assert payload["outputs"]["zip"]["bytes"] > 0


def test_drive_export_mirrors_the_gallery(tmp_path: Path) -> None:
    source_tree(tmp_path / "sources", count=2, size=320)
    config = _config(tmp_path)
    runner = PipelineRunner(config)
    runner.run_all()
    drive = tmp_path / "drive" / "LineScout"
    outputs = runner.run_export(drive_root=drive)
    destination = Path(str(outputs["drive"]["path"]))
    assert destination == drive / config.dataset_version
    assert (destination / "manifest.json").is_file()
    assert (destination / "thumbnails").is_dir()


def test_colab_download_is_a_no_op_off_colab(
    dry_run: tuple[PipelineConfig, PipelineRunner],
) -> None:
    config, runner = dry_run
    archive = config.output_root.parent / "download.zip"
    outputs = runner.run_export(zip_path=archive, download=True)
    assert outputs["zip"]["path"] == str(archive)


# --------------------------------------------------------------------- merging


def test_a_second_run_patches_the_existing_manifest(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    source_tree(root, count=3, size=320)
    config = _config(tmp_path)
    first = PipelineRunner(config)
    first.run_all()
    before = read_manifest(config.manifest_path)
    assert before is not None
    assert len(before.records) == 3

    source_tree(root, count=5, size=320)  # two new drawings appear
    second = PipelineRunner(config)
    second.run_all()
    after = read_manifest(config.manifest_path)
    assert after is not None
    assert len(after.records) == 5
    assert {record.asset_id for record in before.records} <= {
        record.asset_id for record in after.records
    }
    build = next(stage for stage in second.stages if stage.name == "build")
    assert any("merged into 3 existing records" in note for note in build.notes)


def test_overwrite_rebuilds_from_scratch(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    source_tree(root, count=2, size=320)
    config = _config(tmp_path)
    PipelineRunner(config).run_all()
    fresh = PipelineRunner(_config(tmp_path, overwrite=True))
    fresh.run_all()
    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    assert len(manifest.records) == 2


def test_a_broken_manifest_is_reported_not_overwritten(tmp_path: Path) -> None:
    source_tree(tmp_path / "sources", count=2, size=320)
    config = _config(tmp_path)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text("{not json", encoding="utf-8")
    runner = PipelineRunner(config)
    runner.discover()
    runner.run_extract()
    runner.run_measure()
    runner.run_label()
    with pytest.raises(Exception, match="unreadable|invalid"):
        runner.run_build()
    assert config.manifest_path.read_text(encoding="utf-8") == "{not json"


# --------------------------------------------------------------------- labels


def _stub_labeler(monkeypatch: pytest.MonkeyPatch, scores: LabelScores) -> None:
    class StubLabeler:
        encoder = SimpleNamespace(card=SimpleNamespace(name="StubCLIP"))

        def score_batch(self, images: Sequence[Image.Image]) -> list[LabelScores]:
            return [scores] * len(images)

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        ZeroShotLabeler, "load", classmethod(lambda cls, *args, **kwargs: StubLabeler())
    )


def test_zero_shot_labels_reach_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_tree(tmp_path / "sources", count=3, size=320)
    styles = {style: 0.075 for style in PrimaryStyle}
    styles[PrimaryStyle.MANGA_ANIME] = 0.7
    scores = LabelScores(
        styles=styles,
        scopes={
            ScopeLabel.FACE_HEAD: 0.6,
            ScopeLabel.HAIR: 0.25,
            ScopeLabel.EYE: 0.05,
            ScopeLabel.EYEBROW: 0.04,
            ScopeLabel.MOUTH: 0.03,
            ScopeLabel.HAND: 0.03,
            ScopeLabel.FOOT: 0.02,
            ScopeLabel.UPPER_BODY_CLOTHING: 0.02,
            ScopeLabel.FULL_BODY: 0.02,
            ScopeLabel.MULTI_CHARACTER: 0.01,
        },
    )
    _stub_labeler(monkeypatch, scores)
    config = _config(tmp_path, label=True)
    runner = PipelineRunner(config)
    runner.run_all()

    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    for record in manifest.records:
        assert record.primary_style is PrimaryStyle.MANGA_ANIME
        assert record.scopes == [ScopeLabel.FACE_HEAD, ScopeLabel.HAIR]
        assert record.person_count == 1
    labelled = [candidate.labels for candidate in runner.store if candidate.labels]
    assert labelled and all(item.labelled_by == "zero_shot" for item in labelled)
    assert labelled[0].style_scores is not None


def test_a_multi_character_label_implies_two_people(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_tree(tmp_path / "sources", count=2, size=320)
    scopes = {scope: 0.02 for scope in ScopeLabel if scope is not ScopeLabel.UNKNOWN}
    scopes[ScopeLabel.MULTI_CHARACTER] = 0.5
    scopes[ScopeLabel.FULL_BODY] = 0.3
    scores = LabelScores(
        styles={style: 0.2 for style in PrimaryStyle},
        scopes=scopes,
    )
    _stub_labeler(monkeypatch, scores)
    runner = PipelineRunner(_config(tmp_path, label=True))
    runner.run_all()
    manifest = read_manifest(runner.config.manifest_path)
    assert manifest is not None
    for record in manifest.records:
        assert ScopeLabel.MULTI_CHARACTER in record.scopes
        assert record.person_count >= 2  # the manifest's cross-field invariant
        assert len(record.scopes) > 1  # a content scope comes along with it


def test_an_unsafe_source_is_quarantined_and_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_tree(tmp_path / "sources", count=2, size=320)

    class StubClassifier:
        def __init__(self, batch_size: int = 8) -> None:
            self.batch_size = batch_size

        def decision(self, image: Image.Image, *, min_confidence: float) -> SfwDecision:
            return SfwDecision(safe=False, confidence=0.1, method="opennsfw2")

    monkeypatch.setattr(
        OpenNsfw2Classifier, "load", classmethod(lambda cls, *args, **kwargs: StubClassifier())
    )
    config = _config(
        tmp_path,
        sources=[_source(tmp_path / "sources", name="scraped", sfw_method="opennsfw2")],
    )
    runner = PipelineRunner(config)
    runner.run_all()

    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    for record in manifest.records:
        assert record.sfw.safe is False
        assert record.sfw.method == "opennsfw2"
        assert record.enabled is False
        assert record.review.state is ReviewState.QUARANTINED
    assert manifest.enabled_records == []
    label_stage = next(stage for stage in runner.stages if stage.name == "label")
    assert any("opennsfw2" in note for note in label_stage.notes)


def test_labelling_uses_source_defaults_when_disabled(
    dry_run: tuple[PipelineConfig, PipelineRunner],
) -> None:
    config, runner = dry_run
    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    for record in manifest.records:
        assert record.primary_style is PrimaryStyle.GESTURE_SKETCH
        assert record.scopes == [ScopeLabel.FULL_BODY]
    labels = [candidate.labels for candidate in runner.store if candidate.labels]
    assert labels and all(item.labelled_by == "source_default" for item in labels)


# --------------------------------------------------------------------- embeddings


class StubEncoder:
    """Deterministic stand-in for a GPU encoder: pools pixels into a vector.

    The width comes from the real model card, so the shard index's dim check is
    exercised exactly as it would be on a GPU.
    """

    def __init__(self, key: str, dim: int | None = None) -> None:
        self.key = key
        self.dim = dim or MODEL_CARDS[key].dim
        self.released = False

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        rows = []
        for image in images:
            array = np.asarray(image.convert("L"), dtype=np.float32).reshape(-1)
            pooled = np.array(
                [float(chunk.mean()) for chunk in np.array_split(array, self.dim)],
                dtype=np.float32,
            )
            rows.append(pooled / max(float(np.linalg.norm(pooled)), 1e-9))
        return np.asarray(rows, dtype=np.float32)

    def release(self) -> None:
        self.released = True


def test_embeddings_are_written_as_resumable_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_tree(tmp_path / "sources", count=5, size=320)
    monkeypatch.setattr(
        "linescout_ml.colab.runner.load_encoder", lambda key, device="cpu": StubEncoder(key)
    )
    config = _config(tmp_path, embed=True, embedding_shard_size=2, batch_size=2)
    runner = PipelineRunner(config)
    runner.run_all()

    root = config.resolved_embeddings_root() / "mobileclip2_s2"
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    assert index["count"] == 5
    assert index["spec"]["dim"] == MODEL_CARDS["mobileclip2_s2"].dim
    assert len(index["shards"]) == 3  # 2 + 2 + 1
    store = EmbeddingStore.for_key(config.resolved_embeddings_root(), "mobileclip2_s2")
    ids, features = store.load_all()
    assert len(ids) == 5
    assert features.shape == (5, MODEL_CARDS["mobileclip2_s2"].dim)
    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    assert set(ids) == {record.asset_id for record in manifest.records}


def test_embedding_resume_skips_what_is_already_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_tree(tmp_path / "sources", count=4, size=320)
    calls: list[str] = []

    def fake_loader(key: str, device: str = "cpu") -> StubEncoder:
        calls.append(key)
        return StubEncoder(key)

    monkeypatch.setattr("linescout_ml.colab.runner.load_encoder", fake_loader)
    config = _config(tmp_path, embed=True, embedders=["dinov2_vits14"], embedding_shard_size=4)
    PipelineRunner(config).run_all()
    assert calls == ["dinov2_vits14"]

    second = PipelineRunner(config)
    second.run_all()
    stage = next(item for item in second.stages if item.name == "embed")
    assert calls == ["dinov2_vits14"]  # no encoder loaded the second time
    assert any("4 cached, 0 to embed" in note for note in stage.notes)


def test_embedding_shards_record_the_model_licence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_tree(tmp_path / "sources", count=1, size=320)
    monkeypatch.setattr(
        "linescout_ml.colab.runner.load_encoder", lambda key, device="cpu": StubEncoder(key)
    )
    config = _config(tmp_path, embed=True)
    PipelineRunner(config).run_all()
    index = json.loads(
        (config.resolved_embeddings_root() / "mobileclip2_s2" / "index.json").read_text(
            encoding="utf-8"
        )
    )
    assert "Apple ML Research Model License" in index["spec"]["license"]
    assert index["spec"]["citation"]


def test_an_unknown_embedder_fails_loudly(tmp_path: Path) -> None:
    source_tree(tmp_path / "sources", count=1, size=320)
    config = _config(tmp_path, embed=True)
    runner = PipelineRunner(config)
    runner.discover()
    runner.run_extract()
    runner.config.embedders = ["not_a_model"]  # bypasses config validation on purpose
    with pytest.raises(PipelineError, match="unknown embedder"):
        runner.run_embed()


# --------------------------------------------------------------------- build


def test_build_refuses_a_manifest_with_missing_files(
    dry_run: tuple[PipelineConfig, PipelineRunner],
) -> None:
    config, runner = dry_run
    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    victim = config.output_root / manifest.records[0].line_art_path
    victim.unlink()
    with pytest.raises(PipelineError, match="incomplete"):
        runner.run_build()


def test_records_carry_the_crop_the_ink_suggests(
    dry_run: tuple[PipelineConfig, PipelineRunner],
) -> None:
    config, runner = dry_run
    manifest = read_manifest(config.manifest_path)
    assert manifest is not None
    cropped = [record for record in manifest.records if record.crop is not None]
    assert cropped, "the generated drawings do not fill the frame, so a crop is expected"
    for record in cropped:
        crop = record.crop
        assert crop is not None
        assert crop.x + crop.width <= record.width
        assert crop.y + crop.height <= record.height


def test_run_export_without_a_manifest_is_an_error(tmp_path: Path) -> None:
    runner = PipelineRunner(_config(tmp_path))
    with pytest.raises(PipelineError, match="no manifest"):
        runner.run_export(zip_path=tmp_path / "nothing.zip")


def test_records_can_be_rebuilt_from_candidate_state(
    dry_run: tuple[PipelineConfig, PipelineRunner],
) -> None:
    config, runner = dry_run
    records: list[ManifestRecord] = runner.build_records()
    assert len(records) == 6
    assert Manifest(dataset_version=config.dataset_version, records=records).records


def test_building_without_labels_refuses_to_empty_the_gallery(tmp_path: Path) -> None:
    source_tree(tmp_path / "sources", count=2, size=320)
    runner = PipelineRunner(_config(tmp_path))
    runner.discover()
    runner.run_extract()
    runner.run_measure()
    with pytest.raises(PipelineError, match="produced no records"):
        runner.run_build()


def test_a_shard_with_the_wrong_width_is_rejected(tmp_path: Path) -> None:
    store = EmbeddingStore.for_key(tmp_path / "indexes", "dinov2_vits14")
    with pytest.raises(EmbeddingStoreError, match="disagrees"):
        store.append(["ls_sketches_0000000000000000"], np.zeros((1, 7), dtype=np.float32))


def test_shard_rows_must_match_the_id_count(tmp_path: Path) -> None:
    store = EmbeddingStore.for_key(tmp_path / "indexes", "dinov2_vits14")
    dim = MODEL_CARDS["dinov2_vits14"].dim
    with pytest.raises(EmbeddingStoreError, match="ids but features"):
        store.append(["a", "b"], np.zeros((1, dim), dtype=np.float32))


def test_dedupe_is_recomputed_when_the_threshold_changes(tmp_path: Path) -> None:
    """Marks are a pure function of (hashes, threshold), not a sticky one-way flag."""
    root = tmp_path / "sources"
    source_tree(root, count=3, size=384, duplicate_pairs=3)  # 3 distinct + 3 copies
    config = _config(tmp_path)
    runner = PipelineRunner(config)
    runner.discover()
    runner.run_extract()
    runner.run_measure()

    assert runner.run_dedupe().processed == 3
    assert len(runner.store.active) == 3

    # The copies are byte-identical, so even a zero threshold still drops them.
    runner.config.dedupe_threshold = 0
    assert runner.run_dedupe().processed == 3

    # Turning the stage off restores everything, marks included.
    runner.config.dedupe = False
    runner.run_dedupe()
    assert len(runner.store.active) == 6
    assert not any(candidate.duplicate_of for candidate in runner.store)


def test_run_all_reports_its_outputs(tmp_path: Path) -> None:
    source_tree(tmp_path / "sources", count=2, size=320)
    config = _config(tmp_path)
    archive = tmp_path / "gallery.zip"
    report = PipelineRunner(config).run_all(zip_path=archive)

    assert report["outputs"]["zip"]["path"] == str(archive)
    assert report["outputs"]["zip"]["bytes"] > 0
    assert Path(report["outputs"]["run_report"]).is_file()
    with zipfile.ZipFile(archive) as bundle:
        assert "run_report.json" in bundle.namelist()
