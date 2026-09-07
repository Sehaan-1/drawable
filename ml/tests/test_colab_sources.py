"""Discovery, identity, splits, and the resumable candidate store."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from colab_images import line_art, source_tree, write_png

from linescout_ml.colab.config import PipelineConfig, SourceSpec, SplitFractions
from linescout_ml.colab.sources import (
    AssetLabels,
    Candidate,
    CandidateStore,
    Measurements,
    discover,
    discover_source,
    item_id_for,
    split_for_work,
    work_id_for,
)
from linescout_ml.manifest import SfwDecision
from linescout_ml.taxonomy import DatasetSplit, PrimaryStyle, ScopeLabel


def _source(root: Path, **overrides: object) -> SourceSpec:
    fields: dict[str, object] = {
        "name": "sketches",
        "root": root,
        "license_id": "CC0-1.0",
        "origin": "native_line_art",
        "extractor": "none",
    }
    fields.update(overrides)
    return SourceSpec(**fields)  # type: ignore[arg-type]


def _config(root: Path, **overrides: object) -> PipelineConfig:
    fields: dict[str, object] = {
        "dataset_version": "2026.09.06-test",
        "output_root": root / "gallery",
        "sources": [_source(root / "sources")],
        "label": False,
        "embed": False,
    }
    fields.update(overrides)
    return PipelineConfig(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------- discovery


def test_discovery_is_sorted_and_recursive(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    source_tree(root, count=4, subdirs=True)
    found = discover_source(_source(root))
    relative = [image.relative_path for image in found]
    assert relative == sorted(relative)
    assert any("/" in path for path in relative)  # rglob reached the subdirectories
    assert all(image.item_id.endswith(Path(image.relative_path).stem) for image in found)


def test_discovery_honours_patterns_and_limit(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    source_tree(root, count=5)
    (root / "notes.txt").write_text("not an image", encoding="utf-8")
    assert len(discover_source(_source(root))) == 5
    assert len(discover_source(_source(root), limit=2)) == 2
    only_small = discover_source(_source(root, patterns=["*.jpg"]))
    assert only_small == []


def test_discovery_ignores_hidden_files(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    source_tree(root, count=2)
    write_png(root / ".hidden.png", line_art(256, seed=1))
    assert all(not image.path.name.startswith(".") for image in discover_source(_source(root)))


def test_discovery_requires_an_existing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_source(_source(tmp_path / "missing"))


def test_discover_assigns_keys_and_splits(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    source_tree(root, count=6)
    candidates = discover(_config(tmp_path))
    assert len(candidates) == 6
    assert all(candidate.key.startswith("sketches/") for candidate in candidates)
    assert all(candidate.split in set(DatasetSplit) for candidate in candidates)
    assert len({candidate.key for candidate in candidates}) == 6


def test_discover_handles_several_sources(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    source_tree(first, count=2)
    source_tree(second, count=3)
    config = _config(
        tmp_path,
        sources=[
            _source(first, name="alpha"),
            _source(second, name="beta"),
        ],
    )
    candidates = discover(config)
    assert Counter(candidate.source_name for candidate in candidates) == {"alpha": 2, "beta": 3}


# --------------------------------------------------------------------- identity


def test_item_id_drops_the_extension() -> None:
    assert item_id_for("pages/page-001.png") == "pages/page-001"
    assert item_id_for("drawing.png") == "drawing"


def test_work_grouping_by_filename_is_per_image(tmp_path: Path) -> None:
    source = _source(tmp_path)
    assert work_id_for(source, "book-01/page-001.png") == "sketches/book-01/page-001"
    assert work_id_for(source, "book-01/page-002.png") == "sketches/book-01/page-002"


def test_work_grouping_by_parent_dir_is_per_folder(tmp_path: Path) -> None:
    source = _source(tmp_path, work_grouping="parent_dir")
    assert work_id_for(source, "book-01/page-001.png") == "sketches/book-01"
    assert work_id_for(source, "book-01/page-002.png") == "sketches/book-01"
    assert work_id_for(source, "loose.png") == "sketches"


def test_one_work_never_crosses_splits(tmp_path: Path) -> None:
    """The manifest's split rule, guaranteed by construction."""
    root = tmp_path / "sources"
    source_tree(root, count=9, subdirs=True)  # three pages per "book"
    config = _config(tmp_path, sources=[_source(root, work_grouping="parent_dir")])
    candidates = discover(config)
    per_work: dict[str, set[DatasetSplit]] = {}
    for candidate in candidates:
        per_work.setdefault(candidate.work_id, set()).add(candidate.split)
    assert all(len(splits) == 1 for splits in per_work.values())
    assert len(candidates) == 9


# --------------------------------------------------------------------- splits


def test_split_assignment_is_deterministic() -> None:
    fractions = SplitFractions()
    first = split_for_work("sketches/book-01", 7, fractions)
    assert first == split_for_work("sketches/book-01", 7, fractions)


def test_split_assignment_depends_on_the_seed() -> None:
    fractions = SplitFractions()
    seeds = {split_for_work("sketches/book-01", seed, fractions) for seed in range(20)}
    assert len(seeds) > 1


def test_split_assignment_roughly_follows_the_fractions() -> None:
    fractions = SplitFractions()
    counts = Counter(split_for_work(f"work-{index}", 7, fractions) for index in range(4000))
    assert counts[DatasetSplit.TRAIN] / 4000 == pytest.approx(0.70, abs=0.05)
    assert counts[DatasetSplit.VALIDATION] / 4000 == pytest.approx(0.15, abs=0.05)
    assert counts[DatasetSplit.TEST] / 4000 == pytest.approx(0.15, abs=0.05)
    assert counts.get(DatasetSplit.GALLERY_ONLY, 0) == 0


def test_custom_split_fractions_are_honoured() -> None:
    fractions = SplitFractions(train=0.7, validation=0.1, test=0.1, gallery_only=0.1)
    counts = Counter(split_for_work(f"work-{index}", 7, fractions) for index in range(4000))
    assert counts[DatasetSplit.TRAIN] / 4000 == pytest.approx(0.70, abs=0.05)
    assert counts[DatasetSplit.TEST] / 4000 == pytest.approx(0.10, abs=0.05)
    assert len(counts) == 4


def test_a_gallery_only_share_of_one_puts_everything_in_the_gallery() -> None:
    fractions = SplitFractions(train=0.0, validation=0.0, test=0.0, gallery_only=1.0)
    assert all(
        split_for_work(f"work-{index}", 3, fractions) is DatasetSplit.GALLERY_ONLY
        for index in range(50)
    )


# --------------------------------------------------------------------- state


def _measurements(phash_value: str = "0123456789abcdef") -> Measurements:
    return Measurements(
        ink_coverage=0.1,
        text_coverage=0.0,
        background_coverage=0.9,
        quality_score=0.8,
        phash=phash_value,
    )


def _labels() -> AssetLabels:
    return AssetLabels(
        primary_style=PrimaryStyle.MANGA_ANIME,
        scopes=[ScopeLabel.FACE_HEAD],
        person_count=1,
        sfw=SfwDecision(safe=True, confidence=1.0, method="source_rating"),
    )


def test_candidate_store_round_trips(tmp_path: Path) -> None:
    source_tree(tmp_path / "sources", count=3)
    store = CandidateStore(tmp_path / "candidates.jsonl")
    store.upsert(discover(_config(tmp_path)))
    for candidate in store:
        candidate.measurements = _measurements()
    store.save()

    reloaded = CandidateStore(tmp_path / "candidates.jsonl")
    assert reloaded.load() == len(store)
    assert all(candidate.measurements is not None for candidate in reloaded)


def test_upsert_preserves_finished_work(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    source_tree(root, count=3)
    config = _config(tmp_path)
    store = CandidateStore(config.candidates_path)
    store.upsert(discover(config))
    first = next(iter(store))
    first.labels = _labels()
    first.line_art_path = "line_art/x.png"
    first.skip_reason = None
    store.save()

    # Re-discovery (as a resumed run does) must not wipe the stage results.
    resumed = CandidateStore(config.candidates_path)
    resumed.load()
    resumed.upsert(discover(config))
    restored = resumed.get(first.key)
    assert restored is not None
    assert restored.labels == _labels()
    assert restored.line_art_path == "line_art/x.png"


def test_active_excludes_skipped_and_duplicate_candidates(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    source_tree(root, count=3)
    store = CandidateStore(_config(tmp_path).candidates_path)
    store.upsert(discover(_config(tmp_path)))
    candidates = store.candidates
    candidates[0].skip_reason = "too_small: 10x10"
    candidates[1].duplicate_of = "ls_sketches_0000000000000000"
    assert [candidate.key for candidate in store.active] == [candidates[2].key]
    assert candidates[2].is_active


def test_candidate_maps_legacy_checksum_field() -> None:
    candidate = Candidate.model_validate(
        {
            "key": "sketches/a",
            "source_name": "sketches",
            "item_id": "a",
            "work_id": "sketches/a",
            "relative_path": "a.png",
            "source_path": "/tmp/a.png",
            "split": "train",
            "checksum": "a" * 64,
        }
    )
    assert candidate.line_art_checksum == "a" * 64
    assert candidate.source_checksum is None
    assert candidate.thumbnail_checksum is None


def test_a_missing_state_file_is_not_an_error(tmp_path: Path) -> None:
    store = CandidateStore(tmp_path / "nope.jsonl")
    assert store.load() == 0
    assert len(store) == 0


def test_save_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    source_tree(tmp_path / "sources", count=2)
    path = tmp_path / "state" / "candidates.jsonl"
    store = CandidateStore(path)
    store.upsert(discover(_config(tmp_path)))
    store.save()
    assert path.is_file()
    assert list(path.parent.glob("*.tmp")) == []
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert all(line["key"] for line in lines)


def test_candidate_re_resolves_a_moved_source_root(tmp_path: Path) -> None:
    """Drive mounts change between sessions; the relative path is the truth."""
    root = tmp_path / "sources"
    source_tree(root, count=1)
    candidate = discover(_config(tmp_path))[0]
    moved = tmp_path / "remounted"
    moved.mkdir()
    (moved / Path(candidate.relative_path).name).write_bytes(
        Path(candidate.source_path).read_bytes()
    )
    candidate.source_path = str(tmp_path / "gone" / candidate.relative_path)
    assert not Path(candidate.source_path).is_file()
    assert candidate.resolve_source_file(_source(moved)).is_file()
