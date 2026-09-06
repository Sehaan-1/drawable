"""GPU ingestion pipeline for the Milestone 2 Colab notebook.

This subpackage is what ``ml/colab/linescout_gpu_pipeline.ipynb`` drives: it
turns a folder of raw source images into a validated gallery (``originals/``,
``line_art/``, ``thumbnails/``, ``manifest.json``) plus resumable feature
shards, and reports exactly which models and versions produced it.

The notebook stays thin on purpose. Everything here is typed, linted, and unit
tested in CI, so the interesting logic is reviewable in a diff instead of buried
in cells that only run on a GPU.

Installing::

    pip install "linescout-ml[pipeline]"   # numpy + Pillow: CPU stages, tests
    pip install "linescout-ml[gpu]"        # + torch, controlnet-aux, open-clip-torch

Heavy imports are lazy: ``import linescout_ml.colab`` works with only pydantic,
numpy, and Pillow installed, and the torch-dependent stages raise
:class:`~linescout_ml.colab._optional.MissingDependencyError` with the exact
install command if you run them without the GPU extra.

Minimal CPU-only run (the notebook's dry-run cell does this)::

    from linescout_ml.colab import PipelineConfig, PipelineRunner, preset_source

    config = PipelineConfig(
        dataset_version="2026.09.06-dryrun",
        output_root=Path("/tmp/gallery"),
        sources=[preset_source("synthetic", root=FIXTURES / "originals",
                               license_id="synthetic-fixture")],
        label=False,          # no CLIP encoder
        embed=False,          # no feature shards
    )
    report = PipelineRunner(config).run_all()
"""

from __future__ import annotations

from linescout_ml.colab._optional import MissingDependencyError, optional_module
from linescout_ml.colab.assets import (
    GALLERY_DIRS,
    AssetPaths,
    GalleryBuildError,
    asset_id_for,
    asset_paths,
    build_manifest,
    build_record,
    merge_records,
    missing_files,
    read_manifest,
    summarise,
    write_manifest,
    write_png,
)
from linescout_ml.colab.config import (
    SOURCE_PRESETS,
    ExtractorKey,
    PipelineConfig,
    SourcePreset,
    SourceSpec,
    SplitFractions,
    preset_source,
    preset_table,
)
from linescout_ml.colab.embed import EmbeddingStore, EmbeddingStoreError
from linescout_ml.colab.export import (
    StageResult,
    build_run_report,
    colab_download,
    copy_tree,
    describe_file,
    utc_now,
    write_run_report,
    zip_gallery,
    zip_tree,
)
from linescout_ml.colab.extract import EXTRACTOR_SPECS, ExtractorError, LineArtExtractor
from linescout_ml.colab.label import (
    SCOPE_PROMPTS,
    STYLE_PROMPTS,
    LabelScores,
    OpenNsfw2Classifier,
    ZeroShotLabeler,
    select_scopes,
)
from linescout_ml.colab.measure import (
    ImageReadError,
    MeasurementResult,
    duplicate_groups,
    hamming,
    load_gray,
    load_rgb,
    make_thumbnail,
    measure_image,
    phash,
    sha256_file,
)
from linescout_ml.colab.models import MODEL_CARDS, EncoderError, ModelCard, load_encoder
from linescout_ml.colab.runner import (
    PipelineError,
    PipelineRunner,
    ProgressHook,
    candidate_summary,
    extraction_versions,
)
from linescout_ml.colab.runtime import gpu_summary, resolve_device, torch_available
from linescout_ml.colab.sources import (
    AssetLabels,
    Candidate,
    CandidateStore,
    Measurements,
    discover,
    split_for_work,
)

__all__ = [
    "AssetLabels",
    "AssetPaths",
    "Candidate",
    "CandidateStore",
    "EXTRACTOR_SPECS",
    "EmbeddingStore",
    "EmbeddingStoreError",
    "EncoderError",
    "ExtractorError",
    "ExtractorKey",
    "GALLERY_DIRS",
    "GalleryBuildError",
    "ImageReadError",
    "LabelScores",
    "LineArtExtractor",
    "MODEL_CARDS",
    "MeasurementResult",
    "Measurements",
    "MissingDependencyError",
    "ModelCard",
    "OpenNsfw2Classifier",
    "PipelineConfig",
    "PipelineError",
    "PipelineRunner",
    "ProgressHook",
    "SCOPE_PROMPTS",
    "SOURCE_PRESETS",
    "STYLE_PROMPTS",
    "SourcePreset",
    "SourceSpec",
    "SplitFractions",
    "StageResult",
    "ZeroShotLabeler",
    "asset_id_for",
    "asset_paths",
    "build_manifest",
    "build_record",
    "build_run_report",
    "candidate_summary",
    "colab_download",
    "copy_tree",
    "describe_file",
    "discover",
    "duplicate_groups",
    "extraction_versions",
    "gpu_summary",
    "hamming",
    "load_encoder",
    "load_gray",
    "load_rgb",
    "make_thumbnail",
    "measure_image",
    "merge_records",
    "missing_files",
    "optional_module",
    "phash",
    "preset_source",
    "preset_table",
    "read_manifest",
    "resolve_device",
    "select_scopes",
    "sha256_file",
    "split_for_work",
    "summarise",
    "torch_available",
    "utc_now",
    "write_manifest",
    "write_png",
    "write_run_report",
    "zip_gallery",
    "zip_tree",
]
