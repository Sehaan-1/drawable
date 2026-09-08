"""Stage orchestration — the only module the Colab notebook really needs.

Each public ``run_*`` method is one notebook cell's worth of work: it is
idempotent, persists its results into the candidate store, and can be re-run
after a disconnect without redoing finished assets. ``run_all`` chains them for
a hands-off run.

Stage order and why:

1. ``discover`` — find source images, assign deterministic ids and splits.
2. ``extract``  — write ``originals/``, ``line_art/``, ``thumbnails/``.
3. ``measure``  — ink, text, quality, pHash, crop — measured on the *line art*.
4. ``dedupe``   — drop near-duplicate pHashes before paying for labels.
5. ``label``    — zero-shot style/scope from the line art, SFW from the original.
6. ``embed``    — MobileCLIP2 + DINOv2 features into resumable shards.
7. ``build``    — manifest records, merged into any existing manifest.
8. ``export``   — zip, optional Drive copy, run report.

Nothing here prints. Progress goes through an optional hook so the notebook can
drive ``tqdm`` (or a plain counter) without this package depending on either.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image

from linescout_ml.colab.assets import (
    GalleryBuildError,
    asset_id_for,
    asset_paths,
    build_manifest,
    build_record,
    copy_original,
    merge_records,
    missing_files,
    original_suffix_for,
    read_manifest,
    summarise,
    write_manifest,
    write_png,
)
from linescout_ml.colab.config import PipelineConfig, SourceSpec
from linescout_ml.colab.embed import EmbeddingStore
from linescout_ml.colab.export import (
    StageResult,
    build_run_report,
    colab_download,
    copy_tree,
    describe_file,
    utc_now,
    write_run_report,
    zip_gallery,
)
from linescout_ml.colab.extract import (
    EXTRACTOR_SPECS,
    ExtractorError,
    LineArtExtractor,
    library_version,
    native_line_art,
)
from linescout_ml.colab.label import (
    LabelScores,
    OpenNsfw2Classifier,
    ZeroShotLabeler,
    combine_sfw,
    labels_from_scores,
    source_default_labels,
    source_rating_sfw,
)
from linescout_ml.colab.measure import (
    ImageReadError,
    duplicate_groups,
    load_gray,
    load_rgb,
    make_thumbnail,
    measure_image,
    sha256_file,
)
from linescout_ml.colab.models import MODEL_CARDS, load_encoder
from linescout_ml.colab.runtime import (
    clear_gpu_cache,
    gpu_summary,
    resolve_device,
    torch_available,
)
from linescout_ml.colab.sources import Candidate, CandidateStore, Measurements, discover
from linescout_ml.embeddings import ArtifactStamp
from linescout_ml.manifest import Manifest, ManifestRecord, SfwHumanDecision, SfwScreening
from linescout_ml.taxonomy import LineArtOrigin, SfwScreeningMethod, SfwVerdict

#: ``hook(stage_name, completed, total)`` — called after every item.
ProgressHook = Callable[[str, int, int], None]

#: Clear the CUDA cache this often on long runs; fragmentation, not capacity,
#: is what usually kills a multi-hour extraction on a shared T4.
GPU_STAGE_INTERVAL = 32

T = TypeVar("T")


class PipelineError(RuntimeError):
    """A stage could not complete."""


def _chunks(items: Sequence[T], size: int) -> list[list[T]]:
    if size <= 0:
        msg = "chunk size must be positive"
        raise PipelineError(msg)
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


class PipelineRunner:
    """Drives the stages for one :class:`PipelineConfig`."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        progress: ProgressHook | None = None,
        on_note: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.progress = progress
        self.on_note = on_note or (lambda message: None)
        self.store = CandidateStore(config.candidates_path)
        self.stages: list[StageResult] = []
        self.started_at: str | None = None
        self._device: str | None = None
        self._clock = 0.0
        self._sources: dict[str, SourceSpec] = {source.name: source for source in config.sources}

    # ------------------------------------------------------------------ plumbing

    @property
    def device(self) -> str:
        """Resolve once, lazily: a dry run must not need torch installed."""
        if self._device is None:
            self._device = resolve_device(self.config.device) if torch_available() else "cpu"
        return self._device

    def source_for(self, candidate: Candidate) -> SourceSpec:
        try:
            return self._sources[candidate.source_name]
        except KeyError:
            msg = f"candidate {candidate.key} references unknown source {candidate.source_name!r}"
            raise PipelineError(msg) from None

    def gallery_path(self, relative: str) -> Path:
        """Resolve a manifest-relative path against the gallery root."""
        return self.config.output_root / relative

    def _report(self, stage: str, done: int, total: int) -> None:
        if self.progress is not None:
            self.progress(stage, done, total)

    def _start(self, name: str) -> StageResult:
        self._clock = time.perf_counter()
        stage = StageResult(name=name)
        stage.notes.append(f"device={self.device}")
        return stage

    def _finish(self, stage: StageResult) -> StageResult:
        stage.seconds = time.perf_counter() - self._clock
        self.stages.append(stage)
        self.store.save()
        return stage

    def _require_asset_id(self, candidate: Candidate) -> str:
        if candidate.asset_id is None:
            msg = f"candidate {candidate.key} has no asset_id; run discover first"
            raise PipelineError(msg)
        return candidate.asset_id

    # ------------------------------------------------------------------ 1. discover

    def discover(self, *, resume: bool = True) -> StageResult:
        """Find source images and (re)load candidate state.

        With ``resume`` the previous run's state is loaded first and per-stage
        results are preserved, so re-discovery after a disconnect is free.
        """
        stage = self._start("discover")
        resumed = self.store.load() if resume else 0
        if resumed:
            stage.notes.append(f"resumed {resumed} candidates from {self.store.path.name}")

        fresh = discover(self.config)
        known = {candidate.key for candidate in self.store}
        self.store.upsert(fresh)
        stage.processed = len(fresh)
        stage.skipped = sum(1 for candidate in fresh if candidate.key in known)

        for candidate in self.store:
            if candidate.asset_id is None:
                candidate.asset_id = asset_id_for(self.source_for(candidate), candidate.item_id)

        unreadable = sum(
            1
            for candidate in self.store
            if not candidate.resolve_source_file(self.source_for(candidate)).is_file()
        )
        if unreadable:
            stage.notes.append(f"{unreadable} recorded sources are no longer readable")
        return self._finish(stage)

    # ------------------------------------------------------------------ 2. extract

    def _extract_with_retry(self, extractor: LineArtExtractor, rgb: Image.Image) -> Image.Image:
        """Halve the working resolution once on CUDA OOM before giving up.

        A free T4 has 16 GB shared with whatever else the runtime loaded, and one
        4000-px scan can exceed it. Retrying smaller keeps a long run alive
        instead of ending it two hours in.
        """
        try:
            return extractor.extract(rgb, detect_resolution=self.config.line_art_resolution)
        except (ExtractorError, RuntimeError) as error:
            if "out of memory" not in str(error).lower():
                msg = f"extraction failed: {error}"
                raise ExtractorError(msg) from error
            clear_gpu_cache()
            fallback = max(256, self.config.line_art_resolution // 2)
            self.on_note(
                f"CUDA OOM; retrying at {fallback}px instead of {self.config.line_art_resolution}px"
            )
            return extractor.extract(rgb, detect_resolution=fallback)

    def _extract_one(
        self,
        candidate: Candidate,
        source: SourceSpec,
        extractor: LineArtExtractor | None,
    ) -> None:
        """Produce the three gallery files for one candidate."""
        asset_id = self._require_asset_id(candidate)
        source_file = candidate.resolve_source_file(source)
        paths = asset_paths(asset_id, original_suffix=original_suffix_for(source_file))
        if not source_file.is_file():
            candidate.skip_reason = "source_missing"
            return

        try:
            rgb = load_rgb(source_file)
        except ImageReadError as error:
            candidate.skip_reason = str(error)
            return
        gray = rgb.convert("L")

        if min(rgb.size) < self.config.min_short_edge:
            candidate.skip_reason = f"too_small: {rgb.size[0]}x{rgb.size[1]}"
            return

        model: str | None = None
        version: str | None = None
        if source.origin is LineArtOrigin.NATIVE or extractor is None:
            line = native_line_art(gray)
        else:
            try:
                line = self._extract_with_retry(extractor, rgb)
            except ExtractorError as error:
                candidate.skip_reason = f"extraction_failed: {error}"
                return
            model, version = extractor.spec.model, extractor.version

        copy_original(source_file, self.gallery_path(paths.original))
        write_png(line, self.gallery_path(paths.line_art))
        write_png(
            make_thumbnail(line, self.config.thumbnail_size), self.gallery_path(paths.thumbnail)
        )

        candidate.width, candidate.height = rgb.size
        candidate.original_path = paths.original
        candidate.line_art_path = paths.line_art
        candidate.extraction_model = model
        candidate.extraction_version = version
        candidate.source_checksum = sha256_file(self.gallery_path(paths.original))
        candidate.line_art_checksum = sha256_file(self.gallery_path(paths.line_art))
        candidate.thumbnail_checksum = sha256_file(self.gallery_path(paths.thumbnail))
        for image in (rgb, gray, line):
            image.close()

    def _extract_done(self, candidate: Candidate) -> bool:
        """Whether this candidate's three files already exist from a prior run."""
        if not (
            candidate.asset_id
            and candidate.original_path
            and candidate.line_art_path
            and candidate.source_checksum
            and candidate.line_art_checksum
            and candidate.thumbnail_checksum
        ):
            return False
        paths = asset_paths(
            candidate.asset_id,
            original_suffix=Path(candidate.original_path).suffix or ".png",
        )
        return all(
            self.gallery_path(relative).is_file()
            for relative in (paths.original, paths.line_art, paths.thumbnail)
        )

    def run_extract(self) -> StageResult:
        """Write originals, line art, and thumbnails for every active candidate."""
        stage = self._start("extract")
        if not self.config.extract_line_art:
            stage.notes.append("disabled by config")
            return self._finish(stage)

        targets = [
            candidate
            for candidate in self.store.active
            if self.config.overwrite or not self._extract_done(candidate)
        ]
        stage.skipped = len(self.store.active) - len(targets)
        total = len(targets)

        extractors: dict[str, LineArtExtractor | None] = {}
        try:
            for index, candidate in enumerate(targets, start=1):
                source = self.source_for(candidate)
                key = source.extractor if source.origin is LineArtOrigin.EXTRACTED else "none"
                if key not in extractors:
                    extractors[key] = (
                        None if key == "none" else LineArtExtractor.load(key, self.device)
                    )
                    loaded = extractors[key]
                    if loaded is not None:
                        stage.notes.append(f"{key}: controlnet_aux {loaded.version}")
                self._extract_one(candidate, source, extractors[key])
                if candidate.skip_reason:
                    stage.failed += 1
                else:
                    stage.processed += 1
                if index % GPU_STAGE_INTERVAL == 0:
                    clear_gpu_cache()
                self._report("extract", index, total)
        finally:
            for extractor in extractors.values():
                if extractor is not None:
                    extractor.release()
        return self._finish(stage)

    # ------------------------------------------------------------------ 3. measure

    def run_measure(self) -> StageResult:
        """Measure the extracted line art and suggest an initial crop."""
        stage = self._start("measure")
        targets = [
            candidate
            for candidate in self.store.active
            if candidate.line_art_path and (self.config.overwrite or candidate.measurements is None)
        ]
        stage.skipped = len(self.store.active) - len(targets)
        total = len(targets)

        for index, candidate in enumerate(targets, start=1):
            try:
                gray = load_gray(self.gallery_path(str(candidate.line_art_path)))
            except ImageReadError as error:
                candidate.skip_reason = f"line_art_unreadable: {error}"
                stage.failed += 1
                self._report("measure", index, total)
                continue
            if (
                candidate.width
                and candidate.height
                and gray.size != (candidate.width, candidate.height)
            ):
                candidate.skip_reason = (
                    f"geometry_mismatch: line art {gray.size} vs original "
                    f"{(candidate.width, candidate.height)}"
                )
                stage.failed += 1
                self._report("measure", index, total)
                continue

            result = measure_image(gray, analysis_edge=self.config.analysis_edge)
            candidate.measurements = Measurements(
                ink_coverage=result.ink_coverage,
                text_coverage=result.text_coverage,
                background_coverage=result.background_coverage,
                quality_score=result.quality_score,
                phash=result.phash,
            )
            candidate.crop = result.crop
            stage.processed += 1
            gray.close()
            self._report("measure", index, total)
        return self._finish(stage)

    # ------------------------------------------------------------------ 4. dedupe

    def run_dedupe(self) -> StageResult:
        """Mark near-duplicates, keeping the lowest candidate key in each group.

        Marked candidates stay on disk but leave ``store.active``, so they never
        reach the manifest and never get embedded. Their files are excluded from
        the export because :func:`linescout_ml.colab.export.zip_gallery` walks the
        manifest, not the tree.
        """
        stage = self._start("dedupe")
        if not self.config.dedupe:
            # "Off" means nothing is dropped, including marks left by a previous run.
            for candidate in self.store:
                candidate.duplicate_of = None
            stage.notes.append("disabled by config")
            return self._finish(stage)

        # Dedupe is a pure function of (hashes, threshold), so marks are recomputed
        # from scratch over every measured candidate: re-running this stage with a
        # different threshold can both drop and restore assets.
        candidates: list[Candidate] = []
        hashes: list[str] = []
        for candidate in sorted(self.store, key=lambda item: item.key):
            candidate.duplicate_of = None
            if candidate.measurements is not None and not candidate.skip_reason:
                candidates.append(candidate)
                hashes.append(candidate.measurements.phash)
        groups = duplicate_groups(hashes, self.config.dedupe_threshold)
        for group in groups:
            members = sorted((candidates[position] for position in group), key=lambda c: c.key)
            keeper = members[0]
            for member in members[1:]:
                member.duplicate_of = keeper.asset_id
                stage.processed += 1
        stage.skipped = len(candidates) - stage.processed
        stage.notes.append(
            f"{len(groups)} duplicate groups at Hamming <= {self.config.dedupe_threshold}"
        )
        stage.notes.append(f"{len(candidates)} hashes compared")
        return self._finish(stage)

    # ------------------------------------------------------------------ 5. label

    def _sfw_for(
        self,
        candidate: Candidate,
        source: SourceSpec,
        classifier: OpenNsfw2Classifier | None,
    ) -> tuple[SfwScreening | None, SfwHumanDecision | None]:
        """The automated screen and (only for ``manual`` sources) a human decision.

        ``sfw_method: "manual"`` means the operator asserted the whole source
        is SFW by hand. That is a real human decision, so it is recorded as
        one — with the reviewer named after the pipeline rather than silently
        laundered into a screening result.
        """
        method = source.sfw_method
        if method == "manual":
            human = SfwHumanDecision(
                safe=True, reviewer=f"ingestion:{self.config.pipeline_version}"
            )
            return None, human
        if method == "source_rating" or classifier is None:
            return source_rating_sfw(1.0), None

        relative = (
            candidate.original_path or asset_paths(self._require_asset_id(candidate)).original
        )
        original = self.gallery_path(relative)
        try:
            rgb = load_rgb(original)
        except ImageReadError:
            # Nothing to screen: fail closed rather than claim the asset is safe.
            return (
                SfwScreening(
                    verdict=SfwVerdict.UNSAFE, confidence=0.0, method=SfwScreeningMethod.OPENNSFW2
                ),
                None,
            )
        try:
            verdict = classifier.decision(rgb, min_confidence=self.config.sfw_min_confidence)
        finally:
            rgb.close()
        if method == "source_rating+opennsfw2":
            return combine_sfw(source_rating_sfw(1.0), verdict, method=method), None
        return verdict, None

    def run_label(self) -> StageResult:
        """Attach provisional style/scope labels and the SFW verdict."""
        stage = self._start("label")
        targets = [
            candidate
            for candidate in self.store.active
            if candidate.line_art_path and (self.config.overwrite or candidate.labels is None)
        ]
        stage.skipped = len(self.store.active) - len(targets)
        total = len(targets)
        if not targets:
            return self._finish(stage)

        needs_classifier = any(
            "opennsfw2" in self.source_for(candidate).sfw_method for candidate in targets
        )
        classifier = OpenNsfw2Classifier.load(self.config.batch_size) if needs_classifier else None
        if needs_classifier:
            stage.notes.append("sfw: opennsfw2 on the original")

        labeler: ZeroShotLabeler | None = None
        try:
            if self.config.label:
                labeler = ZeroShotLabeler.load(
                    self.config.labeler_model, self.config.labeler_pretrained, self.device
                )
                stage.notes.append(f"zero-shot via {labeler.encoder.card.name}")
            else:
                stage.notes.append("zero-shot disabled; using source defaults")

            seen = 0
            for chunk in _chunks(targets, self.config.batch_size):
                scores = self._score_chunk(chunk, labeler, stage)
                for candidate, candidate_scores in zip(chunk, scores, strict=True):
                    if candidate.skip_reason is not None:
                        continue  # labelling already gave up on this one
                    source = self.source_for(candidate)
                    screening, human = self._sfw_for(candidate, source, classifier)
                    if candidate_scores is None:
                        candidate.labels = source_default_labels(source, screening, human)
                    else:
                        candidate.labels = labels_from_scores(
                            candidate_scores,
                            source,
                            screening,
                            human=human,
                            scope_top_k=self.config.scope_top_k,
                            scope_min_score=self.config.scope_min_score,
                        )
                    stage.processed += 1
                    seen += 1
                clear_gpu_cache()
                self._report("label", seen, total)
        finally:
            if labeler is not None:
                labeler.release()
        return self._finish(stage)

    def _score_chunk(
        self,
        chunk: Sequence[Candidate],
        labeler: ZeroShotLabeler | None,
        stage: StageResult,
    ) -> list[LabelScores | None]:
        """Zero-shot scores aligned with ``chunk``; ``None`` means "not scored"."""
        if labeler is None:
            return [None] * len(chunk)
        images: list[Image.Image] = []
        owners: list[int] = []
        for position, candidate in enumerate(chunk):
            try:
                images.append(load_gray(self.gallery_path(str(candidate.line_art_path))))
                owners.append(position)
            except ImageReadError as error:
                candidate.skip_reason = f"labelling_failed: {error}"
                stage.failed += 1
        scored = labeler.score_batch(images)
        results: list[LabelScores | None] = [None] * len(chunk)
        for position, score in zip(owners, scored, strict=True):
            results[position] = score
        for image in images:
            image.close()
        return results

    # ------------------------------------------------------------------ 6. embed

    def run_embed(self) -> StageResult:
        """Write feature shards for every configured encoder."""
        stage = self._start("embed")
        if not self.config.embed:
            stage.notes.append("disabled by config")
            return self._finish(stage)

        root = self.config.resolved_embeddings_root()
        for key in self.config.embedders:
            card = MODEL_CARDS.get(key)
            if card is None:
                msg = f"unknown embedder {key!r}; available: {sorted(MODEL_CARDS)}"
                raise PipelineError(msg)
            store = EmbeddingStore.for_key(root, key)
            # Artifact stamps: an entry only counts as done when it was
            # computed from the asset's *current* line art.
            artifacts = {
                str(candidate.asset_id): ArtifactStamp(
                    processing_revision=1,
                    line_art_checksum=str(candidate.line_art_checksum),
                )
                for candidate in self.store.active
                if candidate.asset_id and candidate.line_art_checksum
            }
            done = set() if self.config.overwrite else store.existing_ids(artifacts)
            pending = [
                candidate
                for candidate in self.store.active
                if candidate.line_art_path and candidate.asset_id and candidate.asset_id not in done
            ]
            stage.notes.append(f"{key}: {len(done)} cached, {len(pending)} to embed")
            if not pending:
                continue

            shard_size = min(self.config.batch_size, self.config.embedding_shard_size)
            embedded = 0
            encoder = load_encoder(key, self.device)
            try:
                for chunk in _chunks(pending, shard_size):
                    ids: list[str] = []
                    stamps: list[ArtifactStamp] = []
                    images: list[Image.Image] = []
                    for candidate in chunk:
                        try:
                            images.append(
                                load_gray(self.gallery_path(str(candidate.line_art_path)))
                            )
                            ids.append(str(candidate.asset_id))
                            stamps.append(
                                ArtifactStamp(
                                    processing_revision=1,
                                    line_art_checksum=str(candidate.line_art_checksum),
                                )
                            )
                        except ImageReadError as error:
                            candidate.skip_reason = f"embedding_failed: {error}"
                            stage.failed += 1
                    if not ids:
                        continue
                    features = encoder.encode_images(images)
                    store.append(ids, features, stamps)
                    embedded += len(ids)
                    stage.processed += len(ids)
                    for image in images:
                        image.close()
                    self._report(f"embed:{key}", embedded, len(pending))
            finally:
                encoder.release()
        return self._finish(stage)

    # ------------------------------------------------------------------ 7. build

    def build_records(self) -> list[ManifestRecord]:
        """Manifest records for every active, fully processed candidate."""
        records: list[ManifestRecord] = []
        for candidate in self.store.active:
            if candidate.measurements is None or candidate.labels is None:
                continue
            try:
                records.append(build_record(candidate, self.source_for(candidate), self.config))
            except GalleryBuildError as error:
                self.on_note(str(error))
        return records

    def run_build(self) -> Manifest:
        """Merge this run's slice into any existing manifest and write it out."""
        stage = self._start("build")
        records = self.build_records()
        stage.processed = len(records)
        stage.skipped = max(0, len(self.store.active) - len(records))
        for label, count in (
            ("awaiting labels", sum(1 for c in self.store.active if c.labels is None)),
            ("awaiting measurements", sum(1 for c in self.store.active if c.measurements is None)),
        ):
            if count:
                stage.notes.append(f"{count} candidates {label}")
        if not records and self.store.active:
            # Writing an empty manifest over a real gallery would be silent data
            # loss; an empty run is only legitimate when nothing is active.
            stage.failed = len(self.store.active)
            self._finish(stage)
            msg = (
                f"{len(self.store.active)} active candidates produced no records; "
                "run the label stage first"
            )
            raise PipelineError(msg)

        existing = None if self.config.overwrite else read_manifest(self.config.manifest_path)
        merged = merge_records(existing.records, records) if existing else records
        if existing:
            stage.notes.append(f"merged into {len(existing.records)} existing records")

        manifest = build_manifest(merged, self.config.dataset_version)
        problems = missing_files(manifest, self.config.output_root)
        if problems:
            stage.failed = len(problems)
            self._finish(stage)
            raise PipelineError("gallery is incomplete: " + "; ".join(problems[:5]))
        write_manifest(manifest, self.config.manifest_path)
        stage.notes.append(f"content_hash={manifest.content_hash()[:12]}")
        self._finish(stage)
        return manifest

    # ------------------------------------------------------------------ 8. export

    def run_export(
        self,
        *,
        zip_path: Path | None = None,
        drive_root: Path | None = None,
        download: bool = False,
    ) -> dict[str, Any]:
        """Zip the gallery, optionally mirror it to Drive, and write the report."""
        stage = self._start("export")
        manifest = read_manifest(self.config.manifest_path)
        if manifest is None:
            msg = "no manifest to export; run the build stage first"
            raise PipelineError(msg)

        outputs: dict[str, Any] = {}
        report_path = write_run_report(self.config.state_dir, self.report())
        if zip_path is not None:
            archive = zip_gallery(self.config.output_root, manifest, zip_path, extra=[report_path])
            outputs["zip"] = describe_file(archive)
            stage.notes.append(f"zip {archive.name}: {outputs['zip']['bytes']} bytes")
            if download and colab_download(archive):
                stage.notes.append("browser download started")
        if drive_root is not None:
            destination = drive_root / self.config.dataset_version
            copy_tree(self.config.output_root, destination)
            outputs["drive"] = {"path": str(destination)}
            stage.notes.append(f"copied gallery to {destination}")

        stage.processed = len(manifest.records)
        outputs["run_report"] = str(
            write_run_report(self.config.state_dir, self.report(outputs=outputs))
        )
        self._finish(stage)
        return outputs

    # ------------------------------------------------------------------ report

    def report(self, *, outputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """The run report payload, also written to ``_pipeline/run_report.json``."""
        manifest = read_manifest(self.config.manifest_path)
        summary: dict[str, Any] = (
            dict(summarise(manifest.records))
            if manifest
            else {"total": 0, "servable": 0, "trainable": 0}
        )
        summary["candidates"] = candidate_summary(self.store)

        root = self.config.resolved_embeddings_root()
        embedders: list[dict[str, Any]] = []
        for key in self.config.embedders:
            card = MODEL_CARDS.get(key)
            store = EmbeddingStore.for_key(root, key)
            embedders.append(
                {
                    "key": key,
                    "name": card.name if card else "unknown",
                    "dim": store.observed_dim,
                    "license": card.license if card else None,
                    "count": len(store.existing_ids()),
                    "root": str(store.root),
                }
            )

        gpu = gpu_summary() if torch_available() else {"torch": None, "cuda_available": False}
        return build_run_report(
            config_dump=self.config.model_dump(mode="json"),
            stages=self.stages,
            summary=summary,
            gpu=gpu,
            embedders=embedders,
            outputs=outputs,
            started_at=self.started_at or utc_now(),
        )

    # ------------------------------------------------------------------ all of it

    def run_all(
        self,
        *,
        zip_path: Path | None = None,
        drive_root: Path | None = None,
        download: bool = False,
    ) -> dict[str, Any]:
        """Run every enabled stage in order and return the run report."""
        self.started_at = utc_now()
        self.discover()
        self.run_extract()
        self.run_measure()
        self.run_dedupe()
        self.run_label()
        self.run_embed()
        self.run_build()
        outputs = self.run_export(zip_path=zip_path, drive_root=drive_root, download=download)
        return self.report(outputs=outputs)


def extraction_versions() -> dict[str, str]:
    """Provenance helper: what ``extraction_version`` would be recorded now."""
    version = library_version()
    return {key: f"{spec.model}@{version}" for key, spec in EXTRACTOR_SPECS.items()}


def candidate_summary(candidates: Iterable[Candidate]) -> dict[str, Any]:
    """Counts by terminal state — what the notebook shows after a stage."""
    reasons: dict[str, int] = {}
    duplicates = 0
    active = 0
    total = 0
    for candidate in candidates:
        total += 1
        if candidate.duplicate_of:
            duplicates += 1
        elif candidate.skip_reason:
            key = candidate.skip_reason.split(":", 1)[0]
            reasons[key] = reasons.get(key, 0) + 1
        else:
            active += 1
    return {"total": total, "active": active, "duplicates": duplicates, "skipped": reasons}
