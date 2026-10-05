"""Small, frozen, local search over an immutable owner-reviewed prototype batch."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import os
import resource
import shutil
import sys
import tempfile
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from drawable_research_contracts import CurationSnapshotV1, validate_curation_snapshot_files
from PIL import Image

from . import inference
from .inference import MODEL_REGISTRY_PATH, load_frozen_encoder
from .io import (
    MLError,
    load_json,
    resolve_private_path,
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from .models import _expected_source

MODEL_IDS = ("mobileclip2-s2-dfndr2b", "dinov2-small")
RANKING: dict[str, Any] = {
    "id": "prototype-bootstrap-v1",
    "visual_floor": 0.60,
    "tight_weight": 0.5,
}
STYLE_NAMES = {
    "manga_anime": "Manga / anime",
    "western_ink": "Western ink",
    "realistic_academic": "Realistic",
    "cartoon": "Cartoon",
    "gesture_sketch": "Gesture",
}
SCOPE_NAMES = {
    "face_head": "Heads",
    "upper_body_clothing": "Upper bodies",
    "full_body": "Full bodies",
    "multi_character": "Multiple characters",
    "eye": "Eyes",
}


def _models() -> dict[str, Any]:
    return {
        row["id"]: row for row in load_json(MODEL_REGISTRY_PATH)["models"] if row["id"] in MODEL_IDS
    }


def acquire_models(root: Path) -> list[dict[str, Any]]:
    """Explicit setup downloads; inference never downloads or marks terms approved."""
    files = []
    for model_id, row in _models().items():
        source = _expected_source(row)
        if source is None:
            raise MLError(f"No exact upstream artifact pin for {model_id}")
        for name in source["artifact_directory_files"]:
            relative = f"models/{model_id}/files/{name}"
            destination = resolve_private_path(root, relative, must_exist=False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            is_weight = name == source["weights_file"]
            if not destination.exists():
                url = f"https://huggingface.co/{source['repository']}/resolve/{source['revision']}/{name}"
                print(f"Downloading pinned {model_id}/{name}…", file=sys.stderr, flush=True)
                fd, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".part")
                temporary = Path(temporary_name)
                try:
                    digest = hashlib.sha256()
                    count = 0
                    maximum = source["weights_bytes"] if is_weight else 1024 * 1024
                    with (
                        os.fdopen(fd, "wb") as output,
                        urllib.request.urlopen(url, timeout=60) as response,
                    ):
                        while block := response.read(1024 * 1024):
                            count += len(block)
                            if count > maximum:
                                raise MLError(
                                    f"Pinned download exceeds the expected size: {model_id}/{name}"
                                )
                            digest.update(block)
                            output.write(block)
                        output.flush()
                        os.fsync(output.fileno())
                    if is_weight and (
                        count != maximum or digest.hexdigest() != source["weights_sha256"]
                    ):
                        raise MLError(f"Pinned model checksum mismatch: {model_id}/{name}")
                    if not is_weight:
                        load_json(temporary)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            size, digest_text = destination.stat().st_size, sha256_file(destination)
            if is_weight and (
                size != source["weights_bytes"] or digest_text != source["weights_sha256"]
            ):
                raise MLError(f"Local pinned model bytes are corrupt: {model_id}/{name}")
            files.append({"path": relative, "bytes": size, "sha256": digest_text})
    return files


def _snapshot(root: Path, relative: str) -> CurationSnapshotV1:
    path = resolve_private_path(root, relative)
    snapshot = CurationSnapshotV1.model_validate_json(path.read_bytes())
    if snapshot.purpose != "local-educational-prototype" or not 1 <= len(snapshot.items) <= 32:
        raise MLError(
            "Select a saved educational prototype batch containing 1–32 reviewed references"
        )
    if any(not item.permissions.evaluation for item in snapshot.items):
        raise MLError("Saved prototype references must allow local evaluation")
    validate_curation_snapshot_files(snapshot, root)
    return snapshot


def _load_models(root: Path, *, structural: bool) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    with contextlib.redirect_stdout(sys.stderr):
        encoders = {"semantic": load_frozen_encoder(root, MODEL_IDS[0])}
        if structural:
            encoders["structural"] = load_frozen_encoder(root, MODEL_IDS[1])
    return encoders


def _encode(encoder: Any, image: Image.Image) -> Any:
    import torch

    return torch.stack([encoder.encode(image), encoder.encode(image, view="tight")])


def rank_vectors(
    vectors: dict[str, Any], query: dict[str, Any], asset_ids: list[str]
) -> list[tuple[int, float]]:
    """Unknown query scope uses 0.5 full/tight fusion; active branches renormalize."""
    scores = None
    denominator = 0.0
    for key, weight in (("semantic", 0.35), ("structural", 0.25)):
        if key not in query:
            continue
        cosines = (vectors[key] * query[key].unsqueeze(0)).sum(dim=-1).mean(dim=-1)
        branch = (cosines.clamp(-1, 1) + 1) / 2
        scores = branch * weight if scores is None else scores + branch * weight
        denominator += weight
    if scores is None:
        raise MLError("No enabled frozen image branch is available")
    values = (scores / denominator).tolist()
    return sorted(enumerate(values), key=lambda pair: (-pair[1], asset_ids[pair[0]]))


def prepare(root: Path, snapshot_relative: str | None = None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if snapshot_relative is None:
        candidates = sorted(
            (root / "reports/private/curation-snapshots").glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        snapshot_relative = next(
            (
                path.relative_to(root).as_posix()
                for path in candidates
                if load_json(path).get("purpose") == "local-educational-prototype"
            ),
            None,
        )
    if snapshot_relative is None:
        raise MLError(
            "No saved prototype batch exists. Complete image review and save a review-only snapshot first."
        )
    snapshot = _snapshot(root, snapshot_relative)
    snapshot_hash = sha256_file(resolve_private_path(root, snapshot_relative))
    files = acquire_models(root)
    model_pins = {
        key: {
            "artifact_source": row["artifact_source"],
            "preprocessing_profile": row["preprocessing_profile"],
        }
        for key, row in _models().items()
    }
    pointer_path = resolve_private_path(root, "prototype/search/active.json", must_exist=False)
    if pointer_path.exists():
        try:
            active = load_build(root)
            if (
                active["snapshot_sha256"] == snapshot_hash
                and active["model_pins"] == model_pins
                and active["ranking"] == RANKING
                and active["model_files"] == files
            ):
                return {
                    "status": "ready",
                    "build_id": active["build_id"],
                    "item_count": len(snapshot.items),
                    "reused": True,
                }
        except (MLError, ValueError, OSError):
            pass

    import torch
    from safetensors.torch import save_file

    started = time.perf_counter()
    encoders = _load_models(root, structural=True)
    tensors: dict[str, Any] = {}
    model_times: dict[str, list[float]] = {}
    for key, encoder in encoders.items():
        embeddings = []
        times = []
        for item in snapshot.items:
            with Image.open(resolve_private_path(root, item.relative_path)) as source:
                if (
                    source.size != (item.width, item.height)
                    or source.width * source.height > 40_000_000
                ):
                    raise MLError(
                        f"Reviewed image dimensions changed or are too large: {item.asset_id}"
                    )
                source.load()
                tick = time.perf_counter()
                embeddings.append(_encode(encoder, source))
                times.append((time.perf_counter() - tick) * 1000)
        tensors[key] = torch.stack(embeddings).contiguous().float()
        model_times[key] = times
        print(
            f"Indexed {len(embeddings)} reviewed references with {encoder.model_id}.",
            file=sys.stderr,
            flush=True,
        )
    parent = resolve_private_path(root, "prototype/search/builds", must_exist=False)
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".staging-{uuid4()}"
    staging.mkdir(mode=0o700)
    try:
        save_file(tensors, str(staging / "vectors.safetensors"))
        asset_ids = [item.asset_id for item in snapshot.items]
        self_checks = {}
        for profile in ("semantic", "semantic-structural"):
            keys = ["semantic"] if profile == "semantic" else ["semantic", "structural"]
            ranks = [
                rank_vectors(tensors, {key: tensors[key][index] for key in keys}, asset_ids)
                for index in range(len(asset_ids))
            ]
            self_checks[profile] = sum(ranked[0][0] == index for index, ranked in enumerate(ranks))
        body = {
            "schema_version": 1,
            "kind": "drawable-prototype-gallery",
            "purpose": "local-educational-prototype",
            "created_at": datetime.now(UTC).isoformat(),
            "snapshot_path": snapshot_relative,
            "snapshot_sha256": snapshot_hash,
            "asset_ids": asset_ids,
            "model_pins": model_pins,
            "model_files": files,
            "vectors_sha256": sha256_file(staging / "vectors.safetensors"),
            "ranking": RANKING,
            "encoder_code_sha256": sha256_file(Path(inference.__file__)),
            "measurements": {
                "image_views_ms": model_times,
                "total_build_ms": (time.perf_counter() - started) * 1000,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "self_match_top1": self_checks,
                "retrieval_quality_evaluated": False,
            },
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "torchvision",
                    "open_clip_torch",
                    "transformers",
                    "timm",
                    "safetensors",
                )
            },
        }
        build_id = "prototype-" + sha256_json(body)[:24]
        write_json_atomic(staging / "manifest.json", {**body, "build_id": build_id})
        destination = parent / build_id
        staging.rename(destination)
        write_json_atomic(
            pointer_path,
            {"build_id": build_id, "manifest_sha256": sha256_file(destination / "manifest.json")},
            refuse_overwrite=False,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "status": "ready",
        "build_id": build_id,
        "item_count": len(asset_ids),
        "reused": False,
        "self_match_top1": self_checks,
        "training_started": False,
    }


def load_build(root: Path) -> dict[str, Any]:
    pointer = load_json(resolve_private_path(root, "prototype/search/active.json"))
    build_id = pointer.get("build_id", "")
    if (
        not isinstance(build_id, str)
        or not build_id.startswith("prototype-")
        or len(build_id) != 34
        or any(char not in "0123456789abcdef" for char in build_id[10:])
    ):
        raise MLError("Prototype active build ID is malformed")
    relative = f"prototype/search/builds/{build_id}/manifest.json"
    manifest_path = resolve_private_path(root, relative)
    if sha256_file(manifest_path) != pointer.get("manifest_sha256"):
        raise MLError("Prototype gallery manifest checksum changed")
    manifest = load_json(manifest_path)
    body = {key: value for key, value in manifest.items() if key != "build_id"}
    if (
        manifest.get("build_id") != build_id
        or "prototype-" + sha256_json(body)[:24] != build_id
        or manifest.get("purpose") != "local-educational-prototype"
        or manifest.get("kind") != "drawable-prototype-gallery"
        or manifest.get("ranking") != RANKING
    ):
        raise MLError("Prototype gallery identity or ranking recipe changed")
    snapshot_path = resolve_private_path(root, manifest["snapshot_path"])
    if sha256_file(snapshot_path) != manifest["snapshot_sha256"]:
        raise MLError("Prototype review snapshot checksum changed")
    snapshot = _snapshot(root, manifest["snapshot_path"])
    if manifest["asset_ids"] != [item.asset_id for item in snapshot.items]:
        raise MLError("Prototype vector IDs do not match the reviewed batch")
    if (
        sha256_file(
            resolve_private_path(root, f"prototype/search/builds/{build_id}/vectors.safetensors")
        )
        != manifest["vectors_sha256"]
    ):
        raise MLError("Prototype vectors checksum changed")
    current_pins = {
        key: {
            "artifact_source": row["artifact_source"],
            "preprocessing_profile": row["preprocessing_profile"],
        }
        for key, row in _models().items()
    }
    if manifest["model_pins"] != current_pins:
        raise MLError("Pinned model/preprocessing configuration changed; prepare search again")
    if manifest.get("encoder_code_sha256") != sha256_file(Path(inference.__file__)):
        raise MLError("Encoder/preprocessing code changed; prepare search again")
    if any(
        importlib.metadata.version(name) != version
        for name, version in manifest["packages"].items()
    ):
        raise MLError("Model runtime packages changed; prepare search again")
    expected_files = {
        f"models/{model_id}/files/{name}": (
            row["artifact_source"] if name == row["artifact_source"]["weights_file"] else None
        )
        for model_id, row in _models().items()
        for name in row["artifact_source"]["artifact_directory_files"]
    }
    if {entry["path"] for entry in manifest["model_files"]} != set(expected_files):
        raise MLError("Prototype model manifest does not cover the exact pinned files")
    for entry in manifest["model_files"]:
        pin = expected_files[entry["path"]]
        if pin is not None and (
            entry["bytes"] != pin["weights_bytes"] or entry["sha256"] != pin["weights_sha256"]
        ):
            raise MLError("Prototype model manifest differs from the pinned upstream weights")
        path = resolve_private_path(root, entry["path"])
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise MLError("Prototype model artifact checksum changed")
    return manifest


class PrototypeGallery:
    def __init__(self, root: Path, *, structural: bool = False):
        import torch
        from safetensors.torch import load_file

        self.root = root.resolve(strict=True)
        self.manifest = load_build(self.root)
        self.snapshot = _snapshot(self.root, self.manifest["snapshot_path"])
        self.vectors = load_file(
            str(
                resolve_private_path(
                    self.root,
                    f"prototype/search/builds/{self.manifest['build_id']}/vectors.safetensors",
                )
            )
        )
        for key, dimension in (("semantic", 512), ("structural", 384)):
            tensor = self.vectors.get(key)
            if (
                tensor is None
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != (len(self.snapshot.items), 2, dimension)
                or not torch.isfinite(tensor).all()
                or not torch.allclose(
                    tensor.norm(dim=-1), torch.ones((len(self.snapshot.items), 2)), atol=1e-4
                )
            ):
                raise MLError("Prototype vectors are missing, malformed, or not normalized")
        self.encoders = _load_models(self.root, structural=structural)
        with Image.open(
            resolve_private_path(self.root, self.snapshot.items[0].relative_path)
        ) as source:
            for encoder in self.encoders.values():
                _encode(encoder, source)

    def status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "build_id": self.manifest["build_id"],
            "gallery_size": len(self.snapshot.items),
            "enabled_branches": list(self.encoders),
            "model_version": "+".join(encoder.model_id for encoder in self.encoders.values()),
        }

    def search(self, png: bytes) -> dict[str, Any]:
        started = time.perf_counter()
        with Image.open(io.BytesIO(png)) as image:
            if image.format != "PNG" or image.size != (512, 512):
                raise MLError("Prototype query must be a 512 × 512 PNG")
            image.load()
            query = {key: _encode(encoder, image) for key, encoder in self.encoders.items()}
        ranked = rank_vectors(self.vectors, query, self.manifest["asset_ids"])
        groups: dict[str, dict[str, Any]] = {}
        duplicates: set[str] = set()
        for index, score in ranked:
            if score < RANKING["visual_floor"]:
                continue
            item = self.snapshot.items[index]
            try:
                path = resolve_private_path(self.root, item.relative_path)
                if path.stat().st_size != item.byte_length or sha256_file(path) != item.sha256:
                    continue
            except (MLError, OSError):
                continue
            duplicate = item.labels.duplicate_of or item.asset_id
            if duplicate in duplicates:
                continue
            scope = item.labels.primary_scope
            if scope not in groups:
                if len(groups) >= 3:
                    continue
                groups[scope] = {
                    "id": scope,
                    "title": f"Reviewed {SCOPE_NAMES.get(scope, scope.replace('_', ' ')).lower()}",
                    "tentative": True,
                    "results": [],
                }
            if len(groups[scope]["results"]) >= 4:
                continue
            duplicates.add(duplicate)
            groups[scope]["results"].append(
                {
                    "id": item.asset_id,
                    "title": f"{item.source_work_id or item.source_id} · {scope.replace('_', ' ')}{' crop' if item.parent_asset_id else ''}",
                    "image_url": f"/api/v1/assets/{item.asset_id}/preview?build_id={self.manifest['build_id']}",
                    "style": STYLE_NAMES[item.labels.style],
                    "scope": scope,
                    "source": item.attribution,
                    "native": item.parent_asset_id is None,
                    "match": "Tentative",
                    "trace_allowed": item.permissions.trace,
                    "visual_score": score,
                }
            )
        return {
            **self.status(),
            "mode": "provisional" if groups else "insufficient",
            "reason": "experimental_visual_matches" if groups else "no_relevant_candidates",
            "interpretation": "Tentative matches from your reviewed references."
            if groups
            else "No reviewed references passed the visual relevance floor.",
            "groups": list(groups.values()),
            "inference_ms": (time.perf_counter() - started) * 1000,
            "warnings": [
                "Experimental scores are not calibrated confidence.",
                "Query scope and viewpoint compatibility have not been validated.",
            ],
        }
