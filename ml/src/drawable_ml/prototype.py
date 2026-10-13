"""Small, frozen, local search over an immutable owner-reviewed prototype batch."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import resource
import shutil
import sqlite3
import sys
import tempfile
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from drawable_research_contracts import (
    CurationSnapshotV1,
    PrototypeGalleryCatalogV2,
    validate_curation_snapshot_files,
)
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


def acquire_models(root: Path, model_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Explicit setup downloads; inference never downloads or marks terms approved."""
    files = []
    model_registry = _models()
    selected_ids = model_ids or list(model_registry)
    if not set(selected_ids).issubset(model_registry):
        raise MLError("Selected prototype profile references an unknown frozen encoder")
    for model_id in selected_ids:
        row = model_registry[model_id]
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


def _snapshot(
    root: Path,
    relative: str,
    *,
    validate_files: bool = True,
) -> CurationSnapshotV1 | PrototypeGalleryCatalogV2:
    path = resolve_private_path(root, relative)
    try:
        raw_document = path.read_bytes()
        document = json.loads(raw_document)
    except (OSError, json.JSONDecodeError) as error:
        raise MLError("Selected prototype catalog is not valid JSON") from error
    version = document.get("schema_version") if isinstance(document, dict) else None
    if version == 1:
        snapshot: CurationSnapshotV1 | PrototypeGalleryCatalogV2 = CurationSnapshotV1.model_validate_json(raw_document)
    elif version == 2:
        snapshot = PrototypeGalleryCatalogV2.model_validate_json(raw_document)
    else:
        raise MLError("Selected prototype catalog version is unsupported")
    if snapshot.purpose != "local-educational-prototype" or not 1 <= len(snapshot.items) <= 10_000:
        raise MLError(
            "Select a saved educational prototype catalog containing 1–10,000 reviewed references"
        )
    if any(not item.permissions.evaluation for item in snapshot.items):
        raise MLError("Saved prototype references must allow local evaluation")
    if validate_files:
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


def _embedding_implementation_sha256() -> str:
    import importlib.metadata
    import inspect

    from . import inference

    identity = {
        "encode_views": inspect.getsource(_encode),
        "encoder": inspect.getsource(inference.FrozenImageEncoder.encode),
        "image_preparation": inspect.getsource(inference.prepare_model_image),
        "views": ["full", "tight"],
        "normalization": "float32-l2-per-view-v1",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "open-clip-torch")
        },
    }
    return sha256_json(identity)


def _cached_embedding(
    root: Path,
    item: Any,
    encoder: Any,
    image: Image.Image,
    *,
    encoder_sha256: str,
    preprocessing_version: str,
    implementation_sha256: str,
) -> tuple[Any, bool]:
    """Load a checked per-image vector or encode and cache it under all frozen inputs."""
    import torch
    from safetensors import SafetensorError
    from safetensors.torch import load_file, save_file

    cache_identity = {
        "image_sha256": item.sha256,
        "encoder_sha256": encoder_sha256,
        "preprocessing_version": preprocessing_version,
        "implementation_sha256": implementation_sha256,
        "views": ["full", "tight"],
    }
    cache_key = sha256_json(cache_identity)
    database = resolve_private_path(root, "prototype/search/embedding-cache.sqlite3", must_exist=False)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=20)
    try:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS reference_embedding_cache (
                cache_key TEXT PRIMARY KEY,
                image_sha256 TEXT NOT NULL,
                encoder_sha256 TEXT NOT NULL,
                preprocessing_version TEXT NOT NULL,
                implementation_sha256 TEXT NOT NULL,
                vector_path TEXT NOT NULL,
                vector_sha256 TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        os.chmod(database, 0o600)
        row = connection.execute(
            "SELECT * FROM reference_embedding_cache WHERE cache_key=?", (cache_key,)
        ).fetchone()
        if row and (
            row[1] == item.sha256
            and row[2] == encoder_sha256
            and row[3] == preprocessing_version
            and row[4] == implementation_sha256
        ):
            try:
                vector_path = resolve_private_path(root, row[5])
                if sha256_file(vector_path) == row[6]:
                    vector = load_file(str(vector_path), device="cpu").get("embedding")
                    dimension = 512 if encoder.model_id == MODEL_IDS[0] else 384
                    if (
                        vector is not None
                        and row[7] == dimension
                        and tuple(vector.shape) == (2, dimension)
                        and vector.dtype == torch.float32
                        and torch.isfinite(vector).all()
                        and torch.allclose(vector.norm(dim=-1), torch.ones(2), atol=1e-4)
                    ):
                        return vector.contiguous(), True
            except (MLError, OSError, ValueError, RuntimeError, KeyError, TypeError, SafetensorError):
                pass

        vector = _encode(encoder, image).detach().cpu().contiguous().float()
        dimension = 512 if encoder.model_id == MODEL_IDS[0] else 384
        if tuple(vector.shape) != (2, dimension) or not torch.isfinite(vector).all():
            raise MLError(f"Encoder produced an invalid embedding for {item.asset_id}")
        vector = torch.nn.functional.normalize(vector, dim=-1)
        relative = f"prototype/search/embedding-cache/{cache_key[:2]}/{cache_key}.safetensors"
        path = resolve_private_path(root, relative, must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_name(path.name + f".{uuid4()}.part")
        save_file({"embedding": vector}, str(temporary))
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        connection.execute(
            """INSERT OR REPLACE INTO reference_embedding_cache
               (cache_key,image_sha256,encoder_sha256,preprocessing_version,implementation_sha256,
                vector_path,vector_sha256,dimensions,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                cache_key,
                item.sha256,
                encoder_sha256,
                preprocessing_version,
                implementation_sha256,
                relative,
                sha256_file(path),
                dimension,
                datetime.now(UTC).isoformat(),
            ),
        )
        connection.commit()
        return vector, False
    finally:
        connection.close()


def prepare_reference_neighbors(
    root: Path,
    candidate_manifest_relative: str,
    output_relative: str,
) -> dict[str, Any]:
    """Cache MobileCLIP candidate vectors and propose semantically close pairs."""
    from types import SimpleNamespace

    import torch
    from safetensors.torch import load_file

    root = root.resolve(strict=True)
    manifest_path = resolve_private_path(root, candidate_manifest_relative)
    manifest = load_json(manifest_path)
    candidates = manifest.get("items")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "drawable-reference-embedding-candidates"
        or not isinstance(candidates, list)
        or not 1 <= len(candidates) <= 5000
    ):
        raise MLError("Reference embedding candidate manifest is malformed or outside its bounds")
    ids = [item.get("candidate_id") for item in candidates if isinstance(item, dict)]
    if len(ids) != len(candidates) or len(ids) != len(set(ids)):
        raise MLError("Reference embedding candidate identities are malformed")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    encoder = _load_models(root, structural=False)["semantic"]
    registry_row = _models()[MODEL_IDS[0]]
    source = registry_row["artifact_source"]
    preprocessing = json.dumps(
        registry_row["preprocessing_profile"], sort_keys=True, separators=(",", ":")
    )
    implementation_sha256 = sha256_file(Path(__file__))
    embedding_implementation_sha256 = _embedding_implementation_sha256()
    candidate_vectors = []
    cache_hits = cache_misses = 0
    for entry in candidates:
        relative = str(entry.get("relative_path", ""))
        path = resolve_private_path(root, relative)
        if path.stat().st_size != int(entry.get("byte_length", -1)) or sha256_file(path) != entry.get("image_sha256"):
            raise MLError(f"Reference candidate bytes changed before embedding: {entry.get('candidate_id')}")
        with Image.open(path) as image:
            image.load()
            vector, reused = _cached_embedding(
                root,
                SimpleNamespace(
                    asset_id=str(entry["candidate_id"]),
                    sha256=str(entry["image_sha256"]),
                    relative_path=relative,
                ),
                encoder,
                image,
                encoder_sha256=str(source["weights_sha256"]),
                preprocessing_version=preprocessing,
                implementation_sha256=embedding_implementation_sha256,
            )
        candidate_vectors.append(vector)
        cache_hits += int(reused)
        cache_misses += int(not reused)
    candidate_tensor = torch.stack(candidate_vectors).contiguous().float()

    active_manifest = load_build(root, validate_files=False)
    active_snapshot = _snapshot(root, active_manifest["snapshot_path"], validate_files=False)
    vector_path = resolve_private_path(
        root,
        f"prototype/search/builds/{active_manifest['build_id']}/vectors.safetensors",
    )
    active_tensor = load_file(str(vector_path), device="cpu").get("semantic")
    if (
        active_tensor is None
        or tuple(active_tensor.shape) != (len(active_snapshot.items), 2, 512)
        or active_tensor.dtype != torch.float32
        or not torch.isfinite(active_tensor).all()
    ):
        raise MLError("Active prototype semantic vectors are unavailable for duplicate proposals")
    active_pointer = load_json(resolve_private_path(root, "prototype/search/active.json"))
    active_items = list(active_snapshot.items)
    if isinstance(active_snapshot, PrototypeGalleryCatalogV2):
        availability_relative = active_pointer.get("availability_path")
        if not availability_relative:
            active_ids: set[str] = set()
        else:
            availability_path = resolve_private_path(root, availability_relative)
            if sha256_file(availability_path) != active_pointer.get("availability_sha256"):
                active_ids = set()
            else:
                availability = load_json(availability_path)
                rows = availability.get("items")
                by_id = {item.asset_id: item for item in active_items}
                if availability.get("catalog_id") != active_snapshot.catalog_id or not isinstance(rows, dict):
                    active_ids = set()
                else:
                    active_ids = {
                        asset_id
                        for asset_id, row in rows.items()
                        if asset_id in by_id
                        and isinstance(row, dict)
                        and row.get("sha256") == by_id[asset_id].sha256
                        and row.get("reviewed_revision") == by_id[asset_id].asset_revision
                    }
    else:
        active_ids = {item.asset_id for item in active_items}
    active_indexes = [index for index, item in enumerate(active_items) if item.asset_id in active_ids]
    active_embedding_implementation = active_manifest.get("embedding_cache_implementation_sha256")
    if active_indexes and active_embedding_implementation != embedding_implementation_sha256:
        active_vectors = []
        active_cache_hits = active_cache_misses = 0
        for index in active_indexes:
            item = active_items[index]
            image_path = resolve_private_path(root, item.relative_path)
            if sha256_file(image_path) != item.sha256:
                raise MLError(f"Active reference bytes changed before embedding: {item.asset_id}")
            with Image.open(image_path) as image:
                image.load()
                vector, reused = _cached_embedding(
                    root,
                    item,
                    encoder,
                    image,
                    encoder_sha256=str(source["weights_sha256"]),
                    preprocessing_version=preprocessing,
                    implementation_sha256=embedding_implementation_sha256,
                )
            active_vectors.append(vector)
            active_cache_hits += int(reused)
            active_cache_misses += int(not reused)
        active_tensor = torch.stack(active_vectors).contiguous().float()
    else:
        active_cache_hits = active_cache_misses = 0
    if active_indexes:
        active_ids_ordered = [active_items[index].asset_id for index in active_indexes]
        active_hashes = [active_items[index].sha256 for index in active_indexes]
        selected_active_tensor = (
            active_tensor if active_embedding_implementation != embedding_implementation_sha256
            else active_tensor[active_indexes]
        )
        reference_tensor = torch.cat((selected_active_tensor, candidate_tensor), dim=0)
        reference_ids = active_ids_ordered + [str(item["candidate_id"]) for item in candidates]
        reference_hashes = active_hashes + [str(item["image_sha256"]) for item in candidates]
    else:
        reference_tensor = candidate_tensor
        reference_ids = [str(item["candidate_id"]) for item in candidates]
        reference_hashes = [str(item["image_sha256"]) for item in candidates]
    similarities = torch.einsum("ivd,jvd->ijv", candidate_tensor, reference_tensor).mean(dim=-1)
    neighbors: dict[str, list[dict[str, Any]]] = {}
    threshold = float(manifest.get("similarity_threshold", 0.94))
    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate["candidate_id"])
        matches = []
        for target_index in range(len(reference_ids)):
            if reference_ids[target_index] == candidate_id:
                continue
            score = float(similarities[index, target_index].item())
            if score >= threshold:
                matches.append((score, reference_ids[target_index], reference_hashes[target_index]))
        matches.sort(key=lambda row: (-row[0], row[1]))
        neighbors[candidate_id] = [
            {"reference_id": reference_id, "reference_sha256": reference_hash, "cosine": score}
            for score, reference_id, reference_hash in matches[:8]
        ]
    output = {
        "schema_version": 1,
        "kind": "drawable-reference-embedding-neighbors",
        "run_id": manifest.get("run_id"),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "encoder_model_id": encoder.model_id,
        "encoder_sha256": source["weights_sha256"],
        "preprocessing_version": preprocessing,
        "implementation_sha256": implementation_sha256,
        "similarity_threshold": threshold,
        "candidate_count": len(candidates),
        "active_gallery_count": len(active_indexes),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "active_cache_hits": active_cache_hits,
        "active_cache_misses": active_cache_misses,
        "neighbors": neighbors,
    }
    write_json_atomic(resolve_private_path(root, output_relative, must_exist=False), output)
    return {key: value for key, value in output.items() if key != "neighbors"} | {
        "output_path": output_relative
    }


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


def prepare(
    root: Path,
    snapshot_relative: str | None = None,
    *,
    structural: bool = True,
    stage_only: bool = False,
) -> dict[str, Any]:
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
    selected_model_ids = [MODEL_IDS[0]] + ([MODEL_IDS[1]] if structural else [])
    files = acquire_models(root, selected_model_ids)
    model_pins = {
        key: {
            "artifact_source": row["artifact_source"],
            "preprocessing_profile": row["preprocessing_profile"],
        }
        for key, row in _models().items()
        if key in selected_model_ids
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
    encoders = _load_models(root, structural=structural)
    registry = _models()
    implementation_sha256 = sha256_file(Path(__file__))
    embedding_implementation_sha256 = _embedding_implementation_sha256()
    tensors: dict[str, Any] = {}
    model_times: dict[str, list[float]] = {}
    cache_counts: dict[str, dict[str, int]] = {}
    for key, encoder in encoders.items():
        embeddings = []
        times = []
        hits = misses = 0
        model_source = registry[encoder.model_id]["artifact_source"]
        preprocessing = json.dumps(
            registry[encoder.model_id]["preprocessing_profile"], sort_keys=True, separators=(",", ":")
        )
        for item in snapshot.items:
            with Image.open(resolve_private_path(root, item.relative_path)) as image_source:
                if (
                    image_source.size != (item.width, item.height)
                    or image_source.width * image_source.height > 40_000_000
                ):
                    raise MLError(
                        f"Reviewed image dimensions changed or are too large: {item.asset_id}"
                    )
                image_source.load()
                tick = time.perf_counter()
                vector, reused = _cached_embedding(
                    root,
                    item,
                    encoder,
                    image_source,
                    encoder_sha256=model_source["weights_sha256"],
                    preprocessing_version=preprocessing,
                    implementation_sha256=embedding_implementation_sha256,
                )
                embeddings.append(vector)
                hits += int(reused)
                misses += int(not reused)
                times.append((time.perf_counter() - tick) * 1000)
        tensors[key] = torch.stack(embeddings).contiguous().float()
        model_times[key] = times
        cache_counts[key] = {"hits": hits, "misses": misses}
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
        profiles = ["semantic", "semantic-structural"] if "structural" in tensors else ["semantic"]
        for profile in profiles:
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
            "enabled_branches": list(tensors),
            "model_pins": model_pins,
            "model_files": files,
            "vectors_sha256": sha256_file(staging / "vectors.safetensors"),
            "ranking": RANKING,
            "encoder_code_sha256": sha256_file(Path(inference.__file__)),
            "implementation_code_sha256": implementation_sha256,
            "embedding_cache_implementation_sha256": embedding_implementation_sha256,
            "measurements": {
                "image_views_ms": model_times,
                "embedding_cache": cache_counts,
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
        # Re-read checksums and schema before exposing the build. The current active
        # pointer remains untouched when a staged build is requested or validation fails.
        load_build(root, build_id=build_id)
        if not stage_only:
            activate_build(root, build_id)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "status": "staged" if stage_only else "ready",
        "build_id": build_id,
        "item_count": len(asset_ids),
        "reused": False,
        "self_match_top1": self_checks,
        "training_started": False,
    }


def load_build(
    root: Path,
    *,
    build_id: str | None = None,
    validate_files: bool = True,
) -> dict[str, Any]:
    active_pointer = None
    if build_id is None:
        active_pointer = load_json(resolve_private_path(root, "prototype/search/active.json"))
        build_id = active_pointer.get("build_id", "")
    if (
        not isinstance(build_id, str)
        or not build_id.startswith("prototype-")
        or len(build_id) != 34
        or any(char not in "0123456789abcdef" for char in build_id[10:])
    ):
        raise MLError("Prototype active build ID is malformed")
    relative = f"prototype/search/builds/{build_id}/manifest.json"
    manifest_path = resolve_private_path(root, relative)
    manifest_sha256 = sha256_file(manifest_path)
    if active_pointer is not None and manifest_sha256 != active_pointer.get("manifest_sha256"):
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
    snapshot = _snapshot(root, manifest["snapshot_path"], validate_files=validate_files)
    if manifest["asset_ids"] != [item.asset_id for item in snapshot.items]:
        raise MLError("Prototype vector IDs do not match the reviewed batch")
    if (
        sha256_file(
            resolve_private_path(root, f"prototype/search/builds/{build_id}/vectors.safetensors")
        )
        != manifest["vectors_sha256"]
    ):
        raise MLError("Prototype vectors checksum changed")
    model_ids = manifest.get("enabled_branches")
    if not isinstance(model_ids, list):
        # V1 build manifests pinned both frozen encoders and remain readable.
        model_ids = list(manifest["model_pins"])
    branch_to_model_id = {"semantic": MODEL_IDS[0], "structural": MODEL_IDS[1]}
    selected_model_ids = [branch_to_model_id.get(branch, branch) for branch in model_ids]
    if not selected_model_ids or not set(selected_model_ids).issubset(_models()):
        raise MLError("Prototype build has an unsupported enabled encoder branch")
    current_pins = {
        key: {
            "artifact_source": row["artifact_source"],
            "preprocessing_profile": row["preprocessing_profile"],
        }
        for key, row in _models().items()
        if key in selected_model_ids
    }
    if manifest["model_pins"] != current_pins:
        raise MLError("Pinned model/preprocessing configuration changed; prepare search again")
    if manifest.get("encoder_code_sha256") != sha256_file(Path(inference.__file__)):
        raise MLError("Encoder/preprocessing code changed; prepare search again")
    snapshot_document = json.loads(snapshot_path.read_bytes())
    if snapshot_document.get("schema_version") == 2 and manifest.get("implementation_code_sha256") != sha256_file(Path(__file__)):
        raise MLError("Prototype indexing implementation changed; prepare search again")
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
        if model_id in selected_model_ids
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


def activate_build(root: Path, build_id: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest = load_build(root, build_id=build_id)
    pointer: dict[str, Any] = {
        "build_id": build_id,
        "manifest_sha256": sha256_file(resolve_private_path(root, f"prototype/search/builds/{build_id}/manifest.json")),
    }
    snapshot = _snapshot(root, manifest["snapshot_path"])
    if isinstance(snapshot, PrototypeGalleryCatalogV2):
        availability_dir = resolve_private_path(root, "prototype/search/availability", must_exist=False)
        candidates: list[tuple[int, Path]] = []
        if availability_dir.is_dir():
            for path in availability_dir.glob(f"{snapshot.catalog_id}*.json"):
                if path.is_symlink() or not path.is_file():
                    continue
                revision = snapshot.availability_revision
                suffix = path.stem.removeprefix(snapshot.catalog_id)
                if suffix.startswith("-r") and suffix[2:].isdigit():
                    revision = int(suffix[2:])
                candidates.append((revision, path))
        if not candidates:
            raise MLError("V2 gallery availability mask is missing")
        _, availability_path = max(candidates, key=lambda row: (row[0], row[1].name))
        availability_relative = availability_path.relative_to(root).as_posix()
        availability = load_json(availability_path)
        by_id = {item.asset_id: item for item in snapshot.items}
        if (
            availability.get("schema_version") != 1
            or availability.get("catalog_id") != snapshot.catalog_id
            or availability.get("revision", -1) < snapshot.availability_revision
            or not isinstance(availability.get("items"), dict)
            or not set(availability["items"]).issubset(by_id)
        ):
            raise MLError("V2 gallery availability mask is malformed or stale")
        for asset_id, row in availability["items"].items():
            item = by_id[asset_id]
            if not isinstance(row, dict) or row.get("sha256") != item.sha256 or row.get("reviewed_revision") != item.asset_revision:
                raise MLError("V2 gallery availability mask differs from the reviewed catalog")
        pointer["availability_path"] = availability_relative
        pointer["availability_sha256"] = sha256_file(availability_path)
    pointer_path = resolve_private_path(root, "prototype/search/active.json", must_exist=False)
    write_json_atomic(pointer_path, pointer, refuse_overwrite=False)
    return {"status": "active", **pointer}


class PrototypeGallery:
    def __init__(
        self,
        root: Path,
        *,
        structural: bool = False,
        build_id: str | None = None,
        tolerate_corrupt_assets: bool = False,
        staged_validation: bool = False,
    ):
        import torch
        from safetensors.torch import load_file

        self.root = root.resolve(strict=True)
        self.manifest = load_build(
            self.root, build_id=build_id, validate_files=not tolerate_corrupt_assets
        )
        self.snapshot = _snapshot(
            self.root,
            self.manifest["snapshot_path"],
            validate_files=not tolerate_corrupt_assets,
        )
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
            if key == "structural" and tensor is None and not structural:
                continue
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
        self.encoders = _load_models(self.root, structural=structural and "structural" in self.vectors)
        self._active_at_start = self.manifest["build_id"]
        self._staged_validation = staged_validation
        self._fingerprints: dict[str, tuple[int, int, int, int]] = {}
        self._invalid_ids: set[str] = set()
        for item in self.snapshot.items:
            try:
                path = resolve_private_path(self.root, item.relative_path)
                stat = path.stat()
                if not path.is_file() or stat.st_size != item.byte_length:
                    self._invalid_ids.add(item.asset_id)
                    continue
                if getattr(item, "thumbnail_path", None):
                    thumbnail = resolve_private_path(self.root, item.thumbnail_path)
                    if (
                        not thumbnail.is_file()
                        or thumbnail.stat().st_size > 4 * 1024 * 1024
                        or sha256_file(thumbnail) != item.thumbnail_sha256
                    ):
                        self._invalid_ids.add(item.asset_id)
                        continue
                self._fingerprints[item.asset_id] = (
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                )
            except (MLError, OSError, ValueError):
                self._invalid_ids.add(item.asset_id)
        self._initial_availability = self._read_availability()
        warm_image = None
        for item in self.snapshot.items:
            if item.asset_id in self._invalid_ids:
                continue
            try:
                with Image.open(resolve_private_path(self.root, item.relative_path)) as source:
                    source.load()
                    warm_image = source.copy()
                break
            except (MLError, OSError, ValueError):
                self._invalid_ids.add(item.asset_id)
        if warm_image is None:
            raise MLError("No readable reference remains in the selected prototype build")
        for encoder in self.encoders.values():
            _encode(encoder, warm_image)

    def status(self) -> dict[str, Any]:
        scope_coverage: dict[str, int] = {}
        style_coverage: dict[str, int] = {}
        availability = self._read_availability() if hasattr(self, "_initial_availability") else None
        available_ids = (
            set(availability.get("items", {})) - self._invalid_ids
            if availability is not None
            else {item.asset_id for item in self.snapshot.items} - self._invalid_ids
        )
        for item in self.snapshot.items:
            if item.asset_id not in available_ids:
                continue
            scope = item.labels.primary_scope
            style = item.labels.style
            scope_coverage[scope] = scope_coverage.get(scope, 0) + 1
            style_coverage[style] = style_coverage.get(style, 0) + 1
        return {
            "ready": True,
            "build_id": self.manifest["build_id"],
            "gallery_size": len(available_ids),
            "catalog_size": len(self.snapshot.items),
            "scope_coverage": scope_coverage,
            "style_coverage": style_coverage,
            "enabled_branches": list(self.encoders),
            "model_version": "+".join(encoder.model_id for encoder in self.encoders.values()),
        }

    def _read_availability(self) -> dict[str, Any] | None:
        if self._staged_validation and isinstance(self.snapshot, PrototypeGalleryCatalogV2):
            expected_catalog = self.snapshot.catalog_id
            relative = f"prototype/search/availability/{expected_catalog}.json"
            fail_closed = {"items": {}}
            try:
                path = resolve_private_path(self.root, relative)
                mask = load_json(path)
                mask_items = mask.get("items")
                item_by_id = {item.asset_id: item for item in self.snapshot.items}
                if (
                    mask.get("schema_version") != 1
                    or mask.get("catalog_id") != expected_catalog
                    or mask.get("revision", -1) < self.snapshot.availability_revision
                    or not isinstance(mask_items, dict)
                    or not set(mask_items).issubset(item_by_id)
                    or any(
                        not isinstance(row, dict)
                        or row.get("sha256") != item_by_id[asset_id].sha256
                        or row.get("reviewed_revision") != item_by_id[asset_id].asset_revision
                        for asset_id, row in mask_items.items()
                    )
                ):
                    return fail_closed
                return mask
            except (OSError, ValueError, TypeError, MLError):
                return fail_closed
        pointer_path = self.root / "prototype/search/active.json"
        fail_closed = {"items": {}} if isinstance(self.snapshot, PrototypeGalleryCatalogV2) else None
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            if pointer.get("build_id") != self.manifest["build_id"]:
                relative = pointer.get("availability_path")
                expected_hash = pointer.get("availability_sha256")
                if relative and isinstance(expected_hash, str):
                    path = resolve_private_path(self.root, relative)
                    if sha256_file(path) != expected_hash:
                        return fail_closed
                    active_mask = load_json(path)
                    active_items = active_mask.get("items")
                    if not isinstance(active_items, dict):
                        return fail_closed
                    previous_items = {item.asset_id: item for item in self.snapshot.items}
                    filtered = {
                        asset_id: row
                        for asset_id, row in active_items.items()
                        if asset_id in previous_items
                        and isinstance(row, dict)
                        and row.get("sha256") == previous_items[asset_id].sha256
                        and row.get("reviewed_revision") == previous_items[asset_id].asset_revision
                    }
                    return {"schema_version": 1, "catalog_id": getattr(self.snapshot, "catalog_id", None), "items": filtered}
                return getattr(self, "_initial_availability", fail_closed) or fail_closed
            relative = pointer.get("availability_path")
            expected_hash = pointer.get("availability_sha256")
            if not relative:
                return fail_closed
            path = resolve_private_path(self.root, relative)
            if sha256_file(path) != expected_hash:
                return fail_closed
            mask = load_json(path)
            expected_catalog = getattr(self.snapshot, "catalog_id", None)
            mask_items = mask.get("items")
            item_by_id = {item.asset_id: item for item in self.snapshot.items}
            if (
                mask.get("schema_version") != 1
                or mask.get("catalog_id") != expected_catalog
                or not isinstance(mask_items, dict)
                or not set(mask_items).issubset(item_by_id)
                or any(
                    not isinstance(row, dict)
                    or row.get("sha256") != item_by_id[asset_id].sha256
                    or row.get("reviewed_revision") != item_by_id[asset_id].asset_revision
                    for asset_id, row in mask_items.items()
                )
            ):
                return fail_closed
            return mask
        except (OSError, ValueError, TypeError, MLError):
            return getattr(self, "_initial_availability", fail_closed) or fail_closed

    def search(self, png: bytes) -> dict[str, Any]:
        started = time.perf_counter()
        with Image.open(io.BytesIO(png)) as image:
            if image.format != "PNG" or image.size != (512, 512):
                raise MLError("Prototype query must be a 512 × 512 PNG")
            image.load()
            query = {key: _encode(encoder, image) for key, encoder in self.encoders.items()}
        ranked = rank_vectors(self.vectors, query, self.manifest["asset_ids"])
        availability = self._read_availability()
        available_ids = (
            set(availability.get("items", {}))
            if availability is not None
            else {item.asset_id for item in self.snapshot.items}
        ) - self._invalid_ids
        candidates: list[tuple[int, float]] = []
        withdrawn_asset_ids: set[str] = set()
        for index, score in ranked:
            if score < RANKING["visual_floor"]:
                break
            item = self.snapshot.items[index]
            if item.asset_id not in available_ids:
                continue
            try:
                path = resolve_private_path(self.root, item.relative_path)
                stat = path.stat()
                fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if stat.st_size != item.byte_length:
                    self._invalid_ids.add(item.asset_id)
                    withdrawn_asset_ids.add(item.asset_id)
                    continue
                if fingerprint != self._fingerprints.get(item.asset_id):
                    if sha256_file(path) != item.sha256:
                        self._invalid_ids.add(item.asset_id)
                        withdrawn_asset_ids.add(item.asset_id)
                        continue
                    self._fingerprints[item.asset_id] = fingerprint
            except (MLError, OSError):
                self._invalid_ids.add(item.asset_id)
                withdrawn_asset_ids.add(item.asset_id)
                continue
            candidates.append((index, score))

        def make_card(index: int, score: float) -> dict[str, Any]:
            item = self.snapshot.items[index]
            scope = item.labels.primary_scope
            thumbnail = bool(getattr(item, "thumbnail_path", None))
            suffix = "&thumbnail=1" if thumbnail else ""
            return {
                "id": item.asset_id,
                "title": f"{item.source_work_id or item.source_id} · {scope.replace('_', ' ')}{' crop' if item.parent_asset_id else ''}",
                "image_url": f"/api/v1/assets/{item.asset_id}/preview?build_id={self.manifest['build_id']}{suffix}",
                "style": STYLE_NAMES[item.labels.style],
                "scope": scope,
                "source": item.attribution,
                "native": item.parent_asset_id is None,
                "match": "Tentative",
                "trace_allowed": item.permissions.trace,
                "visual_score": score,
            }

        def diverse_row(rows: list[tuple[int, float]], seen: set[str]) -> list[dict[str, Any]]:
            selected: list[dict[str, Any]] = []
            roots: set[str] = set()
            clips: dict[str, int] = {}
            sources: dict[str, int] = {}
            for index, score in rows:
                item = self.snapshot.items[index]
                canonical = item.labels.duplicate_of or item.asset_id
                root_id = str(getattr(item, "root_asset_id", item.parent_asset_id or item.asset_id))
                clip_id = str(getattr(item, "clip_id", None) or item.source_work_id or item.source_id)
                source_id = item.source_work_id or item.source_id
                if canonical in seen or root_id in seen or root_id in roots or clips.get(clip_id, 0) >= 2 or sources.get(source_id, 0) >= 2:
                    continue
                selected.append(make_card(index, score))
                seen.add(canonical)
                seen.add(root_id)
                roots.add(root_id)
                clips[clip_id] = clips.get(clip_id, 0) + 1
                sources[source_id] = sources.get(source_id, 0) + 1
                if len(selected) >= 4:
                    break
            # Hash final response bytes only for the bounded displayed cards.
            verified: list[dict[str, Any]] = []
            for card in selected:
                item = next(entry for entry in self.snapshot.items if entry.asset_id == card["id"])
                try:
                    path = resolve_private_path(self.root, item.relative_path)
                    if sha256_file(path) != item.sha256:
                        self._invalid_ids.add(item.asset_id)
                        withdrawn_asset_ids.add(item.asset_id)
                        continue
                except (MLError, OSError):
                    self._invalid_ids.add(item.asset_id)
                    withdrawn_asset_ids.add(item.asset_id)
                    continue
                verified.append(card)
            return verified

        seen_canonical: set[str] = set()
        best = diverse_row(candidates, seen_canonical)
        groups: list[dict[str, Any]] = []
        if best:
            groups.append({"id": "best", "title": "Best Match", "tentative": True, "results": best})
        for style, title in STYLE_NAMES.items():
            style_rows = [(index, score) for index, score in candidates if self.snapshot.items[index].labels.style == style]
            cards = diverse_row(style_rows, seen_canonical)
            if cards:
                groups.append({"id": f"style_{style}", "title": title, "tentative": True, "results": cards})
            if len(groups) >= 6:
                break
        groups = groups[:6]
        return {
            **self.status(),
            "mode": "provisional" if groups else "insufficient",
            "reason": "experimental_visual_matches" if groups else "no_relevant_candidates",
            "interpretation": "Tentative matches from the available reviewed references."
            if groups
            else "No available reviewed references passed the visual relevance floor.",
            "groups": groups,
            "withdrawn_asset_ids": sorted(withdrawn_asset_ids),
            "inference_ms": (time.perf_counter() - started) * 1000,
            "warnings": [
                "Experimental scores are not calibrated confidence.",
                "Query scope and viewpoint compatibility have not been validated.",
            ],
        }
