from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .contracts import SAFE_ID
from .io import MLError, load_json, sha256_file, sha256_json

CHECKPOINT_NAME = re.compile(r"^checkpoint-step-([0-9]{8})$")
MAX_CHECKPOINT_TENSOR_ELEMENTS = 150_000_000
MAX_CHECKPOINT_FILE_BYTES = 2 * 1024 * 1024 * 1024


def save_checkpoint(
    run_root: Path,
    *,
    run_id: str,
    step: int,
    epoch: int,
    compatibility: dict[str, Any],
    model_state: dict[str, Any],
    optimizer: Any,
    named_parameters: dict[str, Any],
    scaler: Any | None,
    best_metric: float,
    patience: int,
    random_state: Any,
    cuda_rng_states: list[Any],
) -> Path:
    """Write one verified checkpoint generation; model and optimizer use safetensors."""
    if not SAFE_ID.fullmatch(run_id) or step < 0 or epoch < 0:
        raise MLError("Run identity, step, or epoch is invalid")
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as error:
        raise MLError("The locked ML dependency profile is required for checkpoint persistence") from error
    parent = run_root / run_id / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint-step-{step:08d}"
    destination = parent / name
    staging = parent / f".{name}.partial"
    if destination.exists() or destination.is_symlink() or staging.exists() or staging.is_symlink():
        raise MLError("Checkpoint generation already exists; refusing to overwrite or remove it")
    staging.mkdir()
    try:
        model_tensors = {key: value.detach().cpu().contiguous() for key, value in model_state.items()}
        optimizer_tensors: dict[str, Any] = {}
        optimizer_scalars: dict[str, dict[str, int | float | bool | None]] = {}
        param_ids: dict[int, str] = {id(value): name for name, value in named_parameters.items()}
        for parameter, state in optimizer.state.items():
            parameter_name = param_ids.get(id(parameter))
            if not parameter_name:
                raise MLError("Optimizer contains a parameter outside the approved trainable manifest")
            scalars: dict[str, int | float | bool | None] = {}
            for key, value in state.items():
                tensor_key = f"{parameter_name}::{key}"
                if isinstance(value, torch.Tensor):
                    optimizer_tensors[tensor_key] = value.detach().cpu().contiguous()
                elif value is None or type(value) in {int, float, bool}:
                    scalars[key] = value
                else:
                    raise MLError("Optimizer state contains a value type that cannot be safely checkpointed")
            optimizer_scalars[parameter_name] = scalars
        if scaler is not None:
            scaler_state = scaler.state_dict()
            for key, value in scaler_state.items():
                if type(value) in {int, float, bool}:
                    optimizer_scalars.setdefault("__scaler__", {})[str(key)] = value
        rng_tensors = {f"cuda_{index}": state.detach().cpu().contiguous() for index, state in enumerate(cuda_rng_states)}
        rng_tensors["torch_cpu"] = torch.get_rng_state().cpu().contiguous()
        save_file(model_tensors, str(staging / "model.safetensors"))
        save_file(optimizer_tensors or {"__empty__": torch.zeros(1)}, str(staging / "optimizer.safetensors"))
        save_file(rng_tensors or {"__empty__": torch.zeros(1)}, str(staging / "rng.safetensors"))
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "step": step,
            "epoch": epoch,
            "compatibility": compatibility,
            "model_keys": sorted(model_tensors),
            "model_shapes": {key: list(value.shape) for key, value in model_tensors.items()},
            "optimizer_keys": sorted(optimizer_tensors),
            "optimizer_scalars": optimizer_scalars,
            "optimizer_groups": [
                {
                    "param_names": [param_ids[id(parameter)] for parameter in group["params"]],
                    "lr": float(group["lr"]),
                    "weight_decay": float(group.get("weight_decay", 0.0)),
                }
                for group in optimizer.param_groups
            ],
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "best_metric": float(best_metric),
            "early_stop_patience_used": patience,
            "sampler_random_state": random_state,
        }
        metadata_path = staging / "state.json"
        metadata_path.write_text(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        _sync_file(metadata_path)
        files = {path.name: sha256_file(path) for path in sorted(staging.iterdir()) if path.is_file()}
        if sum(path.stat().st_size for path in staging.iterdir() if path.is_file()) > 3 * MAX_CHECKPOINT_FILE_BYTES:
            raise MLError("Checkpoint exceeds the configured total size limit")
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "step": step,
            "files": files,
            "compatibility_sha256": sha256_json(compatibility),
        }
        commit_path = staging / "complete.json"
        commit_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        _sync_file(commit_path)
        _sync_directory(staging)
        os.rename(staging, destination)
        _sync_directory(parent)
        return destination
    except Exception as error:
        # Partial generations remain for inspection and are ignored on resume
        # because they have no published completion marker.
        raise MLError(f"Checkpoint creation failed; incomplete staging data remains at {staging.name}: {error}") from error


def load_checkpoint(checkpoint_path: Path, *, expected_compatibility: dict[str, Any]) -> dict[str, Any]:
    if checkpoint_path.is_symlink() or not checkpoint_path.is_dir():
        raise MLError("Checkpoint must be a regular directory")
    try:
        name = checkpoint_path.name
        parsed_step = int(CHECKPOINT_NAME.fullmatch(name).group(1))  # type: ignore[union-attr]
    except (AttributeError, ValueError) as error:
        raise MLError("Checkpoint directory name is malformed") from error
    marker = load_json(checkpoint_path / "complete.json")
    if marker.get("schema_version") != 1 or type(marker.get("step")) is not int or marker["step"] != parsed_step:
        raise MLError("Checkpoint completion marker is malformed")
    for name, expected_hash in marker.get("files", {}).items():
        if name not in {"model.safetensors", "optimizer.safetensors", "rng.safetensors", "state.json"}:
            raise MLError("Checkpoint lists an unexpected file")
        path = checkpoint_path / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hash:
            raise MLError(f"Checkpoint member is missing, unsafe, or corrupt: {name}")
        if path.stat().st_size > MAX_CHECKPOINT_FILE_BYTES:
            raise MLError("Checkpoint member exceeds the safe read limit")
    state = load_json(checkpoint_path / "state.json")
    if state.get("compatibility") != expected_compatibility:
        raise MLError("Checkpoint code/data/model/recipe profile does not match this resume request")
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as error:
        raise MLError("Install the locked ML dependency profile before loading checkpoints") from error
    for filename in ("model.safetensors", "optimizer.safetensors", "rng.safetensors"):
        path = checkpoint_path / filename
        if not path.is_file():
            raise MLError(f"Checkpoint is incomplete: {filename} is missing")
        total = 0
        with safe_open(path, framework="pt", device="cpu") as safe_file:
            for key in safe_file:
                shape = safe_file.get_slice(key).get_shape()
                count = 1
                for dimension in shape:
                    count *= int(dimension)
                total += count
                if total > MAX_CHECKPOINT_TENSOR_ELEMENTS:
                    raise MLError("Checkpoint tensor header exceeds the safe allocation limit")
        load_file(str(path), device="cpu")
    return {"status": "verified", "step": parsed_step, "state": state, "path": checkpoint_path}


def _sync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
