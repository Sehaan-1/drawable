"""Device resolution and runtime introspection for the GPU stages.

The notebook prints this before doing any work: a free Colab tier hands out T4,
L4, or occasionally A100, and a CPU-only runtime should be detected *before* a
long extraction rather than after. torch is imported lazily through
:mod:`linescout_ml.colab._optional` so importing this module never requires a
GPU stack.
"""

from __future__ import annotations

from typing import Any, Literal

from linescout_ml.colab._optional import has_module, optional_module

TORCH_HINT = "pip install torch torchvision (Colab already ships a CUDA build)"


def _torch() -> Any:
    return optional_module("torch", TORCH_HINT)


def torch_available() -> bool:
    """Whether torch can be imported at all (no GPU stack installed -> False)."""
    return has_module("torch")


def resolve_device(policy: Literal["auto", "cuda", "cpu"] = "auto") -> str:
    """Map the config's device policy onto a concrete torch device string.

    ``cuda`` is an error rather than a silent fallback: a run that was supposed
    to use the GPU should not quietly take an hour on CPU.
    """
    torch = _torch()
    if policy == "cpu":
        return "cpu"
    available = bool(torch.cuda.is_available())
    if policy == "cuda" and not available:
        msg = "CUDA requested but torch.cuda.is_available() is False"
        raise RuntimeError(msg)
    return "cuda" if available else "cpu"


def gpu_summary() -> dict[str, Any]:
    """Facts about the runtime, recorded in the run report."""
    summary: dict[str, Any] = {"torch": None, "cuda_available": False}
    try:
        torch = _torch()
    except RuntimeError as error:
        summary["error"] = str(error)
        return summary

    summary["torch"] = str(torch.__version__)
    summary["cuda_available"] = bool(torch.cuda.is_available())
    if not summary["cuda_available"]:
        return summary
    summary["cuda_version"] = str(torch.version.cuda)
    summary["device_count"] = int(torch.cuda.device_count())
    summary["device_name"] = str(torch.cuda.get_device_name(0))
    capability = torch.cuda.get_device_capability(0)
    summary["compute_capability"] = f"{capability[0]}.{capability[1]}"
    properties = torch.cuda.get_device_properties(0)
    summary["total_memory_gib"] = round(properties.total_memory / (1024**3), 2)
    return summary


def clear_gpu_cache() -> None:
    """Release cached blocks. Cheap insurance against fragmentation on long runs."""
    try:
        torch = _torch()
    except RuntimeError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
