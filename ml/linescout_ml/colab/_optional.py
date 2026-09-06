"""Import helpers for the optional GPU stack.

``linescout_ml`` stays dependency-light on purpose: the manifest schema and the
taxonomy must import with nothing but pydantic installed. The Colab pipeline
needs torch, open_clip, and controlnet_aux, so those are resolved *at call
time* through :mod:`importlib` — which also keeps ``mypy --strict`` usable
without a GPU stack present in CI.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Any


class MissingDependencyError(RuntimeError):
    """A stage needs a package that is not installed in this runtime."""

    def __init__(self, module: str, install_hint: str) -> None:
        super().__init__(f"{module} is required for this stage — install it with: {install_hint}")
        self.module = module
        self.install_hint = install_hint


def optional_module(module: str, install_hint: str) -> ModuleType:
    """Import ``module`` or raise :class:`MissingDependencyError` with a fix."""
    try:
        return importlib.import_module(module)
    except ImportError as error:  # pragma: no cover - exercised only without the GPU stack
        raise MissingDependencyError(module, install_hint) from error


def optional_attr(module: str, attribute: str, install_hint: str) -> Any:
    """Import ``module`` and return ``attribute`` from it."""
    return getattr(optional_module(module, install_hint), attribute)


def has_module(module: str) -> bool:
    """Whether ``module`` can be imported. Never raises, never imports twice."""
    return importlib.util.find_spec(module) is not None
