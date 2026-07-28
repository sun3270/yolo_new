# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


__all__ = ("NAS", "RTDETR", "SAM", "YOLO", "YOLOE", "FastSAM", "YOLOWorld")

_MODEL_MODULES = {
    "FastSAM": "fastsam",
    "NAS": "nas",
    "RTDETR": "rtdetr",
    "SAM": "sam",
    "YOLO": "yolo",
    "YOLOE": "yolo",
    "YOLOWorld": "yolo",
}

if TYPE_CHECKING:
    from .fastsam import FastSAM
    from .nas import NAS
    from .rtdetr import RTDETR
    from .sam import SAM
    from .yolo import YOLO, YOLOE, YOLOWorld


def __getattr__(name: str):
    """Lazy-import model classes on first access."""
    if name in _MODEL_MODULES:
        module = importlib.import_module(f"ultralytics.models.{_MODEL_MODULES[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Return lazily available model names for IDE autocompletion."""
    return sorted(set(globals()) | set(__all__))
