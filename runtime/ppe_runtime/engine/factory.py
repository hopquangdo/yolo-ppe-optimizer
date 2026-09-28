from __future__ import annotations

from pathlib import Path

from ppe_runtime.engine.base import Detector
from ppe_runtime.utils.config import EngineConfig

_SUFFIX_BACKEND = {".pt": "pytorch", ".onnx": "onnx", ".engine": "tensorrt", ".plan": "tensorrt"}


def resolve_backend(cfg: EngineConfig) -> str:
    if cfg.backend != "auto":
        return cfg.backend
    if cfg.weights in (None, "", "synthetic"):
        return "synthetic"
    suffix = Path(cfg.weights).suffix.lower()
    if suffix not in _SUFFIX_BACKEND:
        raise ValueError(f"Cannot infer backend from {cfg.weights!r}; set engine.backend")
    return _SUFFIX_BACKEND[suffix]


def create_engine(cfg: EngineConfig, names: dict[int, str] | None = None) -> Detector:
    """Build the detector for `cfg.backend` (or infer it from the weights suffix when `auto`)."""
    backend = resolve_backend(cfg)
    if backend != "synthetic" and not cfg.weights:
        raise ValueError(f"engine.weights is required for backend {backend!r}")
    if backend == "pytorch":
        from ppe_runtime.engine.pytorch import PyTorchDetector as cls
    elif backend == "onnx":
        from ppe_runtime.engine.onnx import OnnxDetector as cls
    elif backend == "tensorrt":
        from ppe_runtime.engine.tensorrt import TensorRTDetector as cls
    else:
        from ppe_runtime.engine.synthetic import SyntheticDetector as cls
    return cls(cfg, names)
