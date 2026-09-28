"""Inference backends. Heavy dependencies (torch, onnxruntime, tensorrt) are imported only by the chosen backend."""

from ppe_runtime.engine.base import Detector
from ppe_runtime.engine.factory import create_engine, resolve_backend

__all__ = ["Detector", "create_engine", "resolve_backend"]
