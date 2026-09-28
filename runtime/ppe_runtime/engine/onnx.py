from __future__ import annotations

import logging

import numpy as np

from ppe_runtime.engine.base import Detector
from ppe_runtime.pipeline.types import Detection
from ppe_runtime.utils.config import EngineConfig
from ppe_runtime.utils.geometry import to_blob

logger = logging.getLogger(__name__)


class OnnxDetector(Detector):
    """Native ONNX Runtime backend: numpy letterbox → session.run → decode (no torch on the hot path)."""

    backend = "onnx"

    def __init__(self, cfg: EngineConfig, names: dict[int, str] | None = None) -> None:
        super().__init__(cfg, names)
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        providers = [p for p in cfg.providers if p in available] or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(cfg.weights, providers=providers)
        self._input = self._session.get_inputs()[0]
        self._half = "float16" in self._input.type
        shape = self._input.shape  # [1, 3, H, W]; H/W are strings when exported dynamic
        self.imgsz = shape[2] if isinstance(shape[2], int) else cfg.imgsz
        meta = self._session.get_modelmeta().custom_metadata_map or {}
        self.set_names(self.parse_names(meta.get("names")))
        logger.info("ONNX %s on %s, input %s", cfg.weights, self._session.get_providers()[0], shape)

    def predict(self, image: np.ndarray) -> list[Detection]:
        blob, gain, pad = to_blob(image, self.imgsz, self._half)
        output = self._session.run(None, {self._input.name: blob})[0]
        return self.decode(output, gain, pad, image.shape)
