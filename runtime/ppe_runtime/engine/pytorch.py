from __future__ import annotations

import numpy as np

from ppe_runtime.engine.base import Detector
from ppe_runtime.pipeline.types import Detection
from ppe_runtime.utils.config import EngineConfig


class PyTorchDetector(Detector):
    """Ultralytics YOLO on PyTorch (.pt) — reference backend, uses the in-repo `ultralytics` fork."""

    backend = "pytorch"

    def __init__(self, cfg: EngineConfig, names: dict[int, str] | None = None) -> None:
        super().__init__(cfg, names)
        from ultralytics import YOLO

        self._model = YOLO(cfg.weights, task="detect")
        self._kwargs = {
            "conf": cfg.conf,
            "iou": cfg.iou,
            "imgsz": cfg.imgsz,
            "device": cfg.device,
            "half": cfg.half,
            "verbose": False,
        }
        self.set_names(dict(self._model.names))

    def predict(self, image: np.ndarray) -> list[Detection]:
        boxes = self._model.predict(image, **self._kwargs)[0].boxes
        if boxes is None or not len(boxes):
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        return [
            Detection(int(c), self.name_of(int(c)), float(s), (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            for b, s, c in zip(xyxy, conf, cls)
        ]
