from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ppe_runtime.pipeline.types import Detection
from ppe_runtime.utils.config import EngineConfig
from ppe_runtime.utils.geometry import nms, unletterbox, xywh2xyxy


class Detector(ABC):
    """Object detector backend: a BGR frame in, detections in original-frame pixel xyxy out."""

    backend: str = ""

    def __init__(self, cfg: EngineConfig, names: dict[int, str] | None = None) -> None:
        self.cfg = cfg
        self._names_override = dict(names or {})
        self.names: dict[int, str] = {}

    def set_names(self, model_names: dict[int, str] | None) -> None:
        """Model-embedded names, then configs/ppe/classes.yaml overrides on top."""
        self.names = {**(model_names or {}), **self._names_override}

    def name_of(self, class_id: int) -> str:
        return self.names.get(class_id, str(class_id))

    @abstractmethod
    def predict(self, image: np.ndarray) -> list[Detection]: ...

    def warmup(self, runs: int = 3) -> None:
        dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        for _ in range(runs):
            self.predict(dummy)

    def close(self) -> None:  # noqa: B027 - optional hook
        pass

    # --- shared by native (ONNX / TensorRT) backends ---
    def decode(self, output: np.ndarray, gain: float, pad: tuple[float, float], shape: tuple[int, ...]) -> list[Detection]:
        """Decode a YOLO detect head output for one image.

        Handles both export layouts:
          - end2end / NMS-free (YOLO26 default): (1, max_det, 6) rows of x1, y1, x2, y2, score, class
          - raw: (1, 4 + nc, anchors) columns of cx, cy, w, h, class scores… → NMS here
        """
        out = np.asarray(output, dtype=np.float32)
        if out.ndim == 3:
            out = out[0]
        if out.ndim != 2:
            raise ValueError(f"Unexpected detector output shape {output.shape}")
        if out.shape[1] == 6 and out.shape[0] > out.shape[1]:  # end2end
            boxes, scores, classes = out[:, :4], out[:, 4], out[:, 5].astype(int)
            keep = scores >= self.cfg.conf
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        else:
            preds = out.T if out.shape[0] < out.shape[1] else out  # → (anchors, 4 + nc)
            class_scores = preds[:, 4:]
            classes = class_scores.argmax(1)
            scores = class_scores[np.arange(len(classes)), classes]
            keep = scores >= self.cfg.conf
            boxes, scores, classes = xywh2xyxy(preds[keep, :4]), scores[keep], classes[keep]
            if len(boxes):
                # class-aware NMS: offset boxes per class so different classes never suppress each other
                idx = nms(boxes + classes[:, None] * 7680.0, scores, self.cfg.iou)[:300]
                boxes, scores, classes = boxes[idx], scores[idx], classes[idx]
        if not len(boxes):
            return []
        boxes = unletterbox(boxes, gain, pad, shape)
        return [
            Detection(int(c), self.name_of(int(c)), float(s), (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            for b, s, c in zip(boxes, scores, classes)
        ]

    @staticmethod
    def parse_names(value: Any) -> dict[int, str] | None:
        """Names from exported-model metadata: a dict, or its string repr (ONNX), with str or int keys."""
        if value is None:
            return None
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return {int(k): str(v) for k, v in dict(value).items()}
