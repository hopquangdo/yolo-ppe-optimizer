from __future__ import annotations

from abc import ABC, abstractmethod
from types import SimpleNamespace

import numpy as np

from ppe_runtime.pipeline.types import Detection
from ppe_runtime.utils.config import TrackingConfig


class Tracker(ABC):
    """Assigns persistent `track_id`s to detections whose class is in `classes`; others pass through untracked.

    Only people are tracked by default: PPE items are re-associated to a person every frame, so they need no identity,
    and keeping small items out of association stops them from stealing matches from people.
    """

    def __init__(self, cfg: TrackingConfig, classes: set[str] | None = None) -> None:
        self.cfg = cfg
        self.classes = frozenset(classes if classes is not None else {"person"})

    @abstractmethod
    def update(self, detections: list[Detection], image: np.ndarray | None = None) -> list[Detection]: ...

    @abstractmethod
    def reset(self) -> None: ...


class _TrackInput:
    """The subset of ultralytics `Boxes` that BYTETracker/BOTSORT read: `conf`, `cls`, `xywh`, boolean slicing.

    `cls` carries the source detection's list index instead of its class: the trackers never use `cls` for matching,
    only echo it back, while their own `idx` column is relative to an internally filtered subset and can't be mapped
    back to the input.
    """

    def __init__(self, conf: np.ndarray, cls: np.ndarray, xywh: np.ndarray) -> None:
        self.conf, self.cls, self.xywh = conf, cls, xywh

    @classmethod
    def build(cls, indexed: list[tuple[int, Detection]]) -> _TrackInput:
        if not indexed:
            return cls(np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros((0, 4), np.float32))
        rows = [
            ((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2, d.box[2] - d.box[0], d.box[3] - d.box[1])
            for _, d in indexed
        ]
        return cls(
            np.array([d.confidence for _, d in indexed], np.float32),
            np.array([i for i, _ in indexed], np.float32),
            np.array(rows, np.float32),
        )

    def __getitem__(self, idx) -> _TrackInput:
        return _TrackInput(self.conf[idx], self.cls[idx], self.xywh[idx])

    def __len__(self) -> int:
        return len(self.conf)


class UltralyticsTracker(Tracker):
    """Drives an ultralytics tracker class directly with plain detections (no ultralytics predictor involved)."""

    def __init__(self, cfg: TrackingConfig, classes: set[str] | None = None) -> None:
        super().__init__(cfg, classes)
        self._impl = self._make(SimpleNamespace(**cfg.model_dump()))

    def _make(self, args: SimpleNamespace):
        raise NotImplementedError

    def update(self, detections: list[Detection], image: np.ndarray | None = None) -> list[Detection]:
        indexed = [(i, d) for i, d in enumerate(detections) if d.class_name in self.classes]
        out = [d for d in detections if d.class_name not in self.classes]
        rows = self._impl.update(_TrackInput.build(indexed), image)
        for row in rows:  # x1, y1, x2, y2, track_id, score, cls (= source index), idx
            det = detections[int(row[6])]
            out.append(det.with_track(int(row[4]), (float(row[0]), float(row[1]), float(row[2]), float(row[3]))))
        return out

    def reset(self) -> None:
        self._impl.reset()
