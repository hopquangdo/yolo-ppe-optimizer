from __future__ import annotations

import numpy as np

from ppe_runtime.engine.base import Detector
from ppe_runtime.pipeline.types import Detection
from ppe_runtime.utils.config import EngineConfig


class SyntheticDetector(Detector):
    """Deterministic fake detector for tests and dry-runs (no model, no GPU).

    `persons` people walk right 2 px/frame, all wearing helmets; odd-indexed people have no vest.
    """

    backend = "synthetic"

    def __init__(self, cfg: EngineConfig | None = None, names: dict[int, str] | None = None, persons: int = 3) -> None:
        super().__init__(cfg or EngineConfig(backend="synthetic"), names)
        self.set_names({0: "person", 1: "helmet", 2: "vest"})
        self.persons = persons
        self._frame = 0

    def predict(self, image: np.ndarray) -> list[Detection]:
        self._frame += 1
        out = []
        for i in range(self.persons):
            x, y = 50 + i * 150 + self._frame * 2, 100
            out.append(Detection(0, self.name_of(0), 0.9, (x, y, x + 80, y + 200)))
            out.append(Detection(1, self.name_of(1), 0.8, (x + 20, y, x + 60, y + 30)))
            if i % 2 == 0:
                out.append(Detection(2, self.name_of(2), 0.8, (x + 10, y + 50, x + 70, y + 120)))
        return out
