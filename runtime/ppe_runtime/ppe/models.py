from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ppe_runtime.pipeline.types import Detection
from ppe_runtime.utils.geometry import Box


@dataclass(frozen=True, slots=True)
class PersonStatus:
    """One person and the PPE state associated with it on the current frame."""

    person: Detection
    worn: frozenset[str]  # required items found on the person
    missing: frozenset[str]  # required items not found, plus items flagged by negative classes (e.g. no_helmet)

    @property
    def compliant(self) -> bool:
        return not self.missing


@dataclass(frozen=True, slots=True)
class Violation:
    """A person non-compliant for `min_frames` consecutive frames. Raised once per track and missing set."""

    track_id: int | None
    missing: frozenset[str]
    severity: str
    box: Box
    confidence: float
    frames: int  # consecutive non-compliant frames when raised
    frame_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "missing": sorted(self.missing),
            "severity": self.severity,
            "box": [round(v, 1) for v in self.box],
            "confidence": round(self.confidence, 4),
            "frames": self.frames,
            "frame_index": self.frame_index,
        }
