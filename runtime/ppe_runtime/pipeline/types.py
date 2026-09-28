"""Per-frame data passed between engine, tracker, PPE rules and sinks.

Plain dataclasses (not Pydantic): they are created for every detection of every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ppe_runtime.utils.geometry import Box

if TYPE_CHECKING:
    from ppe_runtime.ppe.models import PersonStatus, Violation


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: Box  # pixel xyxy in the original frame
    track_id: int | None = None

    def with_track(self, track_id: int, box: Box) -> Detection:
        return replace(self, track_id=track_id, box=box)


@dataclass(slots=True)
class Timings:
    """Stage latency in milliseconds."""

    detect_ms: float = 0.0
    track_ms: float = 0.0
    rules_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.detect_ms + self.track_ms + self.rules_ms


@dataclass(slots=True)
class FrameResult:
    frame_index: int
    detections: list[Detection]
    persons: list[PersonStatus] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)  # newly raised on this frame
    timings: Timings = field(default_factory=Timings)
    timestamp: float = 0.0  # wall-clock seconds (time.time()) when the frame was processed

    @property
    def tracks(self) -> list[Detection]:
        return [d for d in self.detections if d.track_id is not None]
