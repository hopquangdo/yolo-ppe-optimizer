"""Multi-object tracking (ByteTrack / BoT-SORT from ultralytics, fed with plain detections)."""

from ppe_runtime.tracking.base import Tracker
from ppe_runtime.tracking.factory import create_tracker

__all__ = ["Tracker", "create_tracker"]
