from __future__ import annotations

from ppe_runtime.tracking.base import Tracker
from ppe_runtime.utils.config import TrackingConfig


def create_tracker(cfg: TrackingConfig | None, classes: set[str] | None = None) -> Tracker | None:
    """`None` config = tracking disabled."""
    if cfg is None:
        return None
    if cfg.tracker_type == "botsort":
        from ppe_runtime.tracking.botsort import BotSortTracker

        return BotSortTracker(cfg, classes)
    from ppe_runtime.tracking.bytetrack import ByteTrackTracker

    return ByteTrackTracker(cfg, classes)
