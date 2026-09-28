from __future__ import annotations

from types import SimpleNamespace

from ppe_runtime.tracking.base import UltralyticsTracker


class ByteTrackTracker(UltralyticsTracker):
    """ByteTrack: IoU + Kalman association in two stages (high, then low confidence). Cheapest; default for edge."""

    def _make(self, args: SimpleNamespace):
        from ultralytics.trackers.byte_tracker import BYTETracker

        return BYTETracker(args=args)
