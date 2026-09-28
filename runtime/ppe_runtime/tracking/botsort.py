from __future__ import annotations

from types import SimpleNamespace

from ppe_runtime.tracking.base import UltralyticsTracker


class BotSortTracker(UltralyticsTracker):
    """BoT-SORT: ByteTrack + camera-motion compensation (GMC) and optional ReID. Steadier IDs, more CPU per frame."""

    def _make(self, args: SimpleNamespace):
        from ultralytics.trackers.bot_sort import BOTSORT

        if getattr(args, "with_reid", False) and getattr(args, "model", "auto") == "auto":
            # "auto" reuses detector features through an ultralytics predictor hook, which this runtime doesn't use.
            raise ValueError("botsort with_reid needs an explicit ReID `model` path (e.g. yolo26n-cls.pt), not 'auto'")
        return BOTSORT(args=args)
