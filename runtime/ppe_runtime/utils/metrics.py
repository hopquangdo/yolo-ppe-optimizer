"""Latency statistics and the rolling metrics window reported to sinks."""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from typing import Any

from ppe_runtime.pipeline.types import FrameResult


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(values: list[float], digits: int = 3) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "mean": round(statistics.fmean(values), digits),
        "p50": round(percentile(values, 50), digits),
        "p95": round(percentile(values, 95), digits),
        "p99": round(percentile(values, 99), digits),
    }


# Keys of `snapshot()` that belong to the edge-ops `RuntimeTelemetry` contract; the rest are runtime-only extras.
TELEMETRY_KEYS = frozenset({
    "runtime_uptime_sec", "runtime_queue_size", "runtime_error_rate", "runtime_status",
    "inference_fps", "target_fps", "effective_fps_ratio", "inference_latency_ms",
    "inference_p50_ms", "inference_p95_ms", "inference_p99_ms", "inference_error_rate",
    "camera_fps", "camera_target_fps", "camera_frame_drop_pct", "camera_status",
    "detection_count", "confidence_mean", "confidence_std", "confidence_min", "class_entropy", "output_change_rate",
})  # fmt: skip


class RuntimeMetrics:
    """Counters for one reporting window; `snapshot()` returns the window and starts the next."""

    def __init__(self, target_fps: float = 0.0) -> None:
        self.target_fps = target_fps
        self.violations_total = 0
        self._started = time.monotonic()
        self._prev_classes: Counter | None = None
        self._reset()

    def _reset(self) -> None:
        self._window_start = time.monotonic()
        self._frames = 0
        self._dropped = 0
        self._errors = 0
        self._stages: dict[str, list[float]] = {"detect_ms": [], "track_ms": [], "rules_ms": [], "total_ms": []}
        self._confidences: list[float] = []
        self._classes: Counter = Counter()
        self._persons = 0
        self._non_compliant = 0
        self._violations = 0

    def frame_dropped(self) -> None:
        self._dropped += 1

    def frame_failed(self) -> None:
        self._errors += 1

    def record(self, result: FrameResult) -> None:
        self._frames += 1
        t = result.timings
        for key, value in (("detect_ms", t.detect_ms), ("track_ms", t.track_ms), ("rules_ms", t.rules_ms),
                           ("total_ms", t.total_ms)):
            self._stages[key].append(value)
        for d in result.detections:
            self._confidences.append(d.confidence)
            self._classes[d.class_name] += 1
        self._persons += len(result.persons)
        self._non_compliant += sum(1 for p in result.persons if p.missing)
        self._violations += len(result.violations)
        self.violations_total += len(result.violations)

    def snapshot(self, queue_size: int | None = None, camera_connected: bool | None = None) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = max(now - self._window_start, 1e-3)
        attempts = self._frames + self._errors
        fps = self._frames / elapsed
        lat = summarize(self._stages["total_ms"], 2)
        conf = self._confidences
        error_rate = self._pct(self._errors, attempts)
        ratio = round(fps / self.target_fps, 4) if self.target_fps else None
        snap = {
            "runtime_uptime_sec": round(now - self._started, 1),
            "runtime_queue_size": queue_size,
            "runtime_error_rate": error_rate,
            "runtime_status": "DEGRADED" if error_rate > 10 or (ratio is not None and ratio < 0.5) else "RUNNING",
            "inference_fps": round(fps, 2),
            "target_fps": self.target_fps or None,
            "effective_fps_ratio": ratio,
            "inference_latency_ms": lat["mean"],
            "inference_p50_ms": lat["p50"],
            "inference_p95_ms": lat["p95"],
            "inference_p99_ms": lat["p99"],
            "inference_error_rate": error_rate,
            "camera_fps": round((self._frames + self._dropped + self._errors) / elapsed, 2),
            "camera_target_fps": self.target_fps or None,
            "camera_frame_drop_pct": self._pct(self._dropped, self._frames + self._dropped + self._errors),
            "camera_status": None if camera_connected is None else ("ONLINE" if camera_connected else "DISCONNECTED"),
            "detection_count": sum(self._classes.values()),
            "confidence_mean": round(statistics.fmean(conf), 4) if conf else None,
            "confidence_std": round(statistics.pstdev(conf), 4) if len(conf) > 1 else None,
            "confidence_min": round(min(conf), 4) if conf else None,
            "class_entropy": self._entropy(self._classes),
            "output_change_rate": self._change_rate(),
            # runtime-only extras
            "stage_ms": {k: summarize(v, 2) for k, v in self._stages.items() if k != "total_ms"},
            "persons": self._persons,
            "non_compliant_person_frames": self._non_compliant,
            "violations": self._violations,
            "violations_total": self.violations_total,
        }
        self._prev_classes = self._classes
        self._reset()
        return snap

    def _change_rate(self) -> float | None:
        """Total-variation distance between this window's class distribution and the previous one's."""
        if not self._prev_classes:
            return None
        prev, cur = sum(self._prev_classes.values()), sum(self._classes.values())
        if not cur:
            return None
        keys = set(self._prev_classes) | set(self._classes)
        return round(0.5 * sum(abs(self._classes[k] / cur - self._prev_classes[k] / prev) for k in keys), 4)

    @staticmethod
    def _entropy(counts: Counter) -> float | None:
        total = sum(counts.values())
        if not total:
            return None
        return round(-sum((c / total) * math.log2(c / total) for c in counts.values()), 4)

    @staticmethod
    def _pct(part: int, whole: int) -> float:
        return round(part / whole * 100, 3) if whole else 0.0
