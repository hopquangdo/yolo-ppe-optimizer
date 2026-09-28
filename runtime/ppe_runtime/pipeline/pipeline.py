"""Detect → Track → Associate → Rules, plus the loop that feeds it from a source and reports to sinks."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ppe_runtime.engine import Detector, create_engine
from ppe_runtime.io.sink import MultiSink, Sink, create_sink
from ppe_runtime.io.stream import StreamSource, open_source
from ppe_runtime.io.video import VideoSource
from ppe_runtime.pipeline.types import FrameResult, Timings
from ppe_runtime.ppe import Associator, RuleEngine
from ppe_runtime.ppe.constants import EVENT_SOURCE_LOST, EVENT_SOURCE_RESTORED, EVENT_STARTED, EVENT_STOPPED
from ppe_runtime.tracking import Tracker, create_tracker
from ppe_runtime.utils.config import RuntimeConfig, RulesConfig
from ppe_runtime.utils.metrics import RuntimeMetrics

logger = logging.getLogger(__name__)


class Pipeline:
    """Stateful per-stream processor. Call `reset()` between unrelated videos."""

    def __init__(self, engine: Detector, rules: RulesConfig, tracker: Tracker | None = None) -> None:
        self.engine = engine
        self.tracker = tracker
        self.associator = Associator(rules)
        self.rules = RuleEngine(rules)
        self._frame = 0

    @classmethod
    def from_config(cls, cfg: RuntimeConfig) -> Pipeline:
        rules = cfg.ppe.rules
        engine = create_engine(cfg.engine, cfg.ppe.classes.names)
        return cls(engine, rules, create_tracker(cfg.tracking, set(rules.person_classes)))

    def warmup(self, runs: int = 3) -> None:
        self.engine.warmup(runs)

    def process(self, image: np.ndarray) -> FrameResult:
        self._frame += 1
        t0 = time.perf_counter()
        detections = self.engine.predict(image)
        t1 = time.perf_counter()
        if self.tracker is not None:
            detections = self.tracker.update(detections, image)
        t2 = time.perf_counter()
        persons = self.associator.associate(detections)
        violations = self.rules.evaluate(persons, self._frame)
        t3 = time.perf_counter()
        timings = Timings((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)
        return FrameResult(self._frame, detections, persons, violations, timings, time.time())

    def reset(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()
        self.rules.reset()
        self._frame = 0

    def close(self) -> None:
        self.engine.close()


class Runner:
    """Pulls frames from a source, runs the pipeline, and reports violations / metrics / events to sinks.

    File sources are processed frame by frame until they end. Live sources run until `stop()`, are paced to
    `target_fps`, and skip to the newest frame when inference falls behind.
    """

    def __init__(self, pipeline: Pipeline, source: VideoSource | StreamSource, sink: Sink,
                 target_fps: float = 0.0, report_interval_sec: float = 5.0, on_frame=None) -> None:
        self.pipeline = pipeline
        self.source = source
        self.sink = sink
        self.target_fps = target_fps
        self.report_interval = report_interval_sec
        self.metrics = RuntimeMetrics(target_fps)
        self.on_frame = on_frame  # optional callback(frame, result), e.g. drawing / saving video
        self._stop = threading.Event()

    @classmethod
    def from_config(cls, cfg: RuntimeConfig, on_frame=None, max_frames: int | None = None) -> Runner:
        pipeline = Pipeline.from_config(cfg)
        sink = MultiSink([create_sink(s) for s in cfg.sinks])
        return cls(pipeline, open_source(cfg.source, max_frames), sink, cfg.target_fps, cfg.report_interval_sec,
                   on_frame)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> RuntimeMetrics:
        live = self.source.is_live
        interval = 1.0 / self.target_fps if live and self.target_fps else 0.0
        self.pipeline.warmup()
        source_name = str(getattr(self.source, "source", None) or getattr(self.source, "path", ""))
        self.sink.on_event(EVENT_STARTED, "PPE runtime started", "INFO",
                           {"backend": self.pipeline.engine.backend, "source": source_name})
        next_report = time.monotonic() + self.report_interval
        was_connected, last_dropped = True, 0
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                frame = self.source.wait(0.5) if live else self.source.read()
                if frame is None:
                    if not live:
                        break
                    if was_connected and not self.source.connected:
                        self.sink.on_event(EVENT_SOURCE_LOST, "Frame source disconnected", "ERROR", {})
                        was_connected = False
                else:
                    if not was_connected:
                        self.sink.on_event(EVENT_SOURCE_RESTORED, "Frame source reconnected", "INFO", {})
                        was_connected = True
                    self._handle(frame)

                dropped = self.source.dropped
                for _ in range(dropped - last_dropped):
                    self.metrics.frame_dropped()
                last_dropped = dropped

                now = time.monotonic()
                if now >= next_report:
                    self.sink.on_metrics(self.metrics.snapshot(camera_connected=self.source.connected))
                    next_report = now + self.report_interval
                if interval:
                    self._stop.wait(max(0.0, interval - (time.monotonic() - started)))
        finally:
            self.sink.on_metrics(self.metrics.snapshot(camera_connected=self.source.connected))
            self.sink.on_event(EVENT_STOPPED, "PPE runtime stopped", "INFO",
                               {"violations_total": self.metrics.violations_total})
            self.source.close()
            self.sink.close()
            self.pipeline.close()
        return self.metrics

    def _handle(self, frame: np.ndarray) -> None:
        try:
            result = self.pipeline.process(frame)
        except Exception:
            logger.exception("Pipeline failed on frame")
            self.metrics.frame_failed()
            return
        self.metrics.record(result)
        for violation in result.violations:
            self.sink.on_violation(violation)
        if self.on_frame is not None:
            self.on_frame(frame, result)
