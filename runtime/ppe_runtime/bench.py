"""Per-stage latency / FPS benchmark for comparing model variants (baseline, pruned, QAT; .pt / .onnx / .engine)."""

from __future__ import annotations

import json
import platform
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import numpy as np

from ppe_runtime.io.video import VideoSource
from ppe_runtime.pipeline.pipeline import Pipeline
from ppe_runtime.utils.config import RuntimeConfig
from ppe_runtime.utils.metrics import summarize


def load_frames(source: str | None, count: int, imgsz: int) -> list[np.ndarray]:
    """Up to `count` frames from a video (held in memory, so decoding is not timed), else random noise frames."""
    if source:
        with VideoSource(source, limit=count) as video:
            frames = list(video)
        if not frames:
            raise RuntimeError(f"No frames read from {source!r}")
        return frames
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(8)]


def bench_pipeline(pipeline: Pipeline, frames: list[np.ndarray], n: int, warmup: int) -> dict[str, Any]:
    stages: dict[str, list[float]] = {"detect_ms": [], "track_ms": [], "rules_ms": [], "total_ms": []}
    detections = 0
    for i, frame in enumerate(islice(cycle(frames), warmup + n)):
        result = pipeline.process(frame)
        if i < warmup:
            continue
        t = result.timings
        stages["detect_ms"].append(t.detect_ms)
        stages["track_ms"].append(t.track_ms)
        stages["rules_ms"].append(t.rules_ms)
        stages["total_ms"].append(t.total_ms)
        detections += len(result.detections)
    report: dict[str, Any] = {k: summarize(v) for k, v in stages.items()}
    report["fps"] = round(1000 / report["total_ms"]["mean"], 2) if report["total_ms"]["mean"] else None
    report["frames"] = n
    report["detections_per_frame"] = round(detections / n, 2) if n else 0
    return report


def run_bench(
    cfg: RuntimeConfig, weights: list[str], source: str | None, n: int, warmup: int, out: str | None = None
) -> dict[str, Any]:
    frames = load_frames(source, min(n, 300), cfg.engine.imgsz)
    results = {}
    for w in weights:
        engine_cfg = cfg.engine.model_copy(update={"weights": w, "backend": "auto"})
        pipeline = Pipeline.from_config(cfg.model_copy(update={"engine": engine_cfg}))
        try:
            results[w] = {"backend": pipeline.engine.backend, **bench_pipeline(pipeline, frames, n, warmup)}
        finally:
            pipeline.close()
        r = results[w]
        print(
            f"{Path(w).name:<32} {r['backend']:<9} {r['fps']:>8} FPS | detect p50 {r['detect_ms']['p50']} "
            f"p95 {r['detect_ms']['p95']} ms | track {r['track_ms']['mean']} ms | rules {r['rules_ms']['mean']} ms",
            flush=True,
        )
    report = {
        "host": platform.node(),
        "platform": platform.platform(),
        "source": source,
        "imgsz": cfg.engine.imgsz,
        "device": cfg.engine.device,
        "tracker": cfg.tracking.tracker_type if cfg.tracking else None,
        "frames": n,
        "warmup": warmup,
        "results": results,
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
