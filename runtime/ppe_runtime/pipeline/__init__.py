"""Detect → Track → Associate → Rules.

`Pipeline`/`Runner` are loaded lazily: `pipeline.types` is imported by the engine, tracking and ppe modules, and an
eager import here would make it circular.
"""

from ppe_runtime.pipeline.types import Detection, FrameResult, Timings

__all__ = ["Detection", "FrameResult", "Pipeline", "Runner", "Timings"]


def __getattr__(name: str):
    if name in {"Pipeline", "Runner"}:
        from ppe_runtime.pipeline import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
