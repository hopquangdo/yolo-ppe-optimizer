"""PPE inference runtime: detect → track → associate PPE → violation rules. Independent of the edge-ops app."""

from ppe_runtime.pipeline.types import Detection, FrameResult, Timings
from ppe_runtime.version import __version__

__all__ = ["Detection", "FrameResult", "Pipeline", "Runner", "Timings", "__version__", "load_config"]


def __getattr__(name: str):
    if name in {"Pipeline", "Runner"}:
        from ppe_runtime.pipeline import pipeline

        return getattr(pipeline, name)
    if name == "load_config":
        from ppe_runtime.utils.config import load_config

        return load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
