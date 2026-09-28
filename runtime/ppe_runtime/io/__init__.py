"""Frame sources and result sinks."""

from ppe_runtime.io.sink import MultiSink, Sink, create_sink
from ppe_runtime.io.stream import StreamSource, open_source
from ppe_runtime.io.video import VideoSource, VideoWriter

__all__ = ["MultiSink", "Sink", "StreamSource", "VideoSource", "VideoWriter", "create_sink", "open_source"]
