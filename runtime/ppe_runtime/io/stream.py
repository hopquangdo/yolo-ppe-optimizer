from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ppe_runtime.io.video import VideoSource

logger = logging.getLogger(__name__)


class StreamSource:
    """Live source (RTSP/HTTP URL or camera index) read on a background thread.

    Only the newest frame is kept: `read()` returns it once, or None when nothing new has arrived. Frames overwritten
    before being read count as `dropped`. The reader reconnects with exponential backoff when the stream fails.
    """

    is_live = True

    def __init__(self, source: str | int, reconnect_max_sec: float = 10.0) -> None:
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        self._reconnect_max = reconnect_max_sec
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._fresh = False
        self._stop = threading.Event()
        self.connected = False
        self.dropped = 0
        self.fps = 0.0
        self._cap = self._open()
        self._thread = threading.Thread(target=self._loop, name="stream-reader", daemon=True)
        self._thread.start()

    def _open(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            self.connected = True
            return cap
        cap.release()
        self.connected = False
        return None

    def _loop(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            if self._cap is None:
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self._reconnect_max)
                self._cap = self._open()
                if self._cap is not None:
                    logger.info("Stream %s reconnected", self.source)
                    backoff = 0.5
                continue
            ok, frame = self._cap.read()
            if not ok:
                logger.warning("Stream %s lost; reconnecting", self.source)
                self._cap.release()
                self._cap = None
                self.connected = False
                continue
            with self._lock:
                if self._fresh:
                    self.dropped += 1
                self._frame, self._fresh = frame, True

    def read(self) -> np.ndarray | None:
        with self._lock:
            if not self._fresh:
                return None
            self._fresh = False
            return self._frame

    def wait(self, timeout: float = 1.0) -> np.ndarray | None:
        """Block until a new frame arrives (or `timeout`)."""
        deadline = time.monotonic() + timeout
        while (frame := self.read()) is None and time.monotonic() < deadline and not self._stop.is_set():
            time.sleep(0.002)
        return frame

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()


def open_source(source: str, limit: int | None = None) -> VideoSource | StreamSource:
    """Existing file → VideoSource (every frame, ends); camera index / URL → StreamSource (latest frame, live)."""
    if not source.isdigit() and Path(source).is_file():
        return VideoSource(source, limit)
    return StreamSource(source)
