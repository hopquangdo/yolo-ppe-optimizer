from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoSource:
    """Sequential reader for a video file: every frame is returned, `read()` gives None at the end."""

    is_live = False

    def __init__(self, path: str | Path, limit: int | None = None) -> None:
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video {self.path!r}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        self._limit = limit
        self._read = 0
        self.connected = True
        self.dropped = 0

    def read(self) -> np.ndarray | None:
        if self._limit is not None and self._read >= self._limit:
            return None
        ok, frame = self._cap.read()
        if not ok:
            self.connected = False
            return None
        self._read += 1
        return frame

    def __iter__(self):
        while (frame := self.read()) is not None:
            yield frame

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class VideoWriter:
    """mp4 writer that opens lazily on the first frame (size taken from it)."""

    def __init__(self, path: str | Path, fps: float = 25.0, fourcc: str = "mp4v") -> None:
        self.path = str(path)
        self.fps = fps
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(self.path, self._fourcc, self.fps, (w, h))
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
