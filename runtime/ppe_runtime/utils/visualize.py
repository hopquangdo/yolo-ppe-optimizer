from __future__ import annotations

import cv2
import numpy as np

from ppe_runtime.pipeline.types import FrameResult

_OK, _BAD, _ITEM, _TEXT = (0, 200, 0), (0, 0, 255), (255, 160, 0), (255, 255, 255)


def draw(image: np.ndarray, result: FrameResult, show_items: bool = True) -> np.ndarray:
    """Annotate people (green = compliant, red = missing PPE), other detections, and pipeline FPS."""
    out = image.copy()
    person_ids = {id(s.person) for s in result.persons}
    if show_items:
        for d in result.detections:
            if id(d) in person_ids:
                continue
            x1, y1, x2, y2 = map(int, d.box)
            cv2.rectangle(out, (x1, y1), (x2, y2), _ITEM, 1)
            cv2.putText(out, d.class_name, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _ITEM, 1)
    for s in result.persons:
        p = s.person
        color = _BAD if s.missing else _OK
        x1, y1, x2, y2 = map(int, p.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{p.track_id}" if p.track_id is not None else "person"
        if s.missing:
            label += " no " + ",".join(sorted(s.missing))
        cv2.putText(out, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if result.timings.total_ms:
        fps = 1000 / result.timings.total_ms
        cv2.putText(out, f"{fps:.1f} FPS", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _TEXT, 2, cv2.LINE_AA)
    return out
