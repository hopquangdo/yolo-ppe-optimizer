"""Box utilities (pixel xyxy) and the letterbox pre/post-processing shared by native engines."""

from __future__ import annotations

import cv2
import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def area(box: Box) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def intersection(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def overlap_fraction(inner: Box, outer: Box) -> float:
    """Share of `inner`'s area lying inside `outer` (1.0 = fully contained)."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def center_inside(inner: Box, outer: Box) -> bool:
    cx, cy = center(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy NMS over xyxy boxes; returns kept indices, highest score first."""
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        w = (np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])).clip(0)
        h = (np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])).clip(0)
        inter = w * h
        order = rest[inter / (areas[i] + areas[rest] - inter + 1e-9) <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def letterbox(image: np.ndarray, size: int, color: int = 114) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio, pad to size×size (ultralytics convention). Returns (image, gain, (pad_x, pad_y))."""
    h, w = image.shape[:2]
    gain = min(size / h, size / w)
    nh, nw = round(h * gain), round(w * gain)
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    if (nh, nw) != (h, w):
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color))
    return image, gain, (pad_x, pad_y)


def to_blob(image: np.ndarray, size: int, half: bool = False) -> tuple[np.ndarray, float, tuple[float, float]]:
    """BGR HWC uint8 → letterboxed RGB NCHW float in [0, 1]."""
    img, gain, pad = letterbox(image, size)
    blob = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1)[None]).astype(np.float16 if half else np.float32)
    blob /= 255.0
    return blob, gain, pad


def unletterbox(boxes: np.ndarray, gain: float, pad: tuple[float, float], shape: tuple[int, ...]) -> np.ndarray:
    """Map xyxy boxes from letterboxed input space back to the original image, clipped to it."""
    boxes = boxes.astype(np.float32, copy=True)
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad[0]) / gain).clip(0, shape[1])
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad[1]) / gain).clip(0, shape[0])
    return boxes
