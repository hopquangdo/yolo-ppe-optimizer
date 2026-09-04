"""Objective functions for SHO-based YOLO hyperparameter search."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from ultralytics import YOLO


def build_yolo_objective(
    model_path: str | Path,
    data: str | Path,
    search_space: Mapping[str, tuple[float, float]],
    proxy_epochs: int,
    imgsz: int,
    *,
    batch: int = 16,
    project: str | Path = "runs/sho",
    device: str | None = None,
    model_factory: Callable = YOLO,
) -> Callable[[np.ndarray], float]:
    """Build a minimization objective returning ``1 - mAP50-95``.

    A fresh YOLO instance is created for every candidate, so each candidate
    starts from the same checkpoint rather than continuing a previous run.
    ``model_factory`` is injectable for fast unit tests.
    """
    names = list(search_space)
    if not names or proxy_epochs < 1 or imgsz < 1:
        raise ValueError("search_space must be non-empty and training values must be positive")
    if any(len(bounds) != 2 or bounds[0] >= bounds[1] for bounds in search_space.values()):
        raise ValueError("each search-space entry must be an increasing (lower, upper) pair")
    run_number = 0

    def objective(position: np.ndarray) -> float:
        nonlocal run_number
        values = np.asarray(position, dtype=float)
        if values.shape != (len(names),) or not np.all(np.isfinite(values)):
            raise ValueError(f"candidate must be a finite vector with shape ({len(names)},)")
        overrides = dict(zip(names, values.tolist()))
        kwargs = {
            "data": str(data),
            "epochs": proxy_epochs,
            "imgsz": imgsz,
            "batch": batch,
            "project": str(project),
            "name": f"candidate-{run_number:04d}",
            "exist_ok": True,
            "val": True,
            "plots": False,
            "verbose": False,
            **overrides,
        }
        if device is not None:
            kwargs["device"] = device
        run_number += 1
        model = model_factory(str(model_path))
        metrics = model.train(**kwargs)
        metrics = metrics or getattr(model, "metrics", None)
        box_metrics = getattr(metrics, "box", None)
        map_value = getattr(box_metrics, "map", None)
        if map_value is None or not np.isfinite(map_value):
            raise RuntimeError("YOLO training did not return a finite detection mAP50-95 metric")
        return 1.0 - float(map_value)

    return objective