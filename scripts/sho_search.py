"""SHO hyperparameter search for a YOLO detection model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization.objective import build_yolo_objective
from optimization.sho import sho
from ultralytics import YOLO

SEARCH_SPACE = {
    "lr0": (1e-5, 1e-1),
    "lrf": (0.01, 1.0),
    "momentum": (0.6, 0.98),
    "weight_decay": (0.0, 0.001),
    "warmup_epochs": (0.0, 5.0),
    "warmup_momentum": (0.0, 0.95),
    "hsv_h": (0.0, 0.9),
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
    "degrees": (0.0, 45.0),
    "translate": (0.0, 0.9),
    "scale": (0.0, 0.9),
    "shear": (0.0, 10.0),
    "perspective": (0.0, 0.001),
    "flipud": (0.0, 1.0),
    "mosaic": (0.0, 1.0),
    "mixup": (0.0, 1.0),
    "copy_paste": (0.0, 1.0),
}


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="initial YOLO checkpoint or model YAML")
    parser.add_argument("--data", required=True, help="dataset YAML")
    parser.add_argument("--pop", type=int, default=6)
    parser.add_argument("--gmax", type=int, default=3)
    parser.add_argument("--proxy-epochs", type=int, default=3)
    parser.add_argument("--full-epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/sho")
    parser.add_argument("--output", default="optimized_baseline.yaml")
    return parser.parse_args()


def main(opt):
    lower = [bounds[0] for bounds in SEARCH_SPACE.values()]
    upper = [bounds[1] for bounds in SEARCH_SPACE.values()]
    objective = build_yolo_objective(
        opt.model,
        opt.data,
        SEARCH_SPACE,
        opt.proxy_epochs,
        opt.imgsz,
        batch=opt.batch,
        project=opt.project,
        device=opt.device,
    )
    best_fitness, best_position, convergence = sho(
        opt.pop, opt.gmax, lower, upper, len(SEARCH_SPACE), objective, seed=opt.seed
    )
    best_params = dict(zip(SEARCH_SPACE, best_position.tolist()))
    output = Path(opt.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump({"fitness": best_fitness, "hyperparameters": best_params}, file, sort_keys=False)
    print(f"Best fitness (1 - mAP50-95): {best_fitness:.6f}")
    print("Convergence:", ", ".join(f"{value:.6f}" for value in convergence))
    train_kwargs = {
        "data": opt.data,
        "epochs": opt.full_epochs,
        "imgsz": opt.imgsz,
        "batch": opt.batch,
        "project": opt.project,
        "name": "optimized-baseline",
        **best_params,
    }
    if opt.device is not None:
        train_kwargs["device"] = opt.device
    YOLO(opt.model).train(
        **train_kwargs,
    )


if __name__ == "__main__":
    main(parse_opt())