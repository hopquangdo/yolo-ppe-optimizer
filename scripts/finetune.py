"""Fine-tune a pruned YOLO26 checkpoint (yolo26_pruned.pt with maskbndict).

Alternative to distill.py as the recovery-training step after prune.py — plain
supervised fine-tuning, no teacher.

Example:
    python scripts/finetune.py --weights runs/prune/yolo26_pruned.pt \
        --data coco128.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/yolo26_pruned.pt", help="pruned checkpoint from prune.py")
    parser.add_argument("--data", type=str, default="coco128.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="finetune-yolo26-pruned")
    return parser.parse_args()


def main(opt):
    model = YOLO(opt.weights)
    model.train(
        data=opt.data,
        epochs=opt.epochs,
        finetune=True,
        optimizer="AdamW",
        lr0=opt.lr0,
        momentum=opt.momentum,
        patience=opt.patience,
        batch=opt.batch,
        imgsz=opt.imgsz,
        project=opt.project,
        name=opt.name,
    )


if __name__ == "__main__":
    main(parse_opt())
