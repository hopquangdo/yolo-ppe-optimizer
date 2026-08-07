"""Validate a pruned YOLO26 model.

Usage:
    python scripts/val_pruned.py --weights weights/yolo26_pruned.pt --data coco128.yaml
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/yolo26_pruned.pt")
    # parser.add_argument("--weights", type=str, default="runs/detect/runs/finetune-yolo26-pruned-2/weights/last.pt")
    parser.add_argument("--data", type=str, default="coco128.yaml")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    opt = parser.parse_args()

    model = YOLO(opt.weights)
    results = model.val(data=opt.data, batch=opt.batch, imgsz=opt.imgsz)


if __name__ == "__main__":
    main()
