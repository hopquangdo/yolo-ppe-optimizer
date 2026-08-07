"""Fine-tune a pruned YOLO26 checkpoint (yolo26_pruned.pt with maskbndict)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

weight = "weights/yolo26_pruned.pt"

model = YOLO(weight)

model.train(
    data="coco128.yaml",
    epochs=100,
    finetune=True,
    optimizer="AdamW",
    lr0=1e-3,
    momentum=0.9,
    patience=50,
    batch=16,
    project="runs/detect",
    name="finetune-yolo26-pruned",
)
