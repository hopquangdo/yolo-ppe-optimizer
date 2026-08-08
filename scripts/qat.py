"""QAT (Quantization-Aware Training) cho pruned YOLO26 — NVIDIA TensorRT INT8.

Bước cuối trong pipeline nối tiếp (PLAN_QAT.md):
    sparsity training -> prune -> finetune/distill -> QAT (file này) -> export INT8 TensorRT

Input phải là checkpoint đã pruned VÀ đã phục hồi độ chính xác (finetune.py hoặc
distill.py) — không chạy QAT trực tiếp trên checkpoint vừa prune còn yếu.

Yêu cầu: `pytorch-quantization` (NVIDIA) cài sẵn — thư viện này cần CUDA, không
chạy được trên máy Windows CPU-only (xem PLAN_QAT.md mục 1.5). Script này cần môi
trường Linux/GPU khớp target Jetson Orin Nano + TensorRT.

Example:
    python scripts/qat.py --checkpoint runs/detect/finetune/weights/best.pt \
        --data coco128.yaml --epochs 20 --lr0 1e-3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.qat.nvidia_tensorrt.qat_trainer_yolo26 import QATTrainerYolo26


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="pruned + finetuned/distilled checkpoint")
    parser.add_argument("--data", type=str, default="coco128.yaml")
    parser.add_argument("--epochs", type=int, default=20, help="QAT chỉ cần fine-tune nhẹ, không cần nhiều epoch")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=1e-3, help="bị chia /100 nội bộ trong QATTrainerYolo26")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--calib-batches", type=int, default=256, help="số batch dùng calibrate entropy")
    parser.add_argument("--recalib-every", type=int, default=0, help="0 = tắt recalibrate định kỳ")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="qat-yolo26-pruned")
    return parser.parse_args()


def main(opt):
    model = YOLO(opt.checkpoint)
    model.train(
        trainer=QATTrainerYolo26,
        data=opt.data,
        epochs=opt.epochs,
        optimizer="SGD",
        lr0=opt.lr0,
        momentum=opt.momentum,
        patience=opt.patience,
        batch=opt.batch,
        calib_batches=opt.calib_batches,
        recalib_every=opt.recalib_every,
        project=opt.project,
        name=opt.name,
    )


if __name__ == "__main__":
    main(parse_opt())
