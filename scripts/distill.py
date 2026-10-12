"""Distillation training for pruned YOLO26: teacher(yolo26l/x) -> student(yolo26_pruned.pt).

Replaces finetune.py as the recovery-training step (PLAN.md section 1) — the task
loss (box+cls on ground truth) is already included alongside the distill loss, so
this is not meant to run after finetune.py, it runs instead of it.

Example:
    python scripts/distill.py --teacher yolo26l.pt \
        --student-checkpoint runs/prune_test/yolo26_pruned.pt \
        --data coco128.yaml --lambda-cwd 50 --temperature 4.0 --epochs 50

    # bản B (CrossKD + PKD + B2), dis mặc định 0.14 — xem docs/DISTILL_BAN_B.md
    python scripts/distill.py --kd-method ckd_b --teacher yolo26s.pt \
        --student-checkpoint yolo26n.pt --no-finetune --data data.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.detect.distill_trainer import DistillTrainer


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, required=True, help="teacher checkpoint (e.g. yolo26l.pt)")
    parser.add_argument("--student-checkpoint", type=str, required=True, help="pruned checkpoint from prune.py")
    parser.add_argument("--data", type=str, default="coco128.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lambda-cwd", type=float, default=50.0, help="CWD feature-loss weight (PLAN.md 4.1)")
    parser.add_argument("--temperature", type=float, default=4.0, help="softmax temperature for CWD/cls KD")
    parser.add_argument("--kd-method", type=str, default="cwd_kd", choices=["cwd_kd", "builtin", "ckd_b"],
                        help="cwd_kd (mặc định) | builtin (Ultralytics 8.4.152) | ckd_b (bản B)")
    parser.add_argument("--dis", type=float, default=None, help="hệ số KD cho builtin/ckd_b (mặc định 6.0 / 0.14)")
    parser.add_argument("--kd-params", type=str, default=None, help='JSON cho ckd_b, vd \'{"agree": false}\'')
    parser.add_argument("--no-finetune", action="store_true", help="student không phải checkpoint pruned (vd yolo26n.pt)")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="'auto' để giống thí nghiệm bản B trên Kaggle")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="distill-yolo26-pruned")
    return parser.parse_args()


def main(opt):
    model = YOLO(opt.student_checkpoint)
    model.train(
        trainer=DistillTrainer,
        teacher=opt.teacher,
        lambda_cwd=opt.lambda_cwd,
        kd_temperature=opt.temperature,
        kd_method=opt.kd_method,
        dis=opt.dis,
        kd_params=opt.kd_params,
        data=opt.data,
        epochs=opt.epochs,
        finetune=not opt.no_finetune,
        optimizer=opt.optimizer,
        lr0=opt.lr0,
        momentum=opt.momentum,
        patience=opt.patience,
        batch=opt.batch,
        project=opt.project,
        name=opt.name,
    )


if __name__ == "__main__":
    main(parse_opt())
