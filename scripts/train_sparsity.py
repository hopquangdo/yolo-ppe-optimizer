"""Sparsity training for YOLO26 (BN-gamma L1 regularization).

Run before prune.py.
  After backward: BN_gamma.grad += sr * sign(gamma)
  This is gradient-level L1 (Network Slimming), NOT loss-level.
  sr acts directly on gradient magnitude — typical range: 1e-3 .. 1e-1.

Example:
    python scripts/train_sparsity.py
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.prune_utils import build_ignore_bn_set
from ultralytics.utils.torch_utils import unwrap_model


def _bn_gamma_stats(trainer):
    """Log BN gamma stats each epoch."""
    model = unwrap_model(trainer.model)
    ignore = build_ignore_bn_set(model)
    parts = [
        m.weight.detach().abs().flatten()
        for name, m in model.named_modules()
        if isinstance(m, nn.BatchNorm2d) and name not in ignore
    ]
    if not parts:
        return
    g = torch.cat(parts).cpu()
    mean = g.mean().item()
    if not hasattr(trainer, "_bn_gamma_mean0"):
        trainer._bn_gamma_mean0 = mean
    delta_pct = (trainer._bn_gamma_mean0 - mean) / max(trainer._bn_gamma_mean0, 1e-8) * 100.0
    pct_lt_1e1 = (g < 1e-1).float().mean().item() * 100
    pct_lt_1e2 = (g < 1e-2).float().mean().item() * 100
    status = ""
    if mean > 10 or not torch.isfinite(g).all():
        status = " !! UNSTABLE"
    elif delta_pct < -1:
        status = " !! gamma RISING"
    LOGGER.info(
        f"[BN gamma] epoch={trainer.epoch} sr={trainer.sr:.4f} "
        f"mean={mean:.4f} median={g.median():.4f} min={g.min():.2e} "
        f"delta={delta_pct:+.1f}% |<0.1|={pct_lt_1e1:.1f}% |<0.01|={pct_lt_1e2:.1f}%{status}"
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/yolo26n.pt", help="starting checkpoint")
    parser.add_argument("--data", type=str, default="coco128.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    # sr=1e-2 matches JasonSloan/yolov8-prune's train-sparsity.py (same BN-gamma
    # Network Slimming technique), which reports mAP50=0.964-0.972 after the full
    # sparsity->prune->finetune pipeline — i.e. sparsity training there does NOT
    # crash val to near-zero the way our earlier constant-sr runs did. The other
    # piece of that reference besides sr magnitude: sr decays over training
    # (srtmp = sr*(1-0.9*epoch/epochs), now implemented in engine/trainer.py) so
    # L1 pressure eases off in the back half instead of accumulating the whole run.
    parser.add_argument("--sr", type=float, default=1e-2, help="BN-gamma L1 strength")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="train-sparsity")
    return parser.parse_args()


def main(opt):
    model = YOLO(opt.weights)
    model.add_callback("on_train_epoch_end", _bn_gamma_stats)
    model.train(
        sr=opt.sr,
        data=opt.data,
        epochs=opt.epochs,
        patience=opt.patience,
        batch=opt.batch,
        imgsz=opt.imgsz,
        project=opt.project,
        name=opt.name,
    )


if __name__ == "__main__":
    main(parse_opt())
