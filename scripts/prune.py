"""
YOLO26 structured channel pruning.

Uses dedicated *Pruned module classes* (see ``yolo26-pruned.yaml``) and
``DetectionModelPruned``, mirroring the YOLOv8 flow in ``prune.py``.

New files:
  ultralytics/nn/modules/block_pruned.py
  ultralytics/nn/modules/head_pruned.py
  ultralytics/nn/tasks_pruned.py
  ultralytics/cfg/models/26/yolo26-pruned.yaml
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import yaml

warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # repo root (this file lives in scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # must run before the ultralytics imports below

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules import Concat, Conv
from ultralytics.nn.modules.block import Bottleneck, C2PSA, C3k2
from ultralytics.nn.modules.block_pruned import (
    C2PSAPruned,
    C3k2AttnPruned,
    C3k2C3kPruned,
    C3k2Pruned,
    SPPFPruned,
)
from ultralytics.nn.modules.head_pruned import DetectPruned
from ultralytics.utils.prune_utils import build_ignore_bn_set
from ultralytics.nn.tasks_pruned import DetectionModelPruned
from ultralytics.utils import colorstr


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="sparsity-trained .pt")
    parser.add_argument(
        "--cfg",
        type=str,
        default=str(ROOT / "ultralytics/cfg/models/26/yolo26-pruned.yaml"),
        help="pruned architecture yaml (template)",
    )
    parser.add_argument("--model-size", type=str, default="n", choices=list("nsmlx"))
    parser.add_argument("--prune-ratio", type=float, default=0.05)
    parser.add_argument("--save-dir", type=str, default=str(ROOT / "weights"))
    return parser.parse_args()


def _build_pruned_yaml_template(model_size: str) -> dict:
    """Load yolo26-pruned.yaml template and set scale."""
    cfg_path = ROOT / "ultralytics/cfg/models/26/yolo26-pruned.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    d["scale"] = model_size
    return d


def main(opt):
    weights, prune_ratio, cfg_path, model_size, save_dir = (
        opt.weights,
        opt.prune_ratio,
        opt.cfg,
        opt.model_size,
        opt.save_dir,
    )
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = AutoBackend(weights, fuse=False)
    model.eval()

    # ---- Step 1: collect BN + ignore / chunk lists ---------------------------------
    bn_dict = {}
    ignore_bn_set = build_ignore_bn_set(model.model)
    chunk_bn_list = []

    for name, module in model.model.named_modules():
        if isinstance(module, Bottleneck) and not module.add:
            parent = name.rsplit(".", 2)[0]
            chunk_bn_list.append(f"{parent}.cv1.bn")
        if isinstance(module, C3k2):
            chunk_bn_list.append(f"{name}.cv1.bn")
        if isinstance(module, C2PSA):
            chunk_bn_list.append(f"{name}.cv1.bn")
        if isinstance(module, nn.BatchNorm2d):
            bn_dict[name] = module

    missing_ignore = [n for n in ignore_bn_set if n not in bn_dict]
    if missing_ignore:
        print(f"INFO: {len(missing_ignore)} ignore-list entries not in model (harmless): {missing_ignore}")
    ignore_bn_set = {n for n in ignore_bn_set if n in bn_dict}

    bn_dict = {k: v for k, v in bn_dict.items() if k not in ignore_bn_set}

    # ---- Step 2–5: threshold -------------------------------------------------------
    bn_weights = []
    for module in bn_dict.values():
        bn_weights.extend(module.weight.data.abs().clone().cpu().tolist())
    sorted_bn = torch.sort(torch.tensor(bn_weights))[0]
    highest_thre = min(m.weight.data.abs().clone().cpu().max() for m in bn_dict.values())
    percent_limit = (sorted_bn == highest_thre).nonzero()[0, 0].item() / len(sorted_bn)
    thre = sorted_bn[int(len(sorted_bn) * prune_ratio)]
    print(f"Pruning gamma should be < {colorstr(f'{highest_thre:.4f}')}, yours {colorstr(f'{thre:.4f}')}")
    print(f"Max ratio < {colorstr(f'{percent_limit:.3f}')}, yours {colorstr(f'{prune_ratio:.3f}')}")
    if prune_ratio > percent_limit:
        prune_ratio = percent_limit
        print(f"Ratio capped to {colorstr(f'{prune_ratio:.3f}')}")

    # ---- Step 6: pruned yaml (module names already *Pruned) ------------------------
    with open(cfg_path, encoding="utf-8") as f:
        base_yaml = yaml.safe_load(f)
    pruned_yaml = _build_pruned_yaml_template(model_size)
    pruned_yaml["nc"] = model.model.nc
    pruned_yaml["end2end"] = base_yaml.get("end2end", True)
    pruned_yaml["reg_max"] = base_yaml.get("reg_max", 1)

    # ---- Step 7: masks -------------------------------------------------------------
    print("=" * 94)
    print(f"|\t{'layer':<25}{'|':<10}{'origin':<20}{'|':<10}{'remaining':<20}|")
    maskbndict = {}
    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            origin = module.weight.data.shape[0]
            remaining = origin
            mask = torch.ones(origin)
            if name not in ignore_bn_set:
                # ge, not gt: at prune_ratio=0.0, thre == the global minimum gamma value
                # in the model. A strict `gt` would exclude whichever channel happens to
                # sit exactly at that minimum even when the caller asked to prune nothing.
                mask = module.weight.data.abs().ge(thre).float()
                if name in chunk_bn_list and mask.sum() % 2 == 1:
                    flat = torch.sort(module.weight.data.abs().view(-1))[0]
                    idx = torch.min(torch.nonzero(flat.gt(thre))).item()
                    thre_ = flat[idx - 1] - 1e-6
                    mask = module.weight.data.abs().gt(thre_).float()
                assert mask.sum() > 0, f"bn {name} has no active channels"
                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
                remaining = int(mask.sum().item())
            maskbndict[name] = mask
            print(f"|\t{name:<25}{'|':<10}{origin:<20}{'|':<10}{remaining:<20}|")
    print("=" * 94)

    # ---- Step 8: build pruned model ------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pruned_model = DetectionModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).to(device)
    pruned_model.eval()

    # ---- Step 9: weight transfer (shape-based, order-independent) -------------------
    current_to_prev = pruned_model.current_to_prev
    org_modules = dict(model.model.named_modules())
    pattern_detect = re.compile(r"model\.\d+\.(one2one_)?cv\d\.\d\.2$")
    transferred = {"conv": [], "bn": []}

    def _o2m(bn_name: str) -> str:
        """Map a one2one_* BN name to its one2many (cv2/cv3) counterpart for mask lookup.

        ``DetectPruned`` builds ``one2one_cv2/cv3`` as a deepcopy of ``cv2/cv3``
        (see head_pruned.py), so both branches are forced to share the same channel
        shapes even though one2one is trained independently and its own BN-gamma mask
        may select a different channel count. The mask used to pick *which* channels
        survive must therefore come from one2many (to match the fixed architecture);
        the actual weight *values* copied are still one2one's own trained values.
        """
        return bn_name.replace(".one2one_cv2.", ".cv2.").replace(".one2one_cv3.", ".cv3.")

    def _build_in_mask(curr_bn, org_in_channels):
        """Build boolean input mask matching org conv's in_channels."""
        prev_bn = current_to_prev.get(curr_bn)
        if prev_bn is None:
            return torch.ones(org_in_channels, dtype=torch.bool)
        if isinstance(prev_bn, list):
            raw = torch.cat([maskbndict[_o2m(n)] for n in prev_bn], dim=0)
        else:
            raw = maskbndict[_o2m(prev_bn)]
        raw = raw.to(torch.bool)
        # If mask is shorter than in_channels, tile (SPPF pooling concat)
        if raw.shape[0] < org_in_channels and org_in_channels % raw.shape[0] == 0:
            raw = raw.repeat(org_in_channels // raw.shape[0])
        # If mask is longer than in_channels, take the LAST in_channels elements
        # (C3k2 split: first half = passthrough, second half = bottleneck input)
        if raw.shape[0] > org_in_channels:
            raw = raw[-org_in_channels:]
        return raw

    for name, module_pruned in pruned_model.named_modules():
        if name.endswith(".dfl"):
            continue
        module_org = org_modules.get(name)
        if module_org is None:
            continue

        # --- Detect head final Conv2d (no BN, special out channels) ---
        if pattern_detect.fullmatch(name) and name in current_to_prev:
            prev_bn = current_to_prev[name]
            in_mask = maskbndict[_o2m(prev_bn)].to(torch.bool)
            module_pruned.weight.data.copy_(module_org.weight.data[:, in_mask, :, :])
            if module_org.bias is not None and module_pruned.bias is not None:
                module_pruned.bias.data.copy_(module_org.bias.data)
            transferred["conv"].append(name)
            continue

        # --- Conv2d layers (paired with BN) ---
        if isinstance(module_org, nn.Conv2d) and isinstance(module_pruned, nn.Conv2d):
            curr_bn = name[:-4] + "bn"  # .conv → .bn
            if _o2m(curr_bn) not in maskbndict:
                continue
            out_mask = maskbndict[_o2m(curr_bn)].to(torch.bool)

            if module_org.groups > 1:
                # Depthwise: in_ch == out_ch == groups. Use input mask if available.
                if curr_bn in current_to_prev:
                    dw_mask = _build_in_mask(curr_bn, module_org.weight.shape[0])
                    w = module_org.weight.data[dw_mask]
                else:
                    w = module_org.weight.data[out_mask]
            else:
                in_mask = _build_in_mask(curr_bn, module_org.weight.shape[1])
                assert in_mask.shape[0] == module_org.weight.shape[1], (
                    f"{name}: in_mask {in_mask.shape[0]} != weight in_ch {module_org.weight.shape[1]} "
                    f"(curr_bn={curr_bn}, prev={current_to_prev.get(curr_bn)})"
                )
                w = module_org.weight.data[out_mask][:, in_mask]

            assert w.shape == module_pruned.weight.shape, (
                f"{name}: sliced {w.shape} != pruned {module_pruned.weight.shape}"
            )
            module_pruned.weight.data.copy_(w)
            if module_org.bias is not None and module_pruned.bias is not None:
                module_pruned.bias.data.copy_(module_org.bias.data[out_mask])
            transferred["conv"].append(name)

        # --- BatchNorm2d layers ---
        if isinstance(module_org, nn.BatchNorm2d) and _o2m(name) in maskbndict:
            out_mask = maskbndict[_o2m(name)].to(torch.bool)
            # If pruned size differs from mask (e.g. DWConv BN in ignore set), use input mask
            if out_mask.sum().item() != module_pruned.weight.shape[0] and name in current_to_prev:
                out_mask = _build_in_mask(name, module_org.weight.shape[0])
            module_pruned.weight.data.copy_(module_org.weight.data[out_mask])
            module_pruned.bias.data.copy_(module_org.bias.data[out_mask])
            module_pruned.running_mean.copy_(module_org.running_mean[out_mask])
            module_pruned.running_var.copy_(module_org.running_var[out_mask])
            transferred["bn"].append(name)

    print(f"Transferred: {len(transferred['conv'])} Conv2d, {len(transferred['bn'])} BatchNorm2d")
    missing_bn = [n for n in maskbndict if n not in transferred["bn"] and n not in ignore_bn_set]
    if missing_bn:
        print(f"WARNING: {len(missing_bn)} BN mask keys not transferred: {missing_bn[:5]}...")

    # ---- Step 10: save -------------------------------------------------------------
    save_path = save_dir / "yolo26_pruned.pt"
    pruned_model.task = "detect"
    torch.save(
        {
            "model": pruned_model,
            "maskbndict": maskbndict,
            "pruned_yaml": pruned_yaml,
            "train_args": {"task": "detect"},
        },
        save_path,
    )
    print(f"Saved {save_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pruned_model = pruned_model.to(device)
    dummy = torch.randn(1, 3, 640, 640, device=device)
    with torch.inference_mode():
        pruned_model(dummy)
    n0 = sum(p.numel() for p in model.model.parameters())
    n1 = sum(p.numel() for p in pruned_model.parameters())
    print(f"Params: {n0/1e6:.3f}M -> {n1/1e6:.3f}M ({(1-n1/n0)*100:.1f}% reduction)")
    kept_ratio = sum(v.sum().item() for v in maskbndict.values()) / sum(v.numel() for v in maskbndict.values())
    print(f"BN mask kept: {kept_ratio * 100:.1f}% (use --prune-ratio 0.3 for ~30% channel prune)")



if __name__ == "__main__":
    main(parse_opt())
