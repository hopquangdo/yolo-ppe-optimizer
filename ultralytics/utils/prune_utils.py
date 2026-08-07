"""Pruning utilities for YOLO26 architecture."""

from __future__ import annotations

import re

import torch.nn as nn

from ultralytics.nn.modules import Conv, DWConv
from ultralytics.nn.modules.block import (
    Bottleneck,
    C2PSA,
    PSABlock,
    SPPF,
)


def _bn_name(parent: str, leaf: str) -> str:
    return f"{parent}.{leaf}" if parent else leaf


def build_ignore_bn_set(model: nn.Module) -> set[str]:
    """Return BN layer names excluded from sparsity training and channel pruning."""
    ignore: set[str] = set()

    for name, m in model.named_modules():
        if isinstance(m, Bottleneck) and m.add:
            parent = name.rsplit(".", 2)[0]
            ignore.add(_bn_name(parent, "cv1.bn"))
            ignore.add(_bn_name(name, "cv2.bn"))

        if isinstance(m, SPPF) and getattr(m, "add", False):
            ignore.add(_bn_name(name, "cv2.bn"))

        if isinstance(m, PSABlock):
            ignore.add(_bn_name(name, "attn.qkv.bn"))
            ignore.add(_bn_name(name, "attn.proj.bn"))
            ignore.add(_bn_name(name, "attn.pe.bn"))
            ignore.add(_bn_name(name, "ffn.1.bn"))

        if isinstance(m, C2PSA):
            ignore.add(_bn_name(name, "cv1.bn"))
            ignore.add(_bn_name(name, "cv2.bn"))

        if isinstance(m, DWConv):
            ignore.add(_bn_name(name, "bn"))
        elif isinstance(m, Conv) and isinstance(getattr(m, "conv", None), nn.Conv2d):
            conv = m.conv
            if conv.groups > 1 and conv.groups == conv.in_channels:
                ignore.add(_bn_name(name, "bn"))

        if isinstance(m, nn.BatchNorm2d) and re.match(r"model\.\d+\.m\.\d+\.cv3\.bn$", name):
            ignore.add(name)

    return ignore


def is_prunable_block_output(name: str, module: nn.Module) -> bool:
    """Whether module is a block output whose final BN can be pruned."""
    return isinstance(module, Conv) and not (
        isinstance(module.conv, nn.Conv2d)
        and module.conv.groups > 1
        and module.conv.groups == module.conv.in_channels
    )
