"""Adapter feature cho CWD distillation: channel student -> channel teacher.

Được build lúc runtime từ tensor thật của student/teacher (không bao giờ hardcode),
vì pruning làm số channel thay đổi không đều giữa các layer (xem PLAN.md mục 3).
Adapter chỉ tồn tại trong giai đoạn train distillation và bị bỏ đi trước khi export —
chúng được lưu dưới một key ``ModuleDict`` riêng để dễ strip khỏi state_dict sau này.
"""

from __future__ import annotations

import torch
import torch.nn as nn

class FeatureAdapter(nn.Module):
    """Conv1x1 + BN ánh xạ channel feature của student sang channel feature của teacher."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


@torch.no_grad()
def build_adapters(
    student: nn.Module,
    teacher: nn.Module,
    layer_names: list[str],
    img_size: int = 640,
) -> nn.ModuleDict:
    """Build mỗi layer một adapter bằng cách chạy forward pass giả để đọc shape thật.

    ``layer_names`` phải là key hợp lệ trong ``named_modules()`` của cả student và
    teacher (đúng với mọi cặp scale/bản pruned YOLO26 dùng chung layout yaml —
    xem PLAN.md mục 3).
    """
    from ultralytics.utils.distill_utils import register_feature_hooks, remove_hooks

    device = next(student.parameters()).device
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)

    was_training_s, was_training_t = student.training, teacher.training
    student.eval()
    teacher.eval()

    s_handles, s_feats = register_feature_hooks(student, layer_names)
    t_handles, t_feats = register_feature_hooks(teacher, layer_names)
    try:
        student(dummy)
        teacher(dummy)
    finally:
        remove_hooks(s_handles)
        remove_hooks(t_handles)
        student.train(was_training_s)
        teacher.train(was_training_t)

    adapters = nn.ModuleDict()
    for name in layer_names:
        in_ch = s_feats[name].shape[1]
        out_ch = t_feats[name].shape[1]
        key = name.replace(".", "_")
        adapters[key] = FeatureAdapter(in_ch, out_ch).to(device)
    return adapters
