"""Hàm loss cho distillation: teacher (YOLO26 bản lớn) -> student (YOLO26, có thể đã pruned).

YOLO26 không có DFL (mọi config trong ``ultralytics/cfg/models/26/*.yaml`` đặt
``reg_max: 1``, khiến ``DFL`` module trở thành ``nn.Identity``), nên các công thức
distillation dạng KL-trên-phân-phối-DFL của YOLOv6/v8/v11 không áp dụng được cho
nhánh regression — xem ``reg_kd_loss`` bên dưới để biết cách thay thế.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FeatureHook:
    """Hook forward dạng class để pickle được (closure nội bộ không pickle được,
    trong khi model gắn hook này có thể bị deepcopy khi lưu EMA/checkpoint)."""

    def __init__(self, feats: dict[str, torch.Tensor], name: str):
        self.feats = feats
        self.name = name

    def __call__(self, _module, _inputs, output):
        self.feats[self.name] = output


def register_feature_hooks(model: nn.Module, layer_names: list[str]) -> tuple[list, dict[str, torch.Tensor]]:
    """Đăng ký forward hook lưu output của ``layer_names`` vào một dict dùng chung.

    Trả về danh sách handle (gọi ``.remove()`` từng cái khi xong) và dict sẽ được
    ghi đè bằng activation mới nhất sau mỗi lần forward.
    """
    feats: dict[str, torch.Tensor] = {}
    named = dict(model.named_modules())
    handles = []
    for name in layer_names:
        if name not in named:
            raise KeyError(f"layer '{name}' not found in model (available e.g. {list(named)[:5]}...)")
        handles.append(named[name].register_forward_hook(_FeatureHook(feats, name)))
    return handles, feats


def remove_hooks(handles: list) -> None:
    """Gỡ các forward hook đã đăng ký trước đó."""
    for h in handles:
        h.remove()


def cwd_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor, adapter: nn.Module, T: float = 4.0) -> torch.Tensor:
    r"""Channel-wise Distillation (CWD) cho feature map.

    Nguồn: Shu, Changyong, et al. "Channel-wise Knowledge Distillation for Dense
    Prediction." ICCV 2021. arXiv:2011.13256.

    Với mỗi channel c, softmax theo vị trí không gian i (i = 1..W*H) với nhiệt độ T::

        phi(y_c,i) = exp(y_c,i / T) / sum_j exp(y_c,j / T)

    Loss là KL bất đối xứng, chiều teacher -> student, nhân T^2 để scale lại gradient
    (T càng lớn softmax càng mượt, gradient càng nhỏ nếu không nhân bù)::

        L_CWD = T^2 * sum_c sum_i phi(y_c,i^teacher) * log( phi(y_c,i^teacher) / phi(y_c,i^student) )

    Trọng số khuyến nghị của paper gốc cho feature map: lambda_CWD ~ 50 — nhưng theo
    smoke-test thực tế trên YOLO26 (xem PLAN.md mục 11.3), giá trị này quá lớn và làm
    hỏng training; điểm khởi đầu hợp lý hơn cho YOLO26 là lambda_CWD trong khoảng 1-5.

    ``adapter`` (Conv1x1+BN) ánh xạ channel student sang channel teacher — cần thiết vì
    sau khi prune, số channel giữa 2 bên không còn khớp nhau. Feature map student và
    teacher phải cùng độ phân giải không gian (cùng stride/tầng P3/P4/P5).
    """
    student_feat = adapter(student_feat)
    n, c, h, w = teacher_feat.shape
    t = teacher_feat.reshape(n, c, -1)
    s = student_feat.reshape(n, c, -1)
    phi_t = F.softmax(t / T, dim=-1)
    log_phi_t = F.log_softmax(t / T, dim=-1)
    log_phi_s = F.log_softmax(s / T, dim=-1)
    loss = (phi_t * (log_phi_t - log_phi_s)).sum(dim=-1)  # KL theo từng (batch, channel)
    return (T * T) * loss.sum() / (n * c)


def cls_kd_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor, T: float = 4.0) -> torch.Tensor:
    r"""Response KD cho nhánh classification.

    Nguồn: Li, Chuyi, et al. "YOLOv6: A Single-Stage Object Detection Framework for
    Industrial Applications." arXiv:2209.02976 — phần self-distillation, số hạng
    classification của công thức ``L_KD = KL(p_t^cls || p_s^cls) + KL(p_t^reg || p_s^reg)``.

    Số hạng cls vẫn áp dụng nguyên vẹn cho YOLO26 (khác với số hạng reg, xem
    ``reg_kd_loss``) vì output classification luôn là một phân phối xác suất thật
    (sigmoid/softmax trên nc lớp) bất kể kiến trúc có DFL hay không::

        L_cls_KD = KL(p_teacher^cls || p_student^cls)

    ``student_scores``/``teacher_scores``: logit thô, shape (B, nc, N) — trước sigmoid,
    khớp với ``x["scores"]`` từ ``DetectPruned.forward_head`` (head_pruned.py:94).
    Softmax lấy theo chiều class (dim=1), nhân T^2 theo đúng convention KD chuẩn.
    """
    log_p_s = F.log_softmax(student_scores / T, dim=1)
    log_p_t = F.log_softmax(teacher_scores / T, dim=1)
    p_t = log_p_t.exp()
    loss = (p_t * (log_p_t - log_p_s)).sum(dim=1)  # KL theo từng (batch, anchor)
    return (T * T) * loss.mean()


def reg_kd_loss(
    student_boxes_raw: torch.Tensor,
    teacher_boxes_raw: torch.Tensor,
    student_boxes_decoded: torch.Tensor,
    teacher_boxes_decoded: torch.Tensor,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
) -> torch.Tensor:
    r"""
    Response KD cho nhánh regression — bản thay thế cho head không có DFL.

    YOLOv6 (Li et al., arXiv:2209.02976) dùng ``KL(p_t^reg || p_s^reg)`` trên phân phối
    DFL (box được biểu diễn dưới dạng phân phối rời rạc trên các bin). Công thức đó
    **không dùng được cho YOLO26**: mọi config YOLO26 đặt ``reg_max: 1``
    (``ultralytics/cfg/models/26/*.yaml``), khiến ``DFL`` module là ``nn.Identity``
    (``head_pruned.py:65``) — box regression là 1 giá trị số thực trực tiếp mỗi cạnh,
    không phải phân phối xác suất, nên KL không tính được (softmax trên 1 phần tử luôn
    = 1, KL luôn = 0, loss chết mà không báo lỗi).

    Thay bằng loss khớp trực tiếp trên box, tách làm 2 số hạng theo 2 dạng biểu diễn
    khác nhau (không phải cùng 1 tensor)::

        L_reg_KD = lambda1 * SmoothL1(y_hat_s, y_hat_t) + lambda2 * (1 - GIoU(y_hat_s, y_hat_t))

    - SmoothL1 trên offset thô từng cạnh (``x["boxes"]`` trước ``dist2bbox``), cùng
      đơn vị/scale giữa teacher và student vì cả hai dùng chung stride grid.
    - GIoU (Generalized IoU, Rezatofighi et al., "Generalized Intersection over Union",
      CVPR 2019, arXiv:1902.09630) trên box xyxy/xywh đã decode (sau
      ``dist2bbox(...) * stride``), vì GIoU cần toạ độ hình học thật, không tính được
      trên offset thô.

    ``*_boxes_decoded`` theo layout (B, 4, N) của ``dist2bbox`` (4 thành phần box nằm
    ở dim=1, như trong ``DetectPruned._inference``, head_pruned.py:115) — hàm này tự
    transpose sang (B, N, 4) bên trong để tính GIoU.
    """
    smooth_l1 = F.smooth_l1_loss(student_boxes_raw, teacher_boxes_raw.detach())
    s_dec = student_boxes_decoded.transpose(1, 2)
    t_dec = teacher_boxes_decoded.detach().transpose(1, 2)
    giou = 1.0 - _giou(s_dec, t_dec)
    return lambda1 * smooth_l1 + lambda2 * giou.mean()


def _giou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, eps: float = 1e-7) -> torch.Tensor:
    r"""Generalized IoU (Rezatofighi et Qal., CVPR 2019, arXiv:1902.09630) giữa hai tập box.

    ``GIoU = IoU - (|C \ (A union B)| / |C|)``, với C là bounding box nhỏ nhất bao cả
    A và B — phạt thêm cả khi 2 box không giao nhau (IoU=0 nhưng vẫn phân biệt được xa/gần).
    """
    if xywh:
        x1, y1, w1, h1 = box1.unbind(-1)
        x2, y2, w2, h2 = box2.unbind(-1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp_(0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp_(0)
    inter = inter_w * inter_h

    w1_, h1_ = (b1_x2 - b1_x1), (b1_y2 - b1_y1)
    w2_, h2_ = (b2_x2 - b2_x1), (b2_y2 - b2_y1)
    union = w1_ * h1_ + w2_ * h2_ - inter + eps
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c_area = cw * ch + eps
    return iou - (c_area - union) / c_area


def kd_alpha_schedule(epoch: int, max_epoch: int) -> float:
    r"""Lịch alpha giảm dần theo cosine cho response KD.

    Nguồn: Li, Chuyi, et al. "YOLOv6: A Single-Stage Object Detection Framework for
    Industrial Applications." arXiv:2209.02976 — self-distillation với cosine weight
    decay cho alpha::

        alpha(E_i) = -0.99 * ( (1 - cos(pi * E_i / E_max)) / 2 ) + 1

    Bắt đầu gần 1 (tin soft label/teacher nhiều ở đầu training), giảm dần về ~0.01 ở
    epoch cuối (chuyển sang tin hard label/ground-truth nhiều hơn). Paper gốc ghi rõ:
    *"No performance improvement is attained without the weight decay strategy compared
    with the baseline"* — alpha cố định gần như vô dụng, đây không phải một tinh chỉnh
    tuỳ chọn mà là phần bắt buộc của công thức.
    """
    max_epoch = max(max_epoch, 1)
    e = min(max(epoch, 0), max_epoch)
    return -0.99 * ((1 - math.cos(math.pi * e / max_epoch)) / 2) + 1
