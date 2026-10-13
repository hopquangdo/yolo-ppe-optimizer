"""Hai phương pháp distillation bổ sung cho DistillTrainer (chọn bằng ``kd_method``).

  builtin : KD tích hợp của Ultralytics 8.4.152 (``DistillationModel.loss`` / ``loss_sl2``), port 1-1.
            Projector Conv1×1 → ReLU → Conv1×1 + L2 trên 3 tầng neck, trọng số = độ tin cậy của teacher.
  ckd_b   : "bản B" — CrossKD (CVPR 2024) + PKD (NeurIPS 2022) + trọng số đồng thuận hai nhánh o2m/o2o (B2).

Cả hai giữ đúng quy ước loss của Ultralytics 8.4.152 để các giá trị ``dis`` đã hiệu chỉnh dùng lại được
(builtin dis = 6, ckd_b dis = 0.14)::

    loss_distill = dis * sum_levels(loss_level) * batch_size      (cộng MỘT lần vào tổng loss)

Projector chỉ dùng khi train — gọi ``strip_distill()`` để bỏ trước khi QAT/export.
Tài liệu: docs/DISTILL_BAN_B.md.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from torch import nn

BRANCHES = ("one2many", "one2one")
PROJ_ATTR = "distill_projectors"


# ----------------------------------------------------------------------------- projector
class Projector(nn.Sequential):
    """Giống projector của Ultralytics 8.4.152: Conv1×1 → ReLU → Conv1×1 (c_s → c_t → c_t)."""

    def __init__(self, c_s: int, c_t: int):
        super().__init__(nn.Conv2d(c_s, c_t, 1), nn.ReLU(inplace=True), nn.Conv2d(c_t, c_t, 1))


@torch.no_grad()
def build_projectors(student: nn.Module, teacher: nn.Module, layers: list[str], img_size: int = 256) -> nn.ModuleDict:
    """Đọc số kênh thật bằng một forward giả (student pruned có số kênh lẻ, không hardcode)."""
    from ultralytics.utils.distill_utils import register_feature_hooks, remove_hooks

    dev = next(student.parameters()).device  # student còn ở CPU lúc get_model
    dev_t = next(teacher.parameters()).device  # teacher đã ở GPU (load_checkpoint(device=self.device))
    st, tt = student.training, teacher.training
    student.eval(), teacher.eval()
    hs, fs = register_feature_hooks(student, layers)
    ht, ft = register_feature_hooks(teacher, layers)
    try:
        student(torch.zeros(1, 3, img_size, img_size, device=dev))
        teacher(torch.zeros(1, 3, img_size, img_size, device=dev_t))
    finally:
        remove_hooks(hs), remove_hooks(ht)
        student.train(st), teacher.train(tt)
    return nn.ModuleDict({n.replace(".", "_"): Projector(fs[n].shape[1], ft[n].shape[1]) for n in layers}).to(dev)


def parse_params(raw) -> dict:
    """``kd_params`` nhận dict hoặc chuỗi JSON (vd ``'{"agree": false}'`` từ CLI)."""
    if not raw:
        return {}
    return raw if isinstance(raw, dict) else json.loads(raw)


# ----------------------------------------------------------------------------- helpers
def _branch(preds, name):
    return preds[name] if isinstance(preds, dict) and name in preds else preds


def _split_levels(w: torch.Tensor, feats: list[torch.Tensor]):
    return torch.split(w, [f.shape[-2] * f.shape[-1] for f in feats], dim=-1)


def _weighted_l2(s, t, w):
    n, c = s.shape[:2]
    mse = F.mse_loss(s.view(n, c, -1), t.view(n, c, -1), reduction="none")
    return (mse * w).sum() / (w.sum() * c + 1e-9)


def _pkd(s, t):
    n, c = s.shape[:2]
    s, t = s.reshape(n, c, -1), t.reshape(n, c, -1)
    s = (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-6)
    t = (t - t.mean(-1, keepdim=True)) / (t.std(-1, keepdim=True) + 1e-6)
    return F.mse_loss(s, t) / 2


def _qfl_kd(logits, target, beta=1.0):
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none") * (
        (logits.sigmoid() - target).abs().pow(beta)
    )


# ----------------------------------------------------------------------------- builtin
def builtin_loss(model, teacher_preds) -> torch.Tensor:
    """Port 1-1 của DistillationModel.loss (8.4.152): trọng số = max_c sigmoid((s_o2m + s_o2o) / 2)."""
    t_feats = [model._teacher_feats[n] for n in model._feat_layers]
    sc = (_branch(teacher_preds, "one2many")["scores"] + _branch(teacher_preds, "one2one")["scores"]) / 2
    ws = [p.sigmoid().max(dim=1, keepdim=True).values for p in _split_levels(sc, t_feats)]
    proj = getattr(model, PROJ_ATTR)
    total = 0.0
    for i, n in enumerate(model._feat_layers):
        s = proj[n.replace(".", "_")](model._student_feats[n])
        total = total + _weighted_l2(s, model._teacher_feats[n], ws[i])
    return total


# ----------------------------------------------------------------------------- bản B
def ckd_b_loss(model, teacher_preds, P: dict) -> torch.Tensor:
    """Bản B: PKD trên 3 tầng neck + cross-head qua head teacher (cả o2m và o2o), có trọng số B2.

    L = w_feat · Σ_P PKD(Proj(F_s), F_t) + Σ_{br ∈ o2m, o2o} [ Σ w·QFL(ŝ, σ(s_t)) / (Σw·C) + w_reg · Σ w·|b̂ − b_t| /
    (Σw·4) ] với (ŝ, b̂) = head_teacher_br(Proj(F_s)) và B2: w = (p_m + p_o)/2 · (1 − |p_m − p_o|), p = max_c σ(score
    teacher) tại từng anchor (detach).

    ``P`` (kd_params): w_feat=1, w_reg=4, w_o2o=1, beta=1 (QFL), agree=True (bật B2).
    """
    w_feat, w_reg, w_o2o = float(P.get("w_feat", 1.0)), float(P.get("w_reg", 4.0)), float(P.get("w_o2o", 1.0))
    beta, agree = float(P.get("beta", 1.0)), bool(P.get("agree", True))
    proj = getattr(model, PROJ_ATTR)
    layers = model._feat_layers

    projected = [proj[n.replace(".", "_")](model._student_feats[n]) for n in layers]
    total = sum(_pkd(s, model._teacher_feats[n]) for s, n in zip(projected, layers)) * w_feat

    t_m, t_o = _branch(teacher_preds, "one2many"), _branch(teacher_preds, "one2one")
    if t_m["boxes"].shape[1] != 4:
        raise ValueError("ckd_b cần head reg_max = 1 (YOLO26); head có DFL không dùng được L1 trên 4 khoảng cách.")
    p_m = t_m["scores"].detach().sigmoid().amax(1, keepdim=True)
    p_o = t_o["scores"].detach().sigmoid().amax(1, keepdim=True)
    w = (p_m + p_o) / 2
    if agree:
        w = w * (1.0 - (p_m - p_o).abs())  # B2

    head = model._teacher.model[-1]
    for br, t_out, wb in (("one2many", t_m, 1.0), ("one2one", t_o, w_o2o)):
        if wb == 0 or (br == "one2one" and getattr(head, "one2one_cv2", None) is None):
            continue  # teacher không có nhánh o2o (không end2end) thì chỉ dùng o2m
        d = getattr(head, br)
        ob, oc = [], []
        for i, x in enumerate(projected):  # đặc trưng student (đã chiếu) → head của TEACHER
            fb = fc = x
            for k in range(3):
                fb, fc = d["box_head"][i][k](fb), d["cls_head"][i][k](fc)
            ob.append(fb.flatten(2)), oc.append(fc.flatten(2))
        re_box, re_cls = torch.cat(ob, -1), torch.cat(oc, -1)
        lc = (_qfl_kd(re_cls, t_out["scores"].detach().sigmoid(), beta) * w).sum() / (w.sum() * re_cls.shape[1] + 1e-9)
        l1 = (re_box - t_out["boxes"].detach()).abs()
        lr = (l1 * w).sum() / (w.sum() * l1.shape[1] + 1e-9)
        total = total + wb * (lc + w_reg * lr)
    return total


METHODS = {"builtin": builtin_loss, "ckd_b": ckd_b_loss}
DEFAULT_DIS = {"builtin": 6.0, "ckd_b": 0.14}  # ckd_b: hiệu chỉnh để L_KD ban đầu cùng thang với builtin dis=6


# ----------------------------------------------------------------------------- trước QAT / export
def strip_distill(src: str, dst: str, pruned_ckpt: str | None = None) -> None:
    """Chuẩn hoá checkpoint sau distill để đưa vào QAT / export.

    1. Bỏ mọi module chỉ dùng khi train (``distill_adapters``, ``distill_projectors``).
    2. Trả class về DetectionModelPruned / DetectionModel (bỏ lớp Distill*).
    3. Gán lại ``maskbndict`` từ checkpoint pruned — trainer KHÔNG lưu key này, mà
    QAT (finetune=True) bắt buộc phải có, thiếu sẽ AssertionError.
    """
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.tasks_pruned import DetectionModelPruned

    ck = torch.load(src, map_location="cpu", weights_only=False)
    for key in ("model", "ema"):
        m = ck.get(key)
        if m is None:
            continue
        for attr in ("distill_adapters", PROJ_ATTR):
            if attr in m._modules:
                del m._modules[attr]
        for attr in (
            "_lambda_cwd",
            "_temperature",
            "_kd_alpha",
            "_kd_alpha_max_epoch",
            "_feat_layers",
            "_kd_method",
            "_dis",
            "_kd_params",
        ):
            m.__dict__.pop(attr, None)
        m.__class__ = DetectionModelPruned if isinstance(m, DetectionModelPruned) else DetectionModel
    if pruned_ckpt and "maskbndict" not in ck:
        ck["maskbndict"] = torch.load(pruned_ckpt, map_location="cpu", weights_only=False)["maskbndict"]
    torch.save(ck, dst)
