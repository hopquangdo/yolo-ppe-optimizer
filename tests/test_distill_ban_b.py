"""Kiểm thử đơn vị cho kd_method của DistillTrainer: builtin và ckd_b (bản B).

Chạy trên CPU, không cần data hay checkpoint: student YOLO26n / teacher YOLO26s dựng từ yaml (9 lớp).
    pytest tests/test_distill_ban_b.py -q
"""

import pytest
import torch

from ultralytics.models.yolo.detect.distill_trainer import (
    DistillDetectionModel,
    _teacher_forward_preds,
)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.distill_methods import PROJ_ATTR, builtin_loss, ckd_b_loss, parse_params, strip_distill

IMG = 64


def _pair(kd_method="ckd_b"):
    torch.manual_seed(0)
    student = DetectionModel("yolo26n.yaml", nc=9, verbose=False)
    teacher = DetectionModel("yolo26s.yaml", nc=9, verbose=False).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student.__class__ = DistillDetectionModel
    student.attach_teacher(teacher, kd_method=kd_method, dis=0.14 if kd_method == "ckd_b" else 6.0)
    student.train()
    return student, teacher


def _kd_forward(student, teacher, img):
    student.forward(img)  # hook ghi đặc trưng student
    return _teacher_forward_preds(teacher, img)  # hook ghi đặc trưng teacher


def test_projector_shapes():
    student, _ = _pair()
    proj = getattr(student, PROJ_ATTR)
    # YOLO26n -> YOLO26s: P3 64->128, P4 128->256, P5 256->512
    assert [p[0].in_channels for p in proj.values()] == [64, 128, 256]
    assert [p[-1].out_channels for p in proj.values()] == [128, 256, 512]
    assert sum(x.numel() for x in proj.parameters()) == 517888


def test_ckd_b_gradient_flow():
    """L_KD chỉ cập nhật projector + backbone/neck student; không chạm head student, không chạm teacher."""
    student, teacher = _pair()
    img = torch.rand(2, 3, IMG, IMG)
    tp = _kd_forward(student, teacher, img)
    loss = ckd_b_loss(student, tp, {})
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters()), "teacher không được nhận gradient"
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(student, PROJ_ATTR).parameters())
    head = student.model[-1]
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in head.parameters()), (
        "head student chỉ học từ nhãn thật"
    )
    assert student.model[0].conv.weight.grad.abs().sum() > 0, "gradient KD phải tới backbone"


def test_b2_toggle():
    """kd_params agree=False tắt B2 (chỉ còn trọng số độ tin cậy) → loss cross-head khác khi bật B2."""
    student, teacher = _pair()
    # teacher mới khởi tạo có o2o = deepcopy(o2m) nên hai nhánh trùng nhau → B2 = 1; làm lệch nhánh o2o
    with torch.no_grad():
        for m in teacher.model[-1].one2one_cv3:
            m[-1].bias.add_(torch.randn_like(m[-1].bias))
    img = torch.rand(2, 3, IMG, IMG)
    tp = _kd_forward(student, teacher, img)
    with torch.no_grad():
        a = ckd_b_loss(student, tp, {"w_feat": 0.0})
        b = ckd_b_loss(student, tp, {"w_feat": 0.0, "agree": False})
    assert torch.isfinite(a) and torch.isfinite(b) and not torch.allclose(a, b)


def test_builtin_matches_ultralytics_formula():
    """builtin_loss = sum_P L2 có trọng số max_c sigmoid((s_o2m + s_o2o)/2), như DistillationModel 8.4.152."""
    student, teacher = _pair("builtin")
    img = torch.rand(2, 3, IMG, IMG)
    tp = _kd_forward(student, teacher, img)
    with torch.no_grad():
        got = builtin_loss(student, tp)
        sc = (tp["one2many"]["scores"] + tp["one2one"]["scores"]) / 2
        feats = [student._teacher_feats[n] for n in student._feat_layers]
        parts = torch.split(sc, [f.shape[-2] * f.shape[-1] for f in feats], dim=-1)
        ref = 0.0
        for n, part in zip(student._feat_layers, parts):
            w = part.sigmoid().max(dim=1, keepdim=True).values
            s = getattr(student, PROJ_ATTR)[n.replace(".", "_")](student._student_feats[n])
            t = student._teacher_feats[n]
            c = s.shape[1]
            mse = (s.flatten(2) - t.flatten(2)) ** 2
            ref = ref + (mse * w).sum() / (w.sum() * c + 1e-9)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-7)


def test_parse_params():
    assert parse_params(None) == {}
    assert parse_params('{"agree": false}') == {"agree": False}
    assert parse_params({"w_o2o": 0}) == {"w_o2o": 0}


def test_strip_distill_roundtrip(tmp_path):
    student, _ = _pair()
    src, dst = tmp_path / "best.pt", tmp_path / "best_clean.pt"
    import copy

    torch.save({"model": copy.deepcopy(student), "ema": None, "train_args": {}}, src)
    strip_distill(str(src), str(dst))
    m = torch.load(dst, map_location="cpu", weights_only=False)["model"]
    assert type(m) is DetectionModel
    assert PROJ_ATTR not in m._modules and "distill_adapters" not in m._modules
    assert not any(hasattr(m, a) for a in ("_kd_method", "_dis", "_kd_params", "_feat_layers"))
    m.eval()(torch.rand(1, 3, IMG, IMG))  # chạy được như model thường


def test_invalid_method():
    with pytest.raises(ValueError):
        _pair("khong_co")
