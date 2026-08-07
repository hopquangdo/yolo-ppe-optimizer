"""Distillation trainer: teacher (YOLO26 bản lớn) -> student (YOLO26, pruned hoặc không).

Xem PLAN.md để biết thiết kế đầy đủ. Các ràng buộc tích hợp quan trọng được
tuân thủ ở đây (PLAN.md mục 10):
  - Teacher không bao giờ là submodule đã đăng ký của student, để
    ``build_optimizer`` (duyệt ``self.model.named_modules()`` không lọc theo
    ``requires_grad``) không bao giờ thấy tham số của teacher.
  - Không đụng vào nhánh ``sr`` (sparsity training) của ``BaseTrainer`` hay
    logic backward()/checkpoint-save của nó — distillation phải chạy với ``sr=0``.
  - Tái dùng ``DetectionTrainer.get_model()`` qua ``super()`` thay vì viết lại
    logic load finetune/maskbndict.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.distill_adapters import build_adapters
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.nn.tasks_pruned import DetectionModelPruned
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.distill_utils import (
    cls_kd_loss,
    cwd_loss,
    kd_alpha_schedule,
    register_feature_hooks,
    reg_kd_loss,
    remove_hooks,
)
from ultralytics.utils.tal import dist2bbox, make_anchors

FEAT_LAYERS = ["model.16", "model.19", "model.22"]  # P3/P4/P5, xem PLAN.md mục 3


def _unwrap_preds(preds: Any) -> dict:
    """Trả về dict head one2many bất kể có bọc end2end hay không."""
    return preds["one2many"] if isinstance(preds, dict) and "one2many" in preds else preds


def _decode_boxes(boxes_raw: torch.Tensor, feats: list[torch.Tensor], stride: torch.Tensor) -> torch.Tensor:
    """Decode offset thô từng cạnh thành box xywh theo tỉ lệ ảnh gốc.

    Mô phỏng lại ``DetectPruned._inference`` (head_pruned.py:110-116) nhưng không có DFL
    (reg_max=1 nghĩa là module DFL là ``nn.Identity``, xem PLAN.md mục 3).
    """
    anchors, strides = (a.transpose(0, 1) for a in make_anchors(feats, stride, 0.5))
    return dist2bbox(boxes_raw, anchors.unsqueeze(0), xywh=True, dim=1) * strides


def _teacher_forward_preds(teacher: nn.Module, img: torch.Tensor) -> dict:
    """Forward teacher với BN/dropout ở eval mode nhưng head vẫn trả về preds thô.

    Dùng lại đúng trick đã có trong ``tasks_pruned.py`` (``DetectionModelPruned.__init__``)
    để lấy output đúng shape lúc tính stride: chỉ cần cờ ``training`` của head là True để
    ``forward_head`` bỏ qua bước postprocess.
    """
    head = teacher.model[-1]
    was_head_training = head.training
    head.training = True
    try:
        with torch.no_grad():
            preds = teacher(img)
    finally:
        head.training = was_head_training
    return preds


class DistillMixin:
    """Thêm loss distillation từ teacher lên trên một implementation ``*.loss()`` bình thường.

    Được mix vào ``DetectionModel``/``DetectionModelPruned`` bằng cách đổi class
    (class-swap) trong ``DistillTrainer.get_model()``
    (``model.__class__ = DistillDetectionModelPruned``), nên dùng được cho cả
    trường hợp student đã pruned lẫn student là một scale nhỏ hơn bình thường
    (xem PLAN.md mục "Phạm vi tổng quát").
    """

    def attach_teacher(
        self,
        teacher: nn.Module,
        lambda_cwd: float = 50.0,
        temperature: float = 4.0,
        feat_layers: list[str] | None = None,
    ) -> None:
        feat_layers = feat_layers or FEAT_LAYERS
        object.__setattr__(self, "_teacher", teacher)  # ẩn khỏi named_modules()/parameters()
        self.distill_adapters = build_adapters(self, teacher, feat_layers)  # submodule thật, có thể train
        self._feat_layers = feat_layers
        self._lambda_cwd = lambda_cwd
        self._temperature = temperature
        self._kd_alpha = 1.0  # được callback của DistillTrainer cập nhật mỗi epoch (kd_alpha_schedule)
        self._student_hooks, self._student_feats = register_feature_hooks(self, feat_layers)
        self._teacher_hooks, self._teacher_feats = register_feature_hooks(teacher, feat_layers)

    def detach_teacher(self) -> None:
        """Gỡ hook và bỏ tham chiếu teacher (vd trước khi export)."""
        if hasattr(self, "_student_hooks"):
            remove_hooks(self._student_hooks)
        if hasattr(self, "_teacher_hooks"):
            remove_hooks(self._teacher_hooks)
        for attr in ("_teacher", "_student_hooks", "_teacher_hooks", "_student_feats", "_teacher_feats"):
            if hasattr(self, attr):
                object.__delattr__(self, attr) if attr == "_teacher" else delattr(self, attr)

    _DISTILL_ONLY_ATTRS = ("_teacher", "_student_hooks", "_teacher_hooks", "_student_feats", "_teacher_feats")

    def __deepcopy__(self, memo):
        """Loại bỏ teacher + hook khi bị deepcopy (lúc khởi tạo EMA, lúc lưu checkpoint).

        Cả bản snapshot EMA lẫn checkpoint đã lưu đều không nên mang theo tham chiếu
        teacher (PLAN.md mục 10, rủi ro 1/3) hay closure của hook (forward hook trên
        ``self.model`` chỉ được đăng ký cho student đang sống; một bản copy cũ sẽ hoặc
        giữ tham chiếu chết, hoặc nếu bị pickle sẽ lỗi — ``torch.save`` không pickle
        được hook trỏ ngược về dict của một object khác).
        """
        import copy as _copy

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in self._DISTILL_ONLY_ATTRS:
                continue
            setattr(new, k, _copy.deepcopy(v, memo))
        return new

    def loss(self, batch: dict, preds=None):
        # Validation giữa lúc train chạy trên trainer.ema.ema, một bản deep-copy riêng
        # mà hook forward của nó vẫn đóng (close) qua dict _student_feats của model GỐC
        # (Python deepcopy không rebind lại closure của hàm) — nên dict riêng của bản copy
        # luôn rỗng. Phát hiện việc này và fallback về task loss thuần thay vì crash.
        if getattr(self, "_teacher", None) is None or any(
            n not in getattr(self, "_student_feats", {}) for n in getattr(self, "_feat_layers", ())
        ):
            return super().loss(batch, preds)

        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])  # kích hoạt hook, điền self._student_feats
        task_loss, loss_items = self.criterion(preds, batch)

        teacher_preds = _teacher_forward_preds(self._teacher, batch["img"])  # điền self._teacher_feats

        student_head = _unwrap_preds(preds)
        teacher_head = _unwrap_preds(teacher_preds)

        cwd = torch.zeros((), device=task_loss.device if torch.is_tensor(task_loss) else self.device)
        for name in self._feat_layers:
            adapter = self.distill_adapters[name.replace(".", "_")]
            cwd = cwd + cwd_loss(self._student_feats[name], self._teacher_feats[name], adapter, T=self._temperature)

        cls_kd = cls_kd_loss(student_head["scores"], teacher_head["scores"], T=self._temperature)

        s_boxes_dec = _decode_boxes(student_head["boxes"], student_head["feats"], self.stride)
        t_boxes_dec = _decode_boxes(teacher_head["boxes"], teacher_head["feats"], self._teacher.stride)
        reg_kd = reg_kd_loss(student_head["boxes"], teacher_head["boxes"], s_boxes_dec, t_boxes_dec)

        total = task_loss + self._lambda_cwd * cwd + self._kd_alpha * (cls_kd + reg_kd)
        return total, loss_items


class DistillDetectionModelPruned(DistillMixin, DetectionModelPruned):
    """Student YOLO26 đã pruned, có loss distillation."""


class DistillDetectionModel(DistillMixin, DetectionModel):
    """Student YOLO26 không pruned (scale bất kỳ), có loss distillation."""


class DistillTrainer(DetectionTrainer):
    """Biến thể của DetectionTrainer, train student đối chiếu với một teacher đóng băng.

    Các kwarg thêm cho ``model.train(...)`` (bị pop ở đây, trước khi ``get_cfg``
    kiểm tra key lạ — xem PLAN.md mục 10):
      teacher (str): đường dẫn checkpoint teacher (bắt buộc).
      lambda_cwd (float): trọng số CWD feature-loss, mặc định 50.0 (PLAN.md 4.1).
      kd_temperature (float): nhiệt độ cho CWD/cls KD, mặc định 4.0.
      kd_alpha_max_epoch (int, tuỳ chọn): số epoch dùng cho lịch alpha cosine
        (PLAN.md 4.4); mặc định lấy theo ``epochs`` trong training args.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        overrides = dict(overrides or {})
        self.teacher_weights = overrides.pop("teacher", None)
        self.lambda_cwd = float(overrides.pop("lambda_cwd", 50.0))
        self.kd_temperature = float(overrides.pop("kd_temperature", 4.0))
        self.kd_alpha_max_epoch = overrides.pop("kd_alpha_max_epoch", None)
        if not self.teacher_weights:
            raise ValueError("DistillTrainer requires a 'teacher' checkpoint path.")
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_train_epoch_start", self._update_kd_alpha)

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True, maskbndict=None):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose, maskbndict=maskbndict)
        teacher, _ = load_checkpoint(self.teacher_weights, device=self.device)
        for p in teacher.parameters():
            p.requires_grad = False

        cls = DistillDetectionModelPruned if isinstance(model, DetectionModelPruned) else DistillDetectionModel
        model.__class__ = cls
        model.attach_teacher(teacher, lambda_cwd=self.lambda_cwd, temperature=self.kd_temperature)
        max_epoch = self.kd_alpha_max_epoch or getattr(self.args, "epochs", 100)
        model._kd_alpha_max_epoch = max_epoch
        LOGGER.info(
            colorstr("yellow", f"Distillation enabled: teacher={self.teacher_weights}, "
                     f"lambda_cwd={self.lambda_cwd}, T={self.kd_temperature}, kd_alpha_max_epoch={max_epoch}")
        )
        return model

    @staticmethod
    def _update_kd_alpha(trainer: "DistillTrainer") -> None:
        from ultralytics.utils.torch_utils import unwrap_model

        model = unwrap_model(trainer.model)
        if hasattr(model, "_kd_alpha"):
            model._kd_alpha = kd_alpha_schedule(trainer.epoch, getattr(model, "_kd_alpha_max_epoch", trainer.epochs))
