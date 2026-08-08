"""QAT (Quantization-Aware Training) trainer cho pruned YOLO26 — NVIDIA TensorRT INT8.

Xem ``PLAN_QAT.md`` để biết thiết kế đầy đủ. Port từ repo tham chiếu
(``Pruned-QAT-for-YOLOv8-.../qat_pruned_trainer.py``, viết cho YOLOv8) nhưng
**không** reimplement lại toàn bộ ``_setup_train`` như bản gốc — bản gốc copy gần
như nguyên văn nội bộ ``BaseTrainer`` của ultralytics 8.3.231 (build dataloader,
optimizer, EMA, freeze layer thủ công), rất dễ vỡ khi ultralytics nâng cấp (repo
này đang ở 8.4.52). Ở đây dùng chiến lược tối thiểu hoá override (PLAN_QAT.md 1.5):

- ``get_model()``: tái dùng ``DetectionTrainer.get_model()`` (đã biết load
  ``maskbndict``/pruned checkpoint qua cờ ``finetune``, xem ``train.py:171-198``)
  để dựng ``DetectionModelPruned`` — chỉ thêm bước quant hoá trước/sau.
- ``_setup_train()``: gọi ``super()._setup_train()`` để dùng nguyên toàn bộ logic
  dataloader/optimizer/EMA/freeze của ``BaseTrainer``, chỉ chèn thêm phần
  QAT-specific (BN freeze hook, tắt AMP) sau khi super() chạy xong.

Yêu cầu ``pytorch_quantization`` (NVIDIA) — không cài được trên máy Windows
CPU-only hiện tại, chỉ viết + review logic ở đây; test thật cần môi trường Linux/GPU
khớp target Jetson Orin Nano + TensorRT (PLAN_QAT.md mục 1.5).
"""

from __future__ import annotations

from pytorch_quantization import calib, quant_modules
from pytorch_quantization import nn as quant_nn
from tqdm import tqdm

import torch
import torch.nn as nn

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import TORCH_2_4, is_parallel
from ultralytics.utils.torch_utils import unwrap_model as de_parallel

from .quant_ops_pruned_yolo26 import quant_module_change_pruned_yolo26

# =============================================================================
# Calibration utilities (model-agnostic — giữ nguyên từ repo tham chiếu)
# =============================================================================


def _set_quantizer_state(model, enable_quant: bool, enable_calib: bool):
    """Đặt trạng thái đồng nhất cho tất cả TensorQuantizer trong model.

    quantizer có calibrator -> dùng calib mode hoặc quant mode; quantizer không
    calibrator (vd weight quantizer đã có ``_amax`` từ khởi tạo) -> luôn enable_quant.
    """
    for name, module in model.named_modules():
        if not isinstance(module, quant_nn.TensorQuantizer):
            continue
        if enable_calib:
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.enable_quant()
        else:
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            elif module._amax is None:
                LOGGER.warning(f"[QAT-Yolo26] TensorQuantizer '{name}' không có _amax và không có calibrator — disable.")
                module.disable()
            else:
                module.enable_quant()


def _verify_amax(model, device):
    """Sau calibration: đảm bảo mọi quantizer đang enabled đều có ``_amax`` hợp lệ."""
    fixed = 0
    for name, module in model.named_modules():
        if not isinstance(module, quant_nn.TensorQuantizer) or module._disabled:
            continue
        if module._amax is None:
            LOGGER.warning(f"[QAT-Yolo26] '{name}': _amax là None sau calibration -> disable.")
            module.disable()
            fixed += 1
        else:
            module._amax = module._amax.to(device)
    if fixed:
        LOGGER.warning(f"[QAT-Yolo26] Đã disable {fixed} quantizer thiếu _amax.")


def cal_model(model, data_loader, device, num_batch=256):
    """Calibrate quantization scale bằng entropy method (KL divergence)."""

    def collect_stats(model, data_loader, device, num_batch):
        model.eval()
        _set_quantizer_state(model, enable_quant=False, enable_calib=True)
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader), total=num_batch, desc="[QAT-Yolo26] Collecting calib stats"):
                imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
                model(imgs)
                if i >= num_batch:
                    break
        _set_quantizer_state(model, enable_quant=True, enable_calib=False)

    def compute_amax(model, device, **kwargs):
        for name, module in model.named_modules():
            if not isinstance(module, quant_nn.TensorQuantizer) or module._calibrator is None:
                continue
            if isinstance(module._calibrator, calib.MaxCalibrator):
                module.load_calib_amax(strict=False)
            else:
                module.load_calib_amax(**kwargs)
            if module._amax is not None:
                module._amax = module._amax.to(device)

    collect_stats(model, data_loader, device, num_batch)
    compute_amax(model, device, method="entropy")
    _verify_amax(model, device)


# =============================================================================
# BN freeze hook (model-agnostic — giữ nguyên từ repo tham chiếu)
# =============================================================================


def _bn_eval_hook(module, input):
    """Top-level (picklable) BN forward pre-hook — giữ BN ở eval mode trong QAT."""
    module.eval()


def _register_bn_freeze_hook(model):
    """Đăng ký forward pre-hook giữ BatchNorm ở eval mode trong lúc QAT training.

    Dùng hook (không phải monkey-patch ``model.train()``) vì hook chỉ ảnh hưởng
    forward, sống sót qua ``deepcopy`` (EMA) mà không mang theo state sai — đúng
    bài học đã rút ra ở ``DistillMixin.__deepcopy__`` (distill_trainer.py) và hàm
    này là top-level (không phải closure) để pickle được khi lưu checkpoint.
    """
    bn_layers = [m for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm))]
    handles = [bn.register_forward_pre_hook(_bn_eval_hook) for bn in bn_layers]
    LOGGER.info(f"[QAT-Yolo26] Đăng ký BN freeze hook trên {len(bn_layers)} lớp BatchNorm.")
    return handles


# =============================================================================
# Skip quantizer cho Detect head + attention block — giữ FP32 (PLAN_QAT.md 1.1/1.3)
# =============================================================================


def _skip_detect_quantizers_yolo26(model):
    """Disable quantizer trong DetectPruned (cả one2many lẫn one2one) — giữ FP32.

    Khác repo tham chiếu (chỉ có 1 nhánh Detect): DetectPruned có 2 nhánh
    (``cv2``/``cv3`` one2many dùng lúc train, ``one2one_cv2``/``one2one_cv3`` dùng
    lúc suy luận thật — head_pruned.py:96-108) — phải disable cả hai vì checkpoint
    export cuối cùng cần cả hai FP32 nhất quán (one2many vẫn tồn tại trong graph dù
    không dùng lúc export, xem ``DetectPruned.forward``).
    """
    model_inner = de_parallel(model) if is_parallel(model) else model
    detect_idx = str(len(model_inner.model) - 1)

    disabled = 0
    for name, module in model_inner.named_modules():
        if not isinstance(module, quant_nn.TensorQuantizer):
            continue
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "model" and parts[1] == detect_idx:
            module.disable()
            disabled += 1
    LOGGER.info(f"[QAT-Yolo26] Disabled {disabled} quantizer trong DetectPruned (model.{detect_idx}) -> FP32.")
    return disabled


def _skip_attention_quantizers_yolo26(model):
    """Disable quantizer trong C2PSAPruned/C3k2AttnPruned (Attention nội bộ) — giữ FP32.

    YOLO26 có attention block mà YOLOv8 (repo tham chiếu) không có. Attention có
    softmax/sigmoid nội bộ, nhạy với quantization nhiễu — giữ FP32 ở lần triển khai
    đầu tiên, giống quyết định đã áp dụng cho distillation (không distill riêng
    attention, xem PLAN_DISTILL.md).
    """
    model_inner = de_parallel(model) if is_parallel(model) else model
    attn_prefixes = tuple(
        f"{name}." for name, module in model_inner.named_modules() if module.__class__.__name__ in ("C2PSAPruned", "C3k2AttnPruned")
    )
    if not attn_prefixes:
        return 0

    disabled = 0
    for name, module in model_inner.named_modules():
        if not isinstance(module, quant_nn.TensorQuantizer):
            continue
        if name.startswith(attn_prefixes):
            module.disable()
            disabled += 1
    LOGGER.info(f"[QAT-Yolo26] Disabled {disabled} quantizer trong attention block ({len(attn_prefixes)} khối) -> FP32.")
    return disabled


# =============================================================================
# TRAINER
# =============================================================================


class QATTrainerYolo26(DetectionTrainer):
    """QAT trainer cho pruned YOLO26, output TensorRT INT8 (Jetson Orin Nano).

    Kỹ thuật áp dụng (PLAN_QAT.md mục 1.2 — model-agnostic, giữ nguyên từ repo
    tham chiếu) + phần viết riêng cho YOLO26 (mục 1.3):
      1. Calibration entropy (KL divergence) trên train data.
      2. BN freeze hook (eval mode) trong lúc QAT training.
      3. DetectPruned (cả one2many/one2one) + attention block giữ FP32.
      4. LR scaling: lr0 /= 100 (fine-tuning nhẹ, chỉ để quantizer thích nghi).
      5. ``recalib_every``: định kỳ tính lại calibration trong lúc train.

    Kwarg thêm cho ``model.train(...)`` (pop trước ``super().__init__()``, đúng
    pattern ``DistillTrainer`` đã dùng — tránh ``get_cfg()`` báo lỗi key lạ):
      calib_batches (int): số batch dùng để calibrate, mặc định 256.
      recalib_every (int): số epoch giữa 2 lần recalibrate, 0 = tắt. Mặc định 0
        (khác repo tham chiếu mặc định=1 — recalibrate mỗi epoch tốn thời gian
        đáng kể trên Jetson/edge-dev-loop; bật thủ công khi cần).
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        overrides = dict(overrides or {})
        self.calib_batches = int(overrides.pop("calib_batches", 256))
        self.recalib_every = int(overrides.pop("recalib_every", 0))
        # QAT luôn chạy trên checkpoint đã pruned (PLAN_QAT.md mục 1) -> bắt buộc finetune=True
        # để DetectionTrainer.get_model() load DetectionModelPruned + maskbndict.
        overrides["finetune"] = True
        super().__init__(cfg, overrides, _callbacks)
        self._bn_hook_handles = []

    def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True, maskbndict=None):
        """Dựng model quantized từ pruned checkpoint.

        1. ``quant_modules.initialize()`` — monkey-patch ``nn.Conv2d`` -> ``QuantConv2d``
           TRƯỚC khi model được tạo, để mọi Conv trong ``DetectionModelPruned`` tự
           động là quantized (khác cấu trúc: quant hoá ở cấp global, không phải
           từng layer thủ công).
        2. ``super().get_model(...)`` — tái dùng logic có sẵn của
           ``DetectionTrainer`` (load maskbndict + pruned weights, re-init bias
           detect head) thay vì viết lại (khác repo tham chiếu, xem docstring module).
        3. ``quant_module_change_pruned_yolo26()`` — thêm quantizer cho các khối
           split-based (C3k2*/BottleneckPruned/Concat/Upsample).
        4. Calibrate trên train data (entropy method) — dùng ``self.get_dataloader``
           trực tiếp vì ``self.data`` đã sẵn sàng ở thời điểm này (``BaseTrainer.__init__``
           gọi ``get_dataset()`` trước khi ``engine/model.py`` gọi ``get_model()``).
           ``build_dataset`` (``DetectionTrainer.build_dataset``, train.py:77) đọc
           ``self.model.stride`` để tính grid size — nhưng ``self.model`` (thuộc
           tính trainer) tại thời điểm này vẫn là str/Path (``BaseTrainer.__init__``
           chỉ set path, ``engine/model.py`` mới gán model thật SAU KHI hàm này trả
           về) — nên phải gán tạm ``self.model = model`` trước khi gọi
           ``get_dataloader`` (giống trick repo tham chiếu dùng, chỉ khác: ở đây
           không cần khôi phục lại giá trị cũ vì giá trị cũ (str) không còn được
           dùng nữa sau bước này).
        5. Skip quantizer cho Detect head + attention -> giữ FP32.
        """
        quant_modules.initialize()
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose, maskbndict=maskbndict)
        quant_module_change_pruned_yolo26(model)

        if RANK in (-1, 0):
            model.to(self.device)
            self.model = model  # build_dataset() cần self.model.stride — xem docstring
            calib_bs = max(self.args.batch, 1) * 2 if isinstance(self.args.batch, int) and self.args.batch > 0 else 16
            calib_loader = self.get_dataloader(self.data["train"], batch_size=calib_bs, rank=-1, mode="val")
            LOGGER.info(
                colorstr("yellow", f"[QAT-Yolo26] Calibrating trên train set ({calib_bs} batch, {self.calib_batches} batches) — entropy method")
            )
            cal_model(model, calib_loader, self.device, num_batch=self.calib_batches)

        _skip_detect_quantizers_yolo26(model)
        _skip_attention_quantizers_yolo26(model)
        return model

    def _setup_train(self, world_size=None):
        """Gọi nguyên ``BaseTrainer._setup_train`` rồi chèn thêm phần QAT-specific.

        Khác repo tham chiếu (reimplement toàn bộ dataloader/optimizer/EMA/freeze
        thủ công): ở đây chỉ scale ``lr0`` trước khi ``super()`` build optimizer
        (optimizer đọc ``self.args.lr0`` trong lúc chạy), rồi đăng ký BN freeze
        hook + tắt AMP sau khi model/dataloader đã sẵn sàng — giảm rủi ro vỡ khi
        ultralytics đổi nội bộ ``BaseTrainer`` (PLAN_QAT.md mục 1.5).
        """
        self.args.lr0 /= 100.0
        # QAT không dùng sparsity regularization (BaseTrainer._do_train đọc self.sr).
        self.sr = 0.0

        super()._setup_train(world_size)

        model_inner = de_parallel(self.model) if is_parallel(self.model) else self.model
        self._bn_hook_handles = _register_bn_freeze_hook(model_inner)

        # Fake-quantization không tương thích AMP (giống repo tham chiếu).
        self.amp = False
        self.scaler = torch.amp.GradScaler("cuda", enabled=False) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=False)

    def _recalibrate(self):
        """Re-calibrate quantization scale giữa lúc train (dùng test_loader)."""
        LOGGER.info(f"[QAT-Yolo26] Re-calibrating tại epoch {self.epoch + 1}...")
        model = de_parallel(self.model) if is_parallel(self.model) else self.model

        for h in self._bn_hook_handles:
            h.remove()
        self._bn_hook_handles = []

        model.eval()
        cal_model(model, self.test_loader, self.device, num_batch=self.calib_batches)
        _skip_detect_quantizers_yolo26(model)
        _skip_attention_quantizers_yolo26(model)

        self._bn_hook_handles = _register_bn_freeze_hook(model)
        model.train()

        if self.ema:
            ema_buffers = dict(self.ema.ema.named_buffers())
            for name, buf in model.named_buffers():
                if "_amax" in name and name in ema_buffers:
                    ema_buffers[name].data.copy_(buf.data)
        LOGGER.info("[QAT-Yolo26] Re-calibration xong.")

    def validate(self):
        """Recalibrate trước validation nếu tới hạn ``recalib_every``."""
        if self.recalib_every > 0 and RANK in (-1, 0) and (self.epoch == 0 or (self.epoch + 1) % self.recalib_every == 0):
            self._recalibrate()
        return super().validate()
