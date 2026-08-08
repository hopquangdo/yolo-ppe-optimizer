"""Quantized forward methods cho các module pruned YOLO26 (NVIDIA TensorRT QAT).

Port từ repo tham chiếu (``Pruned-QAT-for-YOLOv8-.../quant_ops_pruned.py``, viết cho
YOLOv8). Khác biệt chính so với bản gốc — xem ``PLAN_QAT.md`` mục 1.3:

- ``C2fPruned`` (YOLOv8, ``.chunk(2, 1)``) không tồn tại trong YOLO26. Thay bằng
  ``C3k2Pruned``/``C3k2C3kPruned``/``C3k2AttnPruned`` — cả ba đều đã dùng
  ``.split(cv1_split_sections, dim=1)`` sẵn (bất đối xứng sau pruning, xem
  ``block_pruned.py``), nên dùng chung một forward quantized duy nhất.
- Attention block (``C2PSAPruned``, ``C3k2AttnPruned``) **không** có forward
  quantized riêng ở đây — theo quyết định giữ FP32 (PLAN_QAT.md mục 1.1), chỉ cần
  đảm bảo quantizer bên trong bị disable (``_skip_attention_quantizers`` trong
  ``qat_trainer_yolo26.py``), không cần thêm ``QuantXxx`` op cho chúng.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils
from pytorch_quantization.tensor_quant import QuantDescriptor

# Các class dùng chung một forward quantized vì cùng pattern
# split(cv1_split_sections) -> submodules -> concat -> cv2 (block_pruned.py).
_C3K2_LIKE_CLASSES = ("C3k2Pruned", "C3k2C3kPruned", "C3k2AttnPruned")


# =============================================================================
# Quantized forward methods
# =============================================================================


def bottleneck_pruned_quant_forward(self, x):
    """Quantized forward cho BottleneckPruned — thêm QuantAdd cho residual."""
    if hasattr(self, "addop"):
        return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def c3k2_pruned_quant_forward(self, x):
    """Quantized forward dùng chung cho C3k2Pruned/C3k2C3kPruned/C3k2AttnPruned."""
    if hasattr(self, "c3k2splitop"):
        y = list(self.c3k2splitop(self.cv1(x)))
    else:
        y = list(self.cv1(x).split(self.cv1_split_sections, dim=1))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))


def concat_quant_forward(self, x):
    """Quantized forward cho Concat."""
    if hasattr(self, "concatop"):
        return self.concatop(x, self.d)
    return torch.cat(x, self.d)


def upsample_quant_forward(self, x):
    """Quantized forward cho Upsample."""
    if hasattr(self, "upsampleop"):
        return self.upsampleop(x)
    return F.interpolate(x, self.size, self.scale_factor, self.mode)


# =============================================================================
# Quantization wrapper modules
# =============================================================================


class QuantAdd(torch.nn.Module, _utils.QuantMixin):
    """Quantized addition cho residual connection trong BottleneckPruned."""

    def __init__(self, quantization):
        super().__init__()
        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y


class QuantC3k2SplitPruned(torch.nn.Module):
    """Quantized split cho C3k2Pruned/C3k2C3kPruned/C3k2AttnPruned.

    ``torch.split`` với sections tùy ý (bất đối xứng sau pruning, vd [80, 40]),
    giống ``QuantC2fSplitPruned`` của repo tham chiếu nhưng đổi tên cho khớp YOLO26.
    """

    def __init__(self, split_sections):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.split_sections = split_sections

    def forward(self, x):
        return torch.split(self._input0_quantizer(x), self.split_sections, dim=1)


class QuantConcat(torch.nn.Module):
    """Quantized concatenation cho FPN neck."""

    def __init__(self, dim):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.dim = dim

    def forward(self, x, dim):
        x_0 = self._input0_quantizer(x[0])
        x_1 = self._input1_quantizer(x[1])
        return torch.cat((x_0, x_1), self.dim)


class QuantUpsample(torch.nn.Module):
    """Quantized upsample cho FPN neck."""

    def __init__(self, size, scale_factor, mode):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())

    def forward(self, x):
        return F.interpolate(self._input_quantizer(x), self.size, self.scale_factor, self.mode)


# =============================================================================
# Apply quantization ops to pruned YOLO26 model
# =============================================================================


def quant_module_change_pruned_yolo26(model):
    """Thêm quantization modules vào pruned YOLO26 model.

    Quét tất cả modules và thay forward method bằng phiên bản quantized:
      - C3k2Pruned/C3k2C3kPruned/C3k2AttnPruned → thêm QuantC3k2SplitPruned
      - BottleneckPruned → thêm QuantAdd (chỉ khi có residual, ``module.add``)
      - Concat → thêm QuantConcat
      - Upsample → thêm QuantUpsample

    C2PSAPruned/C3k2AttnPruned nội bộ (Attention) và DetectPruned KHÔNG được xử lý
    ở đây — quantizer Conv2d bên trong chúng vẫn được thêm bởi
    ``quant_modules.initialize()`` (chạy trước hàm này), việc disable chúng để giữ
    FP32 nằm ở ``_skip_attention_quantizers``/``_skip_detect_quantizers_yolo26``
    trong ``qat_trainer_yolo26.py`` — tách biệt "thêm op quantize" và "bật/tắt
    quantizer" cho rõ trách nhiệm.

    Lưu ý: ``nn.Conv2d`` đã được thay bằng ``QuantConv2d`` qua
    ``quant_modules.initialize()`` trước khi gọi hàm này, nên không cần xử lý Conv ở đây.
    """
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__

        if cls_name in _C3K2_LIKE_CLASSES:
            if not hasattr(module, "c3k2splitop"):
                print(f"Add QuantC3k2SplitPruned to {name}")
                module.c3k2splitop = QuantC3k2SplitPruned(module.cv1_split_sections)
            module.__class__.forward = c3k2_pruned_quant_forward

        if cls_name == "BottleneckPruned":
            if module.add:
                if not hasattr(module, "addop"):
                    print(f"Add QuantAdd to {name}")
                    module.addop = QuantAdd(module.add)
                module.__class__.forward = bottleneck_pruned_quant_forward

        if cls_name == "Concat":
            if not hasattr(module, "concatop"):
                print(f"Add QuantConcat to {name}")
                module.concatop = QuantConcat(module.d)
            module.__class__.forward = concat_quant_forward

        if cls_name == "Upsample":
            if not hasattr(module, "upsampleop"):
                print(f"Add QuantUpsample to {name}")
                module.upsampleop = QuantUpsample(module.size, module.scale_factor, module.mode)
            module.__class__.forward = upsample_quant_forward
