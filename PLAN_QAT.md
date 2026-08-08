# Plan: QAT (Quantization-Aware Training) cho YOLO26 pruned — port từ repo tham chiếu

Trạng thái: **đã viết xong code** (chưa test được — máy hiện tại CPU-only
Windows, `pytorch-quantization` cần CUDA). Cần review + test trên môi trường
Linux/GPU trước khi tin dùng.
Phụ thuộc: checkpoint pruned + phục hồi (xem `PLAN_PRUNE.md`, `PLAN_DISTILL.md`).

## 1. Plan tổng quan

**Xác nhận target thật** (qua paper Purdue,
docs.lib.purdue.edu/cib-conferences/vol2/iss1/48): baseline YOLOv8s, backend
**TensorRT INT8 QAT**, triển khai trên **NVIDIA Jetson Orin Nano**, đạt
mAP@50-95=0.782, tăng 69.1% FPS so với FP32. Paper chạy pruning và QAT như 2
nhánh **song song, độc lập** — nhưng đã chốt với người dùng: dự án này đi
theo **nối tiếp** (khác paper, tiềm năng nén mạnh hơn):

```
sparsity training → prune → finetune/distill (đã làm) → QAT (mục này) → export INT8 TensorRT
```

QAT chạy sau khi checkpoint đã pruned + phục hồi độ chính xác (finetune hoặc
distill) — không chạy QAT trên checkpoint vừa prune còn yếu (đã thấy mAP có
thể gần 0 ngay sau prune, xem mục 11), vì quantize thêm nhiễu lên nền chưa
ổn định sẽ khó hội tụ.

### 1.1 Khác biệt kiến trúc YOLO26 cần lường trước

Đã xác nhận qua toàn bộ quá trình debug distillation (mục 3, 11-12) — áp
dụng nguyên vẹn cho QAT:

- **Không có DFL** (`reg_max=1`, `DFL` = `nn.Identity`) — QAT gốc cho
  YOLOv8/YOLOv6 phải xử lý quantize riêng cho phần DFL softmax (nhạy với
  quantize). YOLO26 **không cần bước này** — đơn giản hơn hẳn bản gốc.
- **End2end dual-head** — lúc inference thật chỉ nhánh `one2one` được dùng
  (xem `head_pruned.py:96-108`, forward trả `y` từ `preds["one2one"]` khi
  không training). Chỉ cần quantize `one2one_cv2/cv3`, có thể bỏ qua hoàn
  toàn `cv2/cv3` (one2many) ở bước export — giảm khối lượng QAT so với repo
  gốc (vốn không có khái niệm one2one/one2many).
- **C2PSA/C3k2Attn (attention)** — có `softmax`/`sigmoid` nội bộ, nhạy với
  quantize. Nên giữ FP32 ở lần đầu (giống cách distillation đã quyết định
  không distill riêng attention — xem mục 1 "Phạm vi tổng quát").
- **Channel bất đối xứng do pruning** — bắt buộc dùng **per-channel
  quantization**, không dùng per-tensor (per-tensor sẽ mất chính xác nặng
  hơn với model đã pruned vì phân phối weight lệch nhiều giữa các channel).
- **`distill_adapters`** (nếu checkpoint đến từ nhánh distill) — phải
  **strip bỏ trước khi QAT**, chỉ tồn tại lúc train distillation, không có
  trong graph suy luận thật.

### 1.2 Kỹ thuật QAT từ repo tham chiếu — giữ nguyên (model-agnostic)

Đọc trực tiếp `qat_pruned_trainer.py` của repo tham chiếu xác nhận các kỹ
thuật sau **không phụ thuộc kiến trúc cụ thể (YOLOv8 hay YOLO26)**, dùng lại
được nguyên vẹn:

1. **Calibration entropy (KL divergence)** — `collect_stats()` chạy forward
   trên `num_batch` batch từ **train data** (không phải val, đa dạng hơn),
   `compute_amax(method="entropy")` tìm range tối ưu cho mỗi quantizer.
2. **BN freeze hook** — `register_forward_pre_hook` (hàm top-level, không
   phải closure) ép BN ở eval mode trong lúc QAT training, dùng
   running_mean/var đã calibrate thay vì batch statistics bị nhiễu bởi fake
   quantization noise. **Đúng bài học đã tự rút ra ở mục 12** (hook phải
   picklable để không vỡ khi `deepcopy` cho EMA) — repo tham chiếu đã làm
   đúng từ đầu, không phải tự phát hiện lại.
3. **Pop custom override keys trước `super().__init__()`** — đúng pattern
   `DistillTrainer.__init__` đã dùng (mục 10, "Cơ chế tách trainer") để
   tránh `get_cfg()`/`check_dict_alignment` báo lỗi key lạ.
4. **LR scaling**: `lr0 /= 100` cho fine-tuning nhẹ trong lúc QAT (không
   train mạnh, chỉ để quantizer thích nghi).
5. **`recalib_every`**: định kỳ tính lại calibration trong lúc train QAT
   (không chỉ 1 lần đầu).

### 1.3 Kỹ thuật cần viết lại riêng cho YOLO26 (kiến trúc-specific)

- **`_skip_detect_quantizers`**: repo gốc disable toàn bộ quantizer trong
  `model.{detect_idx}` (Detect head, layer cuối) để giữ FP32 — với YOLO26,
  cần disable **cả `one2many` lẫn `one2one`** trong `DetectPruned`, không
  chỉ 1 khối Detect như YOLOv8.
- **`quant_module_change_pruned`** (đổi forward method sang bản quantized):
  repo gốc viết cho `C2fPruned` (`.chunk(2,1)`) và `BottleneckPruned`. YOLO26
  cần bản tương ứng cho `C3k2Pruned`/`C3k2C3kPruned` (`.split(sections,dim=1)`
  — đã có `cv1_split_sections` sẵn từ code prune, xem mục 12 code
  `C3k2Pruned.forward`), và **thêm mới** `C2PSAPruned`/`C3k2AttnPruned` (repo
  gốc không có, vì YOLOv8 không có attention block) — nhưng theo 13.1, các
  block attention này dự kiến giữ FP32 nên **không cần** viết quant forward
  riêng cho chúng ở lần đầu, chỉ cần đảm bảo quantizer bị disable đúng cách
  (tương tự kỹ thuật ở `_skip_detect_quantizers`).

### 1.4 File cần tạo

- `ultralytics/qat/nvidia_tensorrt/quant_ops_pruned_yolo26.py` — port từ
  `quant_ops_pruned.py`, đổi `C2fPruned`→`C3k2Pruned`/`C3k2C3kPruned`.
- `ultralytics/qat/nvidia_tensorrt/qat_trainer_yolo26.py` — port từ
  `qat_pruned_trainer.py`, đổi `_skip_detect_quantizers` để xử lý cả
  one2one/one2many, thêm disable quantizer cho `C2PSAPruned`/`C3k2AttnPruned`.
- `scripts/qat.py` (root, cùng style CLI với `distill.py`/`finetune.py`).

### 1.5 Việc cần xác nhận trước khi code

- Cần cài `pytorch-quantization` (NVIDIA) — thư viện này **khó cài trên
  Windows** (thường yêu cầu CUDA + build từ source hoặc container Linux).
  Máy hiện tại đang chạy CPU-only Windows — **không thể test QAT thật ở đây**,
  chỉ có thể viết code + review logic, việc test thật cần môi trường có GPU
  NVIDIA (khớp đúng target Jetson/TensorRT đã xác nhận).
- Checkpoint đầu vào cho QAT: chờ kết quả cuối cùng của benchmark distill vs
  finetune (mục 11 chạy lại) để chọn checkpoint tốt nhất làm input, thay vì
  chọn tuỳ ý.

### 1.6 Code đã viết — khác biệt so với repo tham chiếu

File: `ultralytics/qat/nvidia_tensorrt/{quant_ops_pruned_yolo26.py,
qat_trainer_yolo26.py}`, `scripts/qat.py`.

- **`quant_module_change_pruned_yolo26`**: một forward quantized
  (`c3k2_pruned_quant_forward`) dùng chung cho cả `C3k2Pruned`,
  `C3k2C3kPruned`, `C3k2AttnPruned` vì cả ba đều dùng đúng pattern
  `split(cv1_split_sections) -> submodules -> concat -> cv2`
  (`block_pruned.py`, đã đọc lại forward của cả 3 class để xác nhận giống hệt
  nhau trước khi viết) — đơn giản hơn có 3 hàm riêng như có thể đoán từ mục 1.3.
- **`_skip_detect_quantizers_yolo26`**: disable quantizer trong `model.{idx}`
  cuối cùng — tự động bắt cả `cv2/cv3` (one2many) lẫn `one2one_cv2/cv3`
  (one2one) vì cả hai đều nằm trong cùng `DetectPruned` (cùng prefix
  `model.{idx}.`), không cần tách logic riêng như dự tính ban đầu ở mục 1.3.
- **`_skip_attention_quantizers_yolo26`**: hàm mới (repo tham chiếu không có,
  YOLOv8 không có attention) — disable theo prefix tên module
  `C2PSAPruned`/`C3k2AttnPruned`.
- **Khác biệt kiến trúc lớn nhất so với repo tham chiếu**: `QATTrainerYolo26`
  **không** reimplement `_setup_train` như bản gốc (bản gốc copy gần như
  nguyên văn nội bộ `BaseTrainer` của ultralytics 8.3.231 — dataloader,
  optimizer, EMA, freeze layer thủ công, rất dễ vỡ khi upstream đổi API nội
  bộ). Thay vào đó: gọi `super()._setup_train()` (tái dùng toàn bộ logic có
  sẵn của `BaseTrainer`/`DetectionTrainer` ở phiên bản hiện tại, 8.4.52), chỉ
  chèn thêm phần QAT-specific (scale `lr0 /= 100` trước khi gọi super, đăng ký
  BN freeze hook + tắt AMP sau khi super chạy xong). Tương tự,
  `get_model()` tái dùng `DetectionTrainer.get_model()` (đã có sẵn logic load
  `maskbndict`/pruned checkpoint qua cờ `finetune`, xem `train.py:171-198`)
  thay vì tự parse checkpoint như repo tham chiếu — nhờ đó không cần bước
  "build model tạm để calibrate rồi build lại model cuối" như repo gốc
  (vốn cần vì họ tự quản lý toàn bộ vòng đời checkpoint); ở đây calibrate
  trực tiếp trên model thật ngay trong `get_model()`, dùng
  `self.get_dataloader()` (đã xác nhận `self.data` sẵn sàng ở thời điểm này
  vì `BaseTrainer.__init__` gọi `get_dataset()` trước khi `engine/model.py`
  gọi `get_model()`).
- **Đã xử lý** vấn đề `build_dataset` (train.py:77) đọc `self.model.stride`
  trong lúc `self.model` (thuộc tính trainer) vẫn còn là str path — `get_model()`
  gán tạm `self.model = model` (model quantized vừa build) trước khi gọi
  `self.get_dataloader()` để calibrate.
- **Chưa test được** trên máy này — cần review kỹ trên môi trường GPU trước
  khi chạy thật, đặc biệt tương tác giữa `_recalibrate()` (đổi `_amax`
  buffers trực tiếp trên named_buffers) và `ModelEMA` chưa test được đường
  save/resume thật.
