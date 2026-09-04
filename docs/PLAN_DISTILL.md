# Plan: Distillation YOLO26x/l (teacher) → YOLO26n-pruned (student)

Trạng thái: đã xong phần research/thiết kế, **chưa code**.
Phạm vi: tài liệu này chỉ nói về giai đoạn distillation. Giả định
`train_sparsity.py` → `prune.py` → `finetune.py` đã chạy được (đã verify
riêng — `prune.py` chạy đúng trên layer thật của YOLO26; nó cần checkpoint
đã sparsity-trained, không dùng được checkpoint pretrained thô).

**Phạm vi tổng quát**: mọi công thức/layer-index trong tài liệu này áp
dụng được cho **bất kỳ cặp (teacher lớn hơn, student nhỏ hơn) nào trong họ
YOLO26** — n/s/m/l/x, pruned hoặc không pruned — vì mọi scale dùng chung
layout (`yolo26.yaml` và `yolo26-pruned.yaml` cùng cấu trúc layer, chỉ khác
`scales:` width/depth multiplier) và chung `reg_max: 1` (không có DFL ở bất
kỳ scale nào). Case cụ thể được chọn để triển khai trước là teacher
yolo26x/l → student **yolo26n đã pruned**, vì đây là case khó nhất (channel
split bất đối xứng do mask pruning) — nếu chạy đúng case này, distill giữa
2 scale chuẩn (không pruned, channel split luôn đối xứng) sẽ đơn giản hơn,
không cần thiết kế lại. Khác biệt duy nhất cần code xử lý: loại model
(`DetectionModel` vs `DetectionModelPruned`) phải chọn theo tham số, không
hardcode — xem mục 10.

## 1. Vị trí trong pipeline tổng — 2 study song song

Không chạy 1 pipeline duy nhất — chạy **2 study độc lập, cùng xuất phát từ
1 pruned checkpoint**, cả hai đều đi hết tới QAT + export, để so sánh được
ở điểm cuối cùng (checkpoint thật sự đem deploy), không chỉ so ở bước
recovery-training giữa chừng:

```
                          ┌─ Study A: distill.py  ─┐
train_sparsity.py → prune.py ─┤                         ├─→ QAT → export → so sánh
                          └─ Study B: finetune.py ─┘
```

- **Study A** — `train_sparsity.py → prune.py → distill.py → QAT → export`.
  `distill.py` thay hẳn vai trò recovery-training của `finetune.py`: tổng
  loss ở mục 4.5 đã bao gồm `L_task` (chính là loss của finetune thường
  trên GT) cộng thêm distill loss, cùng một lượt train — không phải chạy
  nối tiếp `finetune.py` rồi mới `distill.py` sau (làm vậy vừa lãng phí
  epoch vừa phá vỡ tính "sạch" của so sánh, vì model đã hội tụ trên GT rồi
  mới học thêm từ teacher).
- **Study B** — `train_sparsity.py → prune.py → finetune.py → QAT → export`.
  Đây **không chỉ là Baseline-0 để tham chiếu tạm** như bản trước của tài
  liệu này viết — mà là **một study hoàn chỉnh, tự nó cũng đi hết tới
  QAT/export**, vì mục tiêu cuối là so sánh 2 checkpoint triển khai được
  thật sự (sau QAT) khác nhau thế nào, không chỉ so mAP ở bước finetune
  giữa chừng. Có thể distill (Study A) giúp trước QAT nhưng bị quantization
  ăn mất lợi thế, hoặc ngược lại giúp QAT ổn định hơn — chỉ so ở bước giữa
  sẽ bỏ lỡ hiệu ứng này.

Cả 2 study dùng **chung 1 pruned checkpoint** (cùng tỷ lệ prune, cùng seed
sparsity-training) và **chung cấu hình QAT** (cùng script `qat.py` ở mục
kế hoạch QAT trước đó) — chỉ khác nhau ở bước recovery-training
(distill vs finetune thường). Nếu không cố định 2 điều này, chênh lệch kết
quả cuối có thể đến từ nhiễu prune/QAT chứ không phải từ distill.

Lúc recovery-training (dù là distill hay finetune), kiến trúc pruned của
student đã cố định (`DetectionModelPruned`), nên có một tập shape tensor cố
định để khớp với teacher (áp dụng cho Study A).

## 2. Câu hỏi nghiên cứu (cố tình tách riêng, không gộp)

- **Q1**: Distillation có phục hồi mAP sau pruning tốt hơn finetune thường
  (không distill) không? → cần baseline không-distill để so sánh.
- **Q2**: Feature-based distillation có đóng góp thêm gì so với chỉ
  response-based (logit) distillation không? → cần tách riêng 2 phần.
- **Q3**: Kết quả nhạy đến mức nào với trọng số loss (λ_CWD, lịch α của
  KD)? → cần quét một lưới nhỏ, không đoán 1 bộ số rồi dùng luôn.

**Không được gộp cả 3 câu hỏi vào 1 lần chạy "distill hết rồi xem sao"** —
nếu nó hoạt động, sẽ không biết phần nào thực sự có tác dụng; nếu không
hoạt động, sẽ không biết sửa ở đâu.

## 3. Fact kiến trúc đã verify bằng code thật (không suy đoán)

- **Layer feature dùng cho CWD**: `ultralytics/cfg/models/26/yolo26-pruned.yaml`
  ghi rõ "Same layout as yolo26.yaml, chỉ khác module class". `DetectPruned`
  nhận `f=[16, 19, 22]` (`yolo26-pruned.yaml:46`). Đây chính là output P3/P4/P5
  của neck, đã xác nhận **cùng chỉ số layer** ở cả config student (pruned)
  và teacher (không pruned) — không cần đoán tên hay hook mù, chỉ khác số
  lượng channel, vị trí khớp 1-1.
- **YOLO26 không có DFL.** Verify trực tiếp bằng code, không suy từ paper
  viết cho YOLOv6/v8/v11:
  - Toàn bộ file `ultralytics/cfg/models/26/*.yaml` đều set `reg_max: 1`.
  - `ultralytics/nn/modules/head_pruned.py:65`:
    `self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()`
    — với `reg_max=1`, dòng này trở thành `nn.Identity()`, tức là box
    regression là **1 giá trị số thực trực tiếp mỗi cạnh**, không phải
    phân phối xác suất rời rạc trên các bin.
  - Hệ quả: `KL(p_reg_teacher || p_reg_student)` (công thức self-distillation
    của YOLOv6 cho box DFL) **không áp dụng được**. Softmax trên 1 phần tử
    luôn = 1 → KL luôn = 0 → loss chết, không sinh gradient nào. Đây là lỗi
    được bắt **trước khi viết code**, đặc biệt nguy hiểm vì nó không crash,
    chỉ âm thầm vô nghĩa (sẽ thấy mAP không cải thiện mà không hiểu vì sao).

## 4. Công thức loss (mỗi công thức đều có nguồn, không đoán)

### 4.1 Feature distillation — CWD (Channel-wise Distillation, Shu et al., arXiv:2011.13256)

Áp dụng tại 3 cặp feature: output teacher/student ở layer 16, 19, 22
(P3/P4/P5), mỗi cặp qua 1 adapter Conv1x1+BN ánh xạ channel student →
channel teacher (số channel khác nhau sau khi prune; adapter chỉ tồn tại
lúc train, bỏ đi trước khi export).

Với mỗi channel `c`, softmax theo không gian với nhiệt độ `T`:

```
phi(y_c,i) = exp(y_c,i / T) / sum_j exp(y_c,j / T)     (softmax theo vị trí không gian i, j = 1..W*H)

L_CWD = T^2 * sum_c sum_i phi(y_c,i^teacher) * log( phi(y_c,i^teacher) / phi(y_c,i^student) )
```

KL bất đối xứng, chiều teacher → student. Nhân `T^2` để scale lại gradient
(kỹ thuật KD chuẩn, vì T càng lớn gradient càng bị co nhỏ nếu không nhân bù).

**Trọng số**: λ_CWD ≈ 50 (theo khuyến nghị gốc của paper CWD cho feature
map — lớn hơn nhiều so với trọng số loss thông thường vì KL trên softmax
không gian cho ra giá trị rất nhỏ). Đây chỉ là điểm khởi đầu cho sweep Q3,
không phải giá trị cuối cùng.

### 4.2 Response distillation — nhánh classification

Vẫn là phân phối xác suất thật (score theo từng class, sigmoid/softmax
trên `nc` lớp) dù có hay không có DFL — không bị ảnh hưởng bởi phát hiện
ở mục 3.

```
L_cls_KD = KL(p_cls_teacher || p_cls_student)
```

Tensor nguồn: `x["scores"]` (logit trước sigmoid, từ `forward_head` trong
`head_pruned.py:94`), softmax theo chiều class.

### 4.3 Response distillation — nhánh regression (đã sửa lại)

YOLO26 không có DFL, nên nhánh box là regression trực tiếp. Công thức
kiểu YOLOv6 `KL(p_reg_t || p_reg_s)` không hợp lệ ở đây (mục 3). Thay bằng
loss khớp trực tiếp trên box đã decode:

```
L_reg_KD = lambda1 * SmoothL1(y_hat_s, y_hat_t) + lambda2 * (1 - GIoU(y_hat_s, y_hat_t))
```

**Chi tiết quan trọng khi code** — 2 số hạng dùng **2 dạng biểu diễn khác
nhau** của box, không phải cùng 1 tensor:
- `SmoothL1` trên `x["boxes"]` thô (`head_pruned.py:92`), **trước**
  `dist2bbox` — cùng đơn vị/scale giữa teacher và student vì cả hai dùng
  chung stride grid tại mỗi tầng P3/P4/P5.
- `GIoU` trên box **sau** `dist2bbox(...) * stride` (`head_pruned.py:115`)
  — GIoU cần toạ độ hình học thật, không tính được trên offset thô.

### 4.4 Trọng số biến thiên theo thời gian cho response KD (YOLOv6 self-distillation, arXiv:2209.02976)

```
alpha(E_i) = -0.99 * ( (1 - cos(pi * E_i / E_max)) / 2 ) + 1
```

Bắt đầu gần 1 (tin soft label nhiều ở đầu training), giảm dần về ~0.01 ở
cuối training (tin hard label/GT nhiều hơn). Chỉ áp dụng cho tổng
`L_cls_KD + L_reg_KD` — **không** áp cho `L_CWD` (CWD là feature-level,
độc lập với lịch epoch của response distill).

Paper gốc ghi rõ: *"No performance improvement is attained without the
weight decay strategy compared with the baseline"* — tức α cố định gần
như vô dụng, đây không phải tinh chỉnh tuỳ chọn mà là phần bắt buộc.

### 4.5 Tổng loss

```
L = L_task(box + cls, GT)                      # giữ nguyên như finetune thường
  + lambda_CWD * L_CWD                          # ~50, tune qua sweep Q3
  + alpha(epoch) * ( L_cls_KD + L_reg_KD )      # cosine-decay theo epoch
```

## 5. Tiền lệ đã tìm được cho đúng dạng pipeline này

arXiv:2509.12918 — "A Novel Compression Framework for YOLOv8: ... via
Structured Pruning and Channel-Wise Distillation". Cùng dạng 3 giai đoạn
(sparsity-aware training → structured pruning → CWD) áp dụng cho YOLOv8,
không phải YOLO26 — nên phần response distillation dựa trên DFL của paper
đó (nếu có) không dùng lại được, nhưng phần CWD feature-distillation và
hình dạng pipeline tổng thể thì dùng được. Con số benchmark thực tế:
YOLOv8m 25.85M → 6.85M tham số (giảm 73.5%), AP50 chỉ giảm 2.7 điểm nhờ
CWD. Đây là mốc kỳ vọng: nếu AP50 của mình giảm nhiều hơn đáng kể ở cùng
tỷ lệ nén, khả năng cao là lỗi implementation, không chỉ đơn giản là
"distillation chưa đủ mạnh".

## 6. Ma trận thí nghiệm (chạy theo đúng thứ tự Q1 → Q2 → Q3, rồi mới sang QAT)

Tất cả thí nghiệm phải xuất phát từ **cùng 1 pruned checkpoint** (cùng tỷ
lệ prune, cùng seed sparsity-training) — nếu không, chênh lệch mAP có thể
đến từ nhiễu của bước prune, không phải từ distillation.

### 6.1 Giai đoạn recovery-training (trước QAT) — trả lời Q1/Q2/Q3

| Run | Config | Trả lời câu hỏi |
|---|---|---|
| Study B | prune → finetune, không distill (dùng `finetune.py` hiện có) | mốc tham chiếu / bản thân cũng là 1 study đầy đủ (mục 1) |
| Exp-1 | Study A với response-only KD (4.2 + 4.3 + 4.4) | Q1 (phần response) |
| Exp-2 | Study A với feature-only CWD (4.1) | Q1 (phần feature) / Q2 |
| Exp-3 | Study A đầy đủ (công thức 4.5) | Q1 đầy đủ / Q2 (so với Exp-1, Exp-2) |
| Sweep | Exp-3 với λ_CWD và trọng số KD thay đổi (lưới nhỏ, vd 3x3) | Q3 |

Cố định seed, số epoch, augmentation giống hệt nhau ở mọi dòng trên. Log
task loss, CWD loss, cls-KD loss, reg-KD loss **riêng biệt** mỗi epoch
(không cộng gộp thành 1 số) — để phát hiện trường hợp total loss giảm
nhưng box loss chững lại/tăng, nghĩa là distillation đang lấn át và "bỏ
đói" task loss.

Sau bước này, chọn cấu hình Exp tốt nhất (thường là Exp-3 hoặc điểm tốt
nhất từ Sweep) làm đại diện cho Study A, đưa cả nó và Study B (finetune)
sang giai đoạn QAT.

### 6.2 Giai đoạn QAT + export — so sánh cuối cùng giữa Study A và Study B

| So sánh | Study A (distill → QAT) | Study B (finetune → QAT) |
|---|---|---|
| Input vào QAT | checkpoint từ Exp tốt nhất ở 6.1 | checkpoint từ `finetune.py` |
| Cấu hình QAT | **giống hệt nhau** — cùng script `qat.py`, cùng epoch, cùng calibration | |
| Metric so sánh | mAP50-95 sau QAT, mức sụt mAP so với trước QAT (đo độ "chịu đựng" quantization), tốc độ hội tụ calibration | |

Đây mới là bảng quyết định thật sự: nếu Study A thắng ở 6.1 nhưng thua ở
6.2 (distill giúp trước QAT nhưng bị quantization ăn mất lợi thế), hoặc
ngược lại (Study A không hơn nhiều ở 6.1 nhưng ổn định hơn qua QAT), thì
kết luận cuối cùng phải dựa trên 6.2, không phải 6.1.

## 7. Rủi ro cần chủ động theo dõi, không chỉ hy vọng không xảy ra

- **Adapter capacity confound**: adapter Conv1x1 dùng cho CWD có thể học
  cách "giả khớp" feature teacher mà không thực sự cải thiện representation
  của student. Cách kiểm tra: sau khi train xong, tắt distillation và đánh
  giá lại feature gốc của student (hoặc so mAP có/không distill ở cùng kiến
  trúc) — không chỉ tin vào việc CWD loss giảm.
- **Khoảng cách năng lực teacher(x/l) vs student(n-pruned) quá lớn**: theo
  literature KD, gap quá lớn đôi khi phản tác dụng (giải pháp thường dùng:
  teacher assistant / progressive distillation). Phương án dự phòng nếu
  yolo26x → yolo26n-pruned không hiệu quả: thử yolo26l (gap nhỏ hơn) làm
  teacher, hoặc distill 2 bước x→l→n.
- **Lệch shape feature sau pruning**: pruning làm số channel thay đổi
  không đều giữa các layer. Không tin tưởng việc khớp theo tên layer suông
  — phải log shape tensor thật từ 1 lần forward pass thật trên cả teacher
  và student trước khi nối adapter, để bắt các chỗ mà logic
  `current_to_prev` trong `tasks_pruned.py` có thể tạo ra shape bất ngờ.

## 8. File cần tạo

- `ultralytics/utils/distill_utils.py`
  - `register_feature_hooks(model, layer_names)` — forward hook lưu
    activation cho cả teacher và student.
  - `cwd_loss(student_feat, teacher_feat, adapter, T=4.0)` (mục 4.1)
  - `cls_kd_loss(student_scores, teacher_scores, T)` (mục 4.2)
  - `reg_kd_loss(student_boxes_raw, teacher_boxes_raw, student_boxes_decoded, teacher_boxes_decoded, lambda1, lambda2)` (mục 4.3)
  - `kd_alpha_schedule(epoch, max_epoch)` (mục 4.4)
- `ultralytics/nn/distill_adapters.py`
  - `nn.ModuleDict` chứa adapter Conv1x1+BN, mỗi cặp (P3, P4, P5) một
    adapter, shape đọc từ model teacher/student thật lúc build (không
    hardcode).
- `ultralytics/models/yolo/detect/distill_trainer.py`
  - Kế thừa `DetectionTrainer`. `get_model()` load student
    (`DetectionModelPruned`, từ checkpoint của `prune.py`) và teacher
    (frozen, `requires_grad=False` toàn bộ). Override bước tính loss để
    forward cả 2 model, tính tổng loss theo mục 4.5, chỉ optimize
    student + tham số adapter.
- `distill.py` (root, cùng style CLI với `finetune.py`/`prune.py`)
  ```
  python distill.py --teacher yolo26l.pt \
      --student-checkpoint runs/prune_test/yolo26_pruned.pt \
      --data coco128.yaml --lambda-cwd 50 --temperature 4.0 --epochs 50
  ```

## 9. Việc còn mở trước/trong khi code

- Xác nhận chính xác tensor nào là "output layer 16/19/22" — phải là
  output của module `C3k2C3kPruned`/`C3k2AttnPruned` tại chỉ số đó,
  **trước khi** bị `Concat`/`DetectPruned` tiêu thụ, không phải sau.
  Verify bằng forward pass thật + in shape, không chỉ đọc yaml.
- λ_CWD=50 và các hằng số trong lịch α là điểm khởi đầu lấy từ paper
  ngoài (kiến trúc/domain khác) — phải được validate qua sweep Q3 trên
  data thật của mình trước khi coi là giá trị cuối cùng.
- Chưa quyết định: dùng yolo26l/x pretrained nguyên bản làm teacher, hay
  fine-tune nó trên dataset đích trước? (ảnh hưởng trực tiếp đến việc
  soft label có "sát domain" hay không).

## 10. Tác động lên pipeline hiện có — checklist tích hợp bắt buộc

Đã verify bằng `git diff` trên `trainer.py`, `train.py`, `tasks.py`, `cfg/*`
(các file đã sửa sẵn cho sparsity/finetune). File distillation là file mới,
thuần cộng thêm — không sửa các file này — nên `train_sparsity.py` →
`prune.py` → `finetune.py` → `val_pruned.py` không bị ảnh hưởng nếu không
chạy `distill.py`. Nhưng 3 điểm sau **bắt buộc phải làm đúng** khi code
`distill_trainer.py`, nếu sai sẽ gây lỗi ngầm khó phát hiện (cùng dạng với
lỗi DFL đã bắt ở mục 3):

1. **Teacher không được là submodule của `self.model`.**
   `build_optimizer()` (`trainer.py:1082`) duyệt toàn bộ
   `self.model.named_modules()` và gom vào param group **không lọc theo
   `requires_grad`**. Nếu teacher bị gắn làm submodule của `self.model`
   (vd `self.model.teacher = teacher_model`), toàn bộ tham số teacher vẫn
   bị đưa vào optimizer state dù đã `requires_grad=False` — tốn bộ nhớ,
   rủi ro vô tình unfreeze sau này. Teacher phải là attribute riêng của
   trainer (`self.teacher_model`), tách biệt khỏi `self.model`. Ngược lại,
   **adapter Conv1x1 phải là submodule của `self.model`** để optimizer
   train được nó.

2. **Không được đụng vào nhánh `self.sr` đã có sẵn.**
   `trainer.py` đã rẽ nhánh dựa trên `self.sr > 0` để thay đổi hẳn
   `backward()` (cộng L1 gradient thủ công cho sparsity training) và
   `optimizer_step()` (bỏ AMP scaler, bỏ EMA update khi sparsity). Vì
   `DistillTrainer` kế thừa `DetectionTrainer` → `BaseTrainer`, distillation
   phải chạy với `sr=0` (không bật đồng thời với sparsity training), và
   loss tổng ở mục 4.5 chỉ được cộng vào `self.loss` **trước** dòng
   `backward()` có sẵn — không tự viết `backward()` riêng.

3. **Checkpoint save/load phải đặt đúng `sr` trong `train_args`.**
   `_save_checkpoint()` (`trainer.py:683`) hiện chỉ có 2 nhánh theo
   `self.sr`: `sr>0` → lưu live weights, else → lưu EMA. `load_checkpoint()`
   (`tasks.py:1614`) đọc lại đúng field đó dựa vào `sr` lưu trong
   `train_args` của checkpoint. Cần quyết định checkpoint sau distill lưu
   EMA hay live weights, và đảm bảo `train_args["sr"]` ghi đúng giá trị
   (nên là 0) — nếu sai, bước load lại checkpoint này cho QAT sau sẽ đọc
   nhầm field.

**Tổng quát hoá cho cặp scale bất kỳ (không chỉ pruned-n)**: `distill_trainer.py`
nên nhận loại model của student qua tham số (`DetectionModel` cho scale
chuẩn, `DetectionModelPruned` cho student đã pruned), không hardcode class
— để cùng 1 trainer dùng được cho mọi cặp teacher/student trong họ YOLO26.
Adapter Conv1x1+BN (đọc shape runtime, không hardcode channel) đã đủ tổng
quát cho cả 2 trường hợp, không cần sửa gì thêm.

**Cơ chế tách trainer, không đụng `finetune.py`**: `engine/model.py:783`
— `self.trainer = (trainer or self._smart_load("trainer"))(...)` — nhận
tham số `trainer=` tuỳ chọn. `finetune.py` gọi `model.train(finetune=True, ...)`
không truyền `trainer=` nên luôn dùng `DetectionTrainer` mặc định, không
liên quan gì tới `DistillTrainer`. `distill.py` sẽ gọi
`model.train(trainer=DistillTrainer, ...)` để chỉ định tường minh. Vì
`DistillTrainer` là class mới kế thừa `DetectionTrainer` (không sửa
`train.py`/`trainer.py` gốc), thêm file này không đổi hành vi của bất kỳ
script nào khác đang dùng `DetectionTrainer` mặc định.

**Không cần lo** (đã verify, tái dùng được nguyên vẹn):
- `get_model()` trong `train.py` đã có sẵn nhánh `self.finetune` load
  `DetectionModelPruned` + `maskbndict` — `DistillTrainer.get_model()` nên
  gọi `super().get_model()` để lấy student, không viết lại logic này (tránh
  phân kỳ khi `train.py` upstream thay đổi).
- Dataloader, augmentation, `DetectionValidator`, `val_pruned.py` — dùng
  lại nguyên vẹn, không cần sửa.
- Field cfg mới cho distill (`teacher`, `lambda_cwd`, `kd_temperature`,
  `kd_alpha_max_epoch`) **không cần đăng ký vào `cfg/__init__.py`** — khác
  với dự đoán ban đầu. `DistillTrainer.__init__` pop các key này khỏi
  `overrides` dict **trước khi** gọi `super().__init__()`, nên
  `get_cfg()`/`check_dict_alignment()` không bao giờ thấy các key lạ này.
  Cách này gọn hơn và giữ thay đổi hoàn toàn cách ly khỏi schema cfg toàn
  cục — đúng tinh thần "file mới, không sửa file gốc" của mục 10.

## 11. Đã code + smoke-test — kết quả thật (không phải suy đoán)

Đã triển khai đủ 4 file ở mục 8 (`distill_utils.py`, `distill_adapters.py`,
`distill_trainer.py`, `distill.py`) và chạy thật trên máy CPU-only (không
GPU) để verify cơ chế trước khi tin vào công thức.

### 11.1 Đơn vị test (unit test loss functions)

`cwd_loss`, `cls_kd_loss`, `reg_kd_loss`, `kd_alpha_schedule` chạy đúng với
tensor giả: gradient backprop qua `cwd_loss` hoạt động, lịch alpha ra đúng
số (1.0 → 0.505 → 0.01 tại epoch 0/25/50 trên `max_epoch=50`, khớp công
thức cosine ở mục 4.4).

### 11.2 2 lỗi tích hợp thật bắt được khi chạy full training loop

1. **Validation giữa lúc train dùng `trainer.ema.ema`** (bản deep-copy
   riêng), không phải `self.model` — hook forward gắn lúc `attach_teacher()`
   không theo đúng cách sang bản copy này (Python `deepcopy` không rebind
   closure của hook về dict mới) → `self._student_feats` trên bản EMA luôn
   rỗng → `KeyError`. **Fix**: `DistillMixin.loss()` tự phát hiện thiếu key
   trong `_student_feats` và fallback về `super().loss()` (task loss thuần)
   — không đụng `trainer.py`/`validator.py`.
2. **Checkpoint save lỗi pickle** vì hook ban đầu viết bằng closure nội bộ
   (`def _hook(...)` lồng trong hàm) — Python không pickle được local
   function. **Fix**: đổi `register_feature_hooks` sang dùng class
   `_FeatureHook` picklable (module-level), và thêm `DistillMixin.__deepcopy__`
   để loại bỏ teacher + hook + feats dict khỏi bản EMA/checkpoint copy (đúng
   tinh thần rủi ro 1/3 ở mục 10 — checkpoint không nên mang teacher).

Cả 2 đều là lỗi tích hợp với vòng lặp train sẵn có của Ultralytics, không
phải lỗi công thức distillation.

### 11.3 Benchmark thật: λ_CWD=50 (giá trị nguồn paper CWD gốc) THẤT BẠI

Setup: CPU, `yolo26n.pt` pretrained làm điểm xuất phát, teacher=`yolo26s.pt`,
data=coco128 (128 ảnh), 5 epoch, `imgsz=320`, `batch=8`, `lr0=1e-4`, cùng
seed cho mọi run. Đây **chỉ là smoke test toy-scale** (không phải sweep Q3
nghiêm túc — cần GPU, nhiều epoch/seed, pruned checkpoint thật).

| Run | mAP50-95 |
|---|---|
| Pretrained gốc (chưa train thêm) | 0.387 |
| Study B: finetune 5 epoch, không distill | 0.248 |
| Study A: distill, λ_CWD=50 (mặc định ban đầu ở mục 4.1) | **0.003** |
| Study A: distill, λ_CWD=2 | **0.251** |

Với λ_CWD=50, `cls_loss` tăng gần gấp đôi so với Study B (~4.2 vs ~2.3) —
đúng hiện tượng đã cảnh báo ở mục 6 ("distillation lấn át và bỏ đói task
loss"). Giảm λ_CWD xuống 2 thì `cls_loss` về mức bình thường và mAP nhỉnh
hơn Study B một chút (0.251 vs 0.248) — xác nhận thực nghiệm giả thuyết,
nhưng chênh lệch quá nhỏ ở quy mô toy này để kết luận "distill thực sự
giúp" (cần sweep Q3 thật với nhiều seed/epoch để có ý nghĩa thống kê).

**Cập nhật giá trị khởi điểm cho sweep Q3** (thay vì λ_CWD≈50 ở mục 4.1,
lấy nguyên từ paper CWD gốc — domain/kiến trúc khác YOLO26): dùng
**λ_CWD trong khoảng 1–5** làm điểm khởi đầu, không phải 50.

### 11.4 Sweep λ_CWD đầy đủ (0.5 → 10) — vẫn còn treo, đáy đường cong chưa chạm

Chạy tiếp sweep 5 giá trị λ_CWD trên cùng toy setup (CPU, 5 epoch, coco128,
cùng seed, teacher=`yolo26s.pt`):

| λ_CWD | mAP50-95 |
|---|---|
| **0.5** | **0.2513** ← duy nhất vượt Study B |
| 1.0 | 0.2264 |
| 2.0 | 0.2082 |
| 5.0 | 0.1881 |
| 10.0 | 0.0756 |
| Study B (không distill, mốc so sánh) | 0.248 |

Xu hướng **giảm đơn điệu, không có điểm uốn** — λ_CWD càng lớn càng tệ ngay
từ 0.5, không phải dạng đường cong tăng-rồi-giảm điển hình của một sweep đã
bao phủ đúng khoảng tối ưu. Điều này cho thấy **khoảng giá trị tốt thật sự
còn thấp hơn 0.5** — 0.5 mới chỉ là giá trị tốt nhất *trong các giá trị đã
thử*, chưa chắc là đáy thật của đường cong.

**Việc cần làm tiếp** (chưa làm, ghi lại để không quên): thu hẹp sweep về
λ_CWD ∈ [0, 0.5] (vd thử thêm {0.05, 0.1, 0.2, 0.5}) để tìm đáy thật, trước
khi kết luận distillation có vượt trội finetune thường hay không — kể cả ở
quy mô toy này. Đồng thời vẫn cần nhắc lại giới hạn: đây là toy-scale (CPU,
128 ảnh, 1 seed, 5 epoch) — không thay thế được sweep Q3 nghiêm túc trên
pruned checkpoint thật với GPU.

### 11.5 Test 3 seed (λ_CWD=0.5, teacher=yolo26s) — kết quả 1-seed trước đó là nhiễu

Lo ngại ở mục 11.3/11.4 (kết luận từ 1 seed không đáng tin) được xác nhận
bằng cách chạy lại Study B và Study A (λ_CWD=0.5) trên 3 seed (0, 1, 2),
cùng toy setup:

| Seed | Study B (finetune) | Study A (distill, teacher=yolo26s, λ=0.5) | Chênh (A−B) |
|---|---|---|---|
| 0 | 0.2481 | 0.2125 | −0.0356 |
| 1 | 0.2410 | 0.2462 | +0.0052 |
| 2 | 0.2107 | 0.2088 | −0.0019 |
| **Trung bình** | **0.2333** | **0.2225** | **−0.0108** |

**Kết luận đảo ngược so với mục 11.3**: qua 3 seed, distill trung bình **TỆ
HƠN** finetune (−0.0108), không phải tốt hơn. Chênh lệch dao động rất mạnh
giữa các seed (−0.036 đến +0.005), không có xu hướng nhất quán — seed=0 ở
mục 11.3 tình cờ rơi vào trường hợp thuận lợi cho distill, không đại diện
cho hành vi trung bình.

**Phát hiện thêm, nghiêm trọng hơn cả biến động giữa seed**: chạy lại đúng
seed=0 với đúng cấu hình (λ_CWD=0.5, teacher=yolo26s) ở 2 lần khác nhau cho
2 kết quả khác nhau (0.2513 ở mục 11.3 vs 0.2125 ở đây) — dù đã
`seed=0, deterministic=True`. Pipeline **không tái lập được hoàn toàn
(not fully reproducible) trên CPU**, nhiều khả năng do các phép toán đa
luồng CPU (MKL/BLAS) không đảm bảo determinism tuyệt đối trong tích luỹ số
thực dấu phẩy động. Hệ quả: nhiễu nền của benchmark này còn lớn hơn cả biến
động giữa các seed đã đo — **mọi so sánh 1 lần chạy (kể cả cùng seed) trong
toàn bộ mục 11 trước đó đều không đáng tin cậy về mặt thống kê**, chỉ có
giá trị xác nhận cơ chế chạy được (không crash), không có giá trị kết luận
"distill có giúp không".

**Kết luận thực tế cho câu hỏi Q1** (tại thời điểm này, trên setup toy này):
không có bằng chứng đáng tin cậy rằng distillation vượt trội finetune —
cần (a) cố định môi trường CPU cho reproducible thật (đặt
`torch.use_deterministic_algorithms(True)` + giới hạn số luồng BLAS) hoặc
chuyển hẳn sang GPU, và (b) test trên đúng case mục tiêu — **student đã
pruned**, không phải yolo26n nguyên vẹn như toàn bộ mục 11 đã test — trước
khi đầu tư thêm công sức dò hyperparameter trên case chưa đại diện này.

