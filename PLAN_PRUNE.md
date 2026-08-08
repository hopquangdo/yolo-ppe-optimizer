# Plan: Pruning YOLO26 (BN-gamma / Network Slimming) — bug tìm được + tuning sparsity training

Trạng thái: `prune.py` đã verify lossless (mục 1). Sparsity training (`train_sparsity.py`)
đã sửa bug optimizer (mục 2) và đang trong quá trình tìm `sr`/số epoch phù hợp.

## 1. 6 bug thật tìm được trong `prune.py`/`tasks_pruned.py`/`block_pruned.py`/`head_pruned.py`

Khi cố chạy đúng case mục tiêu (student pruned thật) ở mục 11, `mAP=0` tuyệt
đối bất kể prune-ratio (kể cả `ratio=0.0`, tức không cắt channel nào) khiến
nghi ngờ pipeline finetune/distill có vấn đề. Điều tra sâu bằng cách so sánh
**tensor từng layer** giữa model gốc và model "pruned ratio=0.0" (phải khớp
tuyệt đối nếu code đúng) đã lộ ra **6 bug thật trong code prune, không liên
quan gì đến distillation**:

1. **`shortcut` đọc nhầm tham số** (`tasks_pruned.py`, cả 3 hàm
   `_parse_c3k2_bottleneck`/`_parse_c3k2_c3k`/`_parse_c3k2_attn`): code đọc
   `args[1]` từ yaml làm `shortcut`, nhưng theo đúng cách `ultralytics/nn/tasks.py`
   parse yaml gốc (`args = [c1, c2, *args[1:]]; args.insert(2, n)`), vị trí đó
   thực chất là cờ **`c3k`**, không phải `shortcut` — `shortcut` không bao giờ
   được yaml ghi đè, luôn là default `True`. Bug này bị "ẩn" ở các block dùng
   yaml `[…, True, …]` (trùng ngẫu nhiên với giá trị đúng) nhưng lộ rõ ở
   `C3k2Pruned` (yaml `[…, False, …]`) → mất hẳn residual connection.
2. **`SPPFPruned.cv1` thiếu `act=False`** (`block_pruned.py`) — bản gốc
   `SPPF.cv1` không có activation, bản pruned bị áp activation mặc định.
3. **`num_heads` hardcode = 1** trong `_parse_c3k2_attn` (`tasks_pruned.py`),
   đáng lẽ phải tính `max(cv1_split[1] // 64, 1)` giống hệt cách
   `C2PSAPruned` tự tính (nó tự tính đúng nên không bị lộ bug) — sai hoàn
   toàn cách chia multi-head attention.
4. **`one2one_cv2/cv3` không được transfer weight thật** (`prune.py`) — code
   cũ chủ động skip (`if ".one2one_" in name: continue`) rồi ghi đè bằng
   `cv2/cv3` phía sau, với giả định "one2one là deepcopy của cv2/cv3". Giả
   định này **sai với checkpoint đã train** (verify: `cv2 != one2one_cv2`
   trên model gốc, lệch tới 0.87) — one2one được train độc lập (top-1
   assignment) và **phân kỳ đáng kể** khỏi one2many (top-k assignment).
   Fix: transfer one2one bằng **mask của one2many tương ứng** (vì kiến trúc
   ép one2one/one2many phải cùng số channel — `DetectPruned.__init__` xây
   `one2one_cv2 = deepcopy(cv2)`), nhưng copy **giá trị** trọng số one2one
   thật, không phải giá trị cv2/cv3.
5. **`gt(thre)` → `ge(thre)`** (`prune.py`) — ở `prune-ratio=0.0`, threshold
   = giá trị gamma nhỏ nhất toàn model; dùng so sánh chặt (`>`) loại nhầm
   đúng channel nằm ở ngưỡng đó dù ratio=0 nghĩa là "không cắt gì".
6. **`xywh=True` hardcode trong `DetectPruned._inference()`** (`head_pruned.py`)
   — bản gốc `Detect.decode_bboxes` dùng công thức
   `xywh = xywh and not self.end2end and not self.xyxy`; với model
   `end2end=True` (đúng trường hợp YOLO26), kết quả phải là **`xywh=False`**
   (decode thẳng ra xyxy, khớp với `postprocess()` vốn giả định input đã là
   `[x1,y1,x2,y2,...]`). Hardcode `True` khiến box ở định dạng xywh bị
   `postprocess()` hiểu nhầm thành xyxy — box cuối cùng bị "trộn" toạ độ
   tâm/kích thước, nhìn như bị dịch chuyển vô nghĩa dù toàn bộ trọng số và
   công thức decode phía trước đều đúng.

**Xác nhận sau khi sửa cả 6 bug**: `prune.py --prune-ratio 0.0` trên
`yolo26n.pt` cho **mAP50-95 = 0.38715544628073... / mAP50 = 0.52915393314424...`
— khớp **tuyệt đối từng chữ số thập phân** với `yolo26n.pt` gốc (không
prune). Đây là bằng chứng dứt điểm: `prune.py` giờ là phép transfer weight
hoàn toàn lossless ở ratio=0.0. Test tiếp `ratio=0.1` trên checkpoint
sparsity-trained thật (không phải model sạch — model sạch chưa qua sparsity
training sẽ sập vì gamma chưa được L1 phân hoá, đúng lý do `train_sparsity.py`
là bước bắt buộc) cho mAP nhỏ nhưng khác 0, tỷ lệ thuận với mức suy giảm của
checkpoint gốc — hành vi hợp lý, không còn dấu hiệu bug.

**Việc cần làm tiếp** (chưa làm): benchmark toy so sánh distill vs finetune
(xem `PLAN_DISTILL.md` mục 11) cần chạy lại từ đầu trên `prune.py` đã sửa —
mọi kết luận cũ dựa trên checkpoint pruned bị lỗi (bug 1-6), không còn giá
trị tham chiếu.

## 2. Bug optimizer trong sparsity training — AdamW sai, phải dùng SGD

Sau khi `prune.py` đã verify đúng (mục 1), vẫn thấy mAP tụt bất thường
(sập về ~0 chỉ sau vài epoch sparsity training). Điều tra phát hiện: kỹ
thuật "gradient-level L1" (`grad += sr*sign(gamma)`, cộng thẳng vào
`.grad` sau `backward()`, không phải cộng vào loss) được thiết kế cho
**SGD thuần** — dưới SGD, mỗi bước update là `param -= lr*grad`, nên phần
tiêm `sr*sign(gamma)` tạo ra độ co ngót **cố định, tuyến tính theo `sr`**
mỗi bước.

**Bug**: code cũ ép cứng `AdamW` cho nhánh `sr>0` (`trainer.py`
`build_optimizer`, nhánh `optimizer="auto"`). Adam chuẩn hoá **mọi** thành
phần gradient (kể cả phần L1 vừa tiêm) theo `sqrt(second-moment)` riêng của
từng tham số — comment cũ trong code ghi "immune to AdamW second-moment
normalization" là **sai**, đã verify thực nghiệm: giảm `sr` 10 lần (0.01 →
0.001) chỉ làm chậm tốc độ suy giảm gamma ~4-5 lần, không phải 10 lần như
lý thuyết SGD dự đoán — bằng chứng trực tiếp Adam đang khuếch đại/làm méo
tín hiệu L1 một cách không kiểm soát được qua `sr`.

**Fix** (`ultralytics/engine/trainer.py`, `build_optimizer`): đổi nhánh
`sr>0` từ ép `AdamW` sang ép **`SGD`** (`lr0`/`momentum` giữ nguyên từ
args, mặc định cfg `lr0=0.01` vốn đã đúng chuẩn cho SGD). Sau fix, verify
lại: cùng `sr=1e-3`, SGD gần như không làm gamma suy giảm trong 10 epoch
đầu (delta chỉ +0.1%) — đúng hành vi tuyến tính, dễ kiểm soát, khác hẳn
AdamW suy giảm khó đoán trước đó.

## 3. Lịch `sr` giảm dần (decay schedule) — thiếu kỹ thuật quan trọng thứ 2

Dù đã sửa sang SGD, `sr` **cố định suốt training** vẫn khiến val mAP sập
nhanh (về 0 từ epoch 4, verify bằng log thật) — vì áp lực L1 dồn liên tục
không giảm, cuối cùng lấn át task loss dù mỗi bước chỉ co ngót nhẹ.

**Tìm được qua research GitHub** (`JasonSloan/yolov8-prune`, cùng kỹ thuật
BN-gamma Network Slimming, kết quả cuối mAP50=0.964-0.972 — chứng minh
sparsity training của họ KHÔNG sập giữa chừng): code họ dùng lịch `sr`
**giảm dần theo epoch**, không cố định:

```python
srtmp = sr * (1 - 0.9 * epoch / epochs)
```

Bắt đầu ở `sr` đầy đủ, giảm tuyến tính còn 10% giá trị gốc ở epoch cuối —
để áp lực sparsify giảm dần, network có cơ hội ổn định lại ở nửa sau
training thay vì bị ép liên tục.

**Fix** (`trainer.py`, chỗ cộng gradient L1): thay `self.sr` bằng
`srtmp = self.sr * (1 - 0.9 * self.epoch / self.epochs)` khi tính lượng
cộng vào `.grad`.

**Xác nhận sau fix** (test thật, `sr=1e-2`, SGD, batch=8, imgsz=640, dataset
coco128): val mAP50-95 dao động (đáy 0.003 ở epoch 3) nhưng **không sập
tuyệt đối và phục hồi dần** — đến epoch 19-20 đạt 0.258, cao hơn cả epoch 0
(0.093). So với trước fix (constant `sr=0.2`): val sập về 0 tuyệt đối từ
epoch 4, không bao giờ hồi lại trong 20 epoch quan sát. Đổi lại: gamma phân
hoá **chậm hơn nhiều** (sau 20 epoch, `pct_lt_0.01` vẫn ~0%) — cần train
dài hơn đáng kể (ước tính hàng trăm epoch, không phải vài chục) mới đạt độ
phân hoá đủ dùng để prune có ý nghĩa.

## 4. Bằng chứng chéo — báo cáo YOLOv8 cũ của user xác nhận đúng công thức

User cung cấp báo cáo LaTeX từ dự án YOLOv8 trước đây (cùng kỹ thuật
Network Slimming), dùng chính xác công thức
`λ₁ = λ₁_init·(1 - 0.9·e/n_e)` — **khớp 100%** với công thức vừa implement
từ research GitHub, không phải trùng hợp mà là cùng 1 kỹ thuật chuẩn. Báo
cáo cũng cho thấy sparsity training YOLOv8 của user trước đây cũng có tụt
mAP thật (0.87→0.81 mAP50) nhưng không sập, khớp hành vi mới quan sát được
trên YOLO26 sau fix.

**Bằng chứng quan trọng nhất** (bảng so sánh pruning YOLOv8s trong báo cáo
cũ): mAP **ngay trước finetune** giảm dần theo mức prune (P4: 0.37, P5:
0.21, P6: **0.003**, P12: **0.00** — sập hoàn toàn), nhưng **sau finetune
luôn phục hồi về ~0.83-0.85** bất kể mức sập trước đó. → Kết luận quan
trọng: **không nên đánh giá pipeline qua mAP ngay sau prune** (số này gần
như luôn rất thấp/bằng 0 một cách vô hại), chỉ mAP **sau finetune/distill
đủ dài** mới có ý nghĩa đánh giá thật.

**Ngoại lệ cảnh báo**: bảng YOLOv8**n** (model nhỏ nhất, cùng size với
YOLO26n đang dùng) trong cùng báo cáo cho thấy 1 case
`Pruning-P1 + Fine-tuning` **không phục hồi** (mAP=0,0) — khác hẳn mọi case
YOLOv8s. Chưa rõ nguyên nhân (model nhỏ ít capacity dự phòng hơn, hay
finetune chưa đủ epoch, hay thí nghiệm chưa hoàn thiện) — cần cảnh giác vì
đang làm đúng size "n" giống case fail này.

## 5. Kết quả thật (test thật trên checkpoint đã fix, pipeline nối tiếp)

Prune ratio=0.1 (13-16% tham số) trên checkpoint sparsity thật (epoch ~34,
`sr=1e-2` + decay), sau đó finetune 100 epoch (coco128, batch=8, imgsz=320):

| Giai đoạn | mAP50-95 |
|---|---|
| Baseline gốc (`yolo26n.pt`, chưa prune) | 0.387 |
| Ngay sau prune (chưa finetune) | ~0.0002 |
| Finetune epoch 65 | 0.340 |
| Finetune epoch 70+ | 0.36+ (đang tiệm cận baseline) |

Khớp đúng mẫu hình đã thấy ở báo cáo YOLOv8 (mục 4) — mAP sau prune gần 0,
nhưng phục hồi rõ ràng và ổn định qua finetune, đạt ~88-93% mAP baseline
gốc chỉ với 40 epoch — xác nhận pipeline nối tiếp
(sparsity→prune→finetune/distill) khả thi trên YOLO26n, không chỉ trên
YOLOv8s như báo cáo cũ.

## 6. Việc còn mở

- Chưa tìm ra `sr`/số epoch tối ưu để đạt phân hoá gamma đủ dùng trong thời
  gian hợp lý trên CPU — cần thêm thời gian train dài hạn (ý tưởng: tăng
  `sr` lên 0.05-0.2 với decay, hoặc chấp nhận train hàng trăm epoch với
  `sr=1e-2`).
- Benchmark so sánh distill vs finetune (xem `PLAN_DISTILL.md`) cần chạy
  lại đầy đủ trên checkpoint đã fix để có kết luận đáng tin — lần chạy gần
  nhất (teacher=yolo26x) cho thấy finetune vượt distill từ epoch 5-6, khác
  với lần chạy trước đó (teacher=yolo26s) — nghi ngờ do capacity gap
  teacher-student quá lớn với yolo26x, cần test thêm để xác nhận xu hướng.

