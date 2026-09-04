# Plan: Thuật toán tìm cấu hình siêu tham số tối ưu (SHO) cho YOLO26

Trạng thái: **đã triển khai bản đầu**. Đây là giai đoạn **đầu tiên** của pipeline
(`optimization/sho.py` đã nhắc tới trong `README.md` nhưng chưa tồn tại):

```
Optimization search (SHO) → Optimized baseline → sparsity training → prune
→ finetune/distill → QAT → export
```

Nguồn tham khảo: Luận án TS Nguyễn Ngọc Thoan (2026, ĐH Xây dựng Hà Nội)
*"Quản lý hành vi không an toàn trên công trường xây dựng sử dụng mô hình
trí tuệ nhân tạo kết hợp tối ưu hoá"* — mục 2.3.2 (thuật toán SHO), mục 4.1
(áp dụng SHO-YOLOv5 cho bài toán PPE), Phụ lục 4 (mã giả Python đầy đủ).
**Đúng cùng bài toán** (PPE detection công trường xây dựng) và **đúng vị
trí trong pipeline** (tối ưu siêu tham số trước khi huấn luyện baseline) mà
đồ án `PPE-YOLO26-Edge` đang cần — khác biệt duy nhất: luận án dùng YOLOv5,
đồ án này dùng YOLO26.

## 1. Thuật toán SHO — tóm tắt từ luận án (mục 2.3.2)

Siêu thuật toán tối ưu dựa trên quần thể (population-based metaheuristic),
lấy cảm hứng hành vi cá ngựa, gồm 3 giai đoạn mỗi vòng lặp:

1. **Di chuyển (movement)** — cá ngựa hoặc bay Lévy quanh nghiệm tốt nhất
   hiện tại (elite) theo chuyển động xoắn ốc (nếu `r1 > 0`), hoặc trôi dạt
   ngẫu nhiên kiểu Brown quanh vị trí hiện tại (nếu `r1 ≤ 0`) — cân bằng
   giữa khai thác (exploitation quanh elite) và khám phá (exploration ngẫu
   nhiên).
2. **Săn mồi (predation)** — kéo nghiệm về gần elite với hệ số suy giảm
   `alpha = (1 - t/T)^(2t/T)` giảm dần theo số vòng lặp `t`/`T` → giai đoạn
   đầu khám phá rộng, giai đoạn cuối hội tụ quanh elite (thành công 90% mô
   phỏng theo mô tả sinh học trong luận án).
3. **Sinh sản (reproduction)** — sắp xếp quần thể theo độ tốt, chia đôi
   thành "cha"/"mẹ", lai ghép tuyến tính `child = r3*father + (1-r3)*mother`
   để sinh nghiệm mới, rồi chọn lọc `pop` nghiệm tốt nhất từ
   `(nghiệm sau săn mồi) ∪ (nghiệm con)` cho vòng lặp kế tiếp.

Mã giả Python đầy đủ (hàm `sho(pop, Max_iter, LB, UB, Dim, fobj)`) đã có sẵn
ở Phụ lục 4 luận án — **port gần như nguyên vẹn** vào `optimization/sho.py`,
vì đây là thuật toán tối ưu tổng quát, không phụ thuộc YOLOv5 hay YOLO26 (chỉ
cần đổi `fobj`, xem mục 3).

## 2. Không gian tìm kiếm (search space)

Luận án tối ưu 18 siêu tham số YOLOv5 (Bảng 4.1) — **toàn bộ 18 tham số này
đã tồn tại sẵn dưới dạng `train()` kwargs trong `ultralytics`** (không đổi
tên qua các version, dùng chung cho YOLO26), nên map trực tiếp không cần
viết lại:

| Nhóm | Tham số (tên trong `ultralytics`) | Giới hạn (theo luận án) |
|---|---|---|
| Optimizer | `lr0` | [1e-5, 1e-1] |
| | `lrf` | [0.01, 1] |
| | `momentum` | [0.6, 0.98] |
| | `weight_decay` | [0.0, 0.001] |
| | `warmup_epochs` | [0.0, 5.0]* |
| | `warmup_momentum` | [0.0, 0.95] |
| Augmentation màu | `hsv_h`, `hsv_s`, `hsv_v` | [0.0, 0.9] (riêng hsv_h theo mặc định ultralytics [0,0.1]) |
| Augmentation hình học | `degrees` | [0.0, 45.0] |
| | `translate` | [0.0, 0.9] |
| | `scale` | [0.0, 0.9] |
| | `shear` | [0.0, 10.0] |
| | `perspective` | [0.0, 0.001] |
| Augmentation tổ hợp | `flipud`, `mosaic`, `mixup`, `copy_paste` | [0.0, 1.0] (xác suất) |

*luận án ghi khoảng "0/0.9" cho warmup_epochs — số liệu bảng bị lệch định
dạng khi trích xuất PDF, cần đối chiếu lại bản gốc trước khi chốt; tạm dùng
khoảng mặc định hợp lý [0, 5] theo `ultralytics/cfg/default.yaml`.

**Khác biệt cần lưu ý khi áp dụng cho YOLO26** (đã rút ra từ toàn bộ quá
trình debug prune/distill/QAT trước đó trong dự án — xem `PLAN_PRUNE.md`,
`PLAN_DISTILL.md`):
- YOLO26 **không có DFL** (`reg_max=1`) — không ảnh hưởng tới search space
  này vì 18 tham số trên đều là optimizer/augmentation, không đụng tới đầu
  DFL.
- YOLO26 dùng **end2end dual-head** (`one2one`/`one2many`) — cũng không
  ảnh hưởng search space, chỉ ảnh hưởng lúc infer/export (đã xử lý ở
  `PLAN_QAT.md`).
- Có thể cân nhắc thêm `box`/`cls`/`dfl` loss-gain vào search space (luận
  án không tối ưu các hệ số loss này) — để mở, không bắt buộc theo đúng
  luận án trước, thêm sau nếu cần.

## 3. Hàm mục tiêu (objective function `fobj`)

Luận án dùng (mục 2.3.7, Hình 4.1 bước 5):

```
f = Σ_i (PPV_i + NPV_i) / (2K)
```

với `PPV` (positive predictive value = precision) và `NPV` (negative
predictive value) tính từ confusion matrix đa lớp, `K` là số lớp — một biến
thể "balanced accuracy" trung bình trên các lớp, cần tự dựng confusion
matrix đa lớp từ kết quả predict (không có sẵn trực tiếp trong
`ultralytics`).

**Đề xuất cho đồ án này**: dùng **`mAP50-95` trên tập validation** (lấy
trực tiếp từ `metrics.box.map` sau `model.val()`) làm `fobj` thay vì tự cài
PPV/NPV — lý do:
- `mAP50-95` là chỉ số chuẩn, đã dùng xuyên suốt toàn bộ các plan khác
  (`PLAN_PRUNE.md`, `PLAN_DISTILL.md`) trong đồ án này, giữ nhất quán cách
  đánh giá giữa các giai đoạn.
- Tránh phải tự viết + kiểm chứng logic confusion-matrix đa lớp (rủi ro bug
  âm thầm, đúng bài học đã rút ra từ 6 bug từng gặp ở `prune.py` khi tự viết
  logic đánh giá không dùng lại API có sẵn).
- SHO tối thiểu hoá `fobj` (thấy trong mã giả: `if SortfitbestN[0] <
  TargetFitness`) — nên dùng `fobj = 1 - mAP50-95` (chuyển bài toán tối đa
  mAP thành tối thiểu hoá, khớp đúng chiều thuật toán gốc).

Giữ nguyên PPV/NPV làm phương án đối chiếu (option `--metric ppv_npv`) nếu
sau này cần so sánh trực tiếp với số liệu luận án — không bắt buộc ở bản
đầu.

## 4. Chi phí tính toán — vấn đề lớn nhất cần giải quyết trước khi code

Luận án chạy `Pop=100`, `Gmax=30` → tối đa 3000 lần huấn luyện YOLOv5 đầy đủ
(có GPU, dataset PPE riêng, không nói rõ epoch/lần). Máy dev hiện tại
**CPU-only** (đã xác nhận nhiều lần ở các plan khác) — 3000 lần train
YOLO26 dù chỉ vài epoch mỗi lần cũng không khả thi trên CPU.

Hướng xử lý (bắt buộc phải quyết trước khi viết `sho.py`, không thể vá sau):

1. **Proxy training rẻ cho mỗi ứng viên** — mỗi cá ngựa chỉ train
   **vài epoch ngắn** (ví dụ 3-5 epoch, ảnh nhỏ `imgsz=320`, dataset con
   `coco128` hoặc subset PPE) để ước lượng tương đối `fobj`, **không** train
   đầy đủ cho từng ứng viên — đúng thực hành chuẩn trong hyperparameter
   search/NAS (một phần vì thời gian, một phần vì mục tiêu chỉ là xếp hạng
   tương đối các bộ tham số, không cần độ chính xác tuyệt đối).
2. **Chỉ train đầy đủ ứng viên tốt nhất cuối cùng** (`λ*`) — khớp bước 4
   trong mã giả luận án ("Chọn λ* tốt nhất, huấn luyện lại mô hình nếu
   cần").
3. **Giảm mạnh `Pop`/`Gmax` so với luận án** cho lần chạy đầu (ví dụ
   `Pop=10-15`, `Gmax=5-8`) — vẫn đủ minh hoạ thuật toán hoạt động đúng,
   tăng dần khi có GPU thật (đúng tinh thần "máy hiện tại chỉ review logic,
   test thật cần môi trường GPU" đã áp dụng nhất quán cho `PLAN_QAT.md`).
4. **Chạy song song nếu có nhiều core/GPU** — mỗi cá ngựa trong 1 thế hệ độc
   lập nhau (không phụ thuộc tuần tự), có thể `multiprocessing`/nhiều
   process train song song — để ngỏ, không bắt buộc bản đầu.

## 5. File cần tạo

- **`optimization/sho.py`** — port hàm `sho()` + `initialization()` +
  `levy()` từ Phụ lục 4 luận án (thuật toán thuần, không phụ thuộc
  ultralytics — nhận `fobj` như một tham số, tách biệt hoàn toàn khỏi phần
  train YOLO, dễ test độc lập bằng hàm chuẩn (Sphere, Rastrigin...) trước
  khi cắm vào YOLO thật).
- **`optimization/objective.py`** — `build_yolo_objective(model_path, data,
  search_space, proxy_epochs, imgsz)` trả về hàm `fobj(x: np.ndarray) ->
  float`: giải mã vector `x` (trong `[LB, UB]`) thành dict hyperparameter
  theo bảng mục 2, gọi `YOLO(model_path).train(**overrides, epochs=
  proxy_epochs)`, lấy `1 - mAP50-95` từ kết quả validate cuối cùng.
- **`scripts/sho_search.py`** — CLI entry point (cùng style
  `scripts/prune.py`/`distill.py`): parse `--pop`, `--gmax`, `--proxy-epochs`,
  chạy `sho()`, in bảng hội tụ (`Convergence_curve`), lưu `λ*` ra YAML
  (`optimized_baseline.yaml` hoặc tương tự) + train lại đầy đủ với `λ*` để
  ra `optimized_baseline.pt` — checkpoint này chính là đầu vào của
  `scripts/train_sparsity.py` (bước kế tiếp trong pipeline).

## 6. Việc cần xác nhận/mở

- Số liệu `warmup_epochs` bảng 4.1 bị lệch khi trích PDF (mục 2) — cần đối
  chiếu bản gốc luận án (file `Luận án TS_Nguyễn Ngọc Thoan_6.3.2026_v1.pdf`
  ở repo root) trước khi chốt giới hạn cuối.
- Chưa quyết `Pop`/`Gmax`/`proxy_epochs` cụ thể cho lần chạy đầu trên máy
  CPU hiện tại — phụ thuộc thời gian chấp nhận được cho 1 lần chạy full
  pipeline thử nghiệm (đề xuất bắt đầu rất nhỏ: `Pop=6`, `Gmax=3`,
  `proxy_epochs=3` để xác nhận code chạy đúng trước, tăng dần sau).
- Chưa quyết dataset cho giai đoạn optimization search — dùng `coco128`
  (đã có sẵn, dùng xuyên suốt các plan khác) hay cần dataset PPE thật (nếu
  đồ án đã có, xem `README.md` mục Dataset hiện vẫn để `TODO`).
- Đối chiếu PPV/NPV vs mAP50-95 làm `fobj` (mục 3) — quyết định dùng
  mAP50-95 là đề xuất, cần xác nhận với người dùng nếu muốn bám sát 100%
  công thức luận án gốc.
