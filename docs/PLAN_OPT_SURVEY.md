# Survey: Các phương pháp tìm cấu hình siêu tham số tối ưu cho YOLO26

Trạng thái: **survey, chưa chốt phương án**. Bổ sung cho `PLAN_SHO.md` —
tài liệu đó đã lock vào SHO (theo luận án tham khảo); tài liệu này khảo sát
rộng hơn để có cơ sở quyết định giữ SHO, đổi phương án, hay dùng SHO kết
hợp phương án khác (ensemble/so sánh).

Tiêu chí đánh giá (áp cho đúng ràng buộc của đồ án — không đánh giá chung
chung):
- **Hiệu quả mẫu (sample efficiency)** — số lần cần train/đánh giá để hội tụ
  tới cấu hình tốt. Quan trọng nhất vì máy hiện tại **CPU-only**, mỗi lần
  đánh giá dù rút ngắn epoch vẫn tốn phút-giờ.
- **Dừng sớm được không (early stopping/pruning)** — có cắt ứng viên tệ
  giữa chừng (trước khi train hết proxy_epochs) hay bắt buộc chạy trọn mỗi
  lần thử.
- **Song song hoá** — các ứng viên trong 1 vòng có độc lập nhau không (chạy
  song song nhiều process/GPU khi có).
- **Độ phức tạp cài đặt** — tự viết từ đầu hay có thư viện production-ready.
- **Khớp tài liệu tham khảo** — có đối chiếu được với luận án gốc
  (SHO-YOLOv5) không, phục vụ mục đích học thuật của đồ án.

## 1. Bảng so sánh

| Phương pháp | Họ thuật toán | Hiệu quả mẫu | Dừng sớm | Song song | Cài đặt | Khớp luận án |
|---|---|---|---|---|---|---|
| **SHO** (Seahorse Optimizer) | Metaheuristic quần thể | Thấp-trung bình — khám phá phần lớn ngẫu nhiên (Lévy flight, Brownian) | Không có cơ chế sẵn | Có (1 thế hệ độc lập) | Tự viết ~150 dòng (đã có mã giả) | **100%** — đúng thuật toán luận án dùng |
| **PSO** (Particle Swarm) | Metaheuristic quần thể | Thấp-trung bình, tương tự SHO | Không | Có | Đơn giản hơn SHO (không có pha sinh sản) | Không — luận án không dùng, nhưng cùng họ, dễ so sánh chéo |
| **GA** (Genetic Algorithm) | Metaheuristic quần thể | Thấp-trung bình | Không (trừ khi tự thêm) | Có | Trung bình (cần thiết kế crossover/mutation) | Không, nhưng README đã liệt kê GA như 1 lựa chọn |
| **GWO** (Grey Wolf), **WOA** (Whale) | Metaheuristic quần thể | Tương tự SHO/PSO | Không | Có | Tương tự SHO | Không, cùng họ "nature-inspired 2010s" như SHO |
| **CMA-ES** | Tiến hoá thích nghi hiệp phương sai | Cao hơn hẳn nhóm trên với bài toán liên tục ít chiều (~18 chiều ở đây là vừa tầm) | Không có sẵn | Có | Có thư viện (`cma` PyPI), không cần tự viết | Không |
| **Random Search** | Baseline | Thấp, nhưng là baseline bắt buộc phải có để chứng minh SHO thực sự tốt hơn ngẫu nhiên | Không | Có, dễ nhất | Rất đơn giản (~10 dòng) | Không, nhưng **nên có** làm đối chứng cho mọi phương án khác |
| **Bayesian Opt (TPE, Optuna)** | Model-based (surrogate) | **Cao** — học phân phối tốt từ lịch sử, ít lần thử hơn hẳn nhóm quần thể khi budget nhỏ | **Có sẵn** (Hyperband/ASHA pruner tích hợp) | Có (Optuna hỗ trợ distributed) | **Thấp** — `pip install optuna`, vài chục dòng wrapper | Không |
| **`model.tune()` của ultralytics** | Wrapper quanh GA nội bộ (mutation-based) | Trung bình | Không | Giới hạn (chạy tuần tự theo mặc định) | **Thấp nhất** — có sẵn, gọi 1 dòng | Không |

## 2. Phân tích theo đúng ràng buộc của đồ án

**Ràng buộc số 1 hiện tại: CPU-only, mỗi lần đánh giá đắt.** Đây là tiêu
chí quyết định nhất, không phải "phương pháp nào mạnh nhất trên benchmark
tổng quát":

- Nhóm metaheuristic quần thể (SHO/PSO/GA/GWO/WOA) đều **tốn mẫu ngang
  nhau về bản chất** — chúng cùng thuộc lớp "khám phá không gian bằng quần
  thể + heuristic cập nhật vị trí", khác nhau chủ yếu ở công thức cập nhật,
  không khác nhau nhiều về số lần đánh giá cần thiết để đạt cùng chất
  lượng nghiệm trên bài toán ít chiều (~18D) như thế này. Nói cách khác:
  **đổi SHO sang PSO/GA/GWO/WOA sẽ không giải quyết được nút thắt compute**
  — nút thắt nằm ở việc thiếu cơ chế dừng sớm, không phải ở công thức di
  chuyển quần thể.
- **CMA-ES** vượt trội hơn nhóm trên về lý thuyết cho bài toán liên tục
  ít chiều, nhưng vẫn **không có pruning**, cùng nhược điểm gốc.
- **Bayesian Optimization (Optuna/TPE + Hyperband)** là lựa chọn duy nhất
  trong bảng giải quyết trực tiếp cả 2 vấn đề: (a) mẫu hiệu quả hơn nhờ mô
  hình hoá surrogate thay vì dò ngẫu nhiên có định hướng yếu, (b) **cắt sớm
  ứng viên tệ** — với proxy_epochs=3-5 như plan ở `PLAN_SHO.md` mục 4,
  pruner có thể dừng 1 ứng viên ngay sau epoch 1 nếu loss đã rõ ràng tệ hơn
  median, tiết kiệm phần lớn thời gian mà nhóm quần thể không làm được.

## 3. Khuyến nghị — chạy 2 phương án song song, không chọn 1

Vì đồ án cần cả **giá trị học thuật** (đối chiếu luận án) lẫn **tính khả
thi** (compute CPU-only), đề xuất không chọn 1 mà chạy **2 track độc lập**,
dùng chung `objective.py` đã plan ở `PLAN_SHO.md` mục 5 (cùng 1 hàm mục
tiêu, cùng search space, chỉ khác thuật toán tìm kiếm — đảm bảo so sánh
công bằng):

1. **Track chính (báo cáo/so sánh với luận án)**: SHO — giữ nguyên plan đã
   có ở `PLAN_SHO.md`.
2. **Track đối chứng (thực dụng, chạy trước vì rẻ hơn)**: Optuna
   (TPE sampler + `HyperbandPruner`) — chạy được ngay trên CPU với budget
   nhỏ hơn nhiều mà vẫn ra cấu hình dùng được, dùng làm **optimized
   baseline thật** cho các bước sau (`sparsity training`/`prune`) trong
   lúc chờ đủ tài nguyên GPU để chạy SHO với Pop/Gmax đúng như luận án.
3. **Random Search** làm đối chứng bắt buộc cho cả 2 (baseline tối thiểu
   — nếu SHO/Optuna không thắng rõ Random Search ở cùng budget, kết luận
   "SHO hiệu quả" sẽ không có cơ sở).

Không khuyến nghị PSO/GA/GWO/WOA/CMA-ES riêng — không giải quyết ràng buộc
compute tốt hơn SHO đáng kể (cùng họ), thêm phương án chỉ tốn thời gian
implement mà không đổi kết luận.

## 4. Việc cần làm nếu triển khai theo mục 3

- `objective.py` (đã plan) cần tách phần "giải mã vector tham số ↔ dict
  train kwargs" ra khỏi phần thuật toán tìm kiếm, để dùng chung được cho
  cả `sho.py` (nhận `x: np.ndarray` trong `[LB, UB]`) và Optuna (nhận
  `trial: optuna.Trial`, gọi `trial.suggest_float(...)` theo cùng bảng
  giới hạn ở `PLAN_SHO.md` mục 2).
- Thêm `optimization/optuna_search.py` + `scripts/optuna_search.py` (cùng
  style CLI với `scripts/sho_search.py`).
- Bảng kết quả cuối (khi có số liệu thật) nên báo cáo: mAP50-95 tốt nhất
  tìm được, số lần đánh giá thực tế cần dùng, tổng thời gian — theo đúng 3
  track (SHO/Optuna/Random) để có cơ sở kết luận phương pháp nào đáng dùng
  cho các lần chạy sau.
