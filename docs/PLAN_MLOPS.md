# Plan: MLOps chuẩn production cho pipeline nén + triển khai edge-cloud

Trạng thái: **plan, chưa code**. Mục tiêu: đóng vòng lặp
`experiment tracking → model registry → CI/CD → monitoring → retrain
trigger` bao quanh pipeline nén đã có (`PLAN_PRUNE.md`/`PLAN_DISTILL.md`/
`PLAN_QAT.md`/`PLAN_SHO.md`) và kiến trúc triển khai đã thiết kế
(`PLAN_EDGE_CLOUD.md`), không phải dựng lại từ đầu.

## 0. Điểm xuất phát thật (đã khảo sát, không giả định)

- **Pipeline nén**: `prune.py` đã verify lossless, còn 1 việc mở (tune
  lịch giảm `sr`). `distill.py`/`finetune.py` mới ở mức thiết kế, **chưa
  code**. `qat_trainer_yolo26.py` đã viết nhưng **chưa test** — máy dev
  hiện tại CPU-only (Windows), `pytorch-quantization` cần CUDA/Linux.
  → Nghĩa là: chưa có checkpoint nào chạy hết pipeline thật, MLOps ở giai
  đoạn này **không nên đặt trọng tâm vào automation cho GPU training** (vì
  chưa có GPU để chạy), mà nên đặt trọng tâm vào **tracking + reproducibility**
  để khi có GPU, mọi run đều so sánh được với nhau.
- **`app/backend`**: FastAPI + SQLAlchemy async + Alembic, model
  `Violation` duy nhất ở `app/backend/app/models/violation.py`, config có
  sẵn field `inference_engine_path` (đang trỏ `weights/yolo26_int8.engine`
  — chỗ nối tự nhiên với model registry). **Chưa có Dockerfile, chưa có
  CI, chưa có mlflow/wandb/dvc.**
- **`.github/workflows/`**: toàn bộ 10 workflow hiện tại là CI của thư
  viện `ultralytics` (test, docs, docker publish lên DockerHub…), **không
  workflow nào động tới `app/backend`/`app/frontend`**. Không được sửa các
  workflow này — chúng phục vụ mục đích khác (release thư viện). MLOps cho
  `app/` cần workflow **mới**, tách riêng.
- **Không có Docker/docker-compose nào** cho `app/` hiện tại — cloud stack
  ở `PLAN_EDGE_CLOUD.md` (Postgres, MQTT broker, object storage, backend)
  hiện chỉ tồn tại trên giấy.

Kết luận: MLOps ở đây không phải "thêm vào" một hệ thống đang chạy, mà là
**dựng nền tảng song song với lúc pipeline nén hoàn thiện** — nên thiết kế
để 2 việc không chặn nhau (training vẫn chạy CPU-only để debug logic, MLOps
scaffold chạy độc lập, ráp lại khi có GPU thật).

## 2. Phạm vi — 2 nhánh MLOps tách biệt, không dùng chung 1 hệ thống

Giống nhận xét ở `PLAN_EDGE_CLOUD.md` (mỗi loại dữ liệu một giao thức),
MLOps ở đây cũng nên tách theo 2 vòng đời khác nhau vì tần suất và rủi ro
rất khác nhau:

| Nhánh | Đối tượng | Tần suất | Rủi ro khi sai |
|---|---|---|---|
| **Model MLOps** | pipeline sparsity→prune→distill→QAT→export, checkpoint, benchmark mAP/latency | Mỗi lần đổi hyperparameter/thuật toán nén | Model tệ hơn deploy nhầm ra Jetson, không biết checkpoint nào đang chạy đâu |
| **App MLOps (DevOps)** | `app/backend`, `app/frontend`, tương lai là Edge Agent | Mỗi lần sửa code app | Backend lỗi, downtime dashboard, không phải sai model |

Không trộn 2 pipeline CI này — ví dụ không chạy benchmark mAP (tốn GPU,
chậm) trong CI của một PR chỉ sửa frontend.

## 3. Nhánh 1 — Model MLOps (experiment tracking + registry)

### 3.1 Experiment tracking: MLflow, tự host

Chọn MLflow (không phải W&B) vì: self-hosted được ngay trên hạ tầng cloud
đã định ở `PLAN_EDGE_CLOUD.md` (không phụ thuộc SaaS bên ngoài, phù hợp dữ
liệu vi phạm nhạy cảm), và có sẵn Model Registry tích hợp — không cần thêm
công cụ thứ 3 riêng cho registry.

- Log ở **mọi** bước của pipeline nén, không chỉ bước cuối:
  - `sparsity training` (BN-gamma, `train_sparsity.py`): log `sr`, sparsity
    ratio đạt được theo epoch, mAP.
  - `prune.py`: log ratio, số param trước/sau, mAP ngay sau prune (chưa
    finetune) — để tách riêng "prune làm mất bao nhiêu" khỏi "finetune bù
    lại được bao nhiêu", đúng tinh thần benchmark distill-vs-finetune đã
    đặt ra ở `PLAN_DISTILL.md`.
  - `distill.py`/`finetune.py` (khi code xong): log theo cả 2 study A/B đã
    định nghĩa trong `PLAN_DISTILL.md`, tag `study=A` / `study=B` để so
    sánh trực tiếp trên MLflow UI.
  - `qat_trainer_yolo26.py`: log mAP INT8 vs FP32 baseline, kèm artifact
    `.engine` khi có GPU chạy được.
  - `optimization/sho.py` / `optuna_search.py` (khi code xong, theo
    `PLAN_SHO.md`/`PLAN_OPT_SURVEY.md`): mỗi trial là 1 MLflow run con,
    nested dưới run cha của thuật toán search — so sánh SHO vs Optuna vs
    Random Search ngay trên UI thay vì tự ghép CSV tay.
- **Không** trạng thái nào của pipeline được coi là "xong" nếu không có
  MLflow run tương ứng — kể cả run thất bại/CPU-only debug cũng log (đánh
  tag `env=cpu_debug`) để không lẫn với kết quả GPU thật.

### 3.2 Model Registry — trả lời câu hỏi "checkpoint nào đang chạy"

`PLAN_QAT.md` mục 1.5 và `PLAN_EDGE_CLOUD.md` mục 5 (bảng `devices` có cột
`model_version`) đều để ngỏ câu hỏi này — MLflow Model Registry là chỗ trả
lời:

- Mỗi checkpoint pass qua QAT + benchmark đạt ngưỡng (mAP drop < X%, latency
  đúng target Jetson) được register vào MLflow Registry với stage
  `Staging`.
- Promote `Staging → Production` là hành động **thủ công có review** (so
  mAP/latency với bản đang chạy), không tự động — sai 1 checkpoint là sai
  trên toàn bộ site đang chạy edge thật.
- Cột `model_version` trong bảng `devices` (thiết kế ở `PLAN_EDGE_CLOUD.md`)
  lưu đúng version string từ MLflow Registry — nối trực tiếp "model nào
  được duyệt" với "model nào Jetson nào đang chạy", trả lời được câu hỏi
  audit khi có sự cố.
- Cơ chế OTA đã thiết kế (MQTT báo "có bản mới" → edge pull HTTPS) trỏ
  thẳng vào artifact `.engine` lấy từ registry, không phải file rời rạc
  trên đĩa cloud.

### 3.3 Việc cần làm (Nhánh 1)

- [ ] Dựng MLflow server (docker-compose, backend store Postgres — dùng
      chung instance Postgres đã định ở `PLAN_EDGE_CLOUD.md` mục 5, khác
      database).
- [ ] Thêm `mlflow.log_*` vào `train_sparsity.py`, `prune.py`,
      `qat_trainer_yolo26.py` — bắt đầu ngay cả khi vẫn CPU-only, để có
      thói quen/schema log đúng trước khi có GPU.
- [ ] Định nghĩa ngưỡng promote `Staging→Production` cụ thể (mAP drop tối
      đa, latency tối đa) — cần input từ yêu cầu thực tế bài toán (chưa có
      trong các PLAN hiện tại, cần hỏi/quyết định riêng).

## 4. Nhánh 2 — App MLOps (CI/CD cho `app/backend`, `app/frontend`)

### 4.1 CI — workflow mới, tách khỏi CI thư viện ultralytics hiện có

Tạo `.github/workflows/app-backend-ci.yml` và `app-frontend-ci.yml` mới
(không đụng 10 workflow hiện có), trigger theo path filter
(`app/backend/**`, `app/frontend/**`) để không chạy lãng phí khi PR chỉ
sửa thư viện.

- Backend: lint (ruff), test (pytest, cần fixture Postgres — dùng service
  container trong Actions), Alembic migration check (`alembic upgrade
  head` chạy sạch trên DB rỗng).
- Frontend: `npm run build`, typecheck, lint.

### 4.2 Docker hoá — chưa có, cần làm trước khi nói "production"

- `app/backend/Dockerfile` (multi-stage, base `python:3.11-slim`).
- `app/frontend/Dockerfile` (build tĩnh + serve qua nginx, hoặc build vào
  chung image với backend nếu muốn đơn giản giai đoạn đầu).
- `docker-compose.yml` ở root cho **dev/staging local**: backend +
  frontend + Postgres + MQTT broker (Mosquitto, bản nhẹ trước, theo đúng
  khuyến nghị "pilot vài thiết bị thì dùng Mosquitto" ở
  `PLAN_EDGE_CLOUD.md` mục 7) + MLflow — 1 lệnh `docker compose up` dựng
  toàn bộ cloud stack đã thiết kế trên giấy thành chạy được thật.

### 4.3 CD — cân nhắc quy mô trước khi làm phức tạp

Vì hệ thống hiện tại **chưa có site thật nào chạy production** (theo đúng
tinh thần "không code trong lượt này" của `PLAN_EDGE_CLOUD.md`), CD nên
dừng ở mức:

- Build + push image lên registry (GHCR) khi merge vào `main`.
- **Không** cần canary/blue-green/multi-region ngay — quá sớm so với quy
  mô hiện tại (vài Jetson pilot). Thêm khi có >1 site thật cần zero-downtime
  deploy.

## 5. Nhánh nối 2 hệ thống — Monitoring & retrain trigger

Đây là phần khiến MLOps khác với chỉ "CI/CD cho model": vòng lặp phải
đóng lại từ dữ liệu thật ngoài hiện trường quay về training.

- **Data drift**: bảng `violations` (mở rộng `device_id`/`site_id` theo
  `PLAN_EDGE_CLOUD.md`) là nguồn dữ liệu giám sát tự nhiên — theo dõi phân
  bố `violation_type`/`confidence` theo thời gian/site. Confidence trung
  bình tụt dần theo site cụ thể là tín hiệu model không còn khớp điều kiện
  camera đó (góc quay đổi, ánh sáng đổi theo mùa…).
- **Model drift / retrain trigger**: chưa cần tự động hoá retrain (rủi ro
  cao, chưa đủ dữ liệu vận hành để tin tưởng ngưỡng tự động) — giai đoạn
  đầu để **cảnh báo cho người** (dashboard đã có `AnalyticsChart`/
  `AlertsPanel` — thêm 1 chart confidence-theo-thời-gian là đủ), người
  quyết định khi nào chạy lại pipeline nén với dữ liệu mới.
- **Feedback loop dữ liệu**: ảnh/clip bằng chứng đã lưu ở object storage
  (MinIO, theo `PLAN_EDGE_CLOUD.md` mục 5) chính là nguồn data để label lại
  và làm tập finetune/distill vòng sau — cần 1 quy trình xuất
  (export ảnh theo khoảng thời gian/site → format training) nhưng **không
  cần build ngay**, chỉ cần đảm bảo lưu trữ hiện tại đã đủ để làm việc này
  sau (đã đủ, vì object storage giữ ảnh gốc).

## 6. Thứ tự triển khai đề xuất

Không làm tất cả cùng lúc — phụ thuộc trạng thái pipeline nén và có GPU
hay chưa:

1. **Ngay bây giờ (không cần GPU)**: MLflow tracking cho các bước đã chạy
   được trên CPU (sparsity training, prune) — mục 3.3. Rẻ, không chặn gì,
   tạo thói quen đúng trước khi có GPU.
2. **Song song, không phụ thuộc GPU**: Docker hoá `app/backend`/
   `app/frontend` + CI mới (mục 4.1, 4.2) — độc lập hoàn toàn với tiến độ
   pipeline nén.
3. **Khi có GPU** (theo `PLAN_QAT.md` mục 1.5): hoàn thiện log MLflow cho
   QAT, bật Model Registry, định nghĩa ngưỡng promote (mục 3.2).
4. **Khi có site pilot thật** (sau khi Edge Agent code xong theo
   `PLAN_EDGE_CLOUD.md` mục 7): bật monitoring/drift alert (mục 5) — không
   có dữ liệu thật thì bước này không kiểm chứng được, làm sớm sẽ phải làm
   lại.
