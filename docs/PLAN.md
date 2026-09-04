# Index — Plan YOLO26 compression pipeline & triển khai edge

Pipeline nén nối tiếp: `sparsity training → prune → finetune/distill → QAT → export`.
Triển khai thực tế: cụm Edge AI (Jetson) + Cloud tập trung.

- **`PLAN_SHO.md`** — tìm cấu hình siêu tham số tối ưu bằng thuật toán
  Seahorse Optimizer (SHO), port từ luận án TS Nguyễn Ngọc Thoan, chạy
  trước sparsity training để tạo optimized baseline.
- **`PLAN_OPT_SURVEY.md`** — survey các phương pháp thay thế/bổ sung SHO
  (PSO/GA/GWO/WOA/CMA-ES/Bayesian Optimization), khuyến nghị chạy song
  song SHO (đối chiếu luận án) + Optuna (thực dụng, có pruning) + Random
  Search (đối chứng bắt buộc).
- **`PLAN_PRUNE.md`** — sparsity training (BN-gamma Network Slimming) +
  `prune.py`: bug đã sửa, tuning `sr`/optimizer, kết quả thật.
- **`PLAN_DISTILL.md`** — distillation (teacher lớn → student pruned): công
  thức, tích hợp trainer, kết quả benchmark.
- **`PLAN_QAT.md`** — QAT (TensorRT INT8, NVIDIA Jetson Orin Nano): port từ
  repo tham chiếu sang YOLO26, đã viết code, chưa test (cần GPU).
- **`PLAN_EDGE_CLOUD.md`** — kiến trúc cụm nhiều Edge AI + 1 Cloud tập
  trung: giao thức giao tiếp (MQTT/HTTPS/WebRTC theo loại dữ liệu), Edge
  Agent, thành phần cloud cần thêm, bảo mật.
- **`PLAN_MLOPS.md`** — MLOps production: tách Model MLOps (MLflow
  tracking + registry cho pipeline nén) và App MLOps (CI/CD, Docker cho
  `app/backend`/`app/frontend`), monitoring/retrain trigger nối 2 nhánh.
