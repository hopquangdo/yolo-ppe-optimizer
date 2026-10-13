# Distillation bản B cho YOLO26 (`kd_method="ckd_b"`)

Phương pháp chưng cất được chốt cho đồ án: **CrossKD + PKD + trọng số đồng thuận hai nhánh (B2)**.
Teacher YOLO26s (đóng băng) → student YOLO26n. Phần code nằm ở `ultralytics/utils/distill_methods.py`, được gọi từ
`DistillTrainer` qua than số `kd_method`. Cơ chế cũ (`cwd_kd`: CWD + response KD) vẫn là mặc định, không đổi.

## Công thức

Chưng cất tại ba đầu ra neck `model.16 / 19 / 22` (P3 / P4 / P5):

```
F′s = Proj(Fs)                                   Proj = Conv1×1 → ReLU → Conv1×1 (c_s → c_t → c_t), chỉ khi train
L_PKD = ½ · MSE(norm(F′s), norm(Ft))             norm: trừ mean, chia std theo từng ảnh, từng kênh trên H×W
(ŝ, b̂) = head_teacher_br(F′s),  br ∈ {o2m, o2o}  cross-head: đặc trưng student đi qua head teacher (đóng băng)
w = (p_m + p_o)/2 · (1 − |p_m − p_o|)            B2, p = max_c σ(score teacher) tại mỗi anchor, detach
L_CH = Σ_br [ Σ w·QFL(ŝ, σ(s_t)) / (Σw·C) + 4 · Σ w·|b̂ − b_t| / (Σw·4) ]
L = L_task + dis · B · Σ_P (L_PKD + L_CH)        B = batch size, dis = 0.14
```

Head student chỉ học từ nhãn thật (L_task); gradient của L_KD đi ngược qua head teacher (trọng số không đổi)
→ projector → neck/backbone student. Khi export, `strip_distill()` bỏ projector nên mô hình triển khai là YOLO26n gốc.

## Cách dùng

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect.distill_trainer import DistillTrainer
from ultralytics.utils.distill_methods import strip_distill

YOLO("yolo26n.pt").train(
    trainer=DistillTrainer,
    teacher="runs/detect/T_s/weights/best.pt",
    kd_method="ckd_b",  # dis mặc định 0.14
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)
strip_distill("runs/detect/train/weights/best.pt", "best_clean.pt")  # trước QAT / export
# student đã pruned: strip_distill(src, dst, pruned_ckpt="yolo26_pruned.pt") để gắn lại maskbndict
```

CLI: `python scripts/distill.py --kd-method ckd_b --teacher T_s.pt --student-checkpoint yolo26n.pt --no-finetune --optimizer auto --data data.yaml --epochs 100`

| Than số     | Mặc định                     | Ý nghĩa                                                                                              |
| ----------- | ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| `kd_method` | `cwd_kd`                     | `cwd_kd` · `builtin` (KD tích hợp Ultralytics 8.4.152, port 1-1) · `ckd_b` (bản B)                   |
| `dis`       | 6.0 (builtin) / 0.14 (ckd_b) | hệ số L_KD; 0.14 hiệu chỉnh để L_KD ban đầu cùng thang với builtin dis = 6                           |
| `kd_params` | `{}`                         | chỉ cho ckd_b: `w_feat`=1, `w_reg`=4, `w_o2o`=1, `beta`=1, `agree`=true (tắt B2: `{"agree": false}`) |

## Kết quả (mAP50-95 trên test, teacher YOLO26s)

| Cấu hình               | ATLD, 45 ep, 5 seed | Data final, 45 ep, 5 seed | Data final, 100 ep, seed 0 |
| ---------------------- | ------------------- | ------------------------- | -------------------------- |
| Không KD               | 0.3451              | 0.2813                    | 0.2877                     |
| builtin (dis = 6)      | 0.3569              | 0.2784                    | 0.2945                     |
| **ckd_b (dis = 0.14)** | **0.3582**          | **0.2852**                | **0.2968**                 |

Data final: ckd_b hơn builtin ở 5/5 seed (sign test p = 0.031). Chi phí train ≈ 101 phút / 45 epoch trên T4
(không KD ≈ 77, builtin ≈ 90); chi phí suy luận không đổi.

## Kiểm thử

`pytest tests/test_distill_ban_b.py` (CPU, không cần data): số than số projector (517 888), gradient KD không chạm
teacher và head student nhưng tới backbone, builtin khớp công thức Ultralytics 8.4.152, B2 bật/tắt, `strip_distill`.

## Lưu ý về `cwd_kd` (chưa sửa trong PR này)

Trong nhánh `cwd_kd`, `task_loss` là vector 3 phần tử (box, cls, l1 — đã nhân batch size), nên
`task_loss + λ·cwd + α·(cls_kd + reg_kd)` cộng phần KD vào **cả 3 phần tử** → KD bị tính 3 lần khi trainer `sum()`,
và không nhân batch size như loss task. Sửa sẽ làm đổi thang của `lambda_cwd` đã chọn, nên để nhóm quyết định.
