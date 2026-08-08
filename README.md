<div align="center">

# PPE-YOLO26-Edge

**Optimization & Compression Pipeline for Real-Time PPE Detection on Edge Devices**

Tối ưu hoá và nén mô hình YOLO26 cho nhận diện thiết bị bảo hộ lao động (PPE), triển khai realtime trên Jetson Orin NX kèm dashboard giám sát và chatbot hỏi đáp.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Model](https://img.shields.io/badge/model-YOLO26-orange)
![Device](https://img.shields.io/badge/edge-Jetson%20Orin%20NX-76B900)
![Status](https://img.shields.io/badge/status-in%20development-yellow)

[Roadmap](#-roadmap) •
[Tổng quan](#-tổng-quan) •
[Pipeline](#-pipeline) •
[Cài đặt](#-cài-đặt) •
[Sử dụng](#-sử-dụng) •
[Kết quả](#-kết-quả) •
[Nhóm thực hiện](#-nhóm-thực-hiện)

</div>

---

## Roadmap

- [x] Chốt pipeline: optimization search → pruning chung → fine-tune / distillation → QAT
- [ ] Hoàn tất optimized baseline (SHO / PSO / GA)
- [ ] Pruned baseline dùng chung (20% / 40% / 60%)
- [ ] Fine-tune + QAT hoàn tất cho cả 3 tỷ lệ prune
- [ ] Distillation + QAT hoàn tất cho cả 3 tỷ lệ prune
- [ ] Benchmark đầy đủ trên Jetson Orin NX
- [ ] Chọn model production
- [ ] Edge inference service
- [ ] Backend + database vi phạm
- [ ] Chatbot (LLM tool-use trên dữ liệu vi phạm)
- [ ] Frontend dashboard realtime
- [ ] Test end-to-end toàn hệ thống
- [ ] Báo cáo & slide bảo vệ đồ án

---

## Tổng quan

**PPE-YOLO26-Edge** là đồ án tốt nghiệp xây dựng pipeline hoàn chỉnh gồm 3 giai đoạn:

1. **Tối ưu hoá kiến trúc** — dùng các thuật toán metaheuristic (SHO, PSO, GA) để tìm cấu hình YOLO26 tốt nhất cho bài toán PPE.
2. **Nén mô hình** — pruning kết hợp 2 phương pháp phục hồi độ chính xác (fine-tune vs. knowledge distillation), sau đó lượng tử hoá (QAT) để chạy realtime trên phần cứng biên.
3. **Triển khai thực tế** — inference service trên Jetson Orin NX, dashboard giám sát trực quan và chatbot trả lời câu hỏi bằng ngôn ngữ tự nhiên dựa trên dữ liệu vi phạm ghi nhận được.

### Tính năng chính

- Nhận diện realtime các lớp PPE: mũ bảo hộ, áo phản quang, khẩu trang, găng tay, giày bảo hộ (và trạng thái vi phạm tương ứng)
- So sánh có kiểm soát (controlled comparison) giữa **fine-tune** và **distillation** như 2 chiến lược phục hồi độ chính xác sau pruning, trên cùng một pruned baseline
- Xuất mô hình INT8 tối ưu cho TensorRT, benchmark trực tiếp trên Jetson Orin NX
- Dashboard realtime hiển thị vi phạm theo thời gian/khu vực, kèm chatbot hỏi đáp bằng ngôn ngữ tự nhiên (LLM tool-use trên dữ liệu có cấu trúc)

---

## Pipeline

```
                    ┌──────────────────────┐
                    │      PPE dataset      │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Optimization search   │   SHO · PSO · GA
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │   Optimized baseline   │   YOLO26 (tuned)
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Structured pruning    │   pruned baseline (20/40/60%)
                    └──────┬────────┬───────┘
                           │        │
                 ┌─────────▼─┐   ┌──▼──────────────┐
                 │ Fine-tune │   │   Distillation   │
                 └─────┬─────┘   └────────┬────────┘
                       │                  │
                 ┌─────▼─────┐      ┌─────▼─────┐
                 │    QAT    │      │    QAT    │
                 └─────┬─────┘      └─────┬─────┘
                       └─────────┬────────┘
                                 │
                      ┌──────────▼──────────┐
                      │  Evaluate & compare   │   mAP · size · FPS
                      └──────────┬──────────┘
                                 │
                      ┌──────────▼──────────┐
                      │  TensorRT INT8 export │
                      └──────────┬──────────┘
                                 │
                      ┌──────────▼──────────┐
                      │  Dashboard + Chatbot  │   Jetson Orin NX
                      └───────────────────────┘
```

Nhánh **Fine-tune** và nhánh **Distillation** cùng xuất phát từ một pruned baseline duy nhất ở mỗi tỷ lệ prune, đảm bảo so sánh công bằng giữa hai phương pháp phục hồi.


---

## Cài đặt

### Yêu cầu

| Thành phần | Phiên bản |
|---|---|
| Python | 3.10+ |
| PyTorch | TODO |
| Ultralytics (YOLO26) | TODO |
| JetPack SDK (Jetson Orin NX) | TODO |
| CUDA / TensorRT | TODO |

### Clone & cài đặt

```bash
git clone https://github.com/<org>/ppe-yolo26-edge.git
cd ppe-yolo26-edge
pip install -r requirements.txt
```

### Môi trường Jetson 
```bash
# Container cho training / QAT
docker pull nvcr.io/nvidia/l4t-pytorch:<tag-khớp-jetpack>

# Container cho export / inference TensorRT
docker pull nvcr.io/nvidia/l4t-tensorrt:<tag-khớp-jetpack>
```

---

## Sử dụng

```bash
# 1. Optimization search — tìm cấu hình YOLO26 tối ưu
python optimization/sho.py --data data/ppe.yaml --generations <N> --population <N>

# 2. Structured pruning trên baseline tối ưu
python compression/pruning/prune.py --weights optimized_baseline.pt --ratio 0.4

# 3a. Fine-tune nhánh phục hồi trực tiếp
python compression/finetune_qat/finetune.py --weights pruned_baseline_0.4.pt

# 3b. Distillation nhánh phục hồi qua teacher
python compression/distill_qat/distill.py --teacher optimized_baseline.pt --student pruned_baseline_0.4.pt

# 4. Quantization-aware training
python compression/finetune_qat/qat.py --weights finetuned_0.4.pt
python compression/distill_qat/qat.py --weights distilled_0.4.pt

# 5. Export TensorRT INT8
python export/export_trt.py --weights <model>.pt --int8

# 6. Benchmark trên Jetson Orin NX
python benchmark/run_benchmark.py --engine <model>.engine --device orin-nx
```

> Các lệnh trên là khung tham chiếu — cập nhật đúng tên tham số khi hoàn thiện code từng module.

---

## Kết quả

### 1. Optimization search

| Thuật toán | mAP50 | mAP50-95 | Thời gian search |
|---|---|---|---|
| Baseline (default) | – | – | – |
| SHO | – | – | – |
| PSO | – | – | – |
| GA | – | – | – |

### 2. Structured pruning (trước phục hồi)

Kết quả pruning thuần (BN-gamma sparsity training + structured channel pruning), trước khi áp dụng bất kỳ phương pháp phục hồi nào — xem chi tiết tại `PLAN_PRUNE.md`.

| Tỷ lệ prune | mAP50 | mAP50-95 | Size (MB) | Δ tham số |
|---|---|---|---|---|
| 20% | – | – | – | – |
| 40% | – | – | – | – |
| 60% | – | – | – | – |

### 3. Fine-tune sau pruning

| Tỷ lệ prune | mAP50 | mAP50-95 | Size (MB) | Δ so với baseline |
|---|---|---|---|---|
| 20% | – | – | – | – |
| 40% | – | – | – | – |
| 60% | – | – | – | – |

### 4. Distillation sau pruning

Teacher = optimized baseline (YOLO26 chưa prune); student = pruned baseline ở mỗi tỷ lệ. Xem công thức loss và thiết kế tại `PLAN_DISTILL.md`.

| Tỷ lệ prune | Teacher | mAP50 | mAP50-95 | Size (MB) | Δ so với baseline |
|---|---|---|---|---|---|
| 20% | – | – | – | – | – |
| 40% | – | – | – | – | – |
| 60% | – | – | – | – | – |

Fine-tune và Distillation cùng xuất phát từ pruned baseline tương ứng ở mục 2, cho phép so sánh trực tiếp hai chiến lược phục hồi ở bảng 3 và 4.

### 5. Benchmark trên Jetson Orin NX

| Model | Format | Size (MB) | FPS | Latency (ms) | mAP50 |
|---|---|---|---|---|---|
| Optimized baseline (FP32) | PyTorch | – | – | – | – |
| Optimized baseline (INT8) | TensorRT | – | – | – | – |
| Fine-tune + QAT (best) | TensorRT | – | – | – | – |
| Distillation + QAT (best) | TensorRT | – | – | – | – |

**Model production:** _TODO — điền model được chọn triển khai + lý do (trade-off tốt nhất)._

> Pipeline nén mô hình (pruning → phục hồi → QAT → TensorRT INT8) tham khảo thiết kế
> và kết quả benchmark từ nghiên cứu waste-detection trên Jetson (YOLOv8s, pruning +
> QAT song song, TensorRT INT8 đạt mAP50-95=0.782, tăng 69.1% FPS so với FP32) — xem
> mục [Trích dẫn](#trích-dẫn). Điểm khác biệt: đồ án này chạy pruning và QAT **nối
> tiếp** thay vì song song, nhằm khai thác tiềm năng nén mạnh hơn.

---

## Dataset

| Thuộc tính | Giá trị |
|---|---|
| Nguồn dữ liệu | TODO |
| Số ảnh (train / val / test) | TODO |
| Số lớp | TODO |
| Kích thước ảnh | TODO |

---

## Đóng góp

Đóng góp luôn được chào đón — báo lỗi, đề xuất tính năng, hoặc Pull Request. Quy trình chung:

1. Tạo branch riêng theo module (`feat/pruning`, `feat/distillation`, `feat/dashboard-backend`...).
2. Test trên baseline chung trước khi mở Pull Request.
3. Mô tả rõ thay đổi và kết quả liên quan (mAP, size, FPS...) trong PR nếu ảnh hưởng đến pipeline nén mô hình.

Mọi ý kiến đóng góp đều hữu ích, kể cả từ ngoài nhóm thực hiện đồ án.

## Giấy phép

Phát hành theo giấy phép [MIT](LICENSE) — cập nhật nếu trường/khoa yêu cầu giấy phép khác.

## Trích dẫn

```bibtex
@article{luong2024edgeai,
  title   = {Edge AI For Real-Time Recyclable Waste Sorting In Viet Nam: A Practical Approach For Smart And Sustainable Cities},
  author  = {Luong, Tam Thanh and Do, Hop Quang and Nguyen, Anh Tien and Trinh, Khanh Bao and Trieu, Dat Quoc and Hoang, Giang Minh and Hoang, Thang Nam},
  journal = {The Purdue Conference for Industrial Engineering and Business},
  year    = {2024},
  url     = {https://docs.lib.purdue.edu/cib-conferences/vol2/iss1/48/}
}
```

