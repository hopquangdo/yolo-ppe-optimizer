# Plan: Kiến trúc cụm Edge AI + Cloud tập trung

Trạng thái: **plan kiến trúc, chưa code**. Mục tiêu: nhiều thiết bị Jetson
(mỗi thiết bị chạy N camera, inference YOLO26 pruned+QAT tại chỗ) gửi dữ
liệu về 1 cloud trung tâm để tổng hợp, lưu trữ, hiển thị (dashboard đã có ở
`app/frontend`, `app/backend`) và phục vụ chatbot/truy vấn.

Không code trong lượt này — tài liệu này để chốt kiến trúc trước khi triển
khai, vì đổi giao thức giữa chừng tốn kém hơn nhiều so với đổi model/pipeline
nén đã làm ở `PLAN_PRUNE.md`/`PLAN_DISTILL.md`/`PLAN_QAT.md`.

## 1. Bài toán và ràng buộc

- Nhiều Jetson Orin NX/Nano rải rác theo site (nhà máy/khu vực), mỗi thiết
  bị nối trực tiếp 1-N camera (RTSP), chạy inference tại chỗ (đã có
  `PLAN_QAT.md`: TensorRT INT8, ~30 FPS, ~42ms latency).
- Cloud không nhận **video thô liên tục** — băng thông site công nghiệp
  thường hạn chế/không ổn định. Cloud chỉ cần: (a) sự kiện vi phạm (nhẹ),
  (b) ảnh/clip ngắn làm bằng chứng (khi có vi phạm), (c) telemetry sức khoẻ
  thiết bị, (d) video trực tiếp **theo yêu cầu** (khi operator mở dashboard
  xem 1 camera cụ thể — đã có UI `LiveMonitorPanel`/`LiveMonitor` grid).
- Mạng site có thể rớt kết nối cloud tạm thời — edge phải **tự chủ**: vẫn
  inference + lưu cục bộ, đồng bộ bù lại khi có mạng trở lại. Đây là ràng
  buộc quan trọng nhất, quyết định toàn bộ lựa chọn giao thức bên dưới.
- Nhiều thiết bị → cần định danh, đăng ký, theo dõi trạng thái tập trung
  (khớp mục "Edge Devices" đã có trong dashboard, hiện đang mock 1 thiết bị
  — kiến trúc này generalize nó thành N thiết bị thật).

## 2. Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│ SITE A                                                            │
│  ┌──────────────┐   RTSP    ┌────────────────────────────────┐  │
│  │ Camera 1..N   │──────────▶│  Edge Agent (Jetson Orin NX)   │  │
│  └──────────────┘           │  - Inference (TensorRT INT8)    │  │
│                              │  - Local buffer (SQLite+disk)   │  │
│                              │  - MQTT client + HTTPS uploader │  │
│                              └───────────┬──────────────────────┘  │
└──────────────────────────────────────────┼─────────────────────┘
                                            │
SITE B, C, ... (mỗi site 1 Edge Agent)      │
                                            │
                     MQTT (mTLS, QoS1)      │      HTTPS (media upload)
                     alert/telemetry/OTA    │      snapshot/clip/model
                                            ▼
                              ┌───────────────────────────┐
                              │   Cloud ingestion layer     │
                              │  - MQTT broker (EMQX/       │
                              │    Mosquitto)                │
                              │  - Media upload API (FastAPI)│
                              └──────────────┬────────────┘
                                             │
                              ┌──────────────▼────────────┐
                              │  Message consumer/worker    │
                              │  (ghi DB, trigger alert)     │
                              └──────────────┬────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                        ▼
           ┌────────────────┐      ┌──────────────────┐     ┌──────────────────┐
           │ PostgreSQL      │      │ Object storage     │     │ app/backend API   │
           │ (violations,    │      │ (ảnh/clip vi phạm, │     │ (đã có, FastAPI)   │
           │ devices, cameras)│      │ S3-compatible/MinIO)│     │ → app/frontend      │
           └────────────────┘      └──────────────────┘     └──────────────────┘
                                                                        │
                                                              ┌─────────▼─────────┐
                                                              │ WebRTC/relay signal │
                                                              │ (xem live theo yêu  │
                                                              │ cầu, không qua MQTT) │
                                                              └───────────────────┘
```

## 3. Giao thức giao tiếp — chọn theo loại dữ liệu, không dùng 1 giao thức cho tất cả

Bài học từ chính pipeline nén đã làm: mỗi bước (sparsity, prune, distill,
QAT) cần kỹ thuật riêng, ghép chung 1 công thức sẽ gãy. Giao tiếp edge-cloud
cũng vậy — 4 loại dữ liệu có đặc tính rất khác nhau:

| Loại dữ liệu | Tần suất | Kích thước | Giao thức đề xuất | Lý do |
|---|---|---|---|---|
| Sự kiện vi phạm (metadata: loại, confidence, bbox, timestamp, personId) | Mỗi lần phát hiện (có debounce) | Vài trăm byte (JSON) | **MQTT QoS 1**, topic `site/{site_id}/device/{device_id}/alert` | Nhẹ, cần đảm bảo gửi (QoS1 = at-least-once), broker tự buffer khi cloud tạm quá tải, client tự buffer khi mất mạng |
| Ảnh/clip bằng chứng | Mỗi vi phạm | 50KB-2MB | **HTTPS multipart upload**, endpoint riêng `POST /media`, kèm `alert_id` để nối lại với event MQTT | Media không hợp với MQTT (giới hạn payload, không tối ưu cho file lớn); upload async, không chặn pipeline inference |
| Telemetry thiết bị (GPU util/mem, nhiệt độ, FPS, latency — đã có mock ở `EdgeDeviceCard`) | Định kỳ 5-15s | Vài trăm byte | **MQTT QoS 0**, topic `site/{site_id}/device/{device_id}/telemetry` | Tần suất cao, mất 1 gói không sao (gói tiếp theo ghi đè), QoS0 giảm overhead |
| Video trực tiếp theo yêu cầu (operator mở `LiveMonitorPanel`) | Chỉ khi có người xem | Liên tục, cao | **WebRTC** (hoặc HLS low-latency nếu WebRTC không khả thi), thiết lập qua signaling riêng, **không** đi qua MQTT/cloud storage | Video liên tục qua REST/MQTT sẽ không scale; WebRTC cho phép (khi hạ tầng cho phép) đi thẳng edge→trình duyệt, cloud chỉ làm signaling, không cõng băng thông video |
| Cập nhật model/config (OTA) | Hiếm (khi có checkpoint mới sau distill/QAT) | Vài-vài chục MB | **MQTT lệnh** (topic `site/{site_id}/device/{device_id}/cmd`) báo "có bản mới" → edge tự **pull** qua HTTPS từ object storage | Không đẩy file lớn qua MQTT; MQTT chỉ làm tín hiệu, kéo file là HTTPS GET có thể resume |

**Vì sao MQTT là xương sống cho alert/telemetry/lệnh** (không phải REST
polling hay gRPC streaming):
- **Publish/subscribe + broker đứng giữa** giải quyết đúng ràng buộc "mất
  mạng tạm thời" ở mục 1 — client MQTT có `clean_session=False` +
  persistent queue phía broker, edge cứ publish, broker giữ hộ đến khi cloud
  consumer online lại (không cần tự viết retry/backoff thủ công như REST).
- **1 broker, N thiết bị** — không cần cloud mở kết nối tới từng edge (edge
  sau NAT/router site, cloud khó gọi ngược vào); mọi thứ edge chủ động kết
  nối ra ngoài, đúng mô hình triển khai công nghiệp thực tế.
- **Topic theo cấu trúc `site/{id}/device/{id}/...`** cho phép cloud
  subscribe wildcard (`site/+/device/+/alert`) để nhận tất cả, hoặc
  dashboard subscribe riêng 1 thiết bị — khớp tự nhiên với việc
  `app/frontend` cần filter theo camera/site đã có ở `IncidentsTable`.

## 4. Edge Agent — thiết kế tối thiểu

Chạy trên mỗi Jetson, là tầng nối giữa pipeline inference (đã có
`qat_trainer_yolo26.py`/export TensorRT) và cloud:

1. **Inference loop** (đã có, ngoài phạm vi plan này) sinh ra sự kiện vi
   phạm nội bộ.
2. **Local write-ahead buffer**: mọi sự kiện + đường dẫn ảnh ghi vào SQLite
   cục bộ **trước**, publish MQTT **sau** — nếu publish thất bại, một
   worker nền retry theo thứ tự từ SQLite. Đảm bảo không mất sự kiện kể cả
   khi mất mạng dài hơn khả năng buffer của broker.
3. **MQTT client**: kết nối mTLS tới broker cloud (client cert riêng theo
   `device_id`, xem mục 6 — bảo mật), publish theo 3 topic ở mục 3, subscribe
   topic `cmd` để nhận lệnh (OTA, đổi cấu hình ROI/threshold — khớp field
   `runtime` đã mock ở `mock/cameras.ts` phía frontend, giờ có nguồn thật).
4. **Media uploader**: worker nền riêng, lấy ảnh/clip từ buffer cục bộ,
   upload HTTPS khi có mạng, retry với backoff, xoá file cục bộ sau khi
   cloud xác nhận nhận (tránh đầy ổ đĩa Jetson).
5. **Heartbeat**: publish telemetry định kỳ ngay cả khi không có vi phạm,
   để cloud phân biệt "thiết bị offline" vs "thiết bị online nhưng không có
   sự kiện" — khớp trạng thái ONLINE/OFFLINE đã có ở `EdgeDeviceCard`/
   `CameraListPanel`.

## 5. Cloud — thành phần cần thêm (bên cạnh `app/backend` đã có)

- **MQTT broker**: EMQX hoặc Mosquitto (EMQX có dashboard quản lý
  connection/ACL sẵn, phù hợp giai đoạn đầu nhiều thiết bị chưa ổn định).
- **Consumer worker**: subscribe broker, mỗi message → ghi PostgreSQL
  (bảng `violations` đã có ở `app/backend/app/models/violation.py`, cần
  thêm cột `device_id`, `site_id`) + publish lại qua WebSocket cho
  `app/frontend` nếu muốn cảnh báo **đẩy realtime** thay vì frontend tự
  poll `GET /violations` (hiện tại đang mock `USE_MOCK`/gọi REST — nâng cấp
  tự nhiên: thêm 1 WebSocket endpoint FastAPI, consumer publish vào đó).
- **Media API**: endpoint `POST /media` nhận multipart, lưu object storage
  (MinIO tự host được trên chính hạ tầng, tương thích S3 API — hợp lý hơn
  lưu file trực tiếp trên đĩa backend).
- **Device registry**: bảng `devices` (device_id, site_id, cert
  fingerprint, model_version, last_heartbeat, status) — cloud dùng để biết
  thiết bị nào tồn tại, đang chạy model/checkpoint nào (nối trực tiếp với
  câu hỏi "checkpoint nào đang chạy" mà `PLAN_QAT.md` mục 1.5 còn để mở).
- **Signaling cho WebRTC live-view**: 1 service nhỏ (hoặc tái dùng
  `app/backend`) để trao đổi SDP/ICE giữa trình duyệt (đang mở
  `LiveMonitorPanel`) và edge agent khi operator bấm xem 1 camera cụ thể.

## 6. Bảo mật

- **mTLS bắt buộc** cho MQTT (không dùng username/password) — mỗi Jetson có
  client certificate riêng, cấp khi provisioning thiết bị, broker chỉ chấp
  nhận cert đã ký bởi CA nội bộ. Thu hồi cert khi thiết bị bị rút khỏi hệ
  thống.
- **Media API dùng short-lived token** (JWT theo `device_id`, edge agent tự
  refresh) thay vì API key tĩnh — giảm rủi ro nếu 1 thiết bị bị chiếm quyền
  vật lý.
- **Không expose port inference/RTSP ra internet** — chỉ port ra ngoài của
  Jetson là kết nối MQTT/HTTPS **outbound**, đúng mô hình "edge chủ động
  gọi ra" đã nói ở mục 3.

## 7. Việc cần làm tiếp (chưa code trong lượt này)

- Chọn cụ thể EMQX vs Mosquitto (EMQX phù hợp hơn nếu cần dashboard quản lý
  ACL/connection nhiều thiết bị ngay từ đầu; Mosquitto nhẹ hơn nếu chỉ vài
  thiết bị pilot).
- Thiết kế schema PostgreSQL đầy đủ (`devices`, `sites`, mở rộng
  `violations` với `device_id`/`site_id`/`media_url`) — mở rộng trực tiếp
  từ `app/backend/app/models/violation.py` đã có.
- Viết Edge Agent thật (Python, chạy cạnh pipeline inference trên Jetson) —
  phụ thuộc pipeline QAT ở `PLAN_QAT.md` đã có checkpoint chạy được trên
  GPU thật trước (hiện máy dev là CPU-only, chưa test được QAT — đã ghi ở
  `PLAN_QAT.md` mục 1.5).
- Đổi `app/frontend` từ `USE_MOCK`/REST-only sang có thêm kênh WebSocket
  cho cảnh báo realtime, và tích hợp WebRTC player cho `LiveMonitorPanel`
  thay video mock hiện tại.
