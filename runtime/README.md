# ppe-runtime

PPE inference runtime: **detect → track → associate PPE to people → violation rules**.
Standalone package; the edge-ops Edge Agent runs it as a container image and only manages its lifecycle.

## Install (dev)

```bash
pip install -e ..            # in-repo ultralytics fork (repo root) — trackers + pytorch engine
pip install -e ".[onnx,dev]" # this package
pytest
```

## Run

```bash
ppe-runtime run --source clip.mp4 --save out.mp4 # configs/default.yaml (pytorch, bytetrack)
ppe-runtime run --engine onnx --weights ../weights/yolo26n.onnx --source rtsp://cam/1
ppe-runtime run --weights synthetic --source clip.mp4 --set ppe.rules.min_frames=3 --set tracking=botsort
ppe-runtime bench ../weights/yolo26n.pt ../weights/yolo26_pruned.pt --source clip.mp4 --json runs/bench.json
```

Violations go to stdout as JSON lines (`{"type": "violation", "track_id": 3, "missing": ["helmet"], ...}`);
logs go to stderr. Other outputs: `sinks:` in the config (`file`, `webhook`, `agent`, `mqtt`, `kafka`).

## Config

`configs/default.yaml` names one file per part; anything can be overridden with `--set key.path=value`.

| File                                  | What                                                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------------------ |
| `engine/{pytorch,onnx,tensorrt}.yaml` | backend, imgsz, conf, iou, device                                                          |
| `tracking/{bytetrack,botsort}.yaml`   | ultralytics tracker parameters (`tracking: none` disables)                                 |
| `ppe/rules.yaml`                      | person classes, required items, aliases, negative classes, association, debounce, severity |
| `ppe/classes.yaml`                    | class id → name override (needed while the dataset has placeholder names)                  |

A tracked person raises **one** violation after `min_frames` consecutive frames missing the same items; it re-arms
when the person becomes compliant or the missing set changes. Without tracking there is no identity to debounce on,
so violations are only raised when `min_frames: 1`.

## Under the Edge Agent

The agent starts the image with `ppe-runtime agent` and injects `RUNTIME_ID`, `NODE_ID`, `AGENT_URL`,
`AGENT_API_KEY`, `MODEL_PATH` (mounted under `/models`) and `RUNTIME_CONFIG` (JSON: `source`, `target_fps`,
`report_interval_sec`, `conf`, `imgsz`, `device`, `engine`, `tracking`, `ppe_rules`, `classes`, `extra_sinks`).
The runtime reports violations and lifecycle as events (`POST /runtimes/{id}/events`) and metrics every
`report_interval_sec` as `RuntimeTelemetry` (`POST /runtimes/{id}/telemetry`). See `.env.example`.

## Images

Build from the repo root (the image needs the ultralytics fork):

```bash
docker build -f runtime/docker/Dockerfile -t edgeops/edge-runtime:latest .                 # amd64/arm64 CPU
docker build -f runtime/docker/Dockerfile.tensorrt -t edgeops/edge-runtime:latest-jetson . # Jetson, JetPack 6
```

## Model export

```bash
python scripts/export_onnx.py ../weights/yolo26n.pt
python scripts/export_tensorrt.py ../weights/yolo26n.pt --half # on the Jetson itself
python scripts/profile_pipeline.py --weights ../weights/yolo26n.onnx --source clip.mp4
```
