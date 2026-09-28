"""Build a TensorRT engine for the native `tensorrt` engine. Run it ON the target device (e.g. the Jetson).

    python scripts/export_tensorrt.py ../weights/yolo26n.pt --half
    python scripts/export_tensorrt.py ../weights/yolo26n.pt --int8 --data ../dataset/ppe-custom.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("weights")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16")
    ap.add_argument("--int8", action="store_true", help="INT8 (needs --data for calibration)")
    ap.add_argument("--data", default=None, help="dataset YAML for INT8 calibration")
    ap.add_argument("--workspace", type=float, default=None, help="builder workspace (GiB)")
    ap.add_argument("--dla", type=int, default=None, help="Jetson DLA core (0/1)")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    if args.int8 and not args.data:
        ap.error("--int8 needs --data for calibration")

    from ultralytics import YOLO

    device = f"dla:{args.dla}" if args.dla is not None else args.device
    path = YOLO(args.weights).export(
        format="engine", imgsz=args.imgsz, half=args.half, int8=args.int8, data=args.data,
        workspace=args.workspace, device=device,
    )
    print(path)


if __name__ == "__main__":
    main()
