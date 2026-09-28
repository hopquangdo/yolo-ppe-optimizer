"""Export YOLO weights to ONNX for the native `onnx` engine.

    python scripts/export_onnx.py ../weights/yolo26n.pt --imgsz 640 [--half] [--dynamic]
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("weights")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 (needs a CUDA device for export)")
    ap.add_argument("--dynamic", action="store_true", help="dynamic input size")
    ap.add_argument("--opset", type=int, default=None)
    ap.add_argument("--no-simplify", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    path = YOLO(args.weights).export(
        format="onnx", imgsz=args.imgsz, half=args.half, dynamic=args.dynamic, opset=args.opset,
        simplify=not args.no_simplify, device=0 if args.half else "cpu",
    )
    print(path)


if __name__ == "__main__":
    main()
