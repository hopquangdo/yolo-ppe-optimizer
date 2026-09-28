"""cProfile the pipeline on a video (or noise frames) to find hot spots outside the model itself.

    python scripts/profile_pipeline.py --weights ../weights/yolo26n.onnx --source clip.mp4 --frames 200 --top 30
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from itertools import cycle, islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe_runtime.bench import load_frames  # noqa: E402
from ppe_runtime.pipeline.pipeline import Pipeline  # noqa: E402
from ppe_runtime.utils.config import load_config, parse_overrides  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default="synthetic")
    ap.add_argument("--source", default=None)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--sort", default="cumulative", help="pstats sort key: cumulative | tottime | ncalls")
    ap.add_argument("--out", default=None, help="save raw stats (open with snakeviz)")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=parse_overrides([f"weights={args.weights}", *args.overrides]))
    pipeline = Pipeline.from_config(cfg)
    frames = load_frames(args.source, min(args.frames, 300), cfg.engine.imgsz)
    pipeline.warmup()

    profiler = cProfile.Profile()
    profiler.enable()
    for frame in islice(cycle(frames), args.frames):
        pipeline.process(frame)
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats(args.sort)
    stats.print_stats(args.top)
    if args.out:
        stats.dump_stats(args.out)


if __name__ == "__main__":
    main()
