"""ppe-runtime entrypoint.

ppe-runtime run   --source video.mp4 [--weights w.pt] [--save out.mp4] [--set engine.conf=0.4 ...]
ppe-runtime bench --weights a.pt b.onnx c.engine [--source video.mp4] [--json out.json]
ppe-runtime agent                 # config from Edge Agent env (RUNTIME_ID, MODEL_PATH, RUNTIME_CONFIG, ...)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from typing import Any

from ppe_runtime.utils.config import load_config, parse_overrides
from ppe_runtime.utils.logging import setup_logging
from ppe_runtime.version import __version__

logger = logging.getLogger("ppe_runtime")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="top-level YAML (default: configs/default.yaml)")
    p.add_argument("--weights", default=None, help=".pt / .onnx / .engine, or 'synthetic'")
    p.add_argument("--engine", default=None, help="engine config name: pytorch | onnx | tensorrt")
    p.add_argument("--tracker", default=None, help="bytetrack | botsort | none | path.yaml")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config field, e.g. engine.conf=0.4 or ppe.rules.min_frames=3",
    )
    p.add_argument("--log-level", default=None)


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    items = list(args.overrides)
    if args.engine:
        items.insert(0, f"engine={args.engine}")
    if args.weights:
        items.append(f"weights={args.weights}")
    if args.tracker:
        items.append(f"tracking={args.tracker}")
    if args.device is not None:
        items.append(f"engine.device={args.device}")
    if getattr(args, "source", None):
        items.append(f"source={args.source}")
    if getattr(args, "target_fps", None) is not None:
        items.append(f"target_fps={args.target_fps}")
    return parse_overrides(items)


def _install_signals(runner) -> None:
    def handler(signum, _frame):
        logger.info("Signal %s received, stopping", signum)
        runner.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):  # not main thread / unsupported on this platform
            pass


def cmd_run(args: argparse.Namespace) -> int:
    from ppe_runtime.io.video import VideoWriter
    from ppe_runtime.pipeline.pipeline import Runner
    from ppe_runtime.utils.visualize import draw

    cfg = load_config(args.config, _overrides(args))
    setup_logging(args.log_level or cfg.log_level)
    writer = VideoWriter(args.save) if args.save else None

    def on_frame(frame, result):
        if not (writer or args.show):
            return
        annotated = draw(frame, result)
        if writer:
            writer.write(annotated)
        if args.show:
            import cv2

            cv2.imshow("ppe-runtime", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                runner.stop()

    runner = Runner.from_config(cfg, on_frame, args.max_frames)
    if writer and getattr(runner.source, "fps", None):
        writer.fps = runner.source.fps
    _install_signals(runner)
    logger.info(
        "Running %s on %s (tracking=%s)",
        runner.pipeline.engine.backend,
        cfg.source,
        cfg.tracking.tracker_type if cfg.tracking else "off",
    )
    try:
        metrics = runner.run()
    finally:
        if writer:
            writer.close()
        if args.show:
            import cv2

            cv2.destroyAllWindows()
    logger.info("Done: %d violations", metrics.violations_total)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from ppe_runtime.bench import run_bench

    cfg = load_config(args.config, _overrides(args))
    setup_logging(args.log_level or "WARNING")
    run_bench(cfg, args.weights_list, args.source, args.frames, args.warmup, args.json)
    return 0


def agent_overrides(env: dict[str, str]) -> tuple[dict[str, Any], str]:
    """Map the Edge Agent container contract (env) onto config overrides. Returns (overrides, log_level)."""
    rc = json.loads(env.get("RUNTIME_CONFIG") or "{}")
    overrides: dict[str, Any] = {}
    if "engine" in rc:
        overrides["engine"] = rc["engine"]
    engine = overrides.get("engine") if isinstance(overrides.get("engine"), dict) else {}
    for key in ("conf", "iou", "imgsz", "device", "half"):
        if key in rc:
            engine[key] = rc[key]
    if engine:
        if isinstance(overrides.get("engine"), str):  # keep the named file, override fields via a second pass
            overrides["_engine_fields"] = engine
        else:
            overrides["engine"] = engine
    if env.get("MODEL_PATH"):
        overrides["weights"] = env["MODEL_PATH"]
    for key in ("source", "target_fps", "report_interval_sec", "tracking"):
        if key in rc:
            overrides[key] = rc[key]
    if isinstance(rc.get("ppe_rules"), dict):
        overrides["ppe"] = {"rules": rc["ppe_rules"]}
    if isinstance(rc.get("classes"), dict):
        overrides.setdefault("ppe", {})["classes"] = {"names": rc["classes"]}
    overrides["sinks"] = [
        {
            "type": "agent",
            "agent_url": env.get("AGENT_URL", "http://localhost:8081"),
            "runtime_id": env["RUNTIME_ID"],
            "node_id": env["NODE_ID"],
            "api_key": env.get("AGENT_API_KEY") or None,
        },
        *list(rc.get("extra_sinks", [])),
    ]
    return overrides, env.get("LOG_LEVEL", "INFO")


def cmd_agent(args: argparse.Namespace) -> int:
    from ppe_runtime.pipeline.pipeline import Runner

    overrides, level = agent_overrides(dict(os.environ))
    setup_logging(args.log_level or level)
    engine_fields = overrides.pop("_engine_fields", None)
    cfg = load_config(args.config, overrides)
    if engine_fields:
        cfg.engine = type(cfg.engine).model_validate({**cfg.engine.model_dump(), **engine_fields})
    runner = Runner.from_config(cfg)
    _install_signals(runner)
    logger.info(
        "Agent-managed runtime %s: %s on %s", os.environ.get("RUNTIME_ID"), runner.pipeline.engine.backend, cfg.source
    )
    runner.run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ppe-runtime", description="PPE detection + tracking + compliance runtime")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="process a video / stream and report violations")
    _common(run)
    run.add_argument("--source", default=None, help="video file, RTSP URL, or camera index")
    run.add_argument("--target-fps", type=float, default=None)
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--show", action="store_true", help="display annotated frames (q to quit)")
    run.add_argument("--save", default=None, help="write annotated video here")
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("bench", help="per-stage latency / FPS for one or more weights")
    _common(bench)
    bench.add_argument("weights_list", nargs="+", metavar="WEIGHTS")
    bench.add_argument("--source", default=None, help="video to sample frames from (default: noise)")
    bench.add_argument("--frames", type=int, default=200)
    bench.add_argument("--warmup", type=int, default=20)
    bench.add_argument("--json", default=None, help="write the report to this JSON file")
    bench.set_defaults(func=cmd_bench)

    agent = sub.add_parser("agent", help="run under the edge-ops Edge Agent (config from env)")
    agent.add_argument("--config", default=None)
    agent.add_argument("--log-level", default=None)
    agent.set_defaults(func=cmd_agent)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
