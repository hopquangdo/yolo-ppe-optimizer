import json

import cv2
import numpy as np
import pytest
from ppe_runtime.cli import agent_overrides, main
from ppe_runtime.io.sink import FileSink
from ppe_runtime.io.video import VideoSource
from ppe_runtime.pipeline import Pipeline, Runner
from ppe_runtime.utils.config import load_config


@pytest.fixture
def cfg():
    return load_config(overrides={"weights": "synthetic", "ppe": {"rules": {"min_frames": 3}}})


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    for i in range(12):
        frame = np.full((480, 640, 3), i * 10, np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_default_config_loads():
    cfg = load_config()
    assert cfg.engine.backend == "pytorch" and cfg.engine.weights.endswith("yolo26n.pt")
    assert cfg.tracking.tracker_type == "bytetrack"
    assert cfg.ppe.rules.required == ["helmet", "vest"]


def test_config_overrides():
    from ppe_runtime.utils.config import parse_overrides

    cfg = load_config(overrides=parse_overrides(["engine=onnx", "tracking=none", "ppe.rules.min_frames=2"]))
    assert cfg.engine.backend == "onnx" and cfg.engine.providers and cfg.tracking is None
    assert cfg.ppe.rules.min_frames == 2


def test_pipeline_detect_track_rules(cfg, image):
    pipeline = Pipeline.from_config(cfg)
    results = [pipeline.process(image) for _ in range(5)]
    last = results[-1]
    assert len(last.persons) == 3 and all(p.person.track_id is not None for p in last.persons)
    # synthetic: persons 0 and 2 wear vests, person 1 does not → exactly one violation, raised at frame 3
    raised = [v for r in results for v in r.violations]
    assert len(raised) == 1 and raised[0].missing == {"vest"} and raised[0].frame_index == 3
    assert last.timings.total_ms > 0


def test_runner_on_video_writes_violations(cfg, video, tmp_path):
    out = tmp_path / "events.jsonl"
    pipeline = Pipeline.from_config(cfg)
    metrics = Runner(pipeline, VideoSource(video), FileSink(str(out)), report_interval_sec=60).run()
    records = [json.loads(line) for line in out.read_text().splitlines()]
    kinds = [r["type"] for r in records]
    assert kinds.count("violation") == 1 and metrics.violations_total == 1
    assert records[0]["event_type"] == "runtime.started" and records[-1]["event_type"] == "runtime.stopped"
    [m] = [r for r in records if r["type"] == "metrics"]
    assert m["inference_fps"] > 0 and m["detection_count"] > 0


def test_cli_run(video, capsys):
    assert main(["run", "--weights", "synthetic", "--source", str(video), "--set", "ppe.rules.min_frames=2"]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert any(r["type"] == "violation" for r in lines)


def test_cli_bench(tmp_path, capsys):
    out = tmp_path / "bench.json"
    assert main(["bench", "synthetic", "--frames", "20", "--warmup", "2", "--json", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["results"]["synthetic"]["fps"] > 0


def test_agent_env_contract():
    env = {
        "RUNTIME_ID": "rt-1",
        "NODE_ID": "node-1",
        "AGENT_URL": "http://agent:8081",
        "MODEL_PATH": "/models/m.onnx",
        "RUNTIME_CONFIG": json.dumps({"source": "rtsp://cam/1", "target_fps": 10, "conf": 0.4, "tracking": "botsort"}),
    }
    overrides, level = agent_overrides(env)
    cfg = load_config(overrides=overrides)
    assert cfg.engine.weights.replace("\\", "/").endswith("/models/m.onnx") and cfg.engine.backend == "auto"
    assert cfg.engine.conf == 0.4 and cfg.source == "rtsp://cam/1" and cfg.target_fps == 10
    assert cfg.tracking.tracker_type == "botsort"
    [sink] = cfg.sinks
    assert sink.type == "agent" and sink.options["runtime_id"] == "rt-1" and level == "INFO"


@pytest.mark.model
def test_real_model_pipeline(weights, image):
    cfg = load_config(overrides={"weights": str(weights), "engine": {"device": "cpu"}})
    result = Pipeline.from_config(cfg).process(image)
    assert result.timings.detect_ms > 0
