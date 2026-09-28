import numpy as np
import pytest

from ppe_runtime.engine import create_engine, resolve_backend
from ppe_runtime.engine.synthetic import SyntheticDetector
from ppe_runtime.utils.config import EngineConfig
from ppe_runtime.utils.geometry import letterbox, nms, unletterbox


@pytest.mark.parametrize(("weights", "backend"), [
    ("a.pt", "pytorch"), ("a.onnx", "onnx"), ("a.engine", "tensorrt"), ("a.plan", "tensorrt"), (None, "synthetic"),
])
def test_resolve_backend(weights, backend):
    assert resolve_backend(EngineConfig(weights=weights)) == backend


def test_resolve_backend_unknown_suffix():
    with pytest.raises(ValueError):
        resolve_backend(EngineConfig(weights="model.bin"))


def test_names_override(image):
    engine = create_engine(EngineConfig(backend="synthetic"), names={2: "safety_vest"})
    assert engine.names == {0: "person", 1: "helmet", 2: "safety_vest"}
    assert {d.class_name for d in engine.predict(image)} == {"person", "helmet", "safety_vest"}


def test_letterbox_roundtrip():
    img = np.zeros((480, 640, 3), np.uint8)
    out, gain, pad = letterbox(img, 640)
    assert out.shape == (640, 640, 3) and gain == 1.0 and pad == (0.0, 80.0)
    boxes = np.array([[10, 90, 110, 190]], np.float32)  # letterboxed space
    np.testing.assert_allclose(unletterbox(boxes, gain, pad, img.shape), [[10, 10, 110, 110]])


def test_nms_keeps_best_of_overlapping():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5).tolist() == [0, 2]


def _decoder(conf=0.25):
    return SyntheticDetector(EngineConfig(backend="synthetic", conf=conf))


def test_decode_end2end_layout():
    # (1, 300, 6): x1 y1 x2 y2 score cls, in 640 letterboxed space of a 480x640 image (pad_y = 80)
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = [100, 180, 200, 380, 0.9, 0]
    out[0, 1] = [0, 0, 5, 5, 0.1, 1]  # below conf
    [d] = _decoder().decode(out, 1.0, (0.0, 80.0), (480, 640, 3))
    assert d.class_name == "person" and d.box == (100, 100, 200, 300)


def test_decode_raw_layout_runs_nms():
    nc, anchors = 3, 50
    out = np.zeros((1, 4 + nc, anchors), np.float32)
    out[0, :4, 0] = [150, 280, 100, 200]  # cx cy w h
    out[0, 4 + 2, 0] = 0.9  # class 2
    out[0, :4, 1] = [152, 281, 100, 200]  # near-duplicate, lower score
    out[0, 4 + 2, 1] = 0.6
    [d] = _decoder().decode(out, 1.0, (0.0, 80.0), (480, 640, 3))
    assert d.class_id == 2 and d.confidence == pytest.approx(0.9)
    assert d.box == pytest.approx((100, 100, 200, 300))


@pytest.mark.model
def test_pytorch_engine(weights, image):
    engine = create_engine(EngineConfig(weights=str(weights), device="cpu"))
    assert engine.backend == "pytorch" and engine.names
    assert isinstance(engine.predict(image), list)
