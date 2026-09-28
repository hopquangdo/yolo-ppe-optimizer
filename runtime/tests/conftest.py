from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RUNTIME_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RUNTIME_DIR.parent
sys.path.insert(0, str(RUNTIME_DIR))  # run tests without installing the package

from ppe_runtime.pipeline.types import Detection  # noqa: E402
from ppe_runtime.utils.config import RulesConfig  # noqa: E402

WEIGHTS = REPO_DIR / "weights" / "yolo26n.pt"


def det(name: str, box, conf: float = 0.9, class_id: int | None = None, track_id: int | None = None) -> Detection:
    ids = {"person": 0, "helmet": 1, "vest": 2, "no_helmet": 3, "no_vest": 4}
    return Detection(ids.get(name, 9) if class_id is None else class_id, name, conf, tuple(map(float, box)), track_id)


@pytest.fixture
def rules() -> RulesConfig:
    return RulesConfig(
        person_classes=["person"],
        required=["helmet", "vest"],
        aliases={"helmet": ["hardhat"]},
        negatives={"no_helmet": "helmet", "no_vest": "vest"},
        min_frames=3,
        forget_after=10,
        severity={"helmet": "CRITICAL", "vest": "WARNING"},
    )


@pytest.fixture
def image() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def weights() -> Path:
    if not WEIGHTS.exists():
        pytest.skip(f"{WEIGHTS} not found")
    return WEIGHTS
