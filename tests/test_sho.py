import numpy as np

from optimization.sho import sho
from optimization.objective import build_yolo_objective


def test_sho_minimizes_sphere():
    best, position, convergence = sho(6, 8, -5, 5, 3, lambda value: np.sum(value**2), seed=7)

    assert best == convergence[-1]
    assert best <= convergence[0]
    assert position.shape == (3,)
    assert np.all(position >= -5)
    assert np.all(position <= 5)


class _BoxMetrics:
    map = 0.75


class _Metrics:
    box = _BoxMetrics()


class _FakeModel:
    def __init__(self):
        self.calls = []

    def train(self, **kwargs):
        self.calls.append(kwargs)
        return _Metrics()


def test_yolo_objective_decodes_candidate():
    models = []

    def factory(path):
        model = _FakeModel()
        models.append((path, model))
        return model

    objective = build_yolo_objective("model.pt", "data.yaml", {"lr0": (0.0, 1.0), "mosaic": (0.0, 1.0)}, 2, 320,
                                     model_factory=factory)
    assert objective(np.array([0.2, 0.8])) == 0.25
    assert models[0][0] == "model.pt"
    assert models[0][1].calls[0]["lr0"] == 0.2
    assert models[0][1].calls[0]["mosaic"] == 0.8