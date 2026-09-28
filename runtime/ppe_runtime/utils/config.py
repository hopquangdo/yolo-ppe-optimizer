"""Load + validate runtime YAML config (configs/default.yaml and the sub-configs it names)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

CONFIG_DIR = Path(os.environ.get("PPE_CONFIG_DIR") or Path(__file__).resolve().parents[2] / "configs")

Backend = Literal["pytorch", "onnx", "tensorrt", "synthetic", "auto"]
SeverityName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Backend = "auto"  # auto = by weights suffix
    weights: str | None = None
    imgsz: int = 640
    conf: float = Field(0.25, ge=0, le=1)
    iou: float = Field(0.7, ge=0, le=1)
    device: str | None = None
    half: bool = False
    providers: list[str] = Field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])

    @field_validator("device", mode="before")
    @classmethod
    def _device_str(cls, v: Any) -> Any:
        return None if v is None else str(v)


class TrackingConfig(BaseModel):
    """Fields of ultralytics' bytetrack.yaml / botsort.yaml; unknown keys are passed through to the tracker."""

    model_config = ConfigDict(extra="allow")

    tracker_type: Literal["bytetrack", "botsort"] = "bytetrack"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True


class AssociationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["overlap", "iou", "center"] = "overlap"
    min_score: float = Field(0.5, ge=0, le=1)


class RulesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_classes: list[str] = Field(default_factory=lambda: ["person"])
    required: list[str] = Field(default_factory=lambda: ["helmet", "vest"])
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    negatives: dict[str, str] = Field(default_factory=dict)
    association: AssociationConfig = Field(default_factory=AssociationConfig)
    min_frames: int = Field(5, ge=1)
    forget_after: int = Field(90, ge=1)
    severity: dict[str, SeverityName] = Field(default_factory=dict)
    default_severity: SeverityName = "WARNING"


class ClassesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: dict[int, str] = Field(default_factory=dict)

    @field_validator("names", mode="before")
    @classmethod
    def _none_empty(cls, v: Any) -> Any:
        return v or {}


class PpeConfig(BaseModel):
    rules: RulesConfig = Field(default_factory=RulesConfig)
    classes: ClassesConfig = Field(default_factory=ClassesConfig)


class SinkConfig(BaseModel):
    """`type` picks the sink; remaining keys are that sink's options (see ppe_runtime.io.sink)."""

    model_config = ConfigDict(extra="allow")

    type: Literal["stdout", "file", "webhook", "agent", "mqtt", "kafka"]

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: EngineConfig = Field(default_factory=EngineConfig)
    tracking: TrackingConfig | None = Field(default_factory=TrackingConfig)  # None = tracking off
    ppe: PpeConfig = Field(default_factory=PpeConfig)
    source: str = "0"
    target_fps: float = Field(15, ge=0)
    report_interval_sec: float = Field(5, gt=0)
    sinks: list[SinkConfig] = Field(default_factory=lambda: [SinkConfig(type="stdout")])
    log_level: str = "INFO"

    @field_validator("source", mode="before")
    @classmethod
    def _source_str(cls, v: Any) -> Any:
        return str(v)


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        out[key] = deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _resolve(value: Any, subdir: str, config_dir: Path) -> Any:
    """A name → contents of configs/<subdir>/<name>.yaml; a path → that file; a mapping → itself."""
    if not isinstance(value, str):
        return value
    path = Path(value)
    if not path.suffix:
        path = config_dir / subdir / f"{value}.yaml"
    elif not path.is_absolute():
        path = config_dir / path
    return read_yaml(path)


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """`["engine.conf=0.4", "tracking=none"]` → nested dict (values parsed as YAML scalars)."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value, got {item!r}")
        node = out
        *parents, leaf = key.strip().split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> RuntimeConfig:
    """Load `path` (default configs/default.yaml), expand sub-config names, apply `overrides`, validate."""
    path = Path(path or os.environ.get("PPE_CONFIG") or CONFIG_DIR / "default.yaml")
    config_dir = path.parent if path.parent.name != "" else CONFIG_DIR
    raw = read_yaml(path)
    overrides = copy.deepcopy(overrides or {})

    engine = raw.get("engine", "pytorch")
    engine_swapped = isinstance(overrides.get("engine"), str)
    if engine_swapped:  # swap the engine file itself, e.g. --set engine=onnx
        engine = overrides.pop("engine")
    engine = dict(_resolve(engine, "engine", config_dir))
    if raw.get("weights") is not None:
        engine.setdefault("weights", str((config_dir / raw["weights"]).resolve()) if raw["weights"] != "synthetic"
                          and not Path(raw["weights"]).is_absolute() else raw["weights"])
    if "weights" in overrides:
        # weights given on the command line / env: relative to the cwd, backend follows the suffix unless the
        # engine file was chosen explicitly too
        w = overrides.pop("weights")
        engine["weights"] = w if w == "synthetic" or Path(w).is_absolute() else str(Path(w).resolve())
        if not engine_swapped:
            engine["backend"] = "auto"

    tracking = overrides.pop("tracking", raw.get("tracking", "bytetrack"))
    if isinstance(tracking, str) and tracking.lower() == "none" or tracking is None or tracking is False:
        tracking = None
    else:
        tracking = _resolve(tracking, "tracking", config_dir)

    ppe_raw = raw.get("ppe") or {}
    ppe = {
        "rules": _resolve(ppe_raw.get("rules", "rules"), "ppe", config_dir),
        "classes": _resolve(ppe_raw.get("classes", "classes"), "ppe", config_dir),
    }

    merged = {k: v for k, v in raw.items() if k not in {"engine", "weights", "tracking", "ppe"}}
    merged.update(engine=engine, tracking=tracking, ppe=ppe)
    merged = deep_merge(merged, overrides)

    cfg = RuntimeConfig.model_validate(merged)
    if cfg.engine.weights == "synthetic":
        cfg.engine.backend = "synthetic"
    return cfg
