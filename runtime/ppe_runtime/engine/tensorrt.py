from __future__ import annotations

import json
import logging

import numpy as np

from ppe_runtime.engine.base import Detector
from ppe_runtime.pipeline.types import Detection
from ppe_runtime.utils.config import EngineConfig
from ppe_runtime.utils.geometry import to_blob

logger = logging.getLogger(__name__)


class TensorRTDetector(Detector):
    """Native TensorRT backend (TRT 8.x and 10.x). Device buffers are torch CUDA tensors, so no pycuda is needed.

    Accepts ultralytics-exported `.engine` files (4-byte length + JSON metadata header) and bare `.plan` files.
    Engines are GPU/TensorRT-version specific: build them on the device that runs them.
    """

    backend = "tensorrt"

    def __init__(self, cfg: EngineConfig, names: dict[int, str] | None = None) -> None:
        super().__init__(cfg, names)
        import tensorrt as trt
        import torch

        self._torch = torch
        self._device = torch.device(f"cuda:{cfg.device}" if cfg.device and cfg.device.isdigit() else "cuda:0")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(cfg.weights, "rb") as f, trt.Runtime(trt_logger) as runtime:
            metadata = self._read_metadata(f)
            if metadata and metadata.get("dla") is not None:
                runtime.DLA_core = int(metadata["dla"])
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize {cfg.weights} (TensorRT version / GPU mismatch?)")
        self._context = self._engine.create_execution_context()
        self.set_names(self.parse_names((metadata or {}).get("names")))

        self._trt10 = not hasattr(self._engine, "num_bindings")
        count = self._engine.num_io_tensors if self._trt10 else self._engine.num_bindings
        self._buffers: list[torch.Tensor] = []
        self._input_index = 0
        self._output_indices: list[int] = []
        for i in range(count):
            if self._trt10:
                name = self._engine.get_tensor_name(i)
                is_input = self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                dtype = trt.nptype(self._engine.get_tensor_dtype(name))
                if is_input and -1 in tuple(self._engine.get_tensor_shape(name)):
                    self._context.set_input_shape(name, (1, 3, cfg.imgsz, cfg.imgsz))
                shape = tuple(self._context.get_tensor_shape(name))
            else:
                is_input = self._engine.binding_is_input(i)
                dtype = trt.nptype(self._engine.get_binding_dtype(i))
                if is_input and -1 in tuple(self._engine.get_binding_shape(i)):
                    self._context.set_binding_shape(i, (1, 3, cfg.imgsz, cfg.imgsz))
                shape = tuple(self._context.get_binding_shape(i))
            self._buffers.append(torch.from_numpy(np.empty(shape, dtype=dtype)).to(self._device))
            if is_input:
                self._input_index = i
            else:
                self._output_indices.append(i)
        in_shape = self._buffers[self._input_index].shape
        self.imgsz = int(in_shape[2])
        self._half = self._buffers[self._input_index].dtype == torch.float16
        self._output_index = max(self._output_indices, key=lambda j: self._buffers[j].numel())  # detection head
        logger.info("TensorRT %s, input %s, fp16=%s", cfg.weights, tuple(in_shape), self._half)

    @staticmethod
    def _read_metadata(f) -> dict | None:
        try:
            length = int.from_bytes(f.read(4), byteorder="little")
            return json.loads(f.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, MemoryError, OverflowError):
            f.seek(0)
            return None

    def predict(self, image: np.ndarray) -> list[Detection]:
        blob, gain, pad = to_blob(image, self.imgsz, self._half)
        self._buffers[self._input_index].copy_(self._torch.from_numpy(blob))
        self._context.execute_v2([int(b.data_ptr()) for b in self._buffers])
        output = self._buffers[self._output_index].float().cpu().numpy()
        return self.decode(output, gain, pad, image.shape)
