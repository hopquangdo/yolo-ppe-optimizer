# PPE runtime image (amd64/arm64, CPU; ONNX Runtime + PyTorch backends). The Edge Agent starts it per deployment.
# Build from the REPO ROOT (needs the ultralytics fork next to runtime/):
#   docker build -f runtime/docker/Dockerfile -t edgeops/edge-runtime:latest .
# Standalone:
#   docker run --rm -v $PWD/weights:/models -v $PWD/videos:/media edgeops/edge-runtime \
#     ppe-runtime run --weights /models/yolo26n.onnx --source /media/clip.mp4
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PPE_CONFIG_DIR=/app/runtime/configs YOLO_OFFLINE=1
WORKDIR /app

# libGL / glib for OpenCV; ffmpeg libs come with opencv wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image ~1 GB smaller than the default CUDA wheels
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# In-repo ultralytics fork (trackers + pytorch backend), then the runtime
COPY pyproject.toml README.md LICENSE ./
COPY ultralytics/ ultralytics/
COPY optimization/ optimization/
RUN pip install .
COPY runtime/ runtime/
RUN pip install "./runtime[onnx]"

# Models are mounted here by the Edge Agent (MODEL_PATH=/models/...)
VOLUME ["/models"]
CMD ["ppe-runtime", "agent"]
