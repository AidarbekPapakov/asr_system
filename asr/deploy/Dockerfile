# ASR service -- GigaAM ASR/emotion + pyannote VAD + the gender CNN,
# served with LitServe.
#
# Build from the repository root:
#   docker compose -f deploy/docker-compose.yml build
#
# CUDA 12.9 because the target GPU (RTX 5070 Ti) is sm_120 -- Blackwell
# support starts at CUDA 12.8. A GPU is not required: the 600M CTC model runs
# at ~11x realtime on CPU, so this image also works with no device attached.
FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04

# ffmpeg: audio.py's fallback decode path shells out to it for whatever
# soundfile refuses. git: pip installs gigaam from a GitHub URL.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# The CUDA wheels are 300-700 MB each: uv's 30 s timeout and parallel
# downloads drop them on a slow or flaky link.
ENV UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_RETRIES=10 \
    UV_CONCURRENT_DOWNLOADS=2 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    PYANNOTE_METRICS_ENABLED=false

WORKDIR /app

# Dependency layer first: source changes below do not re-trigger the ~4 GB
# install.
COPY deploy/requirements.txt /app/requirements.txt
RUN uv venv --python 3.11 /app/.venv \
    && uv pip install --python /app/.venv/bin/python \
        --torch-backend=cu129 \
        -r /app/requirements.txt

COPY src /app/src

# GigaAM checkpoints (~2.8 GB, Sber CDN) and the HF cache for
# pyannote/segmentation-3.0 -- mounted as volumes so a container restart
# does not re-download them.
VOLUME ["/root/.cache/gigaam", "/cache/huggingface"]

EXPOSE 8400

CMD ["python", "src/server.py"]
