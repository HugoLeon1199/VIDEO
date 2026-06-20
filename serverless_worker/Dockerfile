# Production Dockerfile for RunPod Serverless â€” FLUX.2 Klein 4B worker
# Platform: linux/amd64  CUDA 12.9  PyTorch 2.6  Ubuntu 22.04

FROM runpod/pytorch:1.0.7-cu1290-torch260-ubuntu2204

# Unbuffered Python output for clean RunPod logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Model / cache paths â€” must match RunPod Network Volume mount
ENV MODEL_ID=black-forest-labs/FLUX.2-klein-4B \
    HF_HOME=/runpod-volume/hf-cache \
    HF_HUB_CACHE=/runpod-volume/hf-cache \
    RUNPOD_VOLUME_PATH=/runpod-volume

WORKDIR /app

# Install pinned Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py /app/handler.py
COPY model_loader.py /app/model_loader.py
COPY schemas.py /app/schemas.py
COPY image_utils.py /app/image_utils.py

# Build context must be serverless_worker/ in RunPod settings
# HF_TOKEN is injected at runtime via RunPod Secrets â€” never baked into image
# Model weights download to /runpod-volume/hf-cache at first cold-start

CMD ["python3", "-u", "handler.py"]
