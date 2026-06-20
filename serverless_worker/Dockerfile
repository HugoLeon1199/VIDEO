# Production Dockerfile for RunPod Serverless — FLUX.2 Klein 4B worker
# Platform: linux/amd64  CUDA 12.4  Python 3.11

FROM runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Unbuffered Python output for clean RunPod logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Model / cache paths — must match RunPod Network Volume mount
ENV MODEL_ID=black-forest-labs/FLUX.2-klein-4B \
    HF_HOME=/runpod-volume/hf-cache \
    HF_HUB_CACHE=/runpod-volume/hf-cache \
    RUNPOD_VOLUME_PATH=/runpod-volume

WORKDIR /app

# Install pinned Python deps
COPY serverless_worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the worker source — never copy credentials or model weights
COPY serverless_worker/handler.py /app/handler.py
COPY serverless_worker/model_loader.py /app/model_loader.py
COPY serverless_worker/schemas.py /app/schemas.py
COPY serverless_worker/image_utils.py /app/image_utils.py

# HF_TOKEN is injected at runtime via RunPod Secrets — never baked into image
# Model weights download to /runpod-volume/hf-cache at first cold-start

CMD ["python", "-u", "handler.py"]
