# Production Dockerfile for RunPod Serverless — FLUX.2 Klein 4B worker
# Build context: serverless_worker/ directory (RunPod uses the Dockerfile's directory as context)
# Platform: linux/amd64  CUDA 12.4  Python 3.11

FROM runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV MODEL_ID=black-forest-labs/FLUX.2-klein-4B \
    HF_HOME=/runpod-volume/hf-cache \
    HF_HUB_CACHE=/runpod-volume/hf-cache \
    RUNPOD_VOLUME_PATH=/runpod-volume

WORKDIR /app

# COPY paths are relative to serverless_worker/ (RunPod build context = Dockerfile's folder)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py /app/handler.py
COPY model_loader.py /app/model_loader.py
COPY schemas.py /app/schemas.py
COPY image_utils.py /app/image_utils.py

CMD ["python", "-u", "handler.py"]
