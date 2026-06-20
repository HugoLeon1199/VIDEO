# Production Dockerfile for RunPod Serverless — FLUX.2 Klein 4B worker
# Platform: linux/amd64  CUDA 12.9  PyTorch 2.6  Ubuntu 22.04
# Build context in RunPod settings must be: serverless_worker

FROM runpod/pytorch:1.0.7-cu1290-torch260-ubuntu2204

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Model cache lives on the mounted Network Volume so it is downloaded once
ENV MODEL_ID=black-forest-labs/FLUX.2-klein-4B \
    HF_HOME=/runpod-volume/hf-cache \
    HF_HUB_CACHE=/runpod-volume/hf-cache \
    RUNPOD_VOLUME_PATH=/runpod-volume

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py /app/handler.py
COPY model_loader.py /app/model_loader.py
COPY schemas.py /app/schemas.py
COPY image_utils.py /app/image_utils.py

# HF_TOKEN is injected at runtime via RunPod Secrets — never baked into image
#
# The base runpod/pytorch image ships its own ENTRYPOINT (/start.sh for
# SSH/Jupyter). If left in place it wraps our CMD and the serverless handler
# never runs (container exits 127). Reset ENTRYPOINT and invoke python3 by
# absolute path so there is no PATH ambiguity.
ENTRYPOINT []
CMD ["/usr/bin/python3", "-u", "/app/handler.py"]
