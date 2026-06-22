# RunPod Serverless worker — FLUX.2 Klein 9B
# Build context in RunPod settings: serverless_worker_flux2dev
# GPU required: 48GB (29GB VRAM needed)
#
# Same lessons as Klein worker:
#  - ENTRYPOINT not CMD (stray "Container Start Command" overrides CMD)
#  - Model lazy-loaded inside handler(), not at import time
#  - Network Volume mounted at /runpod-volume caches model weights

FROM python:3.11.1-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV MODEL_ID=black-forest-labs/FLUX.2-klein-9B \
    HF_HOME=/runpod-volume/hf-cache \
    HF_HUB_CACHE=/runpod-volume/hf-cache \
    RUNPOD_VOLUME_PATH=/runpod-volume

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY handler.py /handler.py
COPY model_loader.py /model_loader.py
COPY schemas.py /schemas.py
COPY image_utils.py /image_utils.py

# HF_TOKEN injected at runtime via RunPod Secret.
ENTRYPOINT ["python", "-u", "/handler.py"]
