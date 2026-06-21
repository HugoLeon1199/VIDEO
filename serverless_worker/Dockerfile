# RunPod Serverless worker — FLUX.2 Klein 4B  (FINAL working config)
# Build context in RunPod settings: serverless_worker
#
# Key lessons baked in:
#  - python:3.11.1-slim => SYSTEM python (no venv), official RunPod docs base.
#  - ENTRYPOINT (not CMD): the endpoint/template has a stray "Container Start
#    Command" = "Endpoint" that OVERRIDES Docker CMD and crash-loops the worker
#    ("[FATAL tini] exec Endpoint failed"). ENTRYPOINT is not overridden, so the
#    container runs python -u /handler.py (+ ignored extra arg) and the
#    serverless loop actually starts and picks up jobs.
#  - Model is lazy-loaded inside handler() and cached on the network volume.

FROM python:3.11.1-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV MODEL_ID=black-forest-labs/FLUX.2-klein-4B \
    HF_HOME=/runpod-volume/hf-cache \
    HF_HUB_CACHE=/runpod-volume/hf-cache \
    RUNPOD_VOLUME_PATH=/runpod-volume

WORKDIR /

# System libs needed by torch / Pillow / diffusers
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
