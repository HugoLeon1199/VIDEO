# RunPod Serverless worker — F5-TTS voice cloning
# Build context: tts_worker
# Same pattern as FLUX worker: ENTRYPOINT (not CMD), python:3.11.1-slim

FROM python:3.11.1-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV HF_HOME=/runpod-volume/hf-cache \
    HF_HUB_CACHE=/runpod-volume/hf-cache \
    RUNPOD_VOLUME_PATH=/runpod-volume

WORKDIR /

# System libs: ffmpeg for audio processing, git for pip installs
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY handler.py /handler.py

ENTRYPOINT ["python", "-u", "/handler.py"]
