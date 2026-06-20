# Production Dockerfile for RunPod Serverless — FLUX.2 Klein 4B worker
# Platform: linux/amd64  CUDA 12.9  PyTorch 2.6  Ubuntu 22.04

FROM runpod/pytorch:1.0.7-cu1290-torch260-ubuntu2204

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV MODEL_ID=black-forest-labs/FLUX.2-klein-4B \
    HF_HOME=/runpod-volume/hf-cache \
    HF_HUB_CACHE=/runpod-volume/hf-cache \
    RUNPOD_VOLUME_PATH=/runpod-volume

WORKDIR /app

# Debug: show python paths in build log
RUN echo "=== PYTHON PATHS ===" && \
    (which python && echo "python OK") || echo "python NOT FOUND" && \
    (which python3 && echo "python3 OK") || echo "python3 NOT FOUND" && \
    ls /usr/bin/python* 2>/dev/null || echo "no python in /usr/bin" && \
    ls /usr/local/bin/python* 2>/dev/null || echo "no python in /usr/local/bin"

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py /app/handler.py
COPY model_loader.py /app/model_loader.py
COPY schemas.py /app/schemas.py
COPY image_utils.py /app/image_utils.py

CMD ["/bin/sh", "-c", "python3 -u handler.py 2>&1 || python -u handler.py 2>&1"]
