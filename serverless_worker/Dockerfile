# RunPod Serverless worker — official docs pattern (python:slim, system python)
# Build context in RunPod settings: serverless_worker
#
# MINIMAL diagnostic image: only the runpod SDK, NO torch/diffusers. Purpose:
# isolate whether the serverless loop picks up jobs on python:3.11.1-slim
# (system python, clean entrypoint) — runpod/base and runpod/pytorch both went
# "ready" but never picked up jobs. Tiny image => fast build + fast cold start.
# Once the loop is confirmed, torch/diffusers + the real handler are added back.

FROM python:3.11.1-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /

RUN pip install --no-cache-dir runpod==1.7.9

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
