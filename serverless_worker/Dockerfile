# RunPod Serverless worker — FLUX.2 Klein 4B
# Build context in RunPod settings must be: serverless_worker
#
# IMPORTANT: use runpod/base (the SERVERLESS base image), NOT runpod/pytorch
# (which is a POD image for SSH/Jupyter). runpod/base ships the correct
# serverless runtime/entrypoint so `runpod.serverless.start()` attaches to the
# job queue. Using runpod/pytorch made workers go "ready" but never pick up
# jobs (stuck IN_QUEUE), even after fixing exit 127.
#
# runpod/base does NOT include torch, so we install it (plus diffusers stack).

FROM runpod/base:1.0.7-cuda1290-ubuntu2204

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Model cache on the mounted Network Volume (downloaded once)
ENV MODEL_ID=black-forest-labs/FLUX.2-klein-4B \
    HF_HOME=/runpod-volume/hf-cache \
    HF_HUB_CACHE=/runpod-volume/hf-cache \
    RUNPOD_VOLUME_PATH=/runpod-volume

WORKDIR /

COPY requirements.txt /requirements.txt
RUN python -m pip install --no-cache-dir -r /requirements.txt

COPY handler.py /handler.py
COPY model_loader.py /model_loader.py
COPY schemas.py /schemas.py
COPY image_utils.py /image_utils.py

# Do NOT reset ENTRYPOINT — runpod/base's entrypoint sets up the serverless
# runtime, then runs this CMD. HF_TOKEN injected at runtime via RunPod Secret.
CMD ["python", "-u", "/handler.py"]
