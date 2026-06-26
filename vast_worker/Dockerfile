# Vast.ai base image (cuda 12.1, py310, ubuntu22.04). Per Vast's official docs,
# extending vastai/base-image is the recommended way to cut cold start: Vast hosts
# already have these base layers CACHED, so a rented box only pulls our small top
# layers instead of a full ~9GB unknown image (that pull was a big part of the
# slow/hung cold starts when we based on pytorch/pytorch). CUDA 12.1 matches the
# torch 2.4.1 cu121 wheels installed below.
FROM vastai/base-image:cuda-12.1.1-cudnn8-devel-ubuntu22.04-py310-2026-06-15
WORKDIR /workspace

# Install torch 2.4.1 (cu121) first — the pinned diffusers/transformers stack needs
# torch >= 2.4. The vastai base ships CUDA + python but not torch, so we add it.
RUN pip install --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG MODEL_ID=black-forest-labs/FLUX.1-dev
ENV HF_HOME=/workspace/hf_cache
ENV MODEL_ID=${MODEL_ID}
# Off the Xet backend (cas-bridge.xethub.hf.co timed out on every shard); fall back
# to plain-HTTPS CDN by physically removing hf_xet. The model is NOT baked into the
# image (kept light + pushable); the rented box downloads ~24GB at startup over its
# fast US/EU network.
ENV HF_HUB_DISABLE_XET=1
RUN pip uninstall -y hf_xet hf-xet 2>/dev/null || true
# hf_transfer DISABLED. It's fast but fragile: a single dropped connection aborts
# the whole download with "An error occurred while downloading using hf_transfer"
# and it does NOT retry — this was the real cause of the flaky cold starts (some
# runs loaded in ~4min, others hung/failed mid-download). The standard hub
# downloader is a bit slower but resumable and retries per-file; combined with the
# US/EU geo gate (fast link to HF origin) the ~24GB model still lands in a few
# minutes, reliably. Reliability beats raw speed here.
ENV HF_HUB_ENABLE_HF_TRANSFER=0

COPY server.py .
EXPOSE 8080
# vastai/base-image ships an ENTRYPOINT that launches supervisor/caddy. Reset it to
# empty so our CMD runs the worker directly (we rent with runtype="args", which
# keeps the image's ENTRYPOINT+CMD; an inherited supervisor entrypoint would
# swallow our CMD and the worker would never start). Bind 0.0.0.0:8080.
ENTRYPOINT []
# Use the absolute interpreter path: the vastai base has no bare `python` on PATH
# (PATH only carries /usr/bin/python3 + a separate /venv/main that lacks our deps).
# Our pip (/usr/bin/pip) installed torch/diffusers into the SYSTEM python3
# (/usr/local/lib/python3.10/dist-packages), so run that interpreter explicitly.
CMD ["/usr/bin/python3", "server.py", "--port", "8080", "--preload"]
