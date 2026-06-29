# Same vastai/base-image as production 12B worker so rented boxes hit the cached
# base layers. Klein 9B FP8 is ~10GB download vs 24GB for 12B.
FROM vastai/base-image:cuda-12.1.1-cudnn8-devel-ubuntu22.04-py310-2026-06-15

WORKDIR /workspace

# Same torch pin as production — FLUX.2-klein needs torch >= 2.4
RUN pip install --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /workspace/vast_worker_klein/
RUN pip install --no-cache-dir -r /workspace/vast_worker_klein/requirements.txt

# Disable Xet backend (fragile, not retryable — same lesson as production)
ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
RUN pip uninstall -y hf_xet hf-xet 2>/dev/null || true

COPY . /workspace/vast_worker_klein/

RUN mkdir -p /workspace/.cache/huggingface /workspace/model_klein

ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV KLEIN_MODEL_ID=black-forest-labs/FLUX.2-klein-9b-kv-fp8
ENV KLEIN_WORKER_PORT=8081

EXPOSE 8081

ENTRYPOINT []
# Use /usr/bin/python3 (system python with our installed packages) — same as production.
# gpu_worker downloads the model at startup via HF hub then loads into GPU.
CMD ["/usr/bin/python3", "-m", "vast_worker_klein.gpu_worker", "--port", "8081", "--host", "0.0.0.0"]
