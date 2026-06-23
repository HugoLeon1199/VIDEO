FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod \
    diffusers transformers accelerate sentencepiece \
    Pillow

ENV HF_HOME=/runpod-volume/hf-cache
ENV HF_HUB_CACHE=/runpod-volume/hf-cache

COPY handler.py .

ENTRYPOINT ["python", "-u", "handler.py"]
