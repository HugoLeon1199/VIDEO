FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    runpod \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    diffusers transformers accelerate sentencepiece \
    Pillow

ENV HF_HOME=/runpod-volume/hf-cache
ENV HF_HUB_CACHE=/runpod-volume/hf-cache

COPY handler.py .

ENTRYPOINT ["python3", "-u", "handler.py"]
