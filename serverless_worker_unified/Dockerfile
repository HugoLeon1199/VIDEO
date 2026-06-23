FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3-pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    runpod \
    "diffusers==0.30.3" \
    "transformers==4.44.2" \
    accelerate sentencepiece \
    Pillow

ENV HF_HOME=/runpod-volume/hf-cache
ENV HF_HUB_CACHE=/runpod-volume/hf-cache

COPY handler.py .

ENTRYPOINT ["python", "-u", "handler.py"]
