FROM python:3.11.1-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    diffusers transformers accelerate sentencepiece \
    Pillow

ENV HF_HOME=/runpod-volume/hf-cache
ENV HF_HUB_CACHE=/runpod-volume/hf-cache

COPY handler.py .

ENTRYPOINT ["python", "-u", "handler.py"]
