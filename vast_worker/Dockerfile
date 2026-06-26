# torch 2.4 base: the pinned transformers/diffusers require torch >= 2.4.
# On 2.3.0 transformers silently disables PyTorch ("Disabling PyTorch because
# PyTorch >= 2.4 is required but found 2.3.0"), so FluxPipeline load crashes and
# the worker never comes up. torchvision is included (image needs it for the
# CLIP/Siglip image processors).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG MODEL_ID=black-forest-labs/FLUX.1-dev
ENV HF_HOME=/workspace/hf_cache
ENV MODEL_ID=${MODEL_ID}
# Force HF off the Xet transfer backend. cas-bridge.xethub.hf.co timed out on
# every shard and the resume loop re-downloaded the same files (a build hung 5.7h
# then failed). The env var alone was NOT enough — huggingface_hub still routed
# through Xet — so we physically uninstall the hf_xet package, which makes the
# hub fall back to the stable plain-HTTPS CDN. This applies to the model download
# server.py does at startup (the model is NOT baked into the image: that kept the
# image ~33GB and unpushable over the slow home uplink; instead the rented Vast
# box pulls a light image, then downloads the ~24GB model over its fast network).
ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
RUN pip uninstall -y hf_xet hf-xet 2>/dev/null || true

COPY server.py .
EXPOSE 8080
CMD ["python", "server.py", "--port", "8080", "--preload"]
