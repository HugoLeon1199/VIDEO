"""FastAPI worker that runs on a Vast.ai GPU instance.

Accepts POST /generate, runs FLUX.1-dev inference, returns images as base64.
Compatible with VastInstanceBackend payload format.

Start:
    python vast_worker/server.py --port 8080
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import signal
import sys
import time
from typing import Optional

import torch
import uvicorn
from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig
from transformers import T5EncoderModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

# Use 8-bit quantization when available to halve VRAM: FLUX.1-dev bf16 needs ~24GB;
# 8-bit peaks ~13-14GB, fitting on any 16GB+ card and leaving headroom on 24GB.
USE_8BIT = os.getenv("USE_8BIT", "1") == "1"

app = FastAPI(title="Vast Image Worker")
_pipe: Optional[FluxPipeline] = None
_load_error: Optional[str] = None  # set if background load fails

# A single FluxPipeline is NOT thread-safe: two concurrent /generate calls sharing
# it throw "Already borrowed" (the tokenizer's Rust state is borrowed twice). The
# client fans out `--workers N` requests in parallel, so serialize inference behind
# one lock — the GPU runs one image at a time anyway.
import threading
_infer_lock = threading.Lock()

# Graceful shutdown when Vast destroys the instance
signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))


def _load_model() -> FluxPipeline:
    global _pipe
    if _pipe is not None:
        return _pipe

    model_id = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")
    model_revision = os.getenv("HF_MODEL_REVISION", "") or None
    hf_token = os.getenv("HF_TOKEN", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Download the repo first via snapshot_download, then load every submodule from
    # that local dir (offline + instant, no per-subfolder metadata round-trips).
    #
    # ignore_patterns: the FLUX.1-dev repo ships BOTH the diffusers layout
    # (transformer/, text_encoder_2/, vae/ ...) AND redundant single-file weights at
    # root. Diffusers only uses the folders, so we skip the root duplicates —
    # confirmed via HF API ?blobs=true:
    #   flux1-dev.safetensors  23.8GB  (single-file; diffusers uses transformer/)
    #   ae.safetensors          0.34GB (diffusers uses vae/)
    #   dev_grid.jpg            1.3MB  (demo image)
    # That cuts the download ~58GB -> ~34GB (the #1 Vast cost is bandwidth, billed
    # per GB pulled, every rental). Only these 3 EXACT names are skipped — never
    # touch transformer/, text_encoder*, vae/, *.json, tokenizer*.
    #
    # revision: pin a commit SHA so a repo change can't silently alter size/layout
    # (breaking the cost estimate or the pipeline). Empty env = repo default (main).
    from huggingface_hub import snapshot_download
    print(f"Downloading {model_id} (rev={model_revision or 'main'}, skip root single-file)...", flush=True)
    model_path = snapshot_download(
        repo_id=model_id,
        revision=model_revision,
        token=hf_token or None,
        ignore_patterns=[
            "flux1-dev.safetensors",
            "ae.safetensors",
            "dev_grid.jpg",
        ],
    )
    print(f"Download complete: {model_path}", flush=True)

    if USE_8BIT and device == "cuda":
        # 8-bit quantization, mirroring the proven RunPod handler. CRITICAL: quantize
        # BOTH the transformer AND the T5 text_encoder_2 with bitsandbytes. Quantizing
        # only the transformer (and leaving T5 in plain bf16) is what caused the
        # "'T5EncoderModel' object has no attribute '_hf_hook'" crash — the offload
        # hooks expect every big submodule to be device-managed by accelerate. With
        # both quantized, enable_model_cpu_offload places hooks consistently and peak
        # VRAM stays ~13-14GB (safe on 16/24GB cards).
        print(f"Loading {model_id} 8-bit (transformer + T5) on {device}...")
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
        transformer = FluxTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer",
            quantization_config=qcfg, torch_dtype=dtype,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder_2",
            quantization_config=qcfg, torch_dtype=dtype,
        )
        _pipe = FluxPipeline.from_pretrained(
            model_path, transformer=transformer, text_encoder_2=text_encoder_2,
            torch_dtype=dtype,
        )
        # EXACTLY mirror the proven RunPod handler (serverless_worker_unified/handler.py),
        # which generated all 62 images with this same stack: no .to("cuda"), then
        # enable_model_cpu_offload(). bitsandbytes places the 8-bit weights, and
        # model_cpu_offload manages activations. The earlier "index 21" / "sequential
        # offload" errors were from my deviations (.to("cuda"), removing offload) —
        # NOT from this call. Keep it identical to the known-good path.
        _pipe.enable_model_cpu_offload()
    else:
        print(f"Loading {model_id} on {device} dtype={dtype} (no 8-bit)...")
        _pipe = FluxPipeline.from_pretrained(
            model_path, torch_dtype=dtype,
        )
        _pipe.enable_model_cpu_offload()
    print("Model loaded.")
    return _pipe


class GenerateRequest(BaseModel):
    video_id: str
    scene_id: str
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 576
    steps: int = 20
    guidance_scale: float = 3.5
    candidate_seeds: list[int] = [11001]
    output_format: str = "WEBP"
    quality: int = 92
    img2img_base64: Optional[str] = None
    strength: float = 0.75


@app.get("/health")
def health() -> dict:
    # status "ok" means the HTTP server is up (client can stop waiting on boot);
    # model_loaded tells the client whether /generate will succeed immediately.
    return {
        "status": "ok",
        "model_loaded": _pipe is not None,
        "load_error": _load_error,
    }


CLOTHING_RULE = "all people fully clothed in modest traditional clothing"


@app.post("/generate")
def generate(req: GenerateRequest) -> JSONResponse:
    pipe = _load_model()
    images_out = []
    errors = []

    # CLIP encodes `prompt` with a hard 77-token cap; T5 encodes `prompt_2` with
    # 512 tokens. Prepend the clothing constraint to clip_prompt so it always
    # survives the 77-token window, then fold the full text + negative into t5.
    clip_prompt = f"{CLOTHING_RULE}. {req.prompt}"
    t5_prompt = clip_prompt
    if req.negative_prompt:
        t5_prompt = f"{clip_prompt}. Avoid: {req.negative_prompt}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in req.candidate_seeds:
        t0 = time.time()
        try:
            generator = torch.Generator(device=device).manual_seed(seed)
            with _infer_lock:  # serialize: pipeline is not thread-safe
                result = pipe(
                    prompt=clip_prompt,
                    prompt_2=t5_prompt,
                    width=req.width,
                    height=req.height,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance_scale,
                    generator=generator,
                    output_type="pil",
                    max_sequence_length=512,
                )
            img: Image.Image = result.images[0]

            buf = io.BytesIO()
            fmt = req.output_format.upper()
            if fmt == "WEBP":
                img.save(buf, format="WEBP", quality=req.quality)
            elif fmt == "JPEG":
                img.save(buf, format="JPEG", quality=req.quality)
            else:
                img.save(buf, format="PNG")
            raw = buf.getvalue()

            images_out.append({
                "seed": seed,
                "width": img.width,
                "height": img.height,
                "image_base64": base64.b64encode(raw).decode(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "generation_seconds": round(time.time() - t0, 2),
            })
        except Exception as e:
            errors.append(f"seed={seed}: {e}")

    return JSONResponse({"images": images_out, "errors": errors})


def _background_load() -> None:
    """Load the model in a thread so /health is reachable while ~13GB streams in."""
    global _load_error
    try:
        _load_model()
    except Exception as e:  # noqa: BLE001 — surface any load failure via /health
        _load_error = str(e)
        print(f"Model load failed: {e}", flush=True)


if __name__ == "__main__":
    import threading

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--preload", action="store_true", help="Load model at startup")
    args = parser.parse_args()

    # Preload in the background: the HTTP server (and /health) comes up immediately,
    # so the client stops waiting on container boot and then polls model_loaded.
    if args.preload:
        threading.Thread(target=_background_load, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port)
