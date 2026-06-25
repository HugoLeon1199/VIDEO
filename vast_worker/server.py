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
import time
from typing import Optional

import torch
import uvicorn
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="Vast Image Worker")
_pipe: Optional[FluxPipeline] = None


def _load_model() -> FluxPipeline:
    global _pipe
    if _pipe is not None:
        return _pipe

    model_id = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")
    hf_token = os.getenv("HF_TOKEN", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading {model_id} on {device}...")
    _pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=hf_token or None,
    ).to(device)
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
    return {"status": "ok", "model_loaded": _pipe is not None}


@app.post("/generate")
def generate(req: GenerateRequest) -> JSONResponse:
    pipe = _load_model()
    images_out = []
    errors = []

    for seed in req.candidate_seeds:
        t0 = time.time()
        try:
            generator = torch.Generator().manual_seed(seed)
            result = pipe(
                prompt=req.prompt,
                width=req.width,
                height=req.height,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                generator=generator,
                output_type="pil",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--preload", action="store_true", help="Load model at startup")
    args = parser.parse_args()

    if args.preload:
        _load_model()

    uvicorn.run(app, host=args.host, port=args.port)
