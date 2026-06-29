"""Per-GPU FastAPI inference worker for FLUX.2-klein-9B-KV-FP8.

Experimental only — production uses vast_worker/ (FLUX.1-dev 12B).

Differences from the production worker:
- Loads FLUX.2-klein-9B-KV-FP8 (distilled, 4-step, FP8 quantized)
- Supports text-to-image AND img2img via FluxImg2ImgPipeline
- steps: 4 (t2i) or 8 (img2img) — set per request
- guidance_scale: ~1.0 (distilled, CFG-free)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import signal
import sys
import threading
import time
import unicodedata
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="Vast Klein Worker (FLUX.2-klein-9B)")
_t2i_pipe = None
_img2img_pipe = None
_load_error: Optional[str] = None
_infer_lock = threading.Lock()

signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_models(model_path: str) -> None:
    global _t2i_pipe, _img2img_pipe, _load_error
    try:
        from diffusers import FluxPipeline, FluxImg2ImgPipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"[klein_worker] Loading FLUX.2-klein text-to-image from {model_path}...", flush=True)
        _t2i_pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        _t2i_pipe.enable_model_cpu_offload()

        print("[klein_worker] Loading FLUX.2-klein img2img pipeline...", flush=True)
        _img2img_pipe = FluxImg2ImgPipeline(
            **{k: getattr(_t2i_pipe, k) for k in (
                "transformer", "scheduler", "vae", "text_encoder",
                "tokenizer", "text_encoder_2", "tokenizer_2",
            ) if hasattr(_t2i_pipe, k)}
        )
        _img2img_pipe.enable_model_cpu_offload()
        print("[klein_worker] Models ready.", flush=True)
    except Exception as exc:
        _load_error = str(exc)
        print(f"[klein_worker] Model load failed: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    video_id: str
    scene_id: str
    prompt: str
    clip_prompt: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 576
    steps: int = 4
    guidance_scale: float = 1.0
    candidate_seeds: list[int] = [11001]
    output_format: str = "WEBP"
    quality: int = 92
    img2img_base64: Optional[str] = None
    strength: float = 0.65


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _require_worker_token(request: Request) -> None:
    expected = (os.getenv("WORKER_API_TOKEN", "local-worker-token") or "").strip()
    if not expected:
        raise HTTPException(status_code=401, detail="Missing worker token configuration")
    header_token = request.headers.get("x-worker-token", "").strip()
    auth_header = request.headers.get("authorization", "").strip()
    if not header_token and auth_header.lower().startswith("bearer "):
        header_token = auth_header.split(" ", 1)[1].strip()
    if header_token != expected:
        raise HTTPException(status_code=401, detail="Invalid worker token")


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text))
    if "�" in normalized:
        raise HTTPException(status_code=400, detail="Prompt contains U+FFFD replacement character")
    return normalized


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "FLUX.2-klein-9B-KV-FP8",
        "t2i_loaded": _t2i_pipe is not None,
        "img2img_loaded": _img2img_pipe is not None,
        "load_error": _load_error,
    }


@app.post("/generate")
def generate(req: GenerateRequest, request: Request) -> JSONResponse:
    _require_worker_token(request)
    if _t2i_pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    clip_prompt = _normalize(req.clip_prompt or req.prompt)
    t5_prompt = _normalize(req.prompt)

    use_img2img = req.img2img_base64 is not None and _img2img_pipe is not None
    pipe = _img2img_pipe if use_img2img else _t2i_pipe

    # Decode reference image once (shared across seeds)
    reference_image: Optional[Image.Image] = None
    if use_img2img:
        try:
            raw_ref = base64.b64decode(req.img2img_base64)
            reference_image = Image.open(io.BytesIO(raw_ref)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid img2img_base64: {exc}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images_out = []
    errors = []

    for seed in req.candidate_seeds:
        t0 = time.time()
        try:
            generator = torch.Generator(device=device).manual_seed(seed)
            with _infer_lock:
                if use_img2img:
                    result = pipe(
                        image=reference_image,
                        prompt=clip_prompt,
                        prompt_2=t5_prompt,
                        strength=req.strength,
                        width=req.width,
                        height=req.height,
                        num_inference_steps=req.steps,
                        guidance_scale=req.guidance_scale,
                        generator=generator,
                        output_type="pil",
                        max_sequence_length=512,
                    )
                else:
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
                "mode": "img2img" if use_img2img else "text_to_image",
            })
        except Exception as exc:
            errors.append(f"seed={seed}: {exc}")

    return JSONResponse({"images": images_out, "errors": errors})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_model_path: str = "/workspace/model_klein"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/model_klein")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    _model_path = args.model_path
    threading.Thread(target=lambda: _load_models(_model_path), daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port)
