"""Per-GPU FastAPI inference worker.

Loaded by the gateway (server.py). Expects model files already on disk at --model-path.
Listens on a localhost-only port (8090 + gpu_index).

CUDA_VISIBLE_DEVICES is set by the gateway subprocess spawn so this process only sees
the assigned GPU regardless of what 'cuda:0' resolves to.
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
from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig
from transformers import T5EncoderModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

USE_8BIT = os.getenv("USE_8BIT", "1") == "1"

app = FastAPI(title="Vast GPU Worker")
_pipe: Optional[FluxPipeline] = None
_load_error: Optional[str] = None
_infer_lock = threading.Lock()

signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))


# ---------------------------------------------------------------------------
# Model loading (local_files_only=True — gateway already downloaded the model)
# ---------------------------------------------------------------------------

def _load_model(model_path: str) -> FluxPipeline:
    global _pipe
    if _pipe is not None:
        return _pipe

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    from vast_worker.model_loader import validate_local
    revision = (os.getenv("HF_MODEL_REVISION", "") or "").strip()
    if not validate_local(model_path, revision):
        raise RuntimeError(f"Model validation failed at {model_path!r} (revision={revision})")

    if USE_8BIT and device == "cuda":
        print(f"[gpu_worker] Loading FLUX 8-bit from {model_path} on {device}...", flush=True)
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
        transformer = FluxTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer",
            quantization_config=qcfg, torch_dtype=dtype,
            local_files_only=True,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder_2",
            quantization_config=qcfg, torch_dtype=dtype,
            local_files_only=True,
        )
        _pipe = FluxPipeline.from_pretrained(
            model_path, transformer=transformer, text_encoder_2=text_encoder_2,
            torch_dtype=dtype, local_files_only=True,
        )
        _pipe.enable_model_cpu_offload()
    else:
        print(f"[gpu_worker] Loading FLUX from {model_path} on {device} dtype={dtype}...", flush=True)
        _pipe = FluxPipeline.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=True,
        )
        _pipe.enable_model_cpu_offload()
    print("[gpu_worker] Model loaded.", flush=True)
    return _pipe


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    video_id: str
    scene_id: str
    prompt: str
    clip_prompt: str = ""
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

CLOTHING_RULE = "all people fully clothed in modest traditional clothing"


def _normalize_prompt_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text))
    if "�" in normalized:
        raise HTTPException(status_code=400, detail="Prompt contains U+FFFD replacement character")
    return normalized


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _pipe is not None,
        "load_error": _load_error,
    }


@app.post("/generate")
def generate(req: GenerateRequest, request: Request) -> JSONResponse:
    _require_worker_token(request)
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    pipe = _pipe
    images_out = []
    errors = []

    clip_prompt = _normalize_prompt_text(req.clip_prompt or req.prompt)
    t5_prompt = _normalize_prompt_text(req.prompt)
    if req.negative_prompt:
        t5_prompt = f"{t5_prompt}. Avoid: {_normalize_prompt_text(req.negative_prompt)}"

    clip_tokens = _count_tokens(pipe.tokenizer, clip_prompt)
    if clip_tokens > 77:
        raise HTTPException(status_code=400, detail=f"clip_prompt exceeds 77-token limit ({clip_tokens})")

    t5_tokens = _count_tokens(getattr(pipe, "tokenizer_2", pipe.tokenizer), t5_prompt)
    if t5_tokens > 512:
        raise HTTPException(status_code=400, detail=f"prompt_2 exceeds 512-token limit ({t5_tokens})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in req.candidate_seeds:
        t0 = time.time()
        try:
            generator = torch.Generator(device=device).manual_seed(seed)
            with _infer_lock:
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
        except Exception as exc:
            errors.append(f"seed={seed}: {exc}")

    return JSONResponse({"images": images_out, "errors": errors})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_model_path: str = "/workspace/model"


def _background_load() -> None:
    global _load_error
    try:
        _load_model(_model_path)
    except Exception as exc:
        _load_error = str(exc)
        print(f"[gpu_worker] Model load failed: {exc}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/model")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    _model_path = args.model_path
    threading.Thread(target=_background_load, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port)
