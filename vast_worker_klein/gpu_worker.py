"""Per-GPU FastAPI inference worker for FLUX.2-klein-9B-KV-FP8."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import signal
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="Vast Klein Worker (FLUX.2-klein-9B)")
_t2i_pipe = None
_img2img_pipe = None
_load_error: Optional[str] = None
_model_loaded = False
_cache_path = ""
_device_name = ""
_vram_gb = 0.0

MODEL_ID = os.getenv("KLEIN_MODEL_ID") or os.getenv("MODEL_ID") or "black-forest-labs/FLUX.2-klein-9b-kv-fp8"
MODEL_REVISION = os.getenv("KLEIN_HF_REVISION") or os.getenv("MODEL_REVISION") or None
WORKER_PORT = int(os.getenv("KLEIN_WORKER_PORT", "8081"))
HF_CACHE_DIR = os.getenv("HF_HOME", "/workspace/.cache/huggingface")

signal.signal(signal.SIGTERM, lambda _sig, _frame: sys.exit(0))


def _validate_model_cache(model_path: str) -> None:
    required = [
        Path(model_path) / "model_index.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Klein cache validation failed, missing: {missing}")


def _load_models() -> None:
    global _t2i_pipe, _img2img_pipe, _load_error, _model_loaded, _cache_path, _device_name, _vram_gb
    try:
        from diffusers import FluxImg2ImgPipeline, FluxPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device found - Klein 9B requires GPU")
        token = (
            os.getenv("HF_TOKEN")
            or os.getenv("VAST_HF_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
            or None
        )
        _device_name = torch.cuda.get_device_name(0)
        _vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        dtype = torch.bfloat16

        print(f"[klein_worker] Downloading/caching {MODEL_ID} revision={MODEL_REVISION or 'default'}...", flush=True)
        model_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            token=token,
            cache_dir=HF_CACHE_DIR,
        )
        _validate_model_cache(model_path)
        _cache_path = model_path

        print(f"[klein_worker] Loading FLUX.2 Klein t2i from {model_path} ...", flush=True)
        _t2i_pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        _t2i_pipe.enable_model_cpu_offload()

        print("[klein_worker] Building FLUX.2 Klein img2img pipeline ...", flush=True)
        _img2img_pipe = FluxImg2ImgPipeline(
            **{
                key: getattr(_t2i_pipe, key)
                for key in (
                    "transformer",
                    "scheduler",
                    "vae",
                    "text_encoder",
                    "tokenizer",
                    "text_encoder_2",
                    "tokenizer_2",
                )
                if hasattr(_t2i_pipe, key)
            }
        )
        _img2img_pipe.enable_model_cpu_offload()
        _model_loaded = True
        print(f"[klein_worker] Model ready on {_device_name} ({_vram_gb} GB VRAM).", flush=True)
    except Exception as exc:  # noqa: BLE001
        _load_error = str(exc)
        _model_loaded = False
        print(f"[klein_worker] Model load failed: {exc}", flush=True)
        raise


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


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text))
    if "\ufffd" in normalized:
        raise HTTPException(status_code=400, detail="Prompt contains U+FFFD replacement character")
    return normalized


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _model_loaded and not _load_error else "loading",
        "model": MODEL_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_loaded": _model_loaded,
        "t2i_loaded": _t2i_pipe is not None,
        "img2img_loaded": _img2img_pipe is not None,
        "cache_path": _cache_path,
        "device": _device_name or ("cuda" if torch.cuda.is_available() else "cpu"),
        "gpu_vram_gb": _vram_gb,
        "load_error": _load_error,
    }


@app.post("/generate")
def generate(req: GenerateRequest, request: Request) -> JSONResponse:
    _require_worker_token(request)
    if not _model_loaded or _t2i_pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    clip_prompt = _normalize(req.clip_prompt or req.prompt)
    t5_prompt = _normalize(req.prompt)
    use_img2img = req.img2img_base64 is not None and _img2img_pipe is not None
    pipe = _img2img_pipe if use_img2img else _t2i_pipe

    reference_image: Optional[Image.Image] = None
    if use_img2img:
        try:
            raw_ref = base64.b64decode(req.img2img_base64)
            reference_image = Image.open(io.BytesIO(raw_ref)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid img2img_base64: {exc}") from exc

    images_out = []
    errors = []
    for seed in req.candidate_seeds:
        t0 = time.time()
        try:
            generator = torch.Generator(device="cpu").manual_seed(seed)
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
            images_out.append(
                {
                    "seed": seed,
                    "width": img.width,
                    "height": img.height,
                    "image_base64": base64.b64encode(raw).decode(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "generation_seconds": round(time.time() - t0, 2),
                    "mode": "img2img" if use_img2img else "text_to_image",
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"seed={seed}: {exc}")

    return JSONResponse({"images": images_out, "errors": errors})


if __name__ == "__main__":
    import threading

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=WORKER_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # Load models in background thread so /health is reachable during loading
    threading.Thread(target=_load_models, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port)
