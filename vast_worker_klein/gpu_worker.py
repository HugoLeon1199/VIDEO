"""Per-GPU FastAPI worker for FLUX.2-klein-9B-KV-FP8.

Experimental only — production uses vast_worker/ (FLUX.1-dev 12B).

Uses Flux2KleinPipeline (diffusers >= 0.38): one unified pipeline for both
text-to-image (image=None) and reference editing (image=<PIL>).
Always 4 steps, guidance_scale=1.0.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
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

MODEL_ID: str = os.getenv("KLEIN_MODEL_ID", "black-forest-labs/FLUX.2-klein-9b-kv-fp8")
_RAW_REVISION: str = (os.getenv("KLEIN_HF_REVISION") or "").strip()
WORKER_PORT: int = int(os.getenv("KLEIN_WORKER_PORT", "8081"))
HF_CACHE_DIR: str = os.getenv("HF_HOME", "/workspace/.cache/huggingface")
PIPELINE_CLASS: str = "Flux2KleinPipeline"

# Revision must be a commit SHA (40 hex chars) so the image is reproducible.
# Pass KLEIN_HF_REVISION=<sha> when building the Docker image for production.
_SHA_RE = __import__("re").compile(r"^[0-9a-f]{40}$", __import__("re").IGNORECASE)
if _RAW_REVISION and not _SHA_RE.match(_RAW_REVISION):
    raise RuntimeError(
        f"KLEIN_HF_REVISION={_RAW_REVISION!r} is not a 40-char commit SHA. "
        "Set it to the exact commit SHA from the HuggingFace model repo."
    )
MODEL_REVISION: Optional[str] = _RAW_REVISION or None

app = FastAPI(title="Vast Klein Worker (FLUX.2-klein-9B)")

_pipe = None                  # Flux2KleinPipeline instance
_model_loaded: bool = False
_load_error: Optional[str] = None
_device_name: str = ""
_vram_total_gb: float = 0.0
_load_start: float = 0.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_models() -> None:
    global _pipe, _model_loaded, _load_error, _device_name, _vram_total_gb

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device — Klein 9B requires GPU")

        _device_name = torch.cuda.get_device_name(0)
        _vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)

        hf_token = (
            os.getenv("KLEIN_HF_TOKEN")
            or os.getenv("VAST_HF_TOKEN")
            or os.getenv("HF_TOKEN")
            or None
        )

        from vast_worker_klein.model_loader import download_and_validate

        print(f"[klein_worker] Ensuring {MODEL_ID} rev={MODEL_REVISION or 'main'} ...", flush=True)
        model_path = download_and_validate(
            model_id=MODEL_ID,
            model_path=os.path.join(HF_CACHE_DIR, "klein_model"),
            revision=MODEL_REVISION or "",
            hf_token=hf_token,
        )

        print(f"[klein_worker] Loading {PIPELINE_CLASS} from {model_path} on {_device_name} ...", flush=True)
        from diffusers import Flux2KleinPipeline
        _pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        _pipe.enable_model_cpu_offload()

        _model_loaded = True
        elapsed = round(time.time() - _load_start, 1)
        print(f"[klein_worker] Ready in {elapsed}s on {_device_name} ({_vram_total_gb}GB VRAM)", flush=True)

    except Exception as exc:
        _load_error = str(exc)
        _model_loaded = False
        print(f"[klein_worker] Load FAILED: {exc}", flush=True)


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
    output_format: str = "PNG"
    quality: int = 100
    # 0 images = text-to-image; 1-3 images = reference editing
    reference_images_base64: list[str] = []
    # kept for backward compat with VastInstanceBackend
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
    vram_used = None
    if torch.cuda.is_available():
        try:
            vram_used = round(torch.cuda.memory_allocated() / 1024**3, 2)
        except Exception:
            pass
    return {
        "status": "ready" if _model_loaded else ("error" if _load_error else "loading"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pipeline_class": PIPELINE_CLASS,
        "model_loaded": _model_loaded,
        "device": _device_name or ("cuda" if torch.cuda.is_available() else "cpu"),
        "gpu_vram_total_gb": _vram_total_gb,
        "gpu_vram_used_gb": vram_used,
        "load_error": _load_error,
    }


@app.post("/generate")
def generate(req: GenerateRequest, request: Request) -> JSONResponse:
    _require_worker_token(request)
    if not _model_loaded or _pipe is None:
        detail = f"Model not loaded: {_load_error}" if _load_error else "Model loading, try again"
        raise HTTPException(status_code=503, detail=detail)

    prompt = _normalize(req.prompt)

    # Collect reference images: new field takes priority, fall back to legacy img2img_base64
    ref_b64_list = list(req.reference_images_base64)
    if not ref_b64_list and req.img2img_base64:
        ref_b64_list = [req.img2img_base64]

    # Decode reference images
    reference_images: list[Image.Image] = []
    for b64 in ref_b64_list:
        try:
            raw = base64.b64decode(b64)
            reference_images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid reference image: {exc}") from exc

    mode = "reference_editing" if reference_images else "text_to_image"
    # Flux2KleinPipeline: image accepts a single PIL or None
    ref_image = reference_images[0] if reference_images else None

    images_out = []
    errors = []

    for seed in req.candidate_seeds:
        t0 = time.time()
        try:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            result = _pipe(
                image=ref_image,
                prompt=prompt,
                height=req.height,
                width=req.width,
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
                "mode": mode,
                "reference_count": len(reference_images),
            })
        except Exception as exc:
            errors.append(f"seed={seed}: {exc}")

    return JSONResponse({"images": images_out, "errors": errors})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=WORKER_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    _load_start = time.time()
    # Start model loading in background so /health is reachable during loading
    threading.Thread(target=_load_models, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port)
