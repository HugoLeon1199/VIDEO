import runpod
import torch
import time
import base64
import hashlib
import io
import shutil
from pathlib import Path
from PIL import Image
from diffusers import FluxPipeline, FluxImg2ImgPipeline

print("Base model only. No LoRA or external adapter loaded.")

# Remove the old corrupted cache on cold-start to free disk space before
# downloading the clean model. Safe to delete: hf-cache-clean is the new path.
_old_cache = Path("/runpod-volume/hf-cache")
if _old_cache.exists():
    print(f"Removing old cache at {_old_cache} to free disk space...")
    shutil.rmtree(_old_cache, ignore_errors=True)
    print("Old cache removed.")

MODEL_ID = "black-forest-labs/FLUX.1-dev"
_pipe_t2i = None
_pipe_i2i = None


def load_t2i():
    global _pipe_t2i
    if _pipe_t2i is None:
        _pipe_t2i = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir="/runpod-volume/hf-cache-clean",
        ).to("cuda")
    return _pipe_t2i


def load_i2i():
    """Load img2img pipeline — shares UNet weights with t2i to save VRAM."""
    global _pipe_i2i
    if _pipe_i2i is None:
        t2i = load_t2i()
        _pipe_i2i = FluxImg2ImgPipeline(
            scheduler=t2i.scheduler,
            vae=t2i.vae,
            text_encoder=t2i.text_encoder,
            tokenizer=t2i.tokenizer,
            text_encoder_2=t2i.text_encoder_2,
            tokenizer_2=t2i.tokenizer_2,
            transformer=t2i.transformer,
        )
    return _pipe_i2i


def _decode_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_image(pil_img: Image.Image, quality: int = 92) -> tuple:
    buf = io.BytesIO()
    pil_img.save(buf, format="WEBP", quality=quality)
    buf.seek(0)
    raw = buf.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    b64 = "data:image/webp;base64," + base64.b64encode(raw).decode()
    return raw, sha, b64


def handler(job):
    inp = job.get("input", {})

    prompt          = inp["prompt"]
    negative_prompt = inp.get("negative_prompt", "")
    seeds           = inp.get("candidate_seeds", [11001])
    steps           = inp.get("steps", 22)
    guidance_scale  = inp.get("guidance_scale", 3.5)
    width           = inp.get("width", 1280)
    height          = inp.get("height", 720)
    quality         = inp.get("quality", 92)

    # img2img fields
    ref_image_b64   = inp.get("img2img_base64", None)   # base64 reference image
    strength        = float(inp.get("strength", 0.75))   # denoising strength

    # Clamp guidance_scale to flat-vector safe range
    guidance_scale = max(3.5, min(4.0, guidance_scale))

    # Clamp steps
    steps = max(20, min(24, steps))

    # FLUX.1-dev: embed negative_prompt into text (no native support)
    full_prompt = prompt
    if negative_prompt:
        full_prompt = f"{prompt}. Avoid: {negative_prompt}"

    mode = "img2img" if ref_image_b64 else "text_to_image"

    images = []
    errors = []
    t0 = time.time()

    if mode == "img2img":
        pipe = load_i2i()
        ref_image = _decode_image(ref_image_b64).resize((width, height))
    else:
        pipe = load_t2i()
        ref_image = None

    for i, seed in enumerate(seeds):
        try:
            generator = torch.Generator("cuda").manual_seed(seed)
            t1 = time.time()

            if mode == "img2img":
                result = pipe(
                    prompt=full_prompt,
                    image=ref_image,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            else:
                result = pipe(
                    prompt=full_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

            gen_secs = time.time() - t1
            _, sha, b64 = _encode_image(result.images[0], quality)

            images.append({
                "candidate_index": i,
                "seed": seed,
                "mime_type": "image/webp",
                "width": width,
                "height": height,
                "sha256": sha,
                "generation_seconds": gen_secs,
                "base64": b64,
            })
        except Exception as e:
            errors.append(f"seed {seed}: {e}")

    return {
        "video_id": inp.get("video_id", ""),
        "scene_id": inp.get("scene_id", ""),
        "model": MODEL_ID,
        "mode": mode,
        "duration_seconds": time.time() - t0,
        "images": images,
        "errors": errors,
    }


runpod.serverless.start({"handler": handler})
