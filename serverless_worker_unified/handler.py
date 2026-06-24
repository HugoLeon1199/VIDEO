import runpod
import torch
import time
import base64
import hashlib
import io
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-dev"
_pipe = None


def load_model():
    global _pipe
    if _pipe is None:
        _pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir="/runpod-volume/hf-cache",
        ).to("cuda")
    return _pipe


def handler(job):
    inp = job.get("input", {})
    pipe = load_model()

    prompt          = inp["prompt"]
    negative_prompt = inp.get("negative_prompt", "")
    seeds           = inp.get("candidate_seeds", [11001])
    steps           = inp.get("steps", 20)
    guidance_scale  = inp.get("guidance_scale", 3.5)
    width           = inp.get("width", 1024)
    height          = inp.get("height", 576)

    # FLUX.1-dev does not support negative_prompt natively — embed it into the prompt
    full_prompt = prompt
    if negative_prompt:
        full_prompt = f"{prompt}. Avoid: {negative_prompt}"

    images = []
    errors = []
    t0 = time.time()

    for i, seed in enumerate(seeds):
        try:
            generator = torch.Generator("cuda").manual_seed(seed)
            t1 = time.time()
            result = pipe(
                prompt=full_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            gen_secs = time.time() - t1

            buf = io.BytesIO()
            result.images[0].save(buf, format="WEBP", quality=92)
            buf.seek(0)
            raw = buf.getvalue()
            sha = hashlib.sha256(raw).hexdigest()
            b64 = "data:image/webp;base64," + base64.b64encode(raw).decode()

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
        "mode": "text_to_image",
        "duration_seconds": time.time() - t0,
        "images": images,
        "errors": errors,
    }


runpod.serverless.start({"handler": handler})
