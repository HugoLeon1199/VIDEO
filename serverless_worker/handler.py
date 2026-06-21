"""
RunPod Serverless handler — FLUX.2 Klein 4B image generation.

One job = one scene. Generates N candidates sequentially (deterministic seeds).
Model is loaded once at cold-start via model_loader.load_model().
"""

from __future__ import annotations

import logging
import time

import runpod

from image_utils import (
    check_base64_size,
    pil_to_bytes,
    save_to_volume,
    sha256_of,
    to_base64,
)
from model_loader import load_model
from schemas import validate_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("handler")

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    t_job_start = time.time()

    params, errors = validate_input(job_input)
    if errors:
        return {"error": "; ".join(errors)}

    video_id = params["video_id"]
    scene_id = params["scene_id"]
    mode = params["mode"]
    prompt = params["prompt"]
    global_style = params["global_style"]
    negative_prompt = params["negative_prompt"]
    width = params["width"]
    height = params["height"]
    steps = params["steps"]
    guidance_scale = params["guidance_scale"]
    seeds = params["candidate_seeds"]
    output_format = params["output_format"]
    quality = params["quality"]
    output_mode = params["output_mode"]

    # Build full prompt
    full_prompt = f"{prompt}, {global_style}".strip(", ") if global_style else prompt

    # Lazy-load model on first job (NOT at import) so the serverless loop
    # starts immediately and the worker can accept jobs while/after loading.
    pipe = load_model()
    images_out = []
    gen_errors = []

    logger.info(
        "Job start — video=%s scene=%s mode=%s candidates=%d size=%dx%d steps=%d",
        video_id, scene_id, mode, len(seeds), width, height, steps,
    )

    for i, seed in enumerate(seeds, start=1):
        candidate_start = time.time()
        logger.info("Candidate %d/%d seed=%d", i, len(seeds), seed)

        try:
            import torch
            generator = torch.Generator("cuda").manual_seed(seed)

            # FLUX.2 Klein (distilled) does NOT accept negative_prompt.
            result = pipe(
                prompt=full_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=1,
            )
            pil_image = result.images[0]

            img_bytes = pil_to_bytes(pil_image, fmt=output_format, quality=quality)
            checksum = sha256_of(img_bytes)
            gen_seconds = round(time.time() - candidate_start, 2)

            entry: dict = {
                "candidate_index": i,
                "seed": seed,
                "mime_type": f"image/{output_format.lower()}",
                "width": pil_image.width,
                "height": pil_image.height,
                "sha256": checksum,
                "generation_seconds": gen_seconds,
            }

            if output_mode == "base64":
                check_base64_size(img_bytes)
                entry["base64"] = to_base64(img_bytes)
            else:
                rel_path = save_to_volume(
                    img_bytes, video_id, scene_id, i, seed, fmt=output_format
                )
                entry["volume_path"] = rel_path

            images_out.append(entry)
            logger.info("Candidate %d done in %.2fs sha256=%s…", i, gen_seconds, checksum[:12])

        except ValueError as e:
            # e.g. base64 size limit exceeded
            msg = f"Candidate {i} seed={seed}: {e}"
            logger.warning(msg)
            gen_errors.append(msg)
        except Exception as e:
            msg = f"Candidate {i} seed={seed} failed: {e}"
            logger.error(msg, exc_info=True)
            gen_errors.append(msg)

    total_seconds = round(time.time() - t_job_start, 2)
    logger.info("Job done — %d/%d images in %.2fs", len(images_out), len(seeds), total_seconds)

    return {
        "video_id": video_id,
        "scene_id": scene_id,
        "model": MODEL_ID,
        "mode": mode,
        "duration_seconds": total_seconds,
        "images": images_out,
        "errors": gen_errors,
    }


# Start the serverless loop IMMEDIATELY so the worker registers and can
# accept jobs. The model is lazy-loaded inside handler() on the first job
# (load_model() caches the pipeline, so subsequent jobs reuse it).
# Loading the 13GB model before start() makes RunPod report the worker as
# "ready" while the serverless loop never actually starts → jobs stuck IN_QUEUE.
runpod.serverless.start({"handler": handler})
