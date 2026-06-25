"""Step 5: Generate images using RunPod Serverless (FLUX.2 Klein 4B)."""

import json
import sys
from pathlib import Path

from loguru import logger

import config


def _canonical_image_path(video_dir: Path, scene_id: str) -> Path:
    return video_dir / "images" / f"img_{int(scene_id):03d}.png"


def _load_progress(progress_path: Path, video_dir: Path) -> set[str]:
    """Return set of completed scene_ids from progress file."""
    if not progress_path.exists():
        return set()
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        return {
            k
            for k, v in data.items()
            if v.get("status") == "completed" and _canonical_image_path(video_dir, k).exists()
        }
    except (json.JSONDecodeError, AttributeError):
        return set()


def run(video_id: str, n_override: int | None = None) -> None:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    prompts_path = video_dir / "image_prompts.json"
    log_path = video_dir / "generation_log.json"

    if not prompts_path.exists():
        logger.error("image_prompts.json not found: {}", prompts_path)
        sys.exit(1)
    if not config.RUNPOD_API_KEY:
        logger.error("RUNPOD_API_KEY not set — add it to .env or set $env:RUNPOD_API_KEY")
        sys.exit(1)
    if not config.RUNPOD_ENDPOINT_ID:
        logger.error("RUNPOD_ENDPOINT_ID not set — deploy the serverless worker first, then set it in .env")
        sys.exit(1)

    prompts: list[dict] = json.loads(prompts_path.read_text(encoding="utf-8"))
    if n_override:
        prompts = prompts[:n_override]

    # Load resume state
    completed = _load_progress(log_path, video_dir)
    remaining = [p for p in prompts if f"{p['index']:03d}" not in completed]
    logger.info("Scenes: {}/{} done, {} remaining", len(completed), len(prompts), len(remaining))

    if not remaining:
        logger.info("All scenes already generated.")
        return

    from image_generation.runpod_client import RunPodClient
    from image_generation.runpod_serverless_backend import (
        RunPodServerlessBackend,
        promote_candidate_to_render_image,
    )
    from image_generation.schemas import SceneRequest

    client = RunPodClient(
        api_key=config.RUNPOD_API_KEY,
        endpoint_id=config.RUNPOD_ENDPOINT_ID,
        timeout=config.RUNPOD_REQUEST_TIMEOUT,
        poll_interval=config.RUNPOD_POLL_INTERVAL,
        max_retries=config.RUNPOD_MAX_RETRIES,
    )
    backend = RunPodServerlessBackend(client=client)

    # Load or create generation log
    gen_log: dict = {}
    if log_path.exists():
        try:
            gen_log = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    ok = 0
    fail = 0

    for p in remaining:
        scene_id = f"{p['index']:03d}"
        logger.info("Scene {}: {}", scene_id, p["prompt"][:80])

        req = SceneRequest(
            video_id=video_id,
            scene_id=scene_id,
            prompt=p["prompt"],
            width=config.IMAGE_WIDTH,
            height=config.IMAGE_HEIGHT,
            steps=config.IMAGE_STEPS,
            guidance_scale=config.IMAGE_GUIDANCE_SCALE,
            candidate_seeds=config.IMAGE_CANDIDATE_SEEDS,
            output_format=config.IMAGE_OUTPUT_FORMAT,
            quality=config.IMAGE_QUALITY,
            output_mode="base64",
        )

        try:
            result = backend.generate(req)
            selected_image = ""
            if result.candidates:
                selected_image = promote_candidate_to_render_image(
                    result.candidates[0],
                    video_id=video_id,
                    scene_id=scene_id,
                )
            gen_log[scene_id] = {
                "status": "completed" if selected_image and not result.errors else "partial",
                "candidates_saved": len(result.candidates),
                "selected_image": selected_image,
                "errors": result.errors,
                "job_id": result.job_id,
                "duration_seconds": result.duration_seconds,
            }
            log_path.write_text(json.dumps(gen_log, indent=2), encoding="utf-8")

            if result.errors:
                logger.warning("Scene {} partial errors: {}", scene_id, result.errors)
            else:
                logger.info("Scene {} done — {} candidates", scene_id, len(result.candidates))
            ok += 1

        except Exception as e:
            logger.error("Scene {} failed: {}", scene_id, e)
            gen_log[scene_id] = {"status": "failed", "error": str(e)}
            log_path.write_text(json.dumps(gen_log, indent=2), encoding="utf-8")
            fail += 1

    logger.info("Done: {}/{} scenes ok, {} failed", ok, len(remaining), fail)
    if fail:
        sys.exit(1)
