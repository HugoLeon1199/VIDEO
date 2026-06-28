"""Step 5: Generate images using the configured backend."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from loguru import logger

import config


def _canonical_image_path(video_dir: Path, scene_id: str) -> Path:
    return video_dir / "images" / f"img_{int(scene_id):03d}.png"


def _load_progress(progress_path: Path, video_dir: Path) -> set[str]:
    if not progress_path.exists():
        return set()
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        return {
            key
            for key, value in data.items()
            if value.get("status") == "completed" and _canonical_image_path(video_dir, key).exists()
        }
    except (json.JSONDecodeError, AttributeError):
        return set()


def _build_runpod_backend():
    from image_generation.runpod_client import RunPodClient
    from image_generation.runpod_serverless_backend import RunPodServerlessBackend

    if not config.RUNPOD_API_KEY:
        logger.error("RUNPOD_API_KEY not set")
        sys.exit(1)
    if not config.RUNPOD_ENDPOINT_ID:
        logger.error("RUNPOD_ENDPOINT_ID not set")
        sys.exit(1)
    client = RunPodClient(
        api_key=config.RUNPOD_API_KEY,
        endpoint_id=config.RUNPOD_ENDPOINT_ID,
        timeout=config.RUNPOD_REQUEST_TIMEOUT,
        poll_interval=config.RUNPOD_POLL_INTERVAL,
        max_retries=config.RUNPOD_MAX_RETRIES,
    )
    return RunPodServerlessBackend(client=client), None


def _build_vast_backend():
    from image_generation.vast_backend import VastInstanceBackend
    from image_generation.vast_manager import VastManager

    if not config.VAST_API_KEY:
        logger.error("VAST_API_KEY not set - add it to .env")
        sys.exit(1)
    revision = config.require_pinned_hf_model_revision()

    manager = VastManager(api_key=config.VAST_API_KEY, worker_port=config.VAST_WORKER_PORT)
    env_vars = {}
    if config.VAST_HF_TOKEN:
        env_vars["HF_TOKEN"] = config.VAST_HF_TOKEN
    env_vars["HF_MODEL_REVISION"] = revision
    env_vars["WORKER_API_TOKEN"] = config.WORKER_API_TOKEN
    env_vars["USE_8BIT"] = os.getenv("VAST_USE_8BIT", "1")
    if config.VAST_INSTANCE_HOST and config.VAST_INSTANCE_PORT:
        logger.info("Using existing Vast instance: {}:{}", config.VAST_INSTANCE_HOST, config.VAST_INSTANCE_PORT)
        backend = VastInstanceBackend(
            host=config.VAST_INSTANCE_HOST,
            port=config.VAST_INSTANCE_PORT,
            timeout=config.VAST_REQUEST_TIMEOUT,
            worker_token=config.WORKER_API_TOKEN,
        )
        manager.wait_worker_ready(config.VAST_INSTANCE_HOST, config.VAST_INSTANCE_PORT)
        return backend, None

    logger.info(
        "Renting Vast.ai instance (vram>={} GB, max ${}/hr)...",
        config.VAST_MIN_VRAM_GB,
        config.VAST_MAX_PRICE_PER_HOUR,
    )
    cost_caps = tuple(config.VAST_ESTIMATED_TOTAL_COST_FALLBACKS or ()) or (config.VAST_MAX_ESTIMATED_TOTAL_COST,)
    tried_machines: set[int] = set()
    last_error: Exception | None = None
    backend = None
    instance = None
    for total_cap in cost_caps:
        try:
            offer = manager.find_offer(
                min_vram_gb=config.VAST_MIN_VRAM_GB,
                gpu_name=config.VAST_GPU_NAME,
                max_price_per_hour=config.VAST_MAX_PRICE_PER_HOUR,
                min_inet_down_mbps=config.VAST_MIN_INET_DOWN_MBPS,
                min_reliability=config.VAST_MIN_RELIABILITY,
                min_disk_gb=max(60.0, config.VAST_DISK_GB),
                max_inet_cost_per_gb=config.VAST_MAX_INET_DOWN_COST,
                preferred_inet_cost_per_gb=config.VAST_PREFERRED_INET_DOWN_COST,
                expected_download_gb=config.VAST_EXPECTED_DOWNLOAD_GB,
                expected_upload_gb=config.VAST_EXPECTED_UPLOAD_GB,
                max_estimated_total_cost=total_cap,
                n_images=config.IMAGE_CANDIDATES,
                exclude_machine_ids=tried_machines,
            )
        except RuntimeError as exc:
            last_error = exc
            logger.warning("No Vast offer under estimated total cost cap $%.2f: {}", total_cap, exc)
            continue
        tried_machines.add(int(offer.get("machine_id")))
        instance = manager.rent(
            offer_id=offer["id"],
            image=config.VAST_WORKER_IMAGE,
            env_vars=env_vars,
            disk_gb=max(60.0, config.VAST_DISK_GB),
        )
        try:
            instance = manager.wait_until_running(instance.instance_id, timeout=300)
            if not instance.direct_port:
                instance = manager.wait_for_port(instance.instance_id, timeout=120)
            manager.deploy_worker(
                instance,
                hf_token=config.VAST_HF_TOKEN,
                model_revision=revision,
                worker_token=config.WORKER_API_TOKEN,
            )
            manager.wait_worker_ready(instance.ssh_host, instance.direct_port or config.VAST_WORKER_PORT, timeout=600)
            backend = VastInstanceBackend(
                host=instance.ssh_host,
                port=instance.direct_port or config.VAST_WORKER_PORT,
                timeout=config.VAST_REQUEST_TIMEOUT,
                worker_token=config.WORKER_API_TOKEN,
            )
            break
        except Exception as exc:
            last_error = exc
            logger.warning("Vast instance failed to start or warm up: {}", exc)
            manager.destroy(instance.instance_id)
            instance = None
            backend = None
            continue
    if backend is None:
        raise RuntimeError(last_error or "No Vast.ai offer found under configured total cost caps")

    def teardown():
        logger.info("Destroying Vast instance {}...", instance.instance_id)
        manager.destroy(instance.instance_id)

    return backend, teardown


def run(video_id: str, n_override: int | None = None) -> None:
    from image_generation.runpod_serverless_backend import promote_candidate_to_render_image
    from image_generation.schemas import SceneRequest

    video_dir = Path(config.OUTPUT_DIR) / video_id
    prompts_path = video_dir / "image_prompts.json"
    log_path = video_dir / "generation_log.json"
    if not prompts_path.exists():
        logger.error("image_prompts.json not found: {}", prompts_path)
        sys.exit(1)

    prompts: list[dict] = json.loads(prompts_path.read_text(encoding="utf-8"))
    if n_override is not None:
        prompts = prompts[:n_override]
    completed = _load_progress(log_path, video_dir)
    remaining = [item for item in prompts if f"{item['index']:03d}" not in completed]
    logger.info(
        "Backend: {}  |  Scenes: {}/{} done, {} remaining",
        config.IMAGE_BACKEND,
        len(completed),
        len(prompts),
        len(remaining),
    )

    if config.IMAGE_BACKEND == "vast_instance":
        backend, teardown = _build_vast_backend()
    else:
        backend, teardown = _build_runpod_backend()

    gen_log: dict = {}
    if log_path.exists():
        try:
            gen_log = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            gen_log = {}

    ok = 0
    fail = 0
    try:
        for prompt in remaining:
            scene_id = f"{prompt['index']:03d}"
            logger.info("Scene {}: {}", scene_id, prompt["prompt"][:80])
            request = SceneRequest(
                video_id=video_id,
                scene_id=scene_id,
                prompt=prompt["prompt"],
                clip_prompt=prompt.get("clip_prompt", prompt["prompt"]),
                negative_prompt=prompt.get("negative_prompt", ""),
                width=int(prompt.get("width", config.IMAGE_WIDTH)),
                height=int(prompt.get("height", config.IMAGE_HEIGHT)),
                steps=int(prompt.get("steps", config.IMAGE_STEPS)),
                guidance_scale=float(prompt.get("guidance_scale", prompt.get("guidance", config.IMAGE_GUIDANCE_SCALE))),
                candidate_seeds=config.IMAGE_CANDIDATE_SEEDS,
                output_format=config.IMAGE_OUTPUT_FORMAT,
                quality=config.IMAGE_QUALITY,
                output_mode="base64",
            )
            try:
                result = backend.generate(request)
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
                    logger.info("Scene {} done - {} candidates", scene_id, len(result.candidates))
                ok += 1
            except Exception as exc:
                logger.error("Scene {} failed: {}", scene_id, exc)
                gen_log[scene_id] = {"status": "failed", "error": str(exc)}
                log_path.write_text(json.dumps(gen_log, indent=2), encoding="utf-8")
                fail += 1

        thumbnail_prompts_path = video_dir / config.PUBLISHING_DIRNAME / "thumbnail_prompts.json"
        if thumbnail_prompts_path.exists():
            from steps.thumbnails import generate_thumbnail_assets

            logger.info("Thumbnail prompts found - generating publishing thumbnails")
            generate_thumbnail_assets(video_id, backend_override=backend)
    finally:
        if teardown:
            teardown()

    if not remaining:
        logger.info("All scenes already generated.")
    logger.info("Done: {}/{} scenes ok, {} failed", ok, len(remaining), fail)
    if fail:
        sys.exit(1)
