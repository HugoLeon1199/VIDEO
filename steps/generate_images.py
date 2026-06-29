"""Step 5: Generate images using the configured backend."""

from __future__ import annotations

import json
import os
import time
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
        raise RuntimeError("RUNPOD_API_KEY not set")
    if not config.RUNPOD_ENDPOINT_ID:
        raise RuntimeError("RUNPOD_ENDPOINT_ID not set")
    client = RunPodClient(
        api_key=config.RUNPOD_API_KEY,
        endpoint_id=config.RUNPOD_ENDPOINT_ID,
        timeout=config.RUNPOD_REQUEST_TIMEOUT,
        poll_interval=config.RUNPOD_POLL_INTERVAL,
        max_retries=config.RUNPOD_MAX_RETRIES,
    )
    return RunPodServerlessBackend(client=client), None


def _build_vast_backend(planned_image_count: int | None = None):
    from image_generation.vast_backend import VastInstanceBackend
    from image_generation.vast_manager import VastManager

    if not config.VAST_API_KEY:
        raise RuntimeError("VAST_API_KEY not set - add it to .env")
    revision = config.require_pinned_hf_model_revision()
    worker_token = config.require_explicit_worker_token()

    manager = VastManager(api_key=config.VAST_API_KEY, worker_port=config.VAST_WORKER_PORT)
    env_vars = {}
    if config.VAST_HF_TOKEN:
        env_vars["HF_TOKEN"] = config.VAST_HF_TOKEN
    env_vars["HF_MODEL_REVISION"] = revision
    env_vars["WORKER_API_TOKEN"] = worker_token
    env_vars["USE_8BIT"] = os.getenv("VAST_USE_8BIT", "1")

    # Real session image count drives the offer cost estimate.
    n_images_for_offer = planned_image_count if planned_image_count is not None else config.IMAGE_CANDIDATES
    logger.info("Vast offer search: planned_image_count={}", n_images_for_offer)

    def destroy_and_verify(instance_id: int, *, lifecycle=None) -> bool:
        if lifecycle is not None:
            lifecycle.teardown_attempt_count += 1
        try:
            manager.destroy(instance_id)
        except Exception as exc:
            logger.warning("Failed to destroy Vast instance {}: {}", instance_id, exc)
            return False
        for _attempt in range(5):
            try:
                if not any(int(item.get("id", 0)) == int(instance_id) for item in manager.list_instances()):
                    if lifecycle is not None:
                        lifecycle.teardown_verified_count += 1
                        lifecycle.vast_teardown_confirmed = True
                    return True
            except Exception as exc:
                logger.warning("Failed to verify Vast destroy for {}: {}", instance_id, exc)
                return False
            time.sleep(3)
        logger.warning("Vast instance {} did not disappear during bounded destroy verification", instance_id)
        return False

    if config.VAST_INSTANCE_HOST and config.VAST_INSTANCE_PORT:
        logger.info("Using existing Vast instance: {}:{}", config.VAST_INSTANCE_HOST, config.VAST_INSTANCE_PORT)
        backend = VastInstanceBackend(
            host=config.VAST_INSTANCE_HOST,
            port=config.VAST_INSTANCE_PORT,
            timeout=config.VAST_REQUEST_TIMEOUT,
            worker_token=worker_token,
        )
        manager.wait_worker_ready(config.VAST_INSTANCE_HOST, config.VAST_INSTANCE_PORT)
        return backend, None, {
            "managed": False,
            "owned_instance_id": None,
            "worker_boot_id": f"{config.VAST_INSTANCE_HOST}:{config.VAST_INSTANCE_PORT}",
            "worker_ready_count": 1,
            "model_load_count": 1,
            "rent_count": 0,
            "num_gpus": 1,
        }

    logger.info(
        "Renting Vast.ai instance (vram>={} GB, max ${}/hr, gpu_choices={})...",
        config.VAST_MIN_VRAM_GB,
        config.VAST_MAX_PRICE_PER_HOUR,
        config.VAST_NUM_GPUS_CHOICES,
    )
    cost_caps = tuple(config.VAST_ESTIMATED_TOTAL_COST_FALLBACKS or ()) or (config.VAST_MAX_ESTIMATED_TOTAL_COST,)
    tried_machines: set[int] = set()
    last_error: Exception | None = None
    backend = None
    instance = None
    selected_num_gpus = 1
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
                n_images=n_images_for_offer,
                model_load_wall_seconds=config.VAST_MODEL_LOAD_WALL_SECONDS,
                exclude_machine_ids=tried_machines,
                num_gpus_choices=config.VAST_NUM_GPUS_CHOICES,
                min_cpu_ram_per_gpu_gb=config.VAST_MIN_CPU_RAM_PER_GPU_GB,
                min_cpu_cores_per_gpu=config.VAST_MIN_CPU_CORES_PER_GPU,
            )
        except RuntimeError as exc:
            last_error = exc
            logger.warning("No Vast offer under estimated total cost cap $%.2f: {}", total_cap, exc)
            continue
        selected_num_gpus = int(offer.get("_num_gpus", 1))
        tried_machines.add(int(offer.get("machine_id")))
        instance = manager.rent(
            offer_id=offer["id"],
            image=config.VAST_WORKER_IMAGE,
            env_vars=env_vars,
            disk_gb=max(60.0, config.VAST_DISK_GB),
            num_gpus=selected_num_gpus,
        )
        try:
            instance = manager.wait_until_running(instance.instance_id, timeout=config.VAST_INSTANCE_RUNNING_TIMEOUT)
            if not instance.direct_port:
                instance = manager.wait_for_port(instance.instance_id, timeout=config.VAST_PORT_MAPPING_TIMEOUT)
            manager.deploy_worker(
                instance,
                hf_token=config.VAST_HF_TOKEN,
                model_revision=revision,
                worker_token=worker_token,
                custom_image=config.VAST_WORKER_CUSTOM_IMAGE,
            )
            manager.wait_worker_ready(instance.public_ipaddr, instance.direct_port or config.VAST_WORKER_PORT, timeout=600)
            backend = VastInstanceBackend(
                host=instance.public_ipaddr,
                port=instance.direct_port or config.VAST_WORKER_PORT,
                timeout=config.VAST_REQUEST_TIMEOUT,
                worker_token=worker_token,
            )
            break
        except Exception as exc:
            last_error = exc
            logger.warning("Vast instance failed to start or warm up: {}", exc)
            if instance is not None:
                destroy_and_verify(instance.instance_id)
            instance = None
            backend = None
            continue
    if backend is None:
        raise RuntimeError(last_error or "No Vast.ai offer found under configured total cost caps")

    def teardown():
        logger.info("Destroying Vast instance {}...", instance.instance_id)
        destroy_and_verify(instance.instance_id)

    return backend, teardown, {
        "managed": True,
        "owned_instance_id": int(instance.instance_id),
        "worker_boot_id": f"{instance.instance_id}:{instance.public_ipaddr}:{instance.direct_port or config.VAST_WORKER_PORT}",
        "worker_ready_count": 1,
        "model_load_count": 1,
        "rent_count": 1,
        "num_gpus": selected_num_gpus,
    }


def open_backend_with_metadata(backend_name: str | None = None, *, planned_image_count: int | None = None):
    target_backend = backend_name or config.IMAGE_BACKEND
    if target_backend == "vast_instance":
        return _build_vast_backend(planned_image_count=planned_image_count)
    backend, teardown = _build_runpod_backend()
    return backend, teardown, {
        "managed": bool(teardown),
        "owned_instance_id": None,
        "worker_boot_id": "",
        "worker_ready_count": 0,
        "model_load_count": 0,
        "rent_count": 0,
        "num_gpus": 1,
    }


def run(
    video_id: str,
    n_override: int | None = None,
    *,
    backend_override=None,
    manage_backend: bool = True,
    lifecycle=None,
    include_thumbnails: bool = True,
) -> None:
    from image_generation.production import generate_scene_images, pending_scene_prompts, regenerate_failed_scenes

    video_dir = Path(config.OUTPUT_DIR) / video_id
    prompts_path = video_dir / "image_prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(f"image_prompts.json not found: {prompts_path}")

    prompts: list[dict] = json.loads(prompts_path.read_text(encoding="utf-8"))
    if n_override is not None:
        prompts = prompts[:n_override]
    completed = _load_progress(video_dir / "generation_log.json", video_dir)
    remaining = [item for item in prompts if f"{item['index']:03d}" not in completed]
    logger.info(
        "Backend: {}  |  Scenes: {}/{} done, {} remaining",
        config.IMAGE_BACKEND,
        len(completed),
        len(prompts),
        len(remaining),
    )
    effective_backend = backend_override
    teardown = None
    if effective_backend is None and config.IMAGE_BACKEND == "vast_instance":
        effective_backend, teardown, _meta = open_backend_with_metadata("vast_instance")
    try:
        result = generate_scene_images(
            video_id,
            backend_override=effective_backend,
            manage_backend=False if effective_backend is not None else manage_backend,
            lifecycle=lifecycle,
            n_override=n_override,
        )
        retry_result = regenerate_failed_scenes(
            video_id,
            backend_override=effective_backend,
            manage_backend=False,
            lifecycle=lifecycle,
            n_override=n_override,
        )
        fail = result["scene_fail"] + retry_result["scene_fail"]
        if include_thumbnails and (video_dir / config.PUBLISHING_DIRNAME / "thumbnail_prompts.json").exists():
            from steps.thumbnails import generate_thumbnail_assets, generate_thumbnail_backgrounds

            logger.info("Thumbnail prompts found - generating publishing thumbnails")
            generate_thumbnail_backgrounds(
                video_id,
                backend_override=effective_backend,
                manage_backend=False if effective_backend is not None else manage_backend,
                lifecycle=lifecycle,
            )
    finally:
        if teardown:
            teardown()
    if include_thumbnails and (video_dir / config.PUBLISHING_DIRNAME / "thumbnail_prompts.json").exists():
        from steps.thumbnails import generate_thumbnail_assets

        generate_thumbnail_assets(video_id)
    remaining_prompts = pending_scene_prompts(video_id, n_override=n_override)
    if not remaining_prompts:
        logger.info("All scenes already generated.")
    logger.info("Done: {} initial ok, {} retry ok, {} failed", result["scene_ok"], retry_result["scene_ok"], fail)
    if remaining_prompts:
        pending_ids = ", ".join(f"{int(item['index']):03d}" for item in remaining_prompts)
        raise RuntimeError(f"Scene generation incomplete; pending scene IDs: {pending_ids}")
    if fail:
        raise RuntimeError(f"Scene generation failed for {fail} scene attempt(s)")
