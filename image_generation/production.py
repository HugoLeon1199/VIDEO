from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from loguru import logger

import config
from image_generation.schemas import SceneRequest
from steps.creative_package import _atomic_write_json


def _video_dir(video_id: str) -> Path:
    return Path(config.OUTPUT_DIR) / video_id


def _canonical_image_path(video_dir: Path, scene_id: str) -> Path:
    return video_dir / "images" / f"img_{int(scene_id):03d}.png"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_json(path, payload)


def _load_log(path: Path) -> dict:
    value = _read_json(path, {})
    return value if isinstance(value, dict) else {}


def _image_is_readable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
        return width > 0 and height > 0
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def _scene_artifact_ready(entry: dict, video_dir: Path, scene_id: str) -> bool:
    selected_image = str(entry.get("selected_image", "") or "").strip()
    image_path = _canonical_image_path(video_dir, scene_id)
    if entry.get("status") != "completed":
        return False
    if entry.get("errors"):
        return False
    if not selected_image:
        return False
    return _image_is_readable(image_path)


def _scene_done(log: dict, video_dir: Path, scene_id: str) -> bool:
    entry = log.get(scene_id, {})
    return _scene_artifact_ready(entry, video_dir, scene_id)


def _load_prompts(video_id: str, n_override: int | None = None) -> list[dict]:
    prompts_path = _video_dir(video_id) / "image_prompts.json"
    prompts = _read_json(prompts_path, [])
    if not isinstance(prompts, list):
        raise RuntimeError(f"image_prompts.json must contain a list: {prompts_path}")
    if n_override is not None:
        prompts = prompts[:n_override]
    return prompts


def pending_scene_prompts(video_id: str, n_override: int | None = None) -> list[dict]:
    video_dir = _video_dir(video_id)
    prompts = _load_prompts(video_id, n_override=n_override)
    log = _load_log(video_dir / "generation_log.json")
    return [item for item in prompts if not _scene_done(log, video_dir, f"{int(item['index']):03d}")]


def pending_failed_scene_prompts(video_id: str, n_override: int | None = None) -> list[dict]:
    video_dir = _video_dir(video_id)
    prompts = _load_prompts(video_id, n_override=n_override)
    log = _load_log(video_dir / "generation_log.json")
    failed_ids: set[str] = set()
    for item in prompts:
        scene_id = f"{int(item['index']):03d}"
        entry = log.get(scene_id, {})
        if entry.get("status") in {"failed", "partial"}:
            failed_ids.add(scene_id)
            continue
        if not _scene_artifact_ready(entry, video_dir, scene_id):
            failed_ids.add(scene_id)
    return [item for item in prompts if f"{int(item['index']):03d}" in failed_ids]


def pending_thumbnail_prompts(video_id: str, regenerate: list[int] | None = None) -> list[dict]:
    video_dir = _video_dir(video_id)
    prompts = _read_json(video_dir / config.PUBLISHING_DIRNAME / "thumbnail_prompts.json", [])
    if not isinstance(prompts, list):
        return []
    regenerate_set = {int(value) for value in regenerate or []}
    if regenerate_set:
        return [item for item in prompts if int(item["concept_id"]) in regenerate_set]
    pending = []
    for item in prompts:
        concept_id = int(item["concept_id"])
        bg = video_dir / config.PUBLISHING_DIRNAME / "thumbnails" / f"thumbnail_{concept_id:02d}_background.png"
        if not _image_is_readable(bg):
            pending.append(item)
    return pending


def compute_session_image_count(video_ids: list[str]) -> int:
    """Total pending images across all video IDs in a rental session.

    Counts pending scene candidates (scenes × seeds) plus pending thumbnail
    backgrounds. Already-complete images are excluded because pending_* filters
    them out.
    """
    total = 0
    for vid in video_ids:
        total += len(pending_scene_prompts(vid)) * len(config.IMAGE_CANDIDATE_SEEDS)
        total += len(pending_thumbnail_prompts(vid))
    return total


@dataclass
class VastLifecycle:
    vast_session_count: int = 0
    rent_count: int = 0
    backend_create_count: int = 0
    worker_boot_id: str = ""
    worker_ready_count: int = 0
    model_load_count: int = 0
    scene_request_count: int = 0
    thumbnail_request_count: int = 0
    teardown_attempt_count: int = 0
    teardown_verified_count: int = 0
    vast_teardown_confirmed: bool = False

    def to_summary(self) -> dict[str, Any]:
        return {
            "vast_session_count": self.vast_session_count,
            "rent_count": self.rent_count,
            "backend_create_count": self.backend_create_count,
            "worker_boot_id": self.worker_boot_id,
            "worker_ready_count": self.worker_ready_count,
            "model_load_count": self.model_load_count,
            "scene_request_count": self.scene_request_count,
            "thumbnail_request_count": self.thumbnail_request_count,
            "teardown_attempt_count": self.teardown_attempt_count,
            "teardown_verified_count": self.teardown_verified_count,
            "vast_teardown_confirmed": self.vast_teardown_confirmed,
        }


@dataclass
class VastSession(AbstractContextManager):
    planned_image_count: int | None = None   # session-wide pending image count
    video_ids: list[str] | None = None       # optional; used for telemetry
    lifecycle: VastLifecycle = field(default_factory=VastLifecycle)
    backend: Any = None
    teardown = None
    owned_instance_id: int | None = None
    managed: bool = False
    cleanup_error: str = ""
    num_gpus: int = 1                        # set from backend metadata after __enter__

    def __enter__(self):
        from steps import generate_images as step_generate_images

        if config.IMAGE_BACKEND != "vast_instance":
            raise RuntimeError("VastSession requires IMAGE_BACKEND=vast_instance")
        self.lifecycle.vast_session_count += 1
        self.backend, self.teardown, meta = step_generate_images.open_backend_with_metadata(
            "vast_instance",
            planned_image_count=self.planned_image_count,
        )
        self.lifecycle.backend_create_count += 1
        self.lifecycle.worker_boot_id = meta.get("worker_boot_id", "")
        self.lifecycle.worker_ready_count += int(meta.get("worker_ready_count", 0))
        self.lifecycle.model_load_count += int(meta.get("model_load_count", 0))
        self.lifecycle.rent_count += int(meta.get("rent_count", 0))
        self.owned_instance_id = meta.get("owned_instance_id")
        self.managed = bool(meta.get("managed", False))
        self.num_gpus = int(meta.get("num_gpus", 1))
        return self

    def verify_destroyed(self) -> bool:
        if not self.owned_instance_id:
            return False
        from image_generation.vast_manager import VastManager

        manager = VastManager(api_key=config.VAST_API_KEY, worker_port=config.VAST_WORKER_PORT)
        for _attempt in range(5):
            try:
                for item in manager.list_instances():
                    if int(item.get("id", 0)) == int(self.owned_instance_id):
                        break
                else:
                    return True
            except Exception:
                return False
            time.sleep(3)
        return False

    def __exit__(self, exc_type, exc, tb):
        if self.teardown and self.managed:
            self.lifecycle.teardown_attempt_count += 1
            try:
                self.teardown()
            except Exception as teardown_exc:
                self.cleanup_error = f"Vast teardown call failed: {teardown_exc}"
                if exc is None:
                    raise RuntimeError(self.cleanup_error) from teardown_exc
                logger.warning(self.cleanup_error)
                return False
            if self.verify_destroyed():
                self.lifecycle.teardown_verified_count += 1
                self.lifecycle.vast_teardown_confirmed = True
            else:
                self.cleanup_error = f"Vast instance {self.owned_instance_id} could not be destroy-verified"
                if exc is None:
                    raise RuntimeError(self.cleanup_error)
                logger.warning(self.cleanup_error)
        return False


def _scene_request_from_prompt(video_id: str, prompt: dict) -> SceneRequest:
    return SceneRequest(
        video_id=video_id,
        scene_id=f"{int(prompt['index']):03d}",
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


def generate_scene_images(
    video_id: str,
    *,
    backend_override=None,
    manage_backend: bool = True,
    lifecycle: VastLifecycle | None = None,
    n_override: int | None = None,
    prompt_subset: list[dict] | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    from image_generation.runpod_serverless_backend import promote_candidate_to_render_image
    from steps import generate_images as step_generate_images

    video_dir = _video_dir(video_id)
    prompts = list(prompt_subset) if prompt_subset is not None else pending_scene_prompts(video_id, n_override=n_override)
    log_path = video_dir / "generation_log.json"
    gen_log = _load_log(log_path)
    if not prompts:
        return {"scene_ok": 0, "scene_fail": 0, "processed_count": 0}
    backend = backend_override
    teardown = None
    owns_backend = False
    if backend is None:
        backend, teardown, _meta = step_generate_images.open_backend_with_metadata(config.IMAGE_BACKEND)
        owns_backend = manage_backend
        if lifecycle is not None:
            lifecycle.backend_create_count += 1

    # Concurrency: default to 1 (sequential) or the caller-supplied max_workers.
    # Callers that have a VastSession pass session.num_gpus as max_workers.
    concurrency = max(1, max_workers or 1)

    def _process_one(prompt: dict) -> tuple[str, dict]:
        request = _scene_request_from_prompt(video_id, prompt)
        if lifecycle is not None:
            lifecycle.scene_request_count += 1
        try:
            result = backend.generate(request)
            selected_image = ""
            if result.candidates:
                selected_image = promote_candidate_to_render_image(
                    result.candidates[0],
                    video_id=video_id,
                    scene_id=request.scene_id,
                )
            status = (
                "completed"
                if selected_image and not result.errors and _image_is_readable(
                    _canonical_image_path(video_dir, request.scene_id)
                )
                else "partial"
            )
            entry = {
                "status": status,
                "candidates_saved": len(result.candidates),
                "selected_image": selected_image,
                "errors": result.errors,
                "job_id": result.job_id,
                "duration_seconds": result.duration_seconds,
            }
            return request.scene_id, entry
        except Exception as exc:
            logger.error("Scene {} failed: {}", request.scene_id, exc)
            return request.scene_id, {"status": "failed", "error": str(exc)}

    ok = 0
    fail = 0
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_process_one, p): p for p in prompts}
            for fut in as_completed(futures):
                scene_id, entry = fut.result()
                gen_log[scene_id] = entry
                # Only the coordinator thread writes; lock-free because GIL protects
                # dict mutation and _write_json is atomic on disk.
                _write_json(log_path, gen_log)
                if entry.get("status") == "completed":
                    ok += 1
                else:
                    fail += 1
    finally:
        if owns_backend and teardown:
            teardown()
    return {"scene_ok": ok, "scene_fail": fail, "processed_count": len(prompts)}


def regenerate_failed_scenes(
    video_id: str,
    *,
    backend_override=None,
    manage_backend: bool = True,
    lifecycle: VastLifecycle | None = None,
    n_override: int | None = None,
) -> dict[str, Any]:
    prompts = pending_failed_scene_prompts(video_id, n_override=n_override)
    if not prompts:
        return {"scene_ok": 0, "scene_fail": 0, "processed_count": 0}
    return generate_scene_images(
        video_id,
        backend_override=backend_override,
        manage_backend=manage_backend,
        lifecycle=lifecycle,
        n_override=n_override,
        prompt_subset=prompts,
    )
