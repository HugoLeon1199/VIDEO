from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

import config
from image_generation.schemas import SceneRequest


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_log(path: Path) -> dict:
    value = _read_json(path, {})
    return value if isinstance(value, dict) else {}


def _scene_done(log: dict, video_dir: Path, scene_id: str) -> bool:
    entry = log.get(scene_id, {})
    if entry.get("status") != "completed":
        return False
    return _canonical_image_path(video_dir, scene_id).exists()


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
    failed_ids = {
        scene_id
        for scene_id, entry in log.items()
        if entry.get("status") in {"failed", "partial"}
    }
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
        thumb = video_dir / config.PUBLISHING_DIRNAME / "thumbnails" / f"thumbnail_{concept_id:02d}.jpg"
        if not (bg.exists() and thumb.exists()):
            pending.append(item)
    return pending


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
    lifecycle: VastLifecycle = field(default_factory=VastLifecycle)
    backend: Any = None
    teardown = None
    owned_instance_id: int | None = None
    managed: bool = False

    def __enter__(self):
        from steps import generate_images as step_generate_images

        if config.IMAGE_BACKEND != "vast_instance":
            raise RuntimeError("VastSession requires IMAGE_BACKEND=vast_instance")
        self.lifecycle.vast_session_count += 1
        self.backend, self.teardown, meta = step_generate_images.open_backend_with_metadata("vast_instance")
        self.lifecycle.backend_create_count += 1
        self.lifecycle.worker_boot_id = meta.get("worker_boot_id", "")
        self.lifecycle.worker_ready_count += int(meta.get("worker_ready_count", 0))
        self.lifecycle.model_load_count += int(meta.get("model_load_count", 0))
        self.lifecycle.rent_count += int(meta.get("rent_count", 0))
        self.owned_instance_id = meta.get("owned_instance_id")
        self.managed = bool(meta.get("managed", False))
        return self

    def verify_destroyed(self) -> bool:
        if not self.owned_instance_id:
            return False
        from image_generation.vast_manager import VastManager

        manager = VastManager(api_key=config.VAST_API_KEY, worker_port=config.VAST_WORKER_PORT)
        try:
            for item in manager.list_instances():
                if int(item.get("id", 0)) == int(self.owned_instance_id):
                    return False
        except Exception:
            return False
        return True

    def __exit__(self, exc_type, exc, tb):
        if self.teardown and self.managed:
            self.lifecycle.teardown_attempt_count += 1
            self.teardown()
            if self.verify_destroyed():
                self.lifecycle.teardown_verified_count += 1
                self.lifecycle.vast_teardown_confirmed = True
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
    ok = 0
    fail = 0
    try:
        for prompt in prompts:
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
                gen_log[request.scene_id] = {
                    "status": "completed" if selected_image and not result.errors else "partial",
                    "candidates_saved": len(result.candidates),
                    "selected_image": selected_image,
                    "errors": result.errors,
                    "job_id": result.job_id,
                    "duration_seconds": result.duration_seconds,
                }
                ok += 1
            except Exception as exc:
                logger.error("Scene {} failed: {}", request.scene_id, exc)
                gen_log[request.scene_id] = {"status": "failed", "error": str(exc)}
                fail += 1
            _write_json(log_path, gen_log)
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
