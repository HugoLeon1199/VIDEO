"""
Sequential scene chaining: automatically decides text-to-image vs img2img
for each scene in a story sequence.

Rules:
  - First scene (or any scene marked new_background=True): text-to-image
  - Same background, character pose change (strength 0.5-0.6): img2img light
  - Same background, camera angle / composition change (strength 0.7-0.75): img2img heavy

Usage:
    from image_generation.scene_chain import SceneChain, ChainedScene

    scenes = [
        ChainedScene(scene_id="001", prompt="...", transition="new"),
        ChainedScene(scene_id="002", prompt="...", transition="pose"),
        ChainedScene(scene_id="003", prompt="...", transition="angle"),
    ]
    chain = SceneChain(backend, video_id="my-video", steps=22, guidance_scale=3.5)
    results = chain.run(scenes)
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from image_generation.runpod_serverless_backend import RunPodServerlessBackend
from image_generation.schemas import SceneRequest, SceneResult

logger = logging.getLogger(__name__)

# Denoising strength presets
STRENGTH_POSE  = 0.55   # minor pose or arm movement — preserve bg + colors
STRENGTH_ANGLE = 0.72   # camera angle / composition shift — inherit palette only

FLAT2D_PREFIX = (
    "v3ct0r style, simple flat vector art, clean lines, solid colors, "
    "completely flat, no 3D shading, no gradients, "
    "prehistoric humans wearing animal skin clothing, "
)
FLAT2D_SUFFIX = (
    ", 2D educational illustration, dark muted palette, "
    "documentary flat graphic style, 16:9 composition"
)
FLAT2D_NEGATIVE = (
    "photorealistic, photograph, 3D render, CGI, shading, shadow, depth, gradient, "
    "nudity, bare chest, bare skin, exposed torso, extra limbs, "
    "deformed anatomy, text, watermark, logo"
)


@dataclass
class ChainedScene:
    scene_id: str
    prompt: str                                          # scene description only (no style prefix)
    transition: Literal["new", "pose", "angle"] = "new" # how this scene relates to the previous
    negative_prompt: str = ""
    # Override per-scene; None = use chain defaults
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None


@dataclass
class ChainResult:
    scene_id: str
    mode: str          # "text_to_image" | "img2img"
    strength: float    # 0.0 for t2i
    local_path: str    # path to selected candidate .webp
    duration_seconds: float
    errors: list[str] = field(default_factory=list)


class SceneChain:
    """
    Chains scenes sequentially, passing each output image as reference
    to the next scene when transition != 'new'.
    """

    def __init__(
        self,
        backend: RunPodServerlessBackend,
        video_id: str,
        width: int = 1024,
        height: int = 576,
        steps: int = 22,
        guidance_scale: float = 3.5,
        candidate_seeds: list[int] | None = None,
    ):
        self.backend = backend
        self.video_id = video_id
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.candidate_seeds = candidate_seeds or [11001]

    def run(self, scenes: list[ChainedScene]) -> list[ChainResult]:
        results: list[ChainResult] = []
        prev_image_b64: Optional[str] = None  # base64 of last generated image

        for scene in scenes:
            result = self._process_scene(scene, prev_image_b64)
            results.append(result)

            # Load the output image as reference for the next scene
            if result.local_path and not result.errors:
                prev_image_b64 = _load_as_base64(result.local_path)
            else:
                prev_image_b64 = None  # reset chain on error

        return results

    def _process_scene(
        self,
        scene: ChainedScene,
        prev_b64: Optional[str],
    ) -> ChainResult:
        use_img2img = (
            scene.transition in ("pose", "angle")
            and prev_b64 is not None
        )

        if use_img2img:
            strength = STRENGTH_POSE if scene.transition == "pose" else STRENGTH_ANGLE
            mode = "img2img"
        else:
            strength = 0.0
            mode = "text_to_image"

        full_prompt = FLAT2D_PREFIX + scene.prompt + FLAT2D_SUFFIX
        negative = scene.negative_prompt or FLAT2D_NEGATIVE

        req = SceneRequest(
            video_id=self.video_id,
            scene_id=scene.scene_id,
            prompt=full_prompt,
            negative_prompt=negative,
            width=self.width,
            height=self.height,
            steps=scene.steps or self.steps,
            guidance_scale=scene.guidance_scale or self.guidance_scale,
            candidate_seeds=self.candidate_seeds,
            output_format="WEBP",
            quality=92,
            output_mode="base64",
            img2img_base64=prev_b64 if use_img2img else None,
            strength=strength,
        )

        logger.info(
            "Scene %s | mode=%s strength=%.2f | %s",
            scene.scene_id, mode, strength, scene.prompt[:60]
        )

        scene_result: SceneResult = self.backend.generate(req)

        local_path = ""
        if scene_result.candidates:
            local_path = scene_result.candidates[0].local_path or ""

        return ChainResult(
            scene_id=scene.scene_id,
            mode=mode,
            strength=strength,
            local_path=local_path,
            duration_seconds=scene_result.duration_seconds,
            errors=scene_result.errors,
        )


def _load_as_base64(path: str) -> str:
    raw = Path(path).read_bytes()
    ext = Path(path).suffix.lstrip(".").lower()
    mime = "image/webp" if ext == "webp" else f"image/{ext}"
    return f"data:{mime};base64," + base64.b64encode(raw).decode()
