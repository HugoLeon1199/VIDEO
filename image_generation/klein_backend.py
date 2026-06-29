"""HTTP backend for the FLUX.2-klein-9B-KV-FP8 experimental worker.

Extends VastInstanceBackend with:
- scene classification → decide if img2img applies
- reference image loading → encode to base64 and inject into request
- character-scene routing (character scenes: img2img; env/object: text-to-image)

Production FLUX.1-dev 12B path is NOT changed.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from loguru import logger

from image_generation.base_backend import BaseImageBackend
from image_generation.character_bible import detect_characters_in_text
from image_generation.reference_selector import ReferenceDecision, load_master_manifest, select_reference
from image_generation.scene_classifier import classify_scene
from image_generation.schemas import SceneRequest, SceneResult
from image_generation.vast_backend import VastInstanceBackend


def _encode_image_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


class KleinBackend(BaseImageBackend):
    """FLUX.2-klein-9B-KV-FP8 backend with optional img2img reference injection.

    For character scenes: loads a fixed reference image from master_seed_dir and
    sends it as img2img_base64 at the configured strength.
    For environment/object/diagram scenes: text-to-image only (no reference).
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout: int = 300,
        output_root: str | Path = "output",
        worker_token: str = "",
        master_seed_dir: Optional[str] = None,
        steps_t2i: int = 4,
        steps_img2img: int = 8,
        strength_character: float = 0.65,
        strength_continuity: float = 0.55,
    ):
        self._inner = VastInstanceBackend(
            host=host,
            port=port,
            timeout=timeout,
            output_root=output_root,
            worker_token=worker_token,
        )
        self._master_seed_dir = master_seed_dir
        self._master_manifest: dict = {}
        if master_seed_dir:
            self._master_manifest = load_master_manifest(master_seed_dir)

        self._steps_t2i = steps_t2i
        self._steps_img2img = steps_img2img
        self._strength_character = strength_character
        self._strength_continuity = strength_continuity

        # State for same-scene continuity chaining
        self._previous_image: Optional[str] = None
        self._previous_scene_text: Optional[str] = None
        self._previous_continuity_group: Optional[str] = None
        self._chain_depth: int = 0

    def health_check(self) -> bool:
        return self._inner.health_check()

    def generate(self, request: SceneRequest) -> SceneResult:
        scene_text = request.prompt

        # Classify scene to decide mode
        chars = detect_characters_in_text(scene_text)
        continuity_group = self._infer_continuity_group(scene_text)
        classification = classify_scene(
            scene_text,
            previous_scene_text=self._previous_scene_text,
            characters=chars,
            continuity_group=continuity_group,
            previous_continuity_group=self._previous_continuity_group,
        )

        decision = select_reference(
            classification=classification,
            previous_scene_image=self._previous_image,
            master_seed_dir=self._master_seed_dir,
            master_seed_manifest=self._master_manifest,
            chain_depth=self._chain_depth,
        )

        # Build augmented request
        augmented = SceneRequest(
            video_id=request.video_id,
            scene_id=request.scene_id,
            prompt=request.prompt,
            clip_prompt=request.clip_prompt,
            global_style=request.global_style,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            guidance_scale=request.guidance_scale,
            candidate_seeds=request.candidate_seeds,
            output_format=request.output_format,
            quality=request.quality,
            output_mode=request.output_mode,
        )

        if decision.mode == "img2img" and decision.reference_path:
            try:
                augmented.img2img_base64 = _encode_image_b64(decision.reference_path)
                augmented.strength = decision.strength
                augmented.steps = self._steps_img2img
                logger.info(
                    "Klein img2img: scene={} ref={} strength={:.2f} chain={}",
                    request.scene_id, decision.reference_source,
                    decision.strength, decision.chain_depth,
                )
            except OSError as exc:
                logger.warning("Klein: reference load failed ({}), falling back to t2i: {}", decision.reference_path, exc)
                augmented.steps = self._steps_t2i
        else:
            augmented.steps = self._steps_t2i
            logger.info(
                "Klein t2i: scene={} reason={}",
                request.scene_id, decision.reset_reason or "no_reference",
            )

        result = self._inner.generate(augmented)

        # Update continuity state from best candidate
        if result.candidates:
            best = result.candidates[0]
            if best.local_path and Path(best.local_path).exists():
                self._previous_image = best.local_path
                self._chain_depth = (decision.chain_depth + 1) if decision.mode == "img2img" else 0
            else:
                self._previous_image = None
                self._chain_depth = 0
        else:
            self._previous_image = None
            self._chain_depth = 0

        self._previous_scene_text = scene_text
        self._previous_continuity_group = continuity_group
        return result

    @staticmethod
    def _infer_continuity_group(text: str) -> Optional[str]:
        import re
        # Same character in same broad setting = same group
        char_match = re.search(r'\b(karo|luma|the man|the woman)\b', text, re.I)
        env_match = re.search(r'\b(cave|campfire|forest|savanna|valley|shore)\b', text, re.I)
        if char_match and env_match:
            return f"{char_match.group(1).lower()}_{env_match.group(1).lower()}"
        return None
