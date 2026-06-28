"""Shared data classes for image generation requests and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SceneRequest:
    video_id: str
    scene_id: str
    prompt: str
    clip_prompt: str = ""
    global_style: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 576
    steps: int = 4
    guidance_scale: float = 1.0
    candidate_seeds: list[int] = field(default_factory=lambda: [11001, 11002, 11003])
    output_format: str = "WEBP"
    quality: int = 92
    output_mode: str = "base64"
    # img2img fields — None means text-to-image mode
    img2img_base64: Optional[str] = None   # base64 of reference image
    strength: float = 0.75                 # denoising strength (0.5-0.75 typical)

    def to_runpod_input(self) -> dict:
        payload: dict = {
            "video_id": self.video_id,
            "scene_id": self.scene_id,
            "prompt": self.prompt,
            "clip_prompt": self.clip_prompt,
            "global_style": self.global_style,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "candidate_seeds": self.candidate_seeds,
            "output_format": self.output_format,
            "quality": self.quality,
            "output_mode": self.output_mode,
        }
        if self.img2img_base64 is not None:
            payload["img2img_base64"] = self.img2img_base64
            payload["strength"] = self.strength
        return payload


@dataclass
class CandidateResult:
    candidate_index: int
    seed: int
    width: int
    height: int
    sha256: str
    generation_seconds: float
    mime_type: str
    local_path: Optional[str] = None
    volume_path: Optional[str] = None
    base64_data: Optional[str] = None


@dataclass
class SceneResult:
    video_id: str
    scene_id: str
    model: str
    mode: str
    duration_seconds: float
    candidates: list[CandidateResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    job_id: Optional[str] = None
