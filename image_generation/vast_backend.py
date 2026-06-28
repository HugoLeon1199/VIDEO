"""Vast.ai image generation backend.

Implements BaseImageBackend using a self-hosted FastAPI worker running on a
rented Vast.ai GPU instance.

Lifecycle (managed externally by VastManager):
  1. Caller rents instance + waits ready (vast_manager.py)
  2. Constructs VastInstanceBackend(host, port)
  3. Calls generate() per scene — thread-safe (stateless HTTP POST)
  4. Caller destroys instance after all scenes done
"""

from __future__ import annotations

import base64
import hashlib
import time
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

from image_generation.base_backend import BaseImageBackend
from image_generation.schemas import CandidateResult, SceneRequest, SceneResult


class VastInstanceBackend(BaseImageBackend):
    """Send generate requests to a FastAPI worker running on a Vast.ai instance."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout: int = 600,
        output_root: str | Path = "output",
        worker_token: str = "",
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.output_root = Path(output_root)
        token = worker_token.strip()
        self._worker_headers = {}
        if token:
            self._worker_headers = {
                "Authorization": f"Bearer {token}",
                "X-Worker-Token": token,
            }

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10, headers=self._worker_headers)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, request: SceneRequest) -> SceneResult:
        t0 = time.time()
        payload = {
            "video_id": request.video_id,
            "scene_id": request.scene_id,
            "prompt": request.prompt,
            "clip_prompt": request.clip_prompt or request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "candidate_seeds": request.candidate_seeds,
            "output_format": request.output_format,
            "quality": request.quality,
        }
        if request.img2img_base64:
            payload["img2img_base64"] = request.img2img_base64
            payload["strength"] = request.strength

        try:
            resp = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=self.timeout,
                headers=self._worker_headers,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            duration = time.time() - t0
            return SceneResult(
                video_id=request.video_id,
                scene_id=request.scene_id,
                model="vast/flux",
                mode="vast_instance",
                duration_seconds=duration,
                errors=[str(e)],
            )

        duration = time.time() - t0
        candidates: list[CandidateResult] = []
        errors: list[str] = data.get("errors", [])

        images_dir = (
            self.output_root / request.video_id / "images"
            / f"scene_{int(request.scene_id):03d}"
        )
        images_dir.mkdir(parents=True, exist_ok=True)

        for idx, img_data in enumerate(data.get("images", [])):
            seed = img_data.get("seed", request.candidate_seeds[idx] if idx < len(request.candidate_seeds) else 0)
            b64 = img_data.get("image_base64", "")
            if not b64:
                errors.append(f"candidate {idx}: empty image_base64")
                continue

            raw = base64.b64decode(b64)
            sha = hashlib.sha256(raw).hexdigest()
            ext = request.output_format.lower()
            fname = f"candidate_{idx:02d}_seed_{seed}.{ext}"
            local_path = images_dir / fname
            local_path.write_bytes(raw)

            candidates.append(CandidateResult(
                candidate_index=idx,
                seed=seed,
                width=img_data.get("width", request.width),
                height=img_data.get("height", request.height),
                sha256=sha,
                generation_seconds=img_data.get("generation_seconds", 0.0),
                mime_type=f"image/{ext}",
                local_path=str(local_path),
                base64_data=b64,
            ))
            logger.info("Vast: saved {}", local_path)

        return SceneResult(
            video_id=request.video_id,
            scene_id=request.scene_id,
            model="vast/flux",
            mode="vast_instance",
            duration_seconds=duration,
            candidates=candidates,
            errors=errors,
        )
