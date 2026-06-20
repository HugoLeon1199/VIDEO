"""
RunPod Serverless backend — submits jobs, polls results, saves images locally.

Handles:
- base64 decode + SHA-256 verify + Pillow validation
- atomic file write (temp + rename)
- per-candidate failure isolation (one bad candidate keeps the rest)
- job_id persistence for resume
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from image_generation.base_backend import BaseImageBackend
from image_generation.exceptions import ChecksumMismatchError
from image_generation.runpod_client import RunPodClient
from image_generation.schemas import CandidateResult, SceneRequest, SceneResult

logger = logging.getLogger(__name__)

OUTPUT_ROOT = os.environ.get("IMAGE_OUTPUT_ROOT", "output")


class RunPodServerlessBackend(BaseImageBackend):
    def __init__(self, client: Optional[RunPodClient] = None):
        self._client = client or RunPodClient()

    def health_check(self) -> bool:
        return self._client.health_check()

    def generate(self, request: SceneRequest) -> SceneResult:
        job_input = request.to_runpod_input()
        t0 = time.time()

        job_id = self._client.submit(job_input)
        logger.info("Scene %s submitted as job %s", request.scene_id, job_id)

        raw = self._client.poll_until_done(job_id)
        output = raw.get("output", {})

        candidates: list[CandidateResult] = []
        errors: list[str] = list(output.get("errors", []))

        for img_info in output.get("images", []):
            try:
                candidate = self._process_candidate(img_info, request)
                candidates.append(candidate)
            except Exception as e:
                msg = f"Candidate {img_info.get('candidate_index')} processing failed: {e}"
                logger.error(msg)
                errors.append(msg)

        return SceneResult(
            video_id=request.video_id,
            scene_id=request.scene_id,
            model=output.get("model", "unknown"),
            mode=output.get("mode", request.output_mode),
            duration_seconds=round(time.time() - t0, 2),
            candidates=candidates,
            errors=errors,
            job_id=job_id,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_candidate(self, img_info: dict, request: SceneRequest) -> CandidateResult:
        candidate_index = img_info["candidate_index"]
        seed = img_info["seed"]
        expected_sha = img_info["sha256"]
        fmt = request.output_format.lower()

        if request.output_mode == "base64":
            raw_b64 = img_info["base64"]
            img_bytes = base64.b64decode(raw_b64)

            # Verify checksum
            actual_sha = hashlib.sha256(img_bytes).hexdigest()
            if actual_sha != expected_sha:
                raise ChecksumMismatchError(
                    f"SHA mismatch candidate {candidate_index}: expected {expected_sha}, got {actual_sha}"
                )

            # Validate with Pillow
            try:
                pil = Image.open(io.BytesIO(img_bytes))
                pil.verify()
                # Re-open after verify (verify closes the file)
                pil = Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                raise ValueError(f"Corrupted image data for candidate {candidate_index}: {e}")

            local_path = self._save_candidate(
                img_bytes, request.video_id, request.scene_id,
                candidate_index, seed, fmt,
            )
        else:
            local_path = None

        return CandidateResult(
            candidate_index=candidate_index,
            seed=seed,
            width=img_info.get("width", request.width),
            height=img_info.get("height", request.height),
            sha256=expected_sha,
            generation_seconds=img_info.get("generation_seconds", 0.0),
            mime_type=img_info.get("mime_type", f"image/{fmt}"),
            local_path=local_path,
            volume_path=img_info.get("volume_path"),
        )

    @staticmethod
    def _save_candidate(
        data: bytes,
        video_id: str,
        scene_id: str,
        candidate_index: int,
        seed: int,
        fmt: str,
    ) -> str:
        out_dir = Path(OUTPUT_ROOT) / video_id / "images" / f"scene_{scene_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"candidate_{candidate_index:02d}_seed_{seed}.{fmt}"
        dest = out_dir / filename
        meta_dest = out_dir / f"candidate_{candidate_index:02d}_seed_{seed}.json"

        # Atomic write
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        # Write sidecar metadata
        meta = {
            "video_id": video_id,
            "scene_id": scene_id,
            "candidate_index": candidate_index,
            "seed": seed,
            "sha256": hashlib.sha256(data).hexdigest(),
            "file": str(dest),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_dest.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.info("Saved %s", dest)
        return str(dest)
