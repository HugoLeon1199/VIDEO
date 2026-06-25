"""
Unit tests for RunPod Serverless backend — no real GPU required.
All RunPod HTTP calls are mocked.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("RUNPOD_API_KEY", "test_key_abcdef")
os.environ.setdefault("RUNPOD_ENDPOINT_ID", "test_endpoint_123")
os.environ.setdefault("IMAGE_OUTPUT_ROOT", tempfile.mkdtemp())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_webp(width: int = 64, height: int = 36) -> bytes:
    img = Image.new("RGB", (width, height), color=(120, 60, 30))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=80)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_candidate_info(index: int = 1, seed: int = 11001) -> tuple[dict, bytes]:
    data = _make_webp()
    info = {
        "candidate_index": index,
        "seed": seed,
        "mime_type": "image/webp",
        "width": 64,
        "height": 36,
        "sha256": _sha(data),
        "generation_seconds": 1.5,
        "base64": _b64(data),
    }
    return info, data


# ---------------------------------------------------------------------------
# schemas.py — input validation
# ---------------------------------------------------------------------------

class TestValidateInput:
    def _run(self, overrides: dict = {}):
        from serverless_worker.schemas import validate_input
        base = {
            "video_id": "test-video",
            "scene_id": "001",
            "mode": "text_to_image",
            "prompt": "a cave painting",
            "width": 1024,
            "height": 576,
            "steps": 4,
            "guidance_scale": 1.0,
            "candidate_seeds": [11001],
            "output_format": "WEBP",
            "quality": 92,
            "output_mode": "base64",
        }
        base.update(overrides)
        return validate_input(base)

    def test_valid_input_passes(self):
        params, errors = self._run()
        assert not errors
        assert params["prompt"] == "a cave painting"

    def test_empty_prompt_rejected(self):
        _, errors = self._run({"prompt": "   "})
        assert any("prompt" in e for e in errors)

    def test_unsupported_mode_rejected(self):
        _, errors = self._run({"mode": "inpaint"})
        assert any("mode" in e for e in errors)

    def test_bad_dimensions_rejected(self):
        _, errors = self._run({"width": 9999})
        assert any("width" in e for e in errors)

    def test_negative_width_rejected(self):
        _, errors = self._run({"width": -1})
        assert any("width" in e for e in errors)

    def test_too_many_seeds_rejected(self):
        _, errors = self._run({"candidate_seeds": list(range(20))})
        assert any("candidate_seeds" in e for e in errors)

    def test_negative_seed_rejected(self):
        _, errors = self._run({"candidate_seeds": [-1]})
        assert any("seed" in e for e in errors)

    def test_unsupported_format_rejected(self):
        _, errors = self._run({"output_format": "BMP"})
        assert any("output_format" in e for e in errors)

    def test_invalid_output_mode_rejected(self):
        _, errors = self._run({"output_mode": "s3"})
        assert any("output_mode" in e for e in errors)

    def test_quality_out_of_range_rejected(self):
        _, errors = self._run({"quality": 0})
        assert any("quality" in e for e in errors)

    def test_deterministic_seeds_preserved(self):
        params, _ = self._run({"candidate_seeds": [111, 222, 333]})
        assert params["candidate_seeds"] == [111, 222, 333]


# ---------------------------------------------------------------------------
# runpod_client.py — job submission and polling
# ---------------------------------------------------------------------------

class TestRunPodClient:
    def _client(self):
        from image_generation.runpod_client import RunPodClient
        return RunPodClient(api_key="test_key", endpoint_id="ep123")

    def test_submit_returns_job_id(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": "job_abc"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp):
            job_id = client.submit({"prompt": "test"})

        assert job_id == "job_abc"

    def test_submit_retries_on_network_error(self):
        import httpx as _httpx
        client = self._client()
        mock_ok = MagicMock()
        mock_ok.json.return_value = {"id": "job_retry"}
        mock_ok.raise_for_status = MagicMock()

        call_count = 0
        def side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise _httpx.RequestError("timeout")
            return mock_ok

        with patch("httpx.post", side_effect=side_effect):
            with patch("time.sleep"):
                job_id = client.submit({"prompt": "test"})

        assert job_id == "job_retry"
        assert call_count == 2

    def test_poll_returns_on_completed(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "COMPLETED", "output": {"images": []}}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            with patch("time.sleep"):
                result = client.poll_until_done("job_xyz")

        assert result["status"] == "COMPLETED"

    def test_poll_raises_on_failed(self):
        from image_generation.exceptions import JobFailedError
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "FAILED", "error": "OOM"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(JobFailedError, match="FAILED"):
                client.poll_until_done("job_xyz")

    def test_poll_raises_timeout(self):
        from image_generation.exceptions import JobTimeoutError
        client = self._client()
        client._timeout = 0  # instant timeout

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "IN_PROGRESS"}
        mock_resp.raise_for_status = MagicMock()

        cancel_resp = MagicMock()
        cancel_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            with patch("httpx.post", return_value=cancel_resp):
                with patch("time.sleep"):
                    with pytest.raises(JobTimeoutError):
                        client.poll_until_done("job_xyz")

    def test_cancel_called_on_timeout(self):
        from image_generation.exceptions import JobTimeoutError
        client = self._client()
        client._timeout = 0

        mock_get = MagicMock()
        mock_get.json.return_value = {"status": "IN_PROGRESS"}
        mock_get.raise_for_status = MagicMock()

        cancelled = []
        mock_post = MagicMock()
        mock_post.raise_for_status = MagicMock()

        def post_side(*a, **kw):
            cancelled.append(True)
            return mock_post

        with patch("httpx.get", return_value=mock_get):
            with patch("httpx.post", side_effect=post_side):
                with patch("time.sleep"):
                    with pytest.raises(JobTimeoutError):
                        client.poll_until_done("job_xyz")

        assert cancelled, "cancel() was not called on timeout"

    def test_secrets_not_in_debug_output(self, capsys):
        import logging
        client = self._client()
        logging.getLogger("image_generation.runpod_client").setLevel(logging.DEBUG)
        # Just constructing the client should never print the full key
        captured = capsys.readouterr()
        assert "test_key" not in captured.out
        assert "test_key" not in captured.err


# ---------------------------------------------------------------------------
# runpod_serverless_backend.py — image processing
# ---------------------------------------------------------------------------

class TestRunPodServerlessBackend:
    def _backend(self, mock_client):
        from image_generation.runpod_serverless_backend import RunPodServerlessBackend
        return RunPodServerlessBackend(client=mock_client)

    def _mock_client(self, images: list[dict], errors: list[str] = []) -> MagicMock:
        client = MagicMock()
        client.submit.return_value = "job_mock"
        client.poll_until_done.return_value = {
            "status": "COMPLETED",
            "output": {
                "video_id": "v",
                "scene_id": "001",
                "model": "black-forest-labs/FLUX.2-klein-4B",
                "mode": "text_to_image",
                "duration_seconds": 3.0,
                "images": images,
                "errors": errors,
            },
        }
        return client

    def _req(self) -> "SceneRequest":
        from image_generation.schemas import SceneRequest
        return SceneRequest(
            video_id="test-video",
            scene_id="001",
            prompt="cave painting",
            candidate_seeds=[11001],
        )

    def test_successful_generation_saves_file(self):
        info, data = _make_candidate_info(1, 11001)
        client = self._mock_client([info])
        backend = self._backend(client)
        result = backend.generate(self._req())

        assert len(result.candidates) == 1
        assert result.errors == []
        c = result.candidates[0]
        assert c.seed == 11001
        assert Path(c.local_path).exists()

    def test_sha256_mismatch_raises_error(self):
        info, _ = _make_candidate_info(1, 11001)
        info["sha256"] = "deadbeef" * 8  # wrong
        client = self._mock_client([info])
        backend = self._backend(client)
        result = backend.generate(self._req())

        assert len(result.candidates) == 0
        assert any("SHA" in e or "sha" in e.lower() for e in result.errors)

    def test_corrupted_image_rejected(self):
        info = {
            "candidate_index": 1,
            "seed": 11001,
            "mime_type": "image/webp",
            "width": 64,
            "height": 36,
            "sha256": _sha(b"notanimage"),
            "generation_seconds": 1.0,
            "base64": _b64(b"notanimage"),
        }
        client = self._mock_client([info])
        backend = self._backend(client)
        result = backend.generate(self._req())

        assert len(result.candidates) == 0
        assert result.errors

    def test_partial_failure_keeps_successful_candidates(self):
        good_info, _ = _make_candidate_info(1, 11001)
        bad_info = {
            "candidate_index": 2,
            "seed": 11002,
            "mime_type": "image/webp",
            "width": 64,
            "height": 36,
            "sha256": "badhash",
            "generation_seconds": 1.0,
            "base64": _b64(b"corrupt"),
        }
        client = self._mock_client([good_info, bad_info])
        backend = self._backend(client)
        result = backend.generate(self._req())

        assert len(result.candidates) == 1
        assert result.candidates[0].seed == 11001

    def test_atomic_file_write(self):
        info, data = _make_candidate_info(1, 11001)
        client = self._mock_client([info])
        backend = self._backend(client)
        result = backend.generate(self._req())

        path = Path(result.candidates[0].local_path)
        assert path.exists()
        assert path.suffix == ".webp"
        # No .tmp files left
        assert not list(path.parent.glob("*.tmp"))

    def test_resume_skips_completed_scenes(self):
        """generate_images.py skips scenes already in gen_log with status=completed."""
        from scripts.generate_images import _scene_done
        selected = Path(os.environ["IMAGE_OUTPUT_ROOT"]) / "test-video" / "images" / "img_001.png"
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_bytes(_make_webp())

        log = {"001": {"status": "completed", "candidates_saved": 3, "selected_image": str(selected)}}
        assert _scene_done(log, "001", 3) is True
        assert _scene_done(log, "001", 4) is False
        assert _scene_done(log, "002", 3) is False

    def test_resume_requires_selected_image(self):
        """Completed candidate folders alone are not enough for step 6 render."""
        from scripts.generate_images import _scene_done
        log = {"001": {"status": "completed", "candidates_saved": 3, "selected_image": "missing.png"}}
        assert _scene_done(log, "001", 3) is False

    def test_sidecar_json_written(self):
        info, _ = _make_candidate_info(1, 11001)
        client = self._mock_client([info])
        backend = self._backend(client)
        result = backend.generate(self._req())

        img_path = Path(result.candidates[0].local_path)
        meta_path = img_path.with_suffix(".json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["seed"] == 11001
        assert meta["scene_id"] == "001"

    def test_promote_candidate_creates_render_png(self):
        from image_generation.runpod_serverless_backend import promote_candidate_to_render_image

        info, _ = _make_candidate_info(1, 11001)
        client = self._mock_client([info])
        backend = self._backend(client)
        result = backend.generate(self._req())

        selected = promote_candidate_to_render_image(
            result.candidates[0],
            video_id="test-video",
            scene_id="001",
            output_root=os.environ["IMAGE_OUTPUT_ROOT"],
        )

        selected_path = Path(selected)
        assert selected_path.exists()
        assert selected_path.name == "img_001.png"
        with Image.open(selected_path) as img:
            assert img.size == (64, 36)
