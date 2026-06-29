from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import config
from image_generation.vast_manager import VastInstance, VastManager
from image_generation.schemas import CandidateResult, SceneResult
from steps import generate_images, thumbnails


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class _FakeBackend:
    def __init__(self, candidate_path: Path):
        self.calls: list[str] = []
        self.candidate_path = candidate_path

    def generate(self, request):
        self.calls.append(request.scene_id)
        return SceneResult(
            video_id=request.video_id,
            scene_id=request.scene_id,
            model="fake",
            mode="fake",
            duration_seconds=0.1,
            candidates=[
                CandidateResult(
                    candidate_index=0,
                    seed=11,
                    width=1024,
                    height=576,
                    sha256="sha",
                    generation_seconds=0.1,
                    mime_type="image/png",
                    local_path=str(self.candidate_path),
                )
            ],
        )


def test_step5_reuses_one_vast_backend_for_scenes_and_thumbnails(tmp_path: Path, monkeypatch) -> None:
    import image_generation.runpod_serverless_backend as runpod_backend

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(generate_images.config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(thumbnails.config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(generate_images.config, "IMAGE_BACKEND", "vast_instance")
    monkeypatch.setattr(config, "IMAGE_BACKEND", "vast_instance")
    monkeypatch.setattr(runpod_backend, "OUTPUT_ROOT", str(tmp_path))

    video_dir = tmp_path / "demo"
    video_dir.mkdir(parents=True, exist_ok=True)
    _write_json(video_dir / "image_prompts.json", [{"index": 1, "prompt": "full prompt", "clip_prompt": "short prompt", "negative_prompt": ""}])
    _write_json(video_dir / "generation_log.json", {})
    _write_json(video_dir / "publishing" / "thumbnail_prompts.json", [
        {
            "concept_id": 1,
            "type": "human_closeup",
            "clip_prompt": "YouTube thumbnail, human closeup, face",
            "image_prompt": "no text no logo no watermark",
            "negative_prompt": "",
            "thumbnail_text": "WHO WAS IT",
            "subject_side": "left",
            "text_side": "right",
            "paired_title_ids": ["title_1"],
        }
    ])
    _write_json(
        video_dir / "creative_package.json",
        {
            "package_version": "creative-package-v1",
            "language": "en",
            "core_promise": "promise",
            "target_viewer": "viewer",
            "primary_hook": "hook",
            "title_options": [
                {"id": "title_1", "angle": "curiosity", "text": "First Title"},
                {"id": "title_2", "angle": "discovery", "text": "Second Title"},
                {"id": "title_3", "angle": "emotion", "text": "Third Title"},
            ],
            "description_draft": "Description stays.",
            "search_keywords": ["ancient", "history"],
            "chapter_plan": [{"sentence_index": 1, "label": "Intro"}],
            "thumbnail_concepts": [
                {"id": 1, "type": "human_closeup", "visual_hook": "face", "emotional_goal": "shock", "thumbnail_text": "WHO WAS IT", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_1"], "must_show": [], "must_avoid": []},
                {"id": 2, "type": "mystery_reveal", "visual_hook": "cave", "emotional_goal": "mystery", "thumbnail_text": "INSIDE THE CAVE", "subject_side": "right", "text_side": "left", "paired_title_ids": ["title_2"], "must_show": [], "must_avoid": []},
                {"id": 3, "type": "scale_or_danger", "visual_hook": "danger", "emotional_goal": "fear", "thumbnail_text": "TOO BIG", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_3"], "must_show": [], "must_avoid": []},
            ],
        },
    )
    (video_dir / "script.txt").write_text("Sentence one.", encoding="utf-8")
    (video_dir / "audio.mp3").write_bytes(b"audio")

    candidate_path = tmp_path / "candidate.png"
    Image.new("RGB", (1024, 576), (120, 80, 40)).save(candidate_path)
    backend = _FakeBackend(candidate_path)
    teardown_called: list[bool] = []
    thumbnail_background_backends: list[object] = []
    thumbnail_finalize_calls: list[bool] = []

    def fake_build_backend(planned_image_count=None):
        return backend, lambda: teardown_called.append(True), {
            "managed": True, "owned_instance_id": None, "worker_boot_id": "", "worker_ready_count": 1, "model_load_count": 1, "rent_count": 1, "num_gpus": 1,
        }

    def fake_generate_thumbnail_backgrounds(video_id: str, **kwargs):
        thumbnail_background_backends.append(kwargs.get("backend_override"))
        return {"background_generated_count": 1, "thumbnail_failed_ids": []}

    def fake_generate_thumbnail_assets(video_id: str, **kwargs):
        thumbnail_finalize_calls.append(True)
        return {
            "thumbnail_prompt_count": 1,
            "thumbnail_generated_count": 1,
            "thumbnail_failed_ids": [],
            "validation_passed": True,
        }

    monkeypatch.setattr(generate_images, "_build_vast_backend", fake_build_backend)
    monkeypatch.setattr(thumbnails, "generate_thumbnail_backgrounds", fake_generate_thumbnail_backgrounds)
    monkeypatch.setattr(thumbnails, "generate_thumbnail_assets", fake_generate_thumbnail_assets)

    generate_images.run("demo")

    assert backend.calls == ["001"]
    assert thumbnail_background_backends == [backend]
    assert thumbnail_finalize_calls == [True]
    assert teardown_called == [True]


def test_vast_backend_uses_public_ip_when_ssh_host_missing(tmp_path: Path, monkeypatch) -> None:
    import image_generation.vast_manager as vast_manager

    monkeypatch.setattr(config, "VAST_API_KEY", "vast-key")
    monkeypatch.setattr(config, "VAST_HF_TOKEN", "hf-token")
    monkeypatch.setattr(config, "WORKER_API_TOKEN", "worker-token")
    monkeypatch.setattr(config, "HF_MODEL_REVISION", "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")
    monkeypatch.setattr(config, "VAST_REQUEST_TIMEOUT", 123)

    deploy_targets: list[str] = []
    ready_targets: list[tuple[str, int]] = []

    class _FakeManager:
        def __init__(self, api_key: str, worker_port: int):
            self.api_key = api_key
            self.worker_port = worker_port

        def find_offer(self, **_kwargs):
            return {"id": 1, "machine_id": 2}

        def rent(self, *, num_gpus: int = 1, **_kwargs):
            return VastInstance(instance_id=55, ssh_host="", ssh_port=22, direct_port=49000, public_ipaddr="1.2.3.4")

        def wait_until_running(self, _instance_id: int, timeout: int = 300):
            return VastInstance(instance_id=55, ssh_host="", ssh_port=22, direct_port=49000, public_ipaddr="1.2.3.4")

        def wait_for_port(self, _instance_id: int, timeout: int = 120):
            return VastInstance(instance_id=55, ssh_host="", ssh_port=22, direct_port=49000, public_ipaddr="1.2.3.4")

        def deploy_worker(self, instance, **_kwargs):
            deploy_targets.append(instance.ssh_target_host)

        def wait_worker_ready(self, host: str, port: int, timeout: int = 600):
            ready_targets.append((host, port))

        def destroy(self, _instance_id: int):
            return None

        def list_instances(self):
            return []

    monkeypatch.setattr(vast_manager, "VastManager", _FakeManager)

    backend, teardown, meta = generate_images._build_vast_backend(planned_image_count=None)

    assert backend.base_url == "http://1.2.3.4:49000"
    assert deploy_targets == ["1.2.3.4"]
    assert ready_targets == [("1.2.3.4", 49000)]
    assert meta["worker_boot_id"] == "55:1.2.3.4:49000"
    assert callable(teardown)


def test_deploy_worker_uses_uppercase_scp_port_flag(monkeypatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd, check=True):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    manager = VastManager(api_key="vast-key", worker_port=8080)
    instance = VastInstance(instance_id=1, ssh_host="1.2.3.4", ssh_port=22, public_ipaddr="1.2.3.4")
    manager.deploy_worker(instance, worker_dir="vast_worker", hf_token="", model_revision="", worker_token="")

    assert calls[0][:5] == ["scp", "-o", "StrictHostKeyChecking=no", "-P", "22"]
    assert calls[1][:5] == ["ssh", "-o", "StrictHostKeyChecking=no", "-p", "22"]


# ---------------------------------------------------------------------------
# Multi-GPU regression cases
# ---------------------------------------------------------------------------

def test_build_vast_backend_returns_num_gpus_in_meta(tmp_path, monkeypatch) -> None:
    """_build_vast_backend metadata must include num_gpus from the selected offer."""
    import image_generation.vast_manager as vast_manager

    monkeypatch.setattr(config, "VAST_API_KEY", "vast-key")
    monkeypatch.setattr(config, "VAST_HF_TOKEN", "hf-token")
    monkeypatch.setattr(config, "WORKER_API_TOKEN", "worker-token")
    monkeypatch.setattr(config, "HF_MODEL_REVISION", "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")
    monkeypatch.setattr(config, "VAST_NUM_GPUS_CHOICES", [1, 2])
    monkeypatch.setattr(config, "VAST_MODEL_LOAD_WALL_SECONDS", 90.0)
    monkeypatch.setattr(config, "VAST_WORKER_CUSTOM_IMAGE", False)

    class _FakeManagerMultiGPU:
        def __init__(self, api_key: str, worker_port: int):
            pass

        def find_offer(self, **_kwargs):
            return {"id": 1, "machine_id": 2, "_num_gpus": 2}

        def rent(self, *, num_gpus: int = 1, **_kwargs):
            return VastInstance(instance_id=55, ssh_host="", ssh_port=22, direct_port=49000, public_ipaddr="1.2.3.4")

        def wait_until_running(self, _id, **_kw):
            return VastInstance(instance_id=55, ssh_host="", ssh_port=22, direct_port=49000, public_ipaddr="1.2.3.4")

        def wait_for_port(self, _id, **_kw):
            return VastInstance(instance_id=55, ssh_host="", ssh_port=22, direct_port=49000, public_ipaddr="1.2.3.4")

        def deploy_worker(self, instance, **_kw):
            pass

        def wait_worker_ready(self, host, port, timeout=600):
            pass

        def destroy(self, _id):
            pass

        def list_instances(self):
            return []

    monkeypatch.setattr(vast_manager, "VastManager", _FakeManagerMultiGPU)

    from steps import generate_images as step_gen
    _, _, meta = step_gen._build_vast_backend(planned_image_count=500)
    assert meta["num_gpus"] == 2


def test_wait_worker_ready_accepts_missing_workers_ready(monkeypatch) -> None:
    """Legacy /health without workers_ready field must still pass readiness check."""
    import requests
    from image_generation.vast_manager import VastManager

    manager = VastManager(api_key="k", worker_port=8080)

    responses = [
        {"status": "ok", "model_loaded": False},            # still loading
        {"status": "ok", "model_loaded": True},             # no workers_ready key
    ]
    call_idx = [0]

    def fake_get(url, timeout=5):
        data = responses[call_idx[0]]
        call_idx[0] = min(call_idx[0] + 1, len(responses) - 1)
        resp = SimpleNamespace(status_code=200)
        resp.raise_for_status = lambda: None
        resp.json = lambda: data
        return resp

    monkeypatch.setattr(requests, "get", fake_get)

    # Should not raise
    manager.wait_worker_ready("127.0.0.1", 8080, timeout=30)


def test_rent_injects_num_gpus_env(monkeypatch) -> None:
    """rent() must pass NUM_GPUS into the instance environment."""
    import requests
    from image_generation.vast_manager import VastManager

    manager = VastManager(api_key="k", worker_port=8080)
    captured_body: list[dict] = []

    class _FakeResp:
        status_code = 200
        ok = True
        def json(self): return {"new_contract": 42}
        def raise_for_status(self): pass

    class _FakeGetResp:
        status_code = 200
        ok = True
        def json(self): return {"id": 42, "status": "running", "ssh_host": "1.2.3.4", "ssh_port": 22, "actual_status": "running", "public_ipaddr": "1.2.3.4", "ports": {}}
        def raise_for_status(self): pass

    def fake_put(url, json=None, headers=None, timeout=None):
        captured_body.append(json or {})
        return _FakeResp()

    def fake_get(url, headers=None, params=None, timeout=None):
        return _FakeGetResp()

    monkeypatch.setattr(requests, "put", fake_put)
    monkeypatch.setattr(requests, "get", fake_get)

    manager.rent(offer_id=1, image="img:latest", env_vars={}, disk_gb=60.0, num_gpus=2)
    assert len(captured_body) == 1
    assert captured_body[0].get("env", {}).get("NUM_GPUS") == "2"
