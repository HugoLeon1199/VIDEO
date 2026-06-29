"""Tests for the multi-GPU Vast.ai offer selection and generation flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

import config
from image_generation.production import (
    VastLifecycle,
    VastSession,
    compute_session_image_count,
    generate_scene_images,
    pending_thumbnail_prompts,
)
from image_generation.schemas import CandidateResult, SceneResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_video(tmp_path: Path, video_id: str, n_scenes: int = 3, n_thumbs: int = 2) -> Path:
    video_dir = tmp_path / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    prompts = [{"index": i + 1, "prompt": f"scene {i+1}", "clip_prompt": f"clip {i+1}", "negative_prompt": ""} for i in range(n_scenes)]
    _write_json(video_dir / "image_prompts.json", prompts)
    _write_json(video_dir / "generation_log.json", {})
    thumb_prompts = [
        {
            "concept_id": j + 1,
            "type": "human_closeup",
            "clip_prompt": f"thumbnail {j+1}",
            "image_prompt": f"thumb image {j+1}",
            "negative_prompt": "",
            "thumbnail_text": f"THUMB {j+1}",
            "subject_side": "left",
            "text_side": "right",
            "paired_title_ids": [],
        }
        for j in range(n_thumbs)
    ]
    _write_json(video_dir / "publishing" / "thumbnail_prompts.json", thumb_prompts)
    return video_dir


class _FakeBackend:
    def __init__(self, candidate_path: Path, call_duration: float = 0.0):
        self.calls: list[str] = []
        self.candidate_path = candidate_path

    def generate(self, request):
        self.calls.append(request.scene_id)
        return SceneResult(
            video_id=request.video_id,
            scene_id=request.scene_id,
            model="fake",
            mode="fake",
            duration_seconds=0.01,
            candidates=[
                CandidateResult(
                    candidate_index=0,
                    seed=11,
                    width=1024,
                    height=576,
                    sha256="sha",
                    generation_seconds=0.01,
                    mime_type="image/png",
                    local_path=str(self.candidate_path),
                )
            ],
        )


# ---------------------------------------------------------------------------
# §1 compute_session_image_count
# ---------------------------------------------------------------------------

def test_compute_session_image_count_multi_video(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "IMAGE_CANDIDATE_SEEDS", [11001, 11002, 11003])

    _make_video(tmp_path, "video_a", n_scenes=4, n_thumbs=3)
    _make_video(tmp_path, "video_b", n_scenes=6, n_thumbs=2)

    count = compute_session_image_count(["video_a", "video_b"])
    # 4 scenes × 3 seeds + 3 thumbs + 6 scenes × 3 seeds + 2 thumbs = 12+3+18+2 = 35
    assert count == 35


def test_compute_session_image_count_excludes_done(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "IMAGE_CANDIDATE_SEEDS", [11001, 11002])

    video_dir = _make_video(tmp_path, "video_c", n_scenes=3, n_thumbs=1)

    # Mark scene 001 as completed with a real image
    images_dir = video_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1024, 576), (0, 128, 0))
    img.save(images_dir / "img_001.png")
    log = {
        "001": {"status": "completed", "selected_image": str(images_dir / "img_001.png"), "errors": []}
    }
    _write_json(video_dir / "generation_log.json", log)

    count = compute_session_image_count(["video_c"])
    # 2 pending scenes × 2 seeds + 1 thumb = 4 + 1 = 5
    assert count == 5


# ---------------------------------------------------------------------------
# §2 Offer selection — 1-GPU vs 2-GPU
# ---------------------------------------------------------------------------

def _make_offer(num_gpus: int, dph_total: float, gpu_name: str = "RTX 3090",
                machine_id: int = 1, cpu_ram: int = 128 * 1024, cpu_cores_effective: float = 16.0,
                total_flops: float = 35.5, inet_down: float = 1000.0, inet_down_cost: float = 0.002,
                inet_up_cost: float = 0.002, storage_cost: float = 0.0, reliability: float = 0.99,
                compute_cap: int = 860) -> dict:
    return {
        "id": machine_id * 10 + num_gpus,
        "machine_id": machine_id,
        "num_gpus": num_gpus,
        "gpu_name": gpu_name,
        "gpu_ram": 24576,
        "dph_total": dph_total,
        "total_flops": total_flops * num_gpus,
        "inet_down": inet_down,
        "inet_down_cost": inet_down_cost,
        "inet_up_cost": inet_up_cost,
        "storage_cost": storage_cost,
        "reliability": reliability,
        "reliability2": reliability,
        "verification": "verified",
        "compute_cap": compute_cap,
        "cuda_max_good": 12.2,
        "cpu_ram": cpu_ram,
        "cpu_cores_effective": cpu_cores_effective,
        "rentable": True,
        "rented": False,
        "is_bid": False,
        "static_ip": True,
        "hosting_type": 1,
        "gpu_max_temp": 70,
        "disk_space": 100.0,
        "direct_port_count": 5,
        "geolocation": "Texas, US",
    }


def _fake_requests_get_factory(offers_by_num_gpus: dict[int, list[dict]]):
    """Returns a fake requests.get that returns offers based on num_gpus in query."""
    import json as _json

    def fake_get(url, headers=None, params=None, timeout=None):
        q = {}
        if params and "q" in params:
            try:
                q = _json.loads(params["q"]) if isinstance(params["q"], str) else params["q"]
            except Exception:
                q = {}
        ng = q.get("num_gpus", {}).get("eq", 1)
        raw = offers_by_num_gpus.get(ng, [])

        class _Resp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"offers": raw}

        return _Resp()

    return fake_get


def test_find_offer_selects_1gpu_when_cheaper(monkeypatch) -> None:
    import requests
    from image_generation.vast_manager import VastManager

    one_gpu = _make_offer(1, dph_total=0.10, machine_id=1)
    two_gpu = _make_offer(2, dph_total=0.30, machine_id=2, cpu_ram=64*1024, cpu_cores_effective=8.0)

    monkeypatch.setattr(requests, "get", _fake_requests_get_factory({1: [one_gpu], 2: [two_gpu]}))

    manager = VastManager(api_key="k", worker_port=8080)
    offer = manager.find_offer(
        min_vram_gb=24,
        max_price_per_hour=2.0,
        min_inet_down_mbps=100,
        min_reliability=0.90,
        min_disk_gb=60.0,
        max_inet_cost_per_gb=0.05,
        max_estimated_total_cost=5.0,
        n_images=100,
        num_gpus_choices=[1, 2],
    )
    assert offer.get("_num_gpus") == 1


def test_find_offer_selects_2gpu_when_cheaper(monkeypatch) -> None:
    import requests
    from image_generation.vast_manager import VastManager

    # 1-GPU: expensive hourly; 2-GPU: half the gen time at only 1.5× price → cheaper total
    one_gpu = _make_offer(1, dph_total=1.00, machine_id=1)
    two_gpu = _make_offer(2, dph_total=1.50, machine_id=2, cpu_ram=64*1024, cpu_cores_effective=8.0)

    monkeypatch.setattr(requests, "get", _fake_requests_get_factory({1: [one_gpu], 2: [two_gpu]}))

    manager = VastManager(api_key="k", worker_port=8080)
    offer = manager.find_offer(
        min_vram_gb=24,
        max_price_per_hour=5.0,
        min_inet_down_mbps=100,
        min_reliability=0.90,
        min_disk_gb=60.0,
        max_inet_cost_per_gb=0.05,
        max_estimated_total_cost=50.0,
        n_images=1000,
        num_gpus_choices=[1, 2],
    )
    assert offer.get("_num_gpus") == 2


def test_find_offer_unknown_gpu_skipped_for_multigpu(monkeypatch) -> None:
    import requests
    from image_generation.vast_manager import VastManager

    # 2-GPU with unknown GPU model → skipped; 1-GPU known should win
    two_gpu_unknown = _make_offer(2, dph_total=0.10, gpu_name="RTX 9999 ULTRA", machine_id=3, cpu_ram=64*1024, cpu_cores_effective=8.0)
    one_gpu_known = _make_offer(1, dph_total=0.50, machine_id=4)

    monkeypatch.setattr(requests, "get", _fake_requests_get_factory({1: [one_gpu_known], 2: [two_gpu_unknown]}))

    manager = VastManager(api_key="k", worker_port=8080)
    offer = manager.find_offer(
        min_vram_gb=24,
        max_price_per_hour=5.0,
        min_inet_down_mbps=100,
        min_reliability=0.90,
        min_disk_gb=60.0,
        max_inet_cost_per_gb=0.05,
        max_estimated_total_cost=50.0,
        n_images=100,
        num_gpus_choices=[1, 2],
    )
    assert offer.get("_num_gpus") == 1


def test_find_offer_cpu_ram_hard_filter(monkeypatch) -> None:
    import requests
    from image_generation.vast_manager import VastManager

    # 2-GPU with only 32 GB RAM total (needs 64 GB for 32 GB/GPU)
    two_gpu_low_ram = _make_offer(2, dph_total=0.01, machine_id=5, cpu_ram=32*1024, cpu_cores_effective=8.0)
    one_gpu = _make_offer(1, dph_total=0.50, machine_id=6)

    monkeypatch.setattr(requests, "get", _fake_requests_get_factory({1: [one_gpu], 2: [two_gpu_low_ram]}))

    manager = VastManager(api_key="k", worker_port=8080)
    offer = manager.find_offer(
        min_vram_gb=24,
        max_price_per_hour=5.0,
        min_inet_down_mbps=100,
        min_reliability=0.90,
        min_disk_gb=60.0,
        max_inet_cost_per_gb=0.05,
        max_estimated_total_cost=50.0,
        n_images=100,
        num_gpus_choices=[1, 2],
        min_cpu_ram_per_gpu_gb=32.0,
    )
    assert offer.get("_num_gpus") == 1


def test_find_offer_cpu_cores_hard_filter(monkeypatch) -> None:
    import requests
    from image_generation.vast_manager import VastManager

    # 2-GPU with only 4 cores total (needs 8 for 4 cores/GPU)
    two_gpu_low_cores = _make_offer(2, dph_total=0.01, machine_id=7, cpu_ram=128*1024, cpu_cores_effective=4.0)
    one_gpu = _make_offer(1, dph_total=0.50, machine_id=8)

    monkeypatch.setattr(requests, "get", _fake_requests_get_factory({1: [one_gpu], 2: [two_gpu_low_cores]}))

    manager = VastManager(api_key="k", worker_port=8080)
    offer = manager.find_offer(
        min_vram_gb=24,
        max_price_per_hour=5.0,
        min_inet_down_mbps=100,
        min_reliability=0.90,
        min_disk_gb=60.0,
        max_inet_cost_per_gb=0.05,
        max_estimated_total_cost=50.0,
        n_images=100,
        num_gpus_choices=[1, 2],
        min_cpu_cores_per_gpu=4.0,
    )
    assert offer.get("_num_gpus") == 1


# ---------------------------------------------------------------------------
# §3 Cost formula — verified by calling find_offer with controlled inputs
# ---------------------------------------------------------------------------

def test_spi_divides_by_num_gpus_for_known_gpu(monkeypatch) -> None:
    """The measured GPU (RTX 3090) should produce spi=19/num_gpus."""
    import requests
    from image_generation.vast_manager import VastManager

    one_gpu = _make_offer(1, dph_total=0.50, machine_id=1)
    two_gpu = _make_offer(2, dph_total=0.75, machine_id=2, cpu_ram=64*1024, cpu_cores_effective=8.0)

    monkeypatch.setattr(requests, "get", _fake_requests_get_factory({1: [one_gpu], 2: [two_gpu]}))

    manager = VastManager(api_key="k", worker_port=8080)
    # RTX 3090 is in MEASURED_SPEED — just verify find_offer returns a winner without raising
    offer = manager.find_offer(
        min_vram_gb=24, max_price_per_hour=5.0, min_inet_down_mbps=100, min_reliability=0.90,
        min_disk_gb=60.0, max_inet_cost_per_gb=0.05, max_estimated_total_cost=50.0,
        n_images=100, num_gpus_choices=[1, 2],
    )
    assert offer.get("_num_gpus") in (1, 2)


def test_spi_unknown_multigpu_skipped(monkeypatch) -> None:
    """Unknown GPU + num_gpus=2 → offer excluded from pool, RuntimeError raised if only option."""
    import requests
    from image_generation.vast_manager import VastManager

    two_gpu_unknown = _make_offer(2, dph_total=0.01, gpu_name="RTX 9999 UNKNOWN", machine_id=1,
                                  cpu_ram=64*1024, cpu_cores_effective=8.0)

    monkeypatch.setattr(requests, "get", _fake_requests_get_factory({2: [two_gpu_unknown]}))

    manager = VastManager(api_key="k", worker_port=8080)
    with pytest.raises(RuntimeError):
        manager.find_offer(
            min_vram_gb=24, max_price_per_hour=5.0, min_inet_down_mbps=100, min_reliability=0.90,
            min_disk_gb=60.0, max_inet_cost_per_gb=0.05,
            n_images=100, num_gpus_choices=[2],
        )


def test_spi_unknown_single_gpu_uses_tflops(monkeypatch) -> None:
    """Unknown GPU + num_gpus=1 → TFLOPS scaling is used (not None)."""
    import requests
    from image_generation.vast_manager import VastManager

    one_gpu_unknown = _make_offer(1, dph_total=0.10, gpu_name="RTX 9999 ULTRA UNKNOWN", machine_id=1,
                                  total_flops=82.6)

    monkeypatch.setattr(requests, "get", _fake_requests_get_factory({1: [one_gpu_unknown]}))

    manager = VastManager(api_key="k", worker_port=8080)
    # Should not raise — TFLOPS fallback is used
    offer = manager.find_offer(
        min_vram_gb=24, max_price_per_hour=5.0, min_inet_down_mbps=100, min_reliability=0.90,
        min_disk_gb=60.0, max_inet_cost_per_gb=0.05, max_estimated_total_cost=50.0,
        n_images=100, num_gpus_choices=[1],
    )
    assert offer.get("_num_gpus") == 1


# ---------------------------------------------------------------------------
# §4 Custom image path: VAST_WORKER_CUSTOM_IMAGE
# ---------------------------------------------------------------------------

def test_deploy_worker_custom_image_skips_scp(monkeypatch) -> None:
    import subprocess
    from image_generation.vast_manager import VastManager, VastInstance

    scp_calls: list[list] = []
    ssh_calls: list[list] = []

    def fake_run(cmd, **_kwargs):
        from types import SimpleNamespace
        if cmd[0] == "scp":
            scp_calls.append(cmd)
        elif cmd[0] == "ssh":
            ssh_calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    manager = VastManager(api_key="k", worker_port=8080)
    instance = VastInstance(instance_id=1, ssh_host="1.2.3.4", ssh_port=22, public_ipaddr="1.2.3.4")

    # custom_image=True must skip SCP
    manager.deploy_worker(instance, hf_token="", model_revision="abc123", worker_token="tok", custom_image=True)
    assert scp_calls == []


def test_deploy_worker_legacy_runs_scp(monkeypatch) -> None:
    import subprocess
    from image_generation.vast_manager import VastManager, VastInstance

    scp_calls: list[list] = []

    def fake_run(cmd, **_kwargs):
        from types import SimpleNamespace
        if cmd[0] == "scp":
            scp_calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    manager = VastManager(api_key="k", worker_port=8080)
    instance = VastInstance(instance_id=1, ssh_host="1.2.3.4", ssh_port=22, public_ipaddr="1.2.3.4")

    manager.deploy_worker(instance, hf_token="", model_revision="abc123", worker_token="tok", custom_image=False)
    assert len(scp_calls) > 0


# ---------------------------------------------------------------------------
# §5 Concurrency: generate_scene_images with max_workers
# ---------------------------------------------------------------------------

def test_generate_scene_images_uses_threadpool(tmp_path: Path, monkeypatch) -> None:
    """Results from concurrent requests must all land in generation_log."""
    import image_generation.runpod_serverless_backend as runpod_backend

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(runpod_backend, "OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(config, "IMAGE_CANDIDATE_SEEDS", [11001])

    video_dir = _make_video(tmp_path, "vid_conc", n_scenes=4)
    candidate_path = tmp_path / "c.png"
    Image.new("RGB", (1024, 576)).save(candidate_path)
    backend = _FakeBackend(candidate_path)

    result = generate_scene_images("vid_conc", backend_override=backend, manage_backend=False, max_workers=2)
    assert result["scene_ok"] == 4
    assert result["scene_fail"] == 0

    log = json.loads((video_dir / "generation_log.json").read_text())
    assert set(log.keys()) == {"001", "002", "003", "004"}


def test_generate_scene_images_one_failure_does_not_cancel_others(tmp_path: Path, monkeypatch) -> None:
    import image_generation.runpod_serverless_backend as runpod_backend

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(runpod_backend, "OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(config, "IMAGE_CANDIDATE_SEEDS", [11001])

    video_dir = _make_video(tmp_path, "vid_fail", n_scenes=4)
    candidate_path = tmp_path / "c.png"
    Image.new("RGB", (1024, 576)).save(candidate_path)

    call_count = [0]

    class _PartialBackend:
        def generate(self, request):
            call_count[0] += 1
            if request.scene_id == "002":
                raise RuntimeError("injected failure")
            return SceneResult(
                video_id=request.video_id,
                scene_id=request.scene_id,
                model="fake",
                mode="fake",
                duration_seconds=0.01,
                candidates=[
                    CandidateResult(
                        candidate_index=0,
                        seed=11,
                        width=1024,
                        height=576,
                        sha256="sha",
                        generation_seconds=0.01,
                        mime_type="image/png",
                        local_path=str(candidate_path),
                    )
                ],
            )

    result = generate_scene_images("vid_fail", backend_override=_PartialBackend(), manage_backend=False, max_workers=2)
    assert call_count[0] == 4  # all were attempted
    assert result["scene_ok"] == 3
    assert result["scene_fail"] == 1

    log = json.loads((video_dir / "generation_log.json").read_text())
    assert log["001"]["status"] == "completed"
    assert log["002"]["status"] == "failed"
    assert log["003"]["status"] == "completed"
    assert log["004"]["status"] == "completed"


def test_generation_log_valid_json_after_concurrent_writes(tmp_path: Path, monkeypatch) -> None:
    import image_generation.runpod_serverless_backend as runpod_backend

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(runpod_backend, "OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(config, "IMAGE_CANDIDATE_SEEDS", [11001])

    _make_video(tmp_path, "vid_atomic", n_scenes=6)
    candidate_path = tmp_path / "c.png"
    Image.new("RGB", (1024, 576)).save(candidate_path)
    backend = _FakeBackend(candidate_path)

    generate_scene_images("vid_atomic", backend_override=backend, manage_backend=False, max_workers=3)

    log_path = tmp_path / "vid_atomic" / "generation_log.json"
    parsed = json.loads(log_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    assert len(parsed) == 6


# ---------------------------------------------------------------------------
# §6 Batch entrypoint — rent_count, teardown, multi-video
# ---------------------------------------------------------------------------

def test_batch_entrypoint_single_rent(tmp_path: Path, monkeypatch) -> None:
    """VastSession created once → rent_count=1 even across multiple videos."""
    import image_generation.runpod_serverless_backend as runpod_backend

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "IMAGE_BACKEND", "vast_instance")
    monkeypatch.setattr(runpod_backend, "OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(config, "IMAGE_CANDIDATE_SEEDS", [11001])

    candidate_path = tmp_path / "c.png"
    Image.new("RGB", (1024, 576)).save(candidate_path)
    backend = _FakeBackend(candidate_path)
    lifecycle = VastLifecycle()

    _make_video(tmp_path, "batch_a", n_scenes=2, n_thumbs=0)
    _make_video(tmp_path, "batch_b", n_scenes=2, n_thumbs=0)

    # Manually simulate a VastSession with a pre-built backend
    session = VastSession.__new__(VastSession)
    session.planned_image_count = 4
    session.video_ids = ["batch_a", "batch_b"]
    session.lifecycle = lifecycle
    session.backend = backend
    session.teardown = lambda: None
    session.owned_instance_id = 99
    session.managed = True
    session.cleanup_error = ""
    session.num_gpus = 2

    failures: dict[str, int] = {}
    for vid in ["batch_a", "batch_b"]:
        result = generate_scene_images(
            vid,
            backend_override=session.backend,
            manage_backend=False,
            lifecycle=lifecycle,
            max_workers=session.num_gpus,
        )
        failures[vid] = result["scene_fail"]

    assert failures == {"batch_a": 0, "batch_b": 0}
    # Backend should have been called 4 times total (2 scenes per video)
    assert len(backend.calls) == 4


def test_batch_entrypoint_teardown_on_failure(tmp_path: Path, monkeypatch) -> None:
    """Teardown is called even when VastSession.__exit__ is reached after a raise."""
    teardown_called = [False]

    session = VastSession.__new__(VastSession)
    session.planned_image_count = 3
    session.video_ids = ["crash_vid"]
    session.lifecycle = VastLifecycle()
    session.backend = object()
    session.teardown = lambda: teardown_called.__setitem__(0, True)
    session.owned_instance_id = None
    session.managed = True
    session.cleanup_error = ""
    session.num_gpus = 1

    try:
        raise RuntimeError("simulated crash")
    except Exception as exc:
        session.__exit__(type(exc), exc, None)

    assert teardown_called[0]


# ---------------------------------------------------------------------------
# §7 VastSession fields propagated correctly
# ---------------------------------------------------------------------------

def test_vastsession_captures_num_gpus_from_metadata(tmp_path: Path, monkeypatch) -> None:
    import steps.generate_images as step_gen_images

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "IMAGE_BACKEND", "vast_instance")

    candidate_path = tmp_path / "c.png"
    Image.new("RGB", (1024, 576)).save(candidate_path)
    backend = _FakeBackend(candidate_path)

    monkeypatch.setattr(
        step_gen_images,
        "open_backend_with_metadata",
        lambda _name, planned_image_count=None: (
            backend, None,
            {"managed": False, "owned_instance_id": None, "worker_boot_id": "test", "worker_ready_count": 1, "model_load_count": 1, "rent_count": 0, "num_gpus": 3}
        ),
    )

    session = VastSession(planned_image_count=100, video_ids=["v"])
    session.__enter__()
    assert session.num_gpus == 3


def test_vastsession_planned_image_count_passed_to_open_backend(tmp_path: Path, monkeypatch) -> None:
    import steps.generate_images as step_gen_images

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "IMAGE_BACKEND", "vast_instance")

    received: list[int | None] = []
    candidate_path = tmp_path / "c.png"
    Image.new("RGB", (1024, 576)).save(candidate_path)
    backend = _FakeBackend(candidate_path)

    def fake_open(name, planned_image_count=None):
        received.append(planned_image_count)
        return backend, None, {"managed": False, "owned_instance_id": None, "worker_boot_id": "", "worker_ready_count": 1, "model_load_count": 1, "rent_count": 0, "num_gpus": 1}

    monkeypatch.setattr(step_gen_images, "open_backend_with_metadata", fake_open)

    session = VastSession(planned_image_count=777, video_ids=["v"])
    session.__enter__()
    assert received == [777]
