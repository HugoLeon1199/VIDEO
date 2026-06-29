"""P0 hotfix regression tests for commit 8958069.

Covers all 7 production bugs fixed:
  1. Dockerfile copies full package; CMD uses module form
  2. httpx in requirements.txt and legacy pip install string
  3. Custom-image path never calls deploy_worker/SSH/SCP
  4. max_workers propagated through run(), retries, autopilot, batch
  5. generate_thumbnail_assets(allow_gpu_generation=False) never triggers new rental
  6. Per-GPU price gate: dph_total/num_gpus <= max_price_per_gpu_hour for multi-GPU
  7. Direct single-video run() passes real planned_image_count
"""

from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import config
from image_generation.schemas import CandidateResult, SceneResult
from image_generation.vast_manager import VastInstance, VastManager
from steps import generate_images, thumbnails


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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
            model="fake", mode="fake",
            duration_seconds=0.01,
            candidates=[CandidateResult(
                candidate_index=0, seed=11, width=1024, height=576,
                sha256="sha", generation_seconds=0.01,
                mime_type="image/png",
                local_path=str(candidate_path),
            )],
        )


def _make_offer(num_gpus: int, dph_total: float, machine_id: int = 1,
                cpu_ram: int = 128 * 1024, cpu_cores_effective: float = 16.0) -> dict:
    return {
        "id": machine_id * 10 + num_gpus,
        "machine_id": machine_id,
        "num_gpus": num_gpus,
        "gpu_name": "RTX 3090",
        "gpu_ram": 24576,
        "dph_total": dph_total,
        "total_flops": 35.5 * num_gpus,
        "inet_down": 1000.0,
        "inet_down_cost": 0.002,
        "inet_up_cost": 0.002,
        "storage_cost": 0.0,
        "reliability": 0.99,
        "reliability2": 0.99,
        "verification": "verified",
        "compute_cap": 860,
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


# ---------------------------------------------------------------------------
# Fix 1 — Docker image layout
# ---------------------------------------------------------------------------

def test_dockerfile_copies_full_package() -> None:
    dockerfile = Path(__file__).parent.parent / "vast_worker" / "Dockerfile"
    content = dockerfile.read_text(encoding="utf-8")
    # Must copy the whole package directory, not just server.py
    assert "COPY . /workspace/vast_worker/" in content, \
        "Dockerfile must copy entire package dir, not just server.py"


def test_dockerfile_cmd_uses_module_form() -> None:
    dockerfile = Path(__file__).parent.parent / "vast_worker" / "Dockerfile"
    content = dockerfile.read_text(encoding="utf-8")
    assert "vast_worker.server" in content, \
        "CMD must use -m vast_worker.server (module form)"
    assert '"-m"' in content or "'-m'" in content or " -m " in content, \
        "CMD must pass -m flag (module form)"


def test_server_py_spawn_worker_uses_module_form() -> None:
    server_py = Path(__file__).parent.parent / "vast_worker" / "server.py"
    content = server_py.read_text(encoding="utf-8")
    assert "-m\", \"vast_worker.gpu_worker" in content or \
           '"-m", "vast_worker.gpu_worker"' in content or \
           "vast_worker.gpu_worker" in content, \
        "_spawn_worker must use -m vast_worker.gpu_worker, not os.path.join script path"
    assert "os.path.join(os.path.dirname(__file__), \"gpu_worker.py\")" not in content, \
        "_spawn_worker must not use script-path invocation of gpu_worker.py"


# ---------------------------------------------------------------------------
# Fix 2 — httpx dependency
# ---------------------------------------------------------------------------

def test_httpx_in_requirements_txt() -> None:
    reqs = Path(__file__).parent.parent / "vast_worker" / "requirements.txt"
    content = reqs.read_text(encoding="utf-8")
    lines = [l.strip().lower() for l in content.splitlines() if l.strip() and not l.startswith("#")]
    has_httpx = any(l.startswith("httpx") for l in lines)
    assert has_httpx, f"httpx missing from vast_worker/requirements.txt — got: {lines}"


def test_httpx_in_legacy_pip_install_string() -> None:
    vast_manager_py = Path(__file__).parent.parent / "image_generation" / "vast_manager.py"
    content = vast_manager_py.read_text(encoding="utf-8")
    # The pip install string can span multiple f-string lines; check the whole block
    assert "pip install" in content, "deploy_worker must contain a pip install command"
    # Find the region containing the legacy pip install and check httpx is nearby
    pip_idx = content.find("pip install -q fastapi")
    assert pip_idx >= 0, "legacy pip install line not found in vast_manager.py"
    # The install command is at most a few hundred chars; scan 500 chars forward
    pip_region = content[pip_idx: pip_idx + 500]
    assert "httpx" in pip_region, \
        f"httpx missing from legacy pip install region:\n{pip_region}"


# ---------------------------------------------------------------------------
# Fix 3 — Custom image skips deploy_worker / SSH / SCP
# ---------------------------------------------------------------------------

def test_custom_image_skips_deploy_worker(monkeypatch) -> None:
    """With VAST_WORKER_CUSTOM_IMAGE=True, deploy_worker must never be called."""
    import image_generation.vast_manager as vast_manager

    monkeypatch.setattr(config, "VAST_API_KEY", "k")
    monkeypatch.setattr(config, "VAST_HF_TOKEN", "hf")
    monkeypatch.setattr(config, "WORKER_API_TOKEN", "tok")
    monkeypatch.setattr(config, "HF_MODEL_REVISION", "abc123")
    monkeypatch.setattr(config, "VAST_WORKER_CUSTOM_IMAGE", True)
    monkeypatch.setattr(config, "VAST_NUM_GPUS_CHOICES", [1])
    monkeypatch.setattr(config, "VAST_MODEL_LOAD_WALL_SECONDS", 90.0)

    deploy_calls: list[str] = []

    class _FakeMgr:
        def __init__(self, api_key, worker_port): pass

        def find_offer(self, **_kw):
            return {"id": 1, "machine_id": 2, "_num_gpus": 1}

        def rent(self, *, num_gpus=1, **_kw):
            return VastInstance(instance_id=9, ssh_host="h", ssh_port=22,
                                direct_port=49001, public_ipaddr="5.6.7.8")

        def wait_until_running(self, _id, **_kw):
            return VastInstance(instance_id=9, ssh_host="h", ssh_port=22,
                                direct_port=49001, public_ipaddr="5.6.7.8")

        def wait_for_port(self, _id, **_kw):
            return VastInstance(instance_id=9, ssh_host="h", ssh_port=22,
                                direct_port=49001, public_ipaddr="5.6.7.8")

        def deploy_worker(self, instance, **_kw):
            deploy_calls.append("deploy_worker_called")

        def wait_worker_ready(self, host, port, timeout=600):
            pass

        def destroy(self, _id): pass
        def list_instances(self): return []

    monkeypatch.setattr(vast_manager, "VastManager", _FakeMgr)

    _backend, teardown, meta = generate_images._build_vast_backend(planned_image_count=10)
    assert deploy_calls == [], \
        f"deploy_worker must NOT be called when VAST_WORKER_CUSTOM_IMAGE=True, got: {deploy_calls}"
    if teardown:
        teardown()


def test_custom_image_flow_reaches_wait_worker_ready(monkeypatch) -> None:
    """Custom-image path: wait_worker_ready must be called even without deploy_worker."""
    import image_generation.vast_manager as vast_manager

    monkeypatch.setattr(config, "VAST_API_KEY", "k")
    monkeypatch.setattr(config, "VAST_HF_TOKEN", "hf")
    monkeypatch.setattr(config, "WORKER_API_TOKEN", "tok")
    monkeypatch.setattr(config, "HF_MODEL_REVISION", "abc123")
    monkeypatch.setattr(config, "VAST_WORKER_CUSTOM_IMAGE", True)
    monkeypatch.setattr(config, "VAST_NUM_GPUS_CHOICES", [1])
    monkeypatch.setattr(config, "VAST_MODEL_LOAD_WALL_SECONDS", 90.0)

    ready_calls: list[tuple[str, int]] = []

    class _FakeMgr:
        def __init__(self, api_key, worker_port): pass
        def find_offer(self, **_kw): return {"id": 1, "machine_id": 2, "_num_gpus": 1}
        def rent(self, *, num_gpus=1, **_kw):
            return VastInstance(instance_id=9, ssh_host="h", ssh_port=22,
                                direct_port=49001, public_ipaddr="5.6.7.8")
        def wait_until_running(self, _id, **_kw):
            return VastInstance(instance_id=9, ssh_host="h", ssh_port=22,
                                direct_port=49001, public_ipaddr="5.6.7.8")
        def wait_for_port(self, _id, **_kw):
            return VastInstance(instance_id=9, ssh_host="h", ssh_port=22,
                                direct_port=49001, public_ipaddr="5.6.7.8")
        def deploy_worker(self, instance, **_kw):
            pytest.fail("deploy_worker must not be called for custom image")
        def wait_worker_ready(self, host, port, timeout=600):
            ready_calls.append((host, port))
        def destroy(self, _id): pass
        def list_instances(self): return []

    monkeypatch.setattr(vast_manager, "VastManager", _FakeMgr)
    generate_images._build_vast_backend(planned_image_count=5)
    assert len(ready_calls) == 1, f"wait_worker_ready must be called exactly once; got {ready_calls}"


# ---------------------------------------------------------------------------
# Fix 4 — max_workers propagated
# ---------------------------------------------------------------------------

def test_run_accepts_max_workers_param() -> None:
    sig = inspect.signature(generate_images.run)
    assert "max_workers" in sig.parameters, \
        "generate_images.run() must accept max_workers parameter"


def test_regenerate_failed_scenes_accepts_max_workers_param() -> None:
    from image_generation.production import regenerate_failed_scenes
    sig = inspect.signature(regenerate_failed_scenes)
    assert "max_workers" in sig.parameters, \
        "regenerate_failed_scenes() must accept max_workers parameter"


def test_run_threads_max_workers_to_generate_scene_images(tmp_path, monkeypatch) -> None:
    """run() with max_workers=3 must forward that to generate_scene_images."""
    from image_generation import production

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "IMAGE_BACKEND", "runpod_serverless")

    video_dir = tmp_path / "vtest"
    video_dir.mkdir()
    _write_json(video_dir / "image_prompts.json",
                [{"index": 1, "prompt": "p", "clip_prompt": "c", "negative_prompt": ""}])
    _write_json(video_dir / "generation_log.json",
                {"001": {"status": "completed", "selected_image": str(video_dir / "images" / "img_001.png")}})
    (video_dir / "images").mkdir()
    Image.new("RGB", (64, 64)).save(video_dir / "images" / "img_001.png")

    captured_max_workers: list[int | None] = []

    def fake_generate_scene_images(video_id, *, max_workers=None, **_kw):
        captured_max_workers.append(max_workers)
        return {"scene_ok": 0, "scene_fail": 0, "processed_count": 0}

    def fake_regenerate(*_a, **_kw):
        return {"scene_ok": 0, "scene_fail": 0, "processed_count": 0}

    monkeypatch.setattr(production, "generate_scene_images", fake_generate_scene_images)
    monkeypatch.setattr(production, "regenerate_failed_scenes", fake_regenerate)

    generate_images.run("vtest", include_thumbnails=False, max_workers=3)
    assert captured_max_workers and captured_max_workers[0] == 3, \
        f"generate_scene_images must receive max_workers=3, got {captured_max_workers}"


# ---------------------------------------------------------------------------
# Fix 5 — No second GPU rental during thumbnail finalization
# ---------------------------------------------------------------------------

def test_thumbnail_assets_allow_gpu_generation_false_no_backend_call(tmp_path, monkeypatch) -> None:
    """generate_thumbnail_assets(allow_gpu_generation=False) must not call _build_backend."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PUBLISHING_DIRNAME", "publishing")
    monkeypatch.setattr(config, "THUMBNAIL_FONT_SIZE", 48)

    video_dir = tmp_path / "vid"
    video_dir.mkdir()
    (video_dir / "script.txt").write_text("One.\n\nTwo.\n\nThree.", encoding="utf-8")
    _write_json(video_dir / "creative_package.json", {
        "package_version": "creative-package-v1",
        "language": "en",
        "core_promise": "p", "target_viewer": "v", "primary_hook": "h",
        "title_options": [
            {"id": "title_1", "angle": "curiosity", "text": "T1"},
            {"id": "title_2", "angle": "discovery", "text": "T2"},
            {"id": "title_3", "angle": "emotion", "text": "T3"},
        ],
        "description_draft": "d.",
        "search_keywords": [],
        "chapter_plan": [
            {"sentence_index": 1, "label": "Intro"},
            {"sentence_index": 2, "label": "Middle"},
            {"sentence_index": 3, "label": "End"},
        ],
        "thumbnail_concepts": [
            {"id": 1, "type": "human_closeup", "visual_hook": "face", "emotional_goal": "shock",
             "thumbnail_text": "WHO", "subject_side": "left", "text_side": "right",
             "paired_title_ids": ["title_1"], "must_show": [], "must_avoid": []},
            {"id": 2, "type": "mystery_reveal", "visual_hook": "cave", "emotional_goal": "mystery",
             "thumbnail_text": "WHERE", "subject_side": "right", "text_side": "left",
             "paired_title_ids": ["title_2"], "must_show": [], "must_avoid": []},
            {"id": 3, "type": "scale_or_danger", "visual_hook": "scale", "emotional_goal": "danger",
             "thumbnail_text": "HOW", "subject_side": "left", "text_side": "right",
             "paired_title_ids": ["title_3"], "must_show": [], "must_avoid": []},
        ],
    })
    _write_json(video_dir / "publishing" / "thumbnail_prompts.json", [
        {"concept_id": 1, "type": "human_closeup", "image_prompt": "no text",
         "negative_prompt": "", "thumbnail_text": "WHO",
         "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_1"]},
        {"concept_id": 2, "type": "mystery_reveal", "image_prompt": "no text",
         "negative_prompt": "", "thumbnail_text": "WHERE",
         "subject_side": "right", "text_side": "left", "paired_title_ids": ["title_2"]},
        {"concept_id": 3, "type": "scale_or_danger", "image_prompt": "no text",
         "negative_prompt": "", "thumbnail_text": "HOW",
         "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_3"]},
    ])
    thumbnails.load_validated_package(video_dir)

    build_backend_called: list[bool] = []

    monkeypatch.setattr(
        thumbnails, "_build_backend",
        lambda: build_backend_called.append(True) or (_ for _ in ()).throw(
            AssertionError("_build_backend must NOT be called when allow_gpu_generation=False")
        )
    )

    # No background image exists → would normally trigger GPU generation
    diag = thumbnails.generate_thumbnail_assets("vid", allow_gpu_generation=False)

    assert build_backend_called == [], \
        "_build_backend was called despite allow_gpu_generation=False"
    assert diag["validation_passed"] is False, \
        "diagnostics must report validation_passed=False when GPU work is skipped"
    assert 1 in diag["thumbnail_failed_ids"], \
        "concept_id 1 must be in thumbnail_failed_ids when background is missing"


def test_thumbnail_assets_accept_allow_gpu_generation_param() -> None:
    sig = inspect.signature(thumbnails.generate_thumbnail_assets)
    assert "allow_gpu_generation" in sig.parameters, \
        "generate_thumbnail_assets() must accept allow_gpu_generation param"
    assert sig.parameters["allow_gpu_generation"].default is True, \
        "allow_gpu_generation must default to True (no change for existing callers)"


# ---------------------------------------------------------------------------
# Fix 6 — Per-GPU price gate
# ---------------------------------------------------------------------------

def test_2gpu_bundle_passes_per_gpu_price_gate(monkeypatch) -> None:
    """2-GPU bundle at $0.40 total ($0.20/GPU) must pass when max_price_per_gpu_hour=0.20."""
    import requests
    from image_generation.vast_manager import VastManager

    two_gpu = _make_offer(2, dph_total=0.40, machine_id=5,
                          cpu_ram=128 * 1024, cpu_cores_effective=16.0)
    two_gpu["_num_gpus"] = 2

    monkeypatch.setattr(requests, "get",
                        _fake_requests_get_factory({1: [], 2: [two_gpu]}))

    manager = VastManager(api_key="k", worker_port=8080)
    offer = manager.find_offer(
        min_vram_gb=24,
        max_price_per_hour=0.20,
        max_price_per_gpu_hour=0.20,
        min_inet_down_mbps=100,
        min_reliability=0.95,
        min_disk_gb=60.0,
        max_inet_cost_per_gb=0.01,
        max_estimated_total_cost=100.0,
        n_images=100,
        num_gpus_choices=[1, 2],
    )
    assert offer.get("_num_gpus") == 2, \
        f"2-GPU offer at $0.20/GPU must be selected; got {offer}"


def test_single_gpu_old_gate_unchanged(monkeypatch) -> None:
    """1-GPU offer at $0.25/hr must be rejected when max_price_per_hour=0.20."""
    import requests
    from image_generation.vast_manager import VastManager

    one_gpu = _make_offer(1, dph_total=0.25, machine_id=6)

    monkeypatch.setattr(requests, "get",
                        _fake_requests_get_factory({1: [one_gpu], 2: []}))

    manager = VastManager(api_key="k", worker_port=8080)
    with pytest.raises(RuntimeError):
        manager.find_offer(
            min_vram_gb=24,
            max_price_per_hour=0.20,
            max_price_per_gpu_hour=0.20,
            min_inet_down_mbps=100,
            min_reliability=0.95,
            min_disk_gb=60.0,
            max_inet_cost_per_gb=0.01,
            max_estimated_total_cost=100.0,
            n_images=100,
            num_gpus_choices=[1],
        )


def test_find_offer_accepts_max_price_per_gpu_hour_param() -> None:
    sig = inspect.signature(VastManager.find_offer)
    assert "max_price_per_gpu_hour" in sig.parameters, \
        "VastManager.find_offer() must accept max_price_per_gpu_hour param"


def test_config_has_vast_max_price_per_gpu_hour() -> None:
    assert hasattr(config, "VAST_MAX_PRICE_PER_GPU_HOUR"), \
        "config.py must define VAST_MAX_PRICE_PER_GPU_HOUR"
    assert isinstance(config.VAST_MAX_PRICE_PER_GPU_HOUR, float), \
        "VAST_MAX_PRICE_PER_GPU_HOUR must be a float"


# ---------------------------------------------------------------------------
# Fix 7 — Direct single-video run passes real planned_image_count
# ---------------------------------------------------------------------------

def test_direct_run_passes_real_planned_image_count(tmp_path, monkeypatch) -> None:
    """generate_images.run() direct Vast path must compute real count, not fall back to 3."""
    from image_generation import production

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "IMAGE_BACKEND", "vast_instance")
    monkeypatch.setattr(config, "IMAGE_CANDIDATE_SEEDS", [11001, 11002])

    video_dir = tmp_path / "planvid"
    video_dir.mkdir()
    _write_json(video_dir / "image_prompts.json", [
        {"index": i + 1, "prompt": f"s{i}", "clip_prompt": f"c{i}", "negative_prompt": ""}
        for i in range(5)
    ])
    _write_json(video_dir / "generation_log.json", {})

    captured_pic: list[int | None] = []

    def fake_build_backend(planned_image_count=None):
        captured_pic.append(planned_image_count)
        return object(), lambda: None, {
            "managed": True, "owned_instance_id": None, "worker_boot_id": "",
            "worker_ready_count": 1, "model_load_count": 1, "rent_count": 1, "num_gpus": 1,
        }

    compute_calls: list[list[str]] = []

    def fake_compute(video_ids):
        compute_calls.append(list(video_ids))
        # Real answer: 5 pending scenes × 2 seeds + 0 thumbnails = 10
        return 10

    def fake_generate_scene_images(*_a, **_kw):
        return {"scene_ok": 5, "scene_fail": 0, "processed_count": 5}

    def fake_regenerate_failed_scenes(*_a, **_kw):
        return {"scene_ok": 0, "scene_fail": 0, "processed_count": 0}

    monkeypatch.setattr(generate_images, "_build_vast_backend", fake_build_backend)
    monkeypatch.setattr(production, "compute_session_image_count", fake_compute)
    monkeypatch.setattr(production, "generate_scene_images", fake_generate_scene_images)
    monkeypatch.setattr(production, "regenerate_failed_scenes", fake_regenerate_failed_scenes)
    # pending_scene_prompts used in trailing check — return empty to avoid raise
    monkeypatch.setattr(production, "pending_scene_prompts", lambda *_a, **_kw: [])

    generate_images.run("planvid", include_thumbnails=False)

    assert captured_pic, "_build_vast_backend must have been called"
    pic = captured_pic[0]
    # Must receive real count (10), NOT IMAGE_CANDIDATES default (3)
    assert pic == 10, \
        f"planned_image_count must be 10 (5 scenes × 2 seeds), got {pic}"
    assert compute_calls, "compute_session_image_count must have been called by run()"
