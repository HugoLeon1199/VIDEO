"""One-rental quality test: draw scenes 1-10 at 5/10/15/20 steps on ONE Vast box.

Rents a single machine, then runs generate_images 4 times against that SAME running
instance (via VAST_INSTANCE_HOST/PORT, so no re-rent / re-download), each at a
different step count and into a separate output root, so you can eyeball the
quality vs steps trade-off. Destroys the box at the end no matter what.

The model downloads ONCE (~37GB); the 4 runs reuse it. Cost ≈ one rental's GPU
time for 40 images + one download — not 4 separate cold starts.
"""

from __future__ import annotations

import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Load .env into os.environ BEFORE importing config (config reads env at import).
_env = os.path.join(_ROOT, ".env")
if os.path.exists(_env):
    for _line in open(_env, encoding="utf-8").read().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

import config as cfg  # noqa: E402
from image_generation.vast_manager import VastManager  # noqa: E402

VIDEO_ID = "to-tien-ban-lam-gi-ca-ngay-vi"
STEP_LEVELS = [5, 10, 15, 20]
FROM_SCENE, TO_SCENE = 1, 10
PYTHON = sys.executable


def main() -> int:
    if not cfg.VAST_API_KEY:
        print("VAST_API_KEY not set")
        return 1

    mgr = VastManager(api_key=cfg.VAST_API_KEY, worker_port=cfg.VAST_WORKER_PORT)

    # Safety: sweep any orphans first.
    try:
        mgr.destroy_all()
    except Exception as e:  # noqa: BLE001
        print(f"pre-sweep: {e}")

    disk_floor = max(60.0, cfg.VAST_DISK_GB)
    env_vars = {"USE_8BIT": os.getenv("VAST_USE_8BIT", "1")}
    if cfg.VAST_HF_TOKEN:
        env_vars["HF_TOKEN"] = cfg.VAST_HF_TOKEN

    instance = None
    try:
        # --- rent ONE box, ranked by true cost (40 images) ---
        offer = mgr.find_offer(
            min_vram_gb=cfg.VAST_MIN_VRAM_GB,
            max_vram_gb=cfg.VAST_MAX_VRAM_GB,
            gpu_name=cfg.VAST_GPU_NAME,
            max_price_per_hour=cfg.VAST_MAX_PRICE_PER_HOUR,
            min_inet_down_mbps=cfg.VAST_MIN_INET_DOWN_MBPS,
            min_reliability=cfg.VAST_MIN_RELIABILITY,
            min_disk_gb=disk_floor,
            max_inet_cost_per_gb=cfg.VAST_MAX_INET_DOWN_COST,
            preferred_inet_cost_per_gb=cfg.VAST_PREFERRED_INET_DOWN_COST,
            expected_download_gb=cfg.VAST_EXPECTED_DOWNLOAD_GB,
            expected_upload_gb=cfg.VAST_EXPECTED_UPLOAD_GB,
            n_images=len(STEP_LEVELS) * (TO_SCENE - FROM_SCENE + 1),
        )
        print(f"Renting {offer.get('gpu_name')} ${offer.get('dph_total'):.3f}/hr "
              f"dl ${offer.get('inet_down_cost'):.4f}/GB ...")
        instance = mgr.rent(offer_id=offer["id"], image=cfg.VAST_WORKER_IMAGE,
                            env_vars=env_vars, disk_gb=disk_floor)
        instance = mgr.wait_until_running(instance.instance_id, timeout=300)
        if not instance.direct_port:
            instance = mgr.wait_for_port(instance.instance_id, timeout=120)
        mgr.wait_worker_ready(instance.public_ipaddr, instance.direct_port, timeout=600)
        print(f"Ready at {instance.public_ipaddr}:{instance.direct_port}")

        # --- run each step level against the SAME box (reuse model) ---
        run_env = dict(os.environ)
        run_env["VAST_INSTANCE_HOST"] = instance.public_ipaddr
        run_env["VAST_INSTANCE_PORT"] = str(instance.direct_port)
        prompts_file = os.path.join(_ROOT, "output", VIDEO_ID, "image_prompts.json")
        for steps in STEP_LEVELS:
            out_root = f"output_s{steps}"
            print(f"\n===== {steps} steps -> {out_root} =====")
            subprocess.run([
                PYTHON, os.path.join(_ROOT, "scripts", "generate_images.py"),
                "--video-id", VIDEO_ID, "--backend", "vast_instance", "--track", "vi",
                "--prompts", prompts_file,
                "--vast-instances", "1", "--candidates", "1", "--workers", "1",
                "--no-qa", "--from-scene", str(FROM_SCENE), "--to-scene", str(TO_SCENE),
                "--force", "--steps", str(steps), "--output-root", out_root,
            ], env=run_env, check=False)
        print("\nAll step levels done.")
        return 0
    finally:
        # ALWAYS destroy — never leave the box billing.
        try:
            if instance is not None:
                mgr.destroy(instance.instance_id)
        except Exception as e:  # noqa: BLE001
            print(f"destroy error: {e}")
        try:
            mgr.destroy_all()  # belt-and-suspenders
        except Exception as e:  # noqa: BLE001
            print(f"destroy_all error: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
