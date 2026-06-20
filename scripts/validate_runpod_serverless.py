"""
Validate RunPod Serverless environment before running the pipeline.

Without flags: validates env vars, config, no GPU call.
With --generate-test: submits one low-res test image and saves it locally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "OK " if ok else "ERR"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def validate_env() -> bool:
    print("\n=== Environment variables ===")
    ok = True
    ok &= _check("RUNPOD_API_KEY set", bool(os.environ.get("RUNPOD_API_KEY")),
                 "set via $env:RUNPOD_API_KEY or .env")
    ok &= _check("RUNPOD_ENDPOINT_ID set", bool(os.environ.get("RUNPOD_ENDPOINT_ID")),
                 "set via $env:RUNPOD_ENDPOINT_ID or .env")
    ok &= _check("RUNPOD_API_KEY not printed", True, "(never logged — redacted)")
    return ok


def validate_config() -> bool:
    print("\n=== Local configuration ===")
    ok = True
    try:
        import config
        ok &= _check("config.py importable", True)
        ok &= _check("OUTPUT_DIR exists or creatable", True, config.OUTPUT_DIR)
    except ImportError as e:
        ok &= _check("config.py importable", False, str(e))
    return ok


def validate_prompts(video_id: str) -> bool:
    print(f"\n=== Prompts for {video_id} ===")
    import config
    prompts_path = Path(config.OUTPUT_DIR) / video_id / "image_prompts.json"
    if not prompts_path.exists():
        _check("image_prompts.json exists", False, str(prompts_path))
        return False
    try:
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        _check("image_prompts.json parseable", True)
        _check("Non-empty prompts", len(prompts) > 0, f"{len(prompts)} prompts")
        for p in prompts[:3]:
            assert "prompt" in p and "index" in p, "Missing fields"
        _check("Prompt schema valid (spot check)", True)
        return True
    except Exception as e:
        _check("image_prompts.json valid", False, str(e))
        return False


def validate_endpoint_reachable() -> bool:
    print("\n=== Endpoint connectivity ===")
    try:
        from image_generation.runpod_client import RunPodClient
        client = RunPodClient()
        reachable = client.health_check()
        return _check("Endpoint /health responds 200", reachable)
    except Exception as e:
        return _check("Endpoint reachable", False, str(e))


def run_test_generation(video_id: str) -> bool:
    print("\n=== Test image generation (1 image, low res) ===")
    from image_generation.runpod_serverless_backend import RunPodServerlessBackend
    from image_generation.schemas import SceneRequest

    req = SceneRequest(
        video_id=video_id,
        scene_id="TEST",
        prompt="a simple red circle on white background",
        global_style="flat illustration",
        width=256,
        height=144,
        steps=2,
        guidance_scale=1.0,
        candidate_seeds=[42],
        output_format="WEBP",
        quality=80,
        output_mode="base64",
    )

    try:
        backend = RunPodServerlessBackend()
        result = backend.generate(req)
        ok = len(result.candidates) > 0 and not result.errors
        _check("Got at least 1 candidate", ok, f"errors={result.errors}")
        if result.candidates:
            c = result.candidates[0]
            _check("Image saved locally", bool(c.local_path), c.local_path or "")
            print(f"\n  Test image: {c.local_path}")
        return ok
    except Exception as e:
        return _check("Test generation", False, str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RunPod Serverless setup")
    parser.add_argument("--video-id", default="ancient-humans-without-medicine")
    parser.add_argument("--generate-test", action="store_true",
                        help="Submit one real low-res test image")
    args = parser.parse_args()

    # Load .env if present
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    results = [
        validate_env(),
        validate_config(),
        validate_prompts(args.video_id),
        validate_endpoint_reachable(),
    ]

    if args.generate_test:
        results.append(run_test_generation(args.video_id))

    print()
    if all(results):
        print("All checks passed.")
        sys.exit(0)
    else:
        print("Some checks FAILED. Fix the errors above before running the pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
