"""
CLI to generate images for all scenes using the RunPod Serverless backend.

Usage:
    python scripts/generate_images.py \\
        --video-id ancient-humans-without-medicine \\
        --prompts output/ancient-humans-without-medicine/image_prompts.json \\
        --backend runpod_serverless \\
        --resume

Defaults to resume mode. Skips scenes that already have all candidates saved.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("generate_images")


# ---------------------------------------------------------------------------
# Generation log helpers
# ---------------------------------------------------------------------------

def _log_path(video_id: str, output_root: str) -> Path:
    return Path(output_root) / video_id / "generation_log.json"


def _summary_path(video_id: str, output_root: str) -> Path:
    return Path(output_root) / video_id / "generation_summary.json"


def _load_log(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save_log(path: Path, log: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")


def _scene_done(log: dict, scene_id: str, n_candidates: int) -> bool:
    entry = log.get(scene_id, {})
    selected_image = entry.get("selected_image")
    return (
        entry.get("status") == "completed"
        and entry.get("candidates_saved", 0) >= n_candidates
        and bool(selected_image)
        and Path(selected_image).exists()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images via RunPod Serverless")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--prompts", help="Path to image_prompts.json (defaults to output/<video-id>/image_prompts.json)")
    parser.add_argument("--backend", default="runpod_serverless", choices=["runpod_serverless"])
    parser.add_argument("--track", choices=["vi", "en"], default=None,
                        help="Image track: 'vi'=cinematic paleo art, 'en'=ink sketch parchment (both use FLUX.1-dev 12B)")
    parser.add_argument("--scene-id", help="Process only this scene_id")
    parser.add_argument("--from-scene", type=int, help="Start at this scene index (1-based)")
    parser.add_argument("--to-scene", type=int, help="Stop after this scene index (1-based)")
    parser.add_argument("--candidates", type=int, default=3, help="Candidates per scene (default 3)")
    parser.add_argument("--seeds", nargs="+", type=int, help="Explicit seeds (overrides --candidates count)")
    parser.add_argument("--force", action="store_true", help="Regenerate even if already done")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip completed scenes (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run, no API calls")
    parser.add_argument("--validate-only", action="store_true", help="Validate prompts only, no generation")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first scene error")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers (default 10)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-root", default=os.environ.get("IMAGE_OUTPUT_ROOT", "output"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load .env
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    # Apply track-specific config if --track is set
    import config as _config
    _track_steps = None
    _track_guidance = None
    _track_output_subdir = "images"
    if args.track:
        tc = _config.TRACK_CONFIG[args.track]
        _track_steps = tc["steps"]
        _track_guidance = tc["guidance_scale"]
        _track_output_subdir = tc["output_subdir"]
        logger.info("Track '%s': %d steps, guidance %.1f → %s", args.track, _track_steps, _track_guidance, _track_output_subdir)

    # Load prompts
    prompts_path = Path(args.prompts) if args.prompts else (
        Path(args.output_root) / args.video_id / "image_prompts.json"
    )
    if not prompts_path.exists():
        logger.error("Prompts file not found: %s", prompts_path)
        sys.exit(1)

    prompts: list[dict] = json.loads(prompts_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    if args.validate_only:
        logger.info("Validation only — %d prompts look good.", len(prompts))
        return

    # Build seeds
    if args.seeds:
        seeds = args.seeds
    else:
        base = 11000
        seeds = [base + i + 1 for i in range(args.candidates)]

    # Filter scenes
    if args.scene_id:
        # Zero-pad to match prompt index format
        prompts = [p for p in prompts if str(p["index"]).zfill(3) == args.scene_id.zfill(3)]
    else:
        if args.from_scene:
            prompts = prompts[args.from_scene - 1:]
        if args.to_scene:
            prompts = prompts[:args.to_scene]

    if not prompts:
        logger.warning("No prompts matched the filter criteria.")
        return

    logger.info("Processing %d scenes, %d candidates each", len(prompts), len(seeds))

    if args.dry_run:
        for p in prompts:
            print(f"  [dry-run] scene {p['index']:03d}: {p['prompt'][:80]}...")
        return

    # Setup backend
    from image_generation.runpod_serverless_backend import (
        RunPodServerlessBackend,
        promote_candidate_to_render_image,
    )
    from image_generation.schemas import SceneRequest

    backend = RunPodServerlessBackend()

    # generation log is per-track to avoid cross-track resume collisions
    log_suffix = f"_{args.track}" if args.track else ""
    log_path = Path(args.output_root) / args.video_id / f"generation_log{log_suffix}.json"
    gen_log = _load_log(log_path)

    t_start = time.time()
    total_ok = 0
    total_fail = 0
    log_lock = threading.Lock()

    def process_scene(p: dict) -> tuple[str, bool]:
        scene_id = f"{p['index']:03d}"

        if args.resume and not args.force and _scene_done(gen_log, scene_id, len(seeds)):
            logger.info("Scene %s already done — skipping", scene_id)
            return scene_id, True

        logger.info("Scene %s: %s", scene_id, p["prompt"][:80])

        req = SceneRequest(
            video_id=args.video_id,
            scene_id=scene_id,
            prompt=p["prompt"],
            global_style=p.get("global_style", ""),
            negative_prompt=p.get("negative_prompt", ""),
            width=p.get("width", 1024),
            height=p.get("height", 576),
            steps=_track_steps if _track_steps is not None else p.get("steps", 4),
            guidance_scale=_track_guidance if _track_guidance is not None else p.get("guidance_scale", 1.0),
            candidate_seeds=seeds,
            output_mode="base64",
        )

        t_scene = time.time()
        try:
            result = backend.generate(req)
            selected_image = ""
            if result.candidates:
                selected_image = promote_candidate_to_render_image(
                    result.candidates[0],
                    video_id=args.video_id,
                    scene_id=scene_id,
                    output_root=args.output_root,
                    images_subdir=_track_output_subdir,
                )

            entry = {
                "status": "completed" if selected_image and not result.errors else "partial",
                "candidates_saved": len(result.candidates),
                "selected_image": selected_image,
                "errors": result.errors,
                "job_id": result.job_id,
                "duration_seconds": result.duration_seconds,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with log_lock:
                gen_log[scene_id] = entry
                _save_log(log_path, gen_log)

            if result.errors:
                logger.warning("Scene %s partial: %s", scene_id, result.errors)
            else:
                logger.info("Scene %s done — %d candidates in %.1fs",
                            scene_id, len(result.candidates), time.time() - t_scene)
            return scene_id, True

        except Exception as e:
            logger.error("Scene %s FAILED: %s", scene_id, e)
            with log_lock:
                gen_log[scene_id] = {
                    "status": "failed",
                    "error": str(e),
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                _save_log(log_path, gen_log)
            return scene_id, False

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_scene, p): p for p in prompts}
        for future in as_completed(futures):
            scene_id, ok = future.result()
            if ok:
                total_ok += 1
            else:
                total_fail += 1
                if args.fail_fast:
                    logger.error("--fail-fast: cancelling remaining jobs.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    # Write summary
    summary = {
        "video_id": args.video_id,
        "total_scenes": len(prompts),
        "completed": total_ok,
        "failed": total_fail,
        "total_seconds": round(time.time() - t_start, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path = Path(args.output_root) / args.video_id / f"generation_summary{log_suffix}.json"
    _save_log(summary_path, summary)

    logger.info("Done: %d ok, %d failed in %.0fs", total_ok, total_fail, time.time() - t_start)
    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
