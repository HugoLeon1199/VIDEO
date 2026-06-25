"""
Post-generation image review: detect bad images and auto-regenerate them.

Checks each generated image using PIL heuristics (no Gemini API calls):
  - File exists and is a valid image
  - Correct dimensions (1024x576 ± tolerance)
  - Not blank/solid color (low variance)
  - Not mostly black or mostly white
  - File size not suspiciously small (< 10 KB = likely corrupt)

Any scene that fails is added to a regen queue and re-submitted to RunPod.

Usage
-----
# Dry-run: just show which scenes are bad, don't regenerate
python scripts/review_images.py --video-id my-video --track vi --dry-run

# Auto-regen bad scenes (default)
python scripts/review_images.py --video-id my-video --track vi

# Limit scope
python scripts/review_images.py --video-id my-video --track vi --from-scene 5 --to-scene 20

# Use more workers for regen
python scripts/review_images.py --video-id my-video --track vi --workers 5
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
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("review_images")

# ---------------------------------------------------------------------------
# Heuristic checks
# ---------------------------------------------------------------------------

MIN_FILE_SIZE_BYTES = 10_000       # < 10 KB → suspect
BLANK_VARIANCE_THRESHOLD = 18      # low std-dev across channels → blank/solid (night scenes legitimately dark)
DARK_PIXEL_RATIO_THRESHOLD = 0.90  # >90% very dark pixels → black frame
BRIGHT_PIXEL_RATIO_THRESHOLD = 0.90  # >90% very bright pixels → white frame
EXPECTED_WIDTH = 1024
EXPECTED_HEIGHT = 576
SIZE_TOLERANCE = 8                 # allow ±8px


def _check_image(img_path: Path) -> list[str]:
    """Return list of issue strings. Empty = OK."""
    issues = []

    # 1. File existence
    if not img_path.exists():
        return ["file_missing"]

    # 2. File size
    size = img_path.stat().st_size
    if size < MIN_FILE_SIZE_BYTES:
        issues.append(f"file_too_small:{size}B")

    # 3. Open with PIL
    try:
        from PIL import Image, ImageStat
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
    except Exception as e:
        return issues + [f"corrupt:{e}"]

    # 4. Dimensions
    if abs(w - EXPECTED_WIDTH) > SIZE_TOLERANCE or abs(h - EXPECTED_HEIGHT) > SIZE_TOLERANCE:
        issues.append(f"wrong_size:{w}x{h}")

    # 5. Variance (blank/solid color)
    stat = ImageStat.Stat(img)
    mean_stddev = sum(stat.stddev) / max(len(stat.stddev), 1)
    if mean_stddev < BLANK_VARIANCE_THRESHOLD:
        issues.append(f"low_variance:{mean_stddev:.1f}")

    # 6. Black frame check
    import numpy as np
    arr = np.array(img)
    dark_ratio = float((arr.max(axis=2) < 15).mean())
    if dark_ratio > DARK_PIXEL_RATIO_THRESHOLD:
        issues.append(f"mostly_black:{dark_ratio:.2f}")

    # 7. White frame check
    bright_ratio = float((arr.min(axis=2) > 240).mean())
    if bright_ratio > BRIGHT_PIXEL_RATIO_THRESHOLD:
        issues.append(f"mostly_white:{bright_ratio:.2f}")

    return issues


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Review generated images and regen bad ones")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--track", choices=["vi", "en"], required=True)
    parser.add_argument("--output-root", default=os.environ.get("IMAGE_OUTPUT_ROOT", "output"))
    parser.add_argument("--from-scene", type=int, default=None)
    parser.add_argument("--to-scene", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Show bad scenes but don't regenerate")
    parser.add_argument("--workers", type=int, default=5, help="Parallel regen workers")
    parser.add_argument("--max-regen-rounds", type=int, default=2, help="Max regen attempts per scene")
    parser.add_argument("--seed-offset", type=int, default=500000,
                        help="Seed offset for regen (to get different images)")
    parser.add_argument("--verbose", action="store_true")
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

    import config as _config

    tc = _config.TRACK_CONFIG[args.track]
    output_root = args.output_root
    video_dir = Path(output_root) / args.video_id
    images_dir = video_dir / tc["output_subdir"]
    log_path = video_dir / f"generation_log_{args.track}.json"

    gen_log = _load_json(log_path)
    if not gen_log:
        logger.error("No generation log at %s", log_path)
        sys.exit(1)

    # Load prompts
    prompts_path = video_dir / "image_prompts.json"
    prompts_by_id: dict[str, dict] = {}
    if prompts_path.exists():
        try:
            raw = json.loads(prompts_path.read_text(encoding="utf-8"))
            prompts_by_id = {f"{p['index']:03d}": p for p in raw}
        except Exception as e:
            logger.warning("Could not load prompts: %s", e)

    # --- Phase 1: Check all images ---
    all_scene_ids = sorted(gen_log.keys())

    # Apply from/to filter
    if args.from_scene is not None or args.to_scene is not None:
        def _idx(sid: str) -> int:
            try:
                return int(sid)
            except ValueError:
                return 0
        lo = args.from_scene or 1
        hi = args.to_scene or 999999
        all_scene_ids = [s for s in all_scene_ids if lo <= _idx(s) <= hi]

    bad_scenes: list[tuple[str, list[str]]] = []   # (scene_id, issues)
    ok_count = 0

    logger.info("Checking %d scenes...", len(all_scene_ids))
    for scene_id in all_scene_ids:
        entry = gen_log.get(scene_id, {})
        selected = entry.get("selected_image", "")
        if not selected:
            issues = ["no_selected_image"]
        else:
            img_path = Path(selected)
            issues = _check_image(img_path)

        if issues:
            bad_scenes.append((scene_id, issues))
            logger.warning("Scene %s BAD: %s - %s", scene_id, Path(selected).name if selected else "?", issues)
        else:
            ok_count += 1
            logger.debug("Scene %s OK", scene_id)

    logger.info(
        "Check complete: %d OK, %d bad out of %d",
        ok_count, len(bad_scenes), len(all_scene_ids),
    )

    if not bad_scenes:
        logger.info("All images look good!")
        return

    # Print summary table
    print("\n--- Bad Scenes ---")
    for scene_id, issues in bad_scenes:
        entry = gen_log.get(scene_id, {})
        img = entry.get("selected_image", "(none)")
        print(f"  scene {scene_id}: {Path(img).name if img else '?'} -> {', '.join(issues)}")

    if args.dry_run:
        print(f"\n[dry-run] Would regenerate {len(bad_scenes)} scenes. Remove --dry-run to proceed.")
        return

    # --- Phase 2: Regenerate bad scenes ---
    print(f"\nRegenerating {len(bad_scenes)} bad scenes with {args.workers} workers...")

    from image_generation.runpod_serverless_backend import (
        RunPodServerlessBackend,
        promote_candidate_to_render_image,
    )
    from image_generation.schemas import SceneRequest

    backend = RunPodServerlessBackend()
    log_lock = threading.Lock()

    regen_ok = 0
    regen_fail = 0

    def regen_scene(scene_id: str, known_issues: list[str]) -> tuple[str, bool]:
        entry = gen_log.get(scene_id, {})
        p = prompts_by_id.get(scene_id)
        if not p:
            logger.error("Scene %s: no prompt found, cannot regen", scene_id)
            return scene_id, False

        for regen_round in range(1, args.max_regen_rounds + 1):
            seed = 11001 + args.seed_offset * regen_round
            logger.info("Scene %s regen round %d seed=%d", scene_id, regen_round, seed)

            req = SceneRequest(
                video_id=args.video_id,
                scene_id=scene_id,
                prompt=p["prompt"],
                global_style=p.get("global_style", ""),
                negative_prompt=p.get("negative_prompt", ""),
                width=p.get("width", _config.IMAGE_WIDTH),
                height=p.get("height", _config.IMAGE_HEIGHT),
                steps=tc["steps"],
                guidance_scale=tc["guidance_scale"],
                candidate_seeds=[seed],
                output_mode="base64",
            )

            try:
                result = backend.generate(req)
            except Exception as e:
                logger.error("Scene %s regen round %d error: %s", scene_id, regen_round, e)
                continue

            if not result.candidates:
                logger.warning("Scene %s regen round %d: no candidates. Errors: %s",
                               scene_id, regen_round, result.errors)
                continue

            candidate = result.candidates[0]

            # Check the new image
            if not candidate.local_path or not Path(candidate.local_path).exists():
                logger.warning("Scene %s regen round %d: candidate has no local file", scene_id, regen_round)
                continue

            new_issues = _check_image(Path(candidate.local_path))
            if new_issues:
                logger.warning("Scene %s regen round %d: still bad: %s", scene_id, regen_round, new_issues)
                continue

            # Good — promote
            try:
                selected_image = promote_candidate_to_render_image(
                    candidate,
                    video_id=args.video_id,
                    scene_id=scene_id,
                    output_root=output_root,
                    images_subdir=tc["output_subdir"],
                )
            except Exception as e:
                logger.error("Scene %s promote failed: %s", scene_id, e)
                continue

            with log_lock:
                gen_log[scene_id]["selected_image"] = selected_image
                gen_log[scene_id]["selected_seed"] = candidate.seed
                gen_log[scene_id]["status"] = "completed"
                gen_log[scene_id]["review_regen_round"] = regen_round
                gen_log[scene_id]["review_fixed_issues"] = known_issues
                gen_log[scene_id]["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                _save_json(log_path, gen_log)

            logger.info("Scene %s FIXED in round %d seed=%d", scene_id, regen_round, seed)
            return scene_id, True

        logger.error("Scene %s still bad after %d regen rounds", scene_id, args.max_regen_rounds)
        with log_lock:
            gen_log[scene_id]["review_status"] = "still_bad"
            gen_log[scene_id]["review_issues"] = known_issues
            _save_json(log_path, gen_log)
        return scene_id, False

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(regen_scene, sid, issues): sid
            for sid, issues in bad_scenes
        }
        for future in as_completed(futures):
            scene_id, ok = future.result()
            if ok:
                regen_ok += 1
                print(f"  [OK] scene {scene_id} fixed")
            else:
                regen_fail += 1
                print(f"  [FAIL] scene {scene_id} still bad after {args.max_regen_rounds} rounds")

    print(f"\nRegen done: {regen_ok} fixed, {regen_fail} still bad")

    # Save review report
    report = {
        "video_id": args.video_id,
        "track": args.track,
        "total_checked": len(all_scene_ids),
        "ok_count": ok_count,
        "bad_count": len(bad_scenes),
        "regen_ok": regen_ok,
        "regen_fail": regen_fail,
        "bad_scenes": [
            {"scene_id": sid, "issues": issues}
            for sid, issues in bad_scenes
        ],
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report_path = video_dir / f"review_report_{args.track}.json"
    _save_json(report_path, report)
    logger.info("Review report saved: %s", report_path)

    if regen_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
