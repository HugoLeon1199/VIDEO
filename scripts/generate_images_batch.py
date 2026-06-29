"""Multi-video batch image generation using a single Vast.ai session.

Rents one GPU instance, generates scenes + thumbnails for every supplied video ID,
then destroys the instance — exactly one rent, one teardown.

Usage:
    python scripts/generate_images_batch.py --video-ids video_a video_b video_c
    python scripts/generate_images_batch.py --video-ids video_a --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-video Vast.ai batch image generation")
    parser.add_argument(
        "--video-ids",
        nargs="+",
        required=True,
        metavar="VIDEO_ID",
        help="One or more video IDs to process (output/<video-id>/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute planned image count and print offer search params, do not rent",
    )
    args = parser.parse_args()

    video_ids: list[str] = args.video_ids
    for vid in video_ids:
        out_dir = Path(config.OUTPUT_DIR) / vid
        if not out_dir.exists():
            logger.error("Output directory not found for video {}: {}", vid, out_dir)
            sys.exit(1)

    from image_generation.production import (
        VastSession,
        compute_session_image_count,
        generate_scene_images,
        regenerate_failed_scenes,
    )
    from steps import thumbnails as thumb_step

    session_image_count = compute_session_image_count(video_ids)
    logger.info(
        "Batch: {} video(s), ~{} total pending images (scenes × seeds + thumbnails)",
        len(video_ids),
        session_image_count,
    )

    if args.dry_run:
        logger.info("Dry run: VAST_NUM_GPUS_CHOICES={}", config.VAST_NUM_GPUS_CHOICES)
        logger.info("Dry run: VAST_MODEL_LOAD_WALL_SECONDS={}", config.VAST_MODEL_LOAD_WALL_SECONDS)
        logger.info("Dry run: VAST_MAX_ESTIMATED_TOTAL_COST={}", config.VAST_MAX_ESTIMATED_TOTAL_COST)
        logger.info("Dry run: VAST_ESTIMATED_TOTAL_COST_FALLBACKS={}", config.VAST_ESTIMATED_TOTAL_COST_FALLBACKS)
        logger.info("Dry run: VAST_WORKER_CUSTOM_IMAGE={}", config.VAST_WORKER_CUSTOM_IMAGE)
        logger.info("Dry run: No instance rented. Exiting.")
        return

    old_backend = config.IMAGE_BACKEND
    per_video_failures: dict[str, int] = {}
    try:
        config.IMAGE_BACKEND = "vast_instance"
        with VastSession(planned_image_count=session_image_count, video_ids=video_ids) as session:
            logger.info(
                "Vast session open: num_gpus={}, instance_id={}",
                session.num_gpus,
                session.owned_instance_id,
            )
            for vid in video_ids:
                vid_failures = 0
                logger.info("--- Processing video: {} ---", vid)
                try:
                    result = generate_scene_images(
                        vid,
                        backend_override=session.backend,
                        manage_backend=False,
                        lifecycle=session.lifecycle,
                        max_workers=session.num_gpus,
                    )
                    vid_failures += result.get("scene_fail", 0)
                    retry = regenerate_failed_scenes(
                        vid,
                        backend_override=session.backend,
                        manage_backend=False,
                        lifecycle=session.lifecycle,
                        max_workers=session.num_gpus,
                    )
                    vid_failures += retry.get("scene_fail", 0)
                except Exception as exc:
                    logger.error("Scene generation error for {}: {}", vid, exc)
                    vid_failures += 1

                try:
                    from image_generation.production import pending_thumbnail_prompts

                    if pending_thumbnail_prompts(vid):
                        thumb_step.generate_thumbnail_backgrounds(
                            vid,
                            backend_override=session.backend,
                            manage_backend=False,
                            lifecycle=session.lifecycle,
                            max_workers=session.num_gpus,
                        )
                except Exception as exc:
                    logger.error("Thumbnail generation error for {}: {}", vid, exc)
                    vid_failures += 1

                per_video_failures[vid] = vid_failures

    finally:
        config.IMAGE_BACKEND = old_backend

    # Thumbnail finalization (CPU-only, no GPU needed)
    for vid in video_ids:
        try:
            thumb_diag = thumb_step.generate_thumbnail_assets(vid, allow_gpu_generation=False)
            failed_ids = thumb_diag.get("thumbnail_failed_ids", [])
            if failed_ids or not thumb_diag.get("validation_passed"):
                logger.error(
                    "Thumbnail finalization failed for {}: failed_ids={}, passed={}",
                    vid, failed_ids, thumb_diag.get("validation_passed"),
                )
                per_video_failures[vid] = per_video_failures.get(vid, 0) + len(failed_ids or [1])
        except Exception as exc:
            logger.error("Thumbnail asset finalization error for {}: {}", vid, exc)
            per_video_failures[vid] = per_video_failures.get(vid, 0) + 1

    total_failures = sum(per_video_failures.values())
    logger.info("Batch complete. Per-video failure counts: {}", per_video_failures)
    if total_failures:
        logger.error("Batch finished with {} total failure(s)", total_failures)
        sys.exit(1)
    logger.info("All videos processed successfully.")


if __name__ == "__main__":
    main()
