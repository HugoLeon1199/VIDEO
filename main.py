import argparse
import os
import sys
from pathlib import Path

from loguru import logger

import config


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    logger.add(
        config.LOGS_FILE,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
    )


def get_video_dir(video_id: str) -> Path:
    return Path(config.OUTPUT_DIR) / video_id


def detect_resume_step(video_dir: Path, n_images: int) -> int:
    """Return the next step number to run based on existing output files."""
    images_dir = video_dir / "images"

    checks = [
        (video_dir / "metadata.json", 8),
        (video_dir / "final.mp4", 7),
        (video_dir / "image_prompts.json", 6),
        (video_dir / "timestamps.json", 4),
        (video_dir / "audio.mp3", 3),
        (video_dir / "script.txt", 2),
    ]

    if (video_dir / "image_prompts.json").exists():
        completed_images = len(list(images_dir.glob("img_*.png"))) if images_dir.exists() else 0
        if completed_images >= n_images:
            for path, next_step in checks:
                if path == video_dir / "final.mp4":
                    if path.exists():
                        return next_step
                    return 6
            return 6
        elif completed_images > 0:
            return 5

    for path, next_step in checks:
        if path.exists():
            return next_step

    return 2


def run_step(step_num: int, video_id: str, subtitles: bool = False, n_override: int | None = None) -> None:
    if step_num == 1:
        from steps.generate_script import run as _run
        _run(video_id)
    elif step_num == 2:
        from steps.tts import run as _run
        _run(video_id)
    elif step_num == 3:
        from steps.transcribe import run as _run
        _run(video_id)
    elif step_num == 4:
        from steps.image_prompts import run as _run
        _run(video_id, n_override=n_override)
    elif step_num == 5:
        from steps.generate_images import run as _run
        _run(video_id, n_override=n_override)
    elif step_num == 6:
        from steps.render_video import run as _run
        _run(video_id, subtitles=subtitles)
    elif step_num == 7:
        from steps.metadata import run as _run
        _run(video_id)
    else:
        raise ValueError(f"Unknown step: {step_num}")


def validate_environment(video_id: str) -> None:
    video_dir = get_video_dir(video_id)
    if not video_dir.exists():
        logger.error("Video directory not found: {}", video_dir)
        logger.info("Create it and add script.txt: output/{}/script.txt", video_id)
        sys.exit(1)

    if not config.ANTHROPIC_API_KEY and not config.GEMINI_API_KEY:
        logger.warning("No API keys set — steps 4, 5, and 7 will fail")
    elif not config.ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — step 7 will fail (step 4 uses Gemini)")
    if not config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — steps 4 and 5 will fail")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YouTube Autopilot Pipeline — Ancient Humans channel"
    )
    parser.add_argument("--video-id", required=True, help="Video slug (e.g. what-ancient-humans-did-all-day)")
    parser.add_argument("--step", type=int, help="Run only this step (1–7)")
    parser.add_argument("--from-step", type=int, help="Run from this step to step 7")
    parser.add_argument("--resume", action="store_true", help="Auto-detect and resume from last completed step")
    parser.add_argument("--subtitles", action="store_true", help="Burn subtitles into video (step 6)")
    parser.add_argument(
        "--demo",
        type=int,
        metavar="N",
        nargs="?",
        const=10,
        help="Demo mode: generate only N images (default 10, ~30s video)",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("=== YouTube Autopilot Pipeline ===")
    logger.info("Video ID: {}", args.video_id)

    n_images = args.demo if args.demo else config.IMAGES_PER_VIDEO
    if args.demo:
        logger.info("DEMO MODE: generating {} images (~{}s video)", n_images, n_images * 3)

    validate_environment(args.video_id)

    if args.step:
        logger.info("Running single step: {}", args.step)
        run_step(args.step, args.video_id, subtitles=args.subtitles, n_override=args.demo)

    elif args.from_step:
        logger.info("Running from step {} to 7", args.from_step)
        for step in range(args.from_step, 8):
            logger.info("--- Step {} ---", step)
            run_step(step, args.video_id, subtitles=args.subtitles, n_override=args.demo)

    elif args.resume:
        next_step = detect_resume_step(get_video_dir(args.video_id), n_images)
        if next_step >= 8:
            logger.info("Pipeline already complete for: {}", args.video_id)
            return
        logger.info("Resuming from step {}", next_step)
        for step in range(next_step, 8):
            logger.info("--- Step {} ---", step)
            run_step(step, args.video_id, subtitles=args.subtitles, n_override=args.demo)

    else:
        logger.info("Running full pipeline (steps 2–7)")
        script_path = get_video_dir(args.video_id) / "script.txt"
        if not script_path.exists():
            logger.error("script.txt not found at: {}", script_path)
            sys.exit(1)

        for step in range(2, 8):
            logger.info("--- Step {} ---", step)
            run_step(step, args.video_id, subtitles=args.subtitles, n_override=args.demo)

    logger.info("=== Pipeline finished ===")


if __name__ == "__main__":
    main()
