import argparse
import os
import shutil
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


def _setup_demo_dir(base_video_id: str, n: int) -> str:
    """Create output/{base_id}_demo{n}/ and copy/trim input files from base folder.
    Returns the demo video_id to use for all subsequent steps.
    """
    import json as _json

    demo_id = f"{base_video_id}_demo{n}"
    base_dir = get_video_dir(base_video_id)
    demo_dir = get_video_dir(demo_id)
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Copy script and timestamps
    for fname in ("script.txt", "timestamps.json"):
        src = base_dir / fname
        dst = demo_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            logger.debug("Demo: copied {} → {}", src.name, dst)

    # Trim audio to the end of the last demo prompt / last demo sentence
    # Use timestamps.json to find the cut point (n-th sentence end time)
    src_audio = base_dir / "audio.mp3"
    dst_audio = demo_dir / "audio_trimmed.mp3"
    if src_audio.exists() and not dst_audio.exists():
        ts_path = base_dir / "timestamps.json"
        prompts_path = demo_dir / "image_prompts.json"
        cut_time = None

        # Prefer existing demo image_prompts end time
        if prompts_path.exists():
            try:
                items = _json.loads(prompts_path.read_text(encoding="utf-8"))
                cut_time = items[-1]["end"] + 1.0  # 1s tail
            except Exception:
                pass

        # Fall back to timestamps
        if cut_time is None and ts_path.exists():
            try:
                ts = _json.loads(ts_path.read_text(encoding="utf-8"))
                demo_ts = ts[:n]
                cut_time = demo_ts[-1]["end"] + 1.0
            except Exception:
                pass

        if cut_time:
            import subprocess as _sp
            import os as _os
            ffmpeg_path = shutil.which("ffmpeg") or str(
                Path(r"C:\Users\LEON_RM\AppData\Local\Microsoft\WinGet\Packages"
                     r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin") / "ffmpeg.exe"
            )
            _os.environ["PATH"] = str(Path(ffmpeg_path).parent) + _os.pathsep + _os.environ.get("PATH", "")
            cmd = ["ffmpeg", "-y", "-i", str(src_audio), "-t", f"{cut_time:.3f}", "-c:a", "copy", str(dst_audio)]
            r = _sp.run(cmd, capture_output=True)
            if r.returncode == 0:
                logger.info("Demo: trimmed audio to {:.1f}s → {}", cut_time, dst_audio.name)
            else:
                logger.warning("Demo: audio trim failed, falling back to full audio")
                shutil.copy2(src_audio, demo_dir / "audio.mp3")
        else:
            shutil.copy2(src_audio, demo_dir / "audio.mp3")
            logger.debug("Demo: copied full audio (no cut time)")

    logger.info("Demo folder: output/{}/", demo_id)
    return demo_id


def detect_resume_step(video_dir: Path) -> int:
    """Return the next step number to run based on existing output files."""
    import json as _json

    images_dir = video_dir / "images"
    prompts_path = video_dir / "image_prompts.json"
    ts_path = video_dir / "timestamps.json"

    if prompts_path.exists() and ts_path.exists():
        try:
            n_prompts = len(_json.loads(prompts_path.read_text(encoding="utf-8")))
            n_timestamps = len(_json.loads(ts_path.read_text(encoding="utf-8")))
        except Exception:
            n_prompts = n_timestamps = 0

        # Mismatch means prompts were generated for a different (demo) run — redo step 4
        if n_prompts != n_timestamps:
            logger.warning(
                "prompts ({}) != timestamps ({}) — resuming from step 4",
                n_prompts, n_timestamps,
            )
            return 4

        completed_images = (
            sum(1 for p_path in images_dir.glob("img_*.png")) if images_dir.exists() else 0
        )
        if completed_images >= n_prompts > 0:
            if (video_dir / "final.mp4").exists():
                if (video_dir / "metadata.json").exists():
                    return 8
                return 7
            return 6
        elif completed_images > 0:
            return 5
        return 5

    checks = [
        (prompts_path, 5),
        (ts_path, 4),
        (video_dir / "audio.mp3", 3),
        (video_dir / "script.txt", 2),
    ]
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
        help="Demo mode: run steps 4–6 in output/{video_id}_demo{N}/ (default N=10)",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("=== YouTube Autopilot Pipeline ===")

    # Demo mode: redirect to isolated subfolder, steps 4–6 only
    if args.demo:
        n = args.demo
        validate_environment(args.video_id)
        demo_id = _setup_demo_dir(args.video_id, n)
        logger.info("Video ID (demo): {} ({} images)", demo_id, n)
        start_step = args.from_step or (args.step or 4)
        steps = [args.step] if args.step else range(start_step, 7)  # steps 4,5,6 only
        for step in steps:
            logger.info("--- Step {} ---", step)
            run_step(step, demo_id, subtitles=args.subtitles, n_override=n)
        logger.info("=== Demo finished → output/{}/ ===", demo_id)
        return

    # Full / single / resume mode — always uses base video_id, no n_override
    logger.info("Video ID: {}", args.video_id)
    validate_environment(args.video_id)

    if args.step:
        logger.info("Running single step: {}", args.step)
        run_step(args.step, args.video_id, subtitles=args.subtitles)

    elif args.from_step:
        logger.info("Running from step {} to 7", args.from_step)
        for step in range(args.from_step, 8):
            logger.info("--- Step {} ---", step)
            run_step(step, args.video_id, subtitles=args.subtitles)

    elif args.resume:
        video_dir = get_video_dir(args.video_id)
        next_step = detect_resume_step(video_dir)
        if next_step >= 8:
            logger.info("Pipeline already complete for: {}", args.video_id)
            return
        logger.info("Resuming from step {}", next_step)
        for step in range(next_step, 8):
            logger.info("--- Step {} ---", step)
            run_step(step, args.video_id, subtitles=args.subtitles)

    else:
        logger.info("Running full pipeline (steps 2–7)")
        script_path = get_video_dir(args.video_id) / "script.txt"
        if not script_path.exists():
            logger.error("script.txt not found at: {}", script_path)
            sys.exit(1)
        for step in range(2, 8):
            logger.info("--- Step {} ---", step)
            run_step(step, args.video_id, subtitles=args.subtitles)

    logger.info("=== Pipeline finished ===")


if __name__ == "__main__":
    main()
