import argparse
import os
import shutil
import sys
from pathlib import Path

from loguru import logger

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

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
    import json as _json
    import subprocess as _sp

    demo_id = f"{base_video_id}_demo{n}"
    base_dir = get_video_dir(base_video_id)
    demo_dir = get_video_dir(demo_id)
    demo_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("script.txt", "timestamps.json"):
        src = base_dir / fname
        dst = demo_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    src_audio = base_dir / "audio.mp3"
    dst_audio = demo_dir / "audio_trimmed.mp3"
    if src_audio.exists() and not dst_audio.exists():
        cut_time = None
        prompts_path = demo_dir / "image_prompts.json"
        ts_path = base_dir / "timestamps.json"
        if prompts_path.exists():
            try:
                items = _json.loads(prompts_path.read_text(encoding="utf-8"))
                cut_time = items[-1]["end"] + 1.0
            except Exception:
                cut_time = None
        if cut_time is None and ts_path.exists():
            try:
                ts = _json.loads(ts_path.read_text(encoding="utf-8"))
                cut_time = ts[:n][-1]["end"] + 1.0
            except Exception:
                cut_time = None
        if cut_time:
            ffmpeg_path = shutil.which("ffmpeg") or str(
                Path(r"C:\Users\LEON_RM\AppData\Local\Microsoft\WinGet\Packages"
                     r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin") / "ffmpeg.exe"
            )
            os.environ["PATH"] = str(Path(ffmpeg_path).parent) + os.pathsep + os.environ.get("PATH", "")
            cmd = ["ffmpeg", "-y", "-i", str(src_audio), "-t", f"{cut_time:.3f}", "-c:a", "copy", str(dst_audio)]
            result = _sp.run(cmd, capture_output=True)
            if result.returncode != 0:
                shutil.copy2(src_audio, demo_dir / "audio.mp3")
        else:
            shutil.copy2(src_audio, demo_dir / "audio.mp3")
    logger.info("Demo folder: output/{}/", demo_id)
    return demo_id


def detect_resume_step(video_dir: Path) -> int:
    import json as _json

    images_dir = video_dir / "images"
    prompts_path = video_dir / "image_prompts.json"
    ts_path = video_dir / "timestamps.json"
    audio_path = video_dir / "audio.mp3"
    script_path = video_dir / "script.txt"
    effects_path = video_dir / "effects_plan.json"

    if prompts_path.exists():
        try:
            n_prompts = len(_json.loads(prompts_path.read_text(encoding="utf-8")))
        except Exception:
            n_prompts = 0
        if n_prompts > 0:
            completed_images = sum(1 for _ in images_dir.glob("img_*.png")) if images_dir.exists() else 0
            if completed_images >= n_prompts:
                if (video_dir / "final.mp4").exists():
                    if (video_dir / "metadata.json").exists():
                        return 9
                    return 8
                if (video_dir / "soundscape.json").exists() and effects_path.exists():
                    return 7
                if (video_dir / "soundscape.json").exists() or effects_path.exists():
                    return 6
                return 6
            if completed_images > 0:
                return 5
            return 5
        return 4

    checks = [(ts_path, 4), (audio_path, 3), (script_path, 2)]
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
        from steps.post_production_design import run as _run
        _run(video_id)
    elif step_num == 7:
        from steps.render_video import run as _run
        _run(video_id, subtitles=subtitles)
    elif step_num == 8:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube Autopilot Pipeline - Ancient Humans channel")
    parser.add_argument("--video-id", required=True, help="Video slug (e.g. what-ancient-humans-did-all-day)")
    parser.add_argument("--autopilot", action="store_true", help="Run full production autopilot from a script file")
    parser.add_argument("--script-file", help="Path to the narration script used by --autopilot")
    parser.add_argument("--step", type=int, help="Run only this step (1-8)")
    parser.add_argument("--from-step", type=int, help="Run from this step to step 8")
    parser.add_argument("--resume", action="store_true", help="Auto-detect and resume from last completed step")
    parser.add_argument("--subtitles", action="store_true", help="Also burn subtitles into final_subbed.mp4 during step 7")
    parser.add_argument("--demo", type=int, metavar="N", nargs="?", const=10, help="Demo mode: run steps 4-7 in output/{video_id}_demo{N}/")
    args = parser.parse_args()

    setup_logging()
    logger.info("=== YouTube Autopilot Pipeline ===")

    if args.autopilot:
        if not args.script_file:
            logger.error("--autopilot requires --script-file")
            sys.exit(1)
        from steps.autopilot import run as run_autopilot

        logger.info("Running production autopilot for {}", args.video_id)
        run_autopilot(args.video_id, args.script_file, resume=args.resume)
        logger.info("=== Autopilot finished ===")
        return

    if args.demo:
        validate_environment(args.video_id)
        demo_id = _setup_demo_dir(args.video_id, args.demo)
        start_step = args.from_step or (args.step or 4)
        steps = [args.step] if args.step else range(start_step, 8)
        for step in steps:
            logger.info("--- Step {} ---", step)
            run_step(step, demo_id, subtitles=args.subtitles, n_override=args.demo)
        logger.info("=== Demo finished -> output/{}/ ===", demo_id)
        return

    logger.info("Video ID: {}", args.video_id)
    validate_environment(args.video_id)

    if args.step:
        run_step(args.step, args.video_id, subtitles=args.subtitles)
    elif args.from_step:
        for step in range(args.from_step, 9):
            logger.info("--- Step {} ---", step)
            run_step(step, args.video_id, subtitles=args.subtitles)
    elif args.resume:
        next_step = detect_resume_step(get_video_dir(args.video_id))
        if next_step >= 9:
            logger.info("Pipeline already complete for: {}", args.video_id)
            return
        for step in range(next_step, 9):
            logger.info("--- Step {} ---", step)
            run_step(step, args.video_id, subtitles=args.subtitles)
    else:
        script_path = get_video_dir(args.video_id) / "script.txt"
        if not script_path.exists():
            logger.error("script.txt not found at: {}", script_path)
            sys.exit(1)
        for step in range(2, 9):
            logger.info("--- Step {} ---", step)
            run_step(step, args.video_id, subtitles=args.subtitles)

    logger.info("=== Pipeline finished ===")


if __name__ == "__main__":
    main()
