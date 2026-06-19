"""Step 6: Render final video using FFmpeg with Ken Burns effect and optional subtitles."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

import config


def _check_ffmpeg() -> None:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if result.returncode != 0:
        logger.error("FFmpeg not found. Install FFmpeg and ensure it is in PATH.")
        sys.exit(1)


def _timestamps_to_srt(timestamps: list[dict], srt_path: Path) -> None:
    """Convert timestamps.json to SRT subtitle file."""
    lines = []
    for seg in timestamps:
        idx = seg["index"]
        start = _seconds_to_srt_time(seg["start"])
        end = _seconds_to_srt_time(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def _seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_filter_complex(prompts: list[dict], n: int, subtitles: bool, srt_path: str) -> str:
    """
    Build FFmpeg filter_complex string.

    For each image:
      - scale + pad to 1920x1080
      - zoompan for Ken Burns (5% zoom over duration)
      - setpts to sync timing

    Then concat all clips + add audio + optional subtitles.
    """
    filters = []
    inputs = []

    for i, item in enumerate(prompts):
        duration = item["end"] - item["start"]
        if duration <= 0:
            duration = 3.0

        frames = int(duration * config.VIDEO_FPS)
        zoom_end = 1.0 + config.KEN_BURNS_ZOOM
        # zoompan: zoom from 1.0 to zoom_end over `frames` frames
        zp = (
            f"[{i}:v]"
            f"scale={config.IMAGE_WIDTH}:{config.IMAGE_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={config.IMAGE_WIDTH}:{config.IMAGE_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"scale={config.VIDEO_WIDTH*2}:{config.VIDEO_HEIGHT*2},"
            f"zoompan=z='min(zoom+{config.KEN_BURNS_ZOOM/frames:.6f},{zoom_end})'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}:fps={config.VIDEO_FPS},"
            f"setpts=PTS-STARTPTS"
            f"[v{i}]"
        )
        filters.append(zp)
        inputs.append(f"[v{i}]")

    # Concat all video streams
    n_inputs = "".join(inputs)
    filters.append(f"{n_inputs}concat=n={n}:v=1:a=0[vout]")

    # Add subtitles if requested
    if subtitles and srt_path:
        srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
        filters.append(f"[vout]subtitles='{srt_escaped}':force_style='FontSize={config.SUBTITLE_FONT_SIZE},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2'[vfinal]")
        final_video = "[vfinal]"
    else:
        final_video = "[vout]"

    return ";".join(filters), final_video


def run(video_id: str, subtitles: bool = False) -> None:
    _check_ffmpeg()

    video_dir = Path(config.OUTPUT_DIR) / video_id
    prompts_path = video_dir / "image_prompts.json"
    audio_path = video_dir / "audio.mp3"
    images_dir = video_dir / "images"
    output_path = video_dir / "final.mp4"

    for path, name in [(prompts_path, "image_prompts.json"), (audio_path, "audio.mp3")]:
        if not path.exists():
            logger.error("{} not found: {}", name, path)
            sys.exit(1)

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    n = len(prompts)

    # Verify all images exist
    missing = [p["index"] for p in prompts if not (images_dir / f"img_{p['index']:03d}.png").exists()]
    if missing:
        logger.error("Missing {} images: {}", len(missing), missing[:10])
        logger.info("Run step 5 first (or --resume to generate missing images).")
        sys.exit(1)

    logger.info("Rendering video: {} images + audio → {}", n, output_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        srt_path = ""
        if subtitles:
            timestamps_path = video_dir / "timestamps.json"
            if timestamps_path.exists():
                timestamps = json.loads(timestamps_path.read_text(encoding="utf-8"))
                srt_file = Path(tmpdir) / "subtitles.srt"
                _timestamps_to_srt(timestamps, srt_file)
                srt_path = str(srt_file)
                logger.info("Subtitle burn-in enabled")
            else:
                logger.warning("timestamps.json not found — subtitles disabled")
                subtitles = False

        filter_complex, final_video = _build_filter_complex(prompts, n, subtitles, srt_path)

        # Build input args: one -i per image + audio
        input_args = []
        for item in prompts:
            img_file = images_dir / f"img_{item['index']:03d}.png"
            input_args += ["-loop", "1", "-i", str(img_file)]
        input_args += ["-i", str(audio_path)]

        cmd = [
            "ffmpeg", "-y",
            *input_args,
            "-filter_complex", filter_complex,
            "-map", final_video,
            "-map", f"{n}:a",           # audio from last input
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", config.VIDEO_BITRATE,
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-r", str(config.VIDEO_FPS),
            str(output_path),
        ]

        logger.debug("FFmpeg command: {}", " ".join(cmd))
        logger.info("FFmpeg rendering started (this may take several minutes)...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("FFmpeg failed:\n{}", result.stderr[-3000:])
            sys.exit(1)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Video rendered: {} ({:.1f} MB)", output_path, file_size_mb)
