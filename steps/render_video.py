"""Step 6: Render final video — image slideshow synced to audio with correct gaps."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

import config


FFMPEG_WINGET_PATH = (
    r"C:\Users\LEON_RM\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
)


def _ensure_ffmpeg_path() -> None:
    if not shutil.which("ffmpeg"):
        os.environ["PATH"] = FFMPEG_WINGET_PATH + os.pathsep + os.environ.get("PATH", "")


def _check_ffmpeg() -> None:
    _ensure_ffmpeg_path()
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if result.returncode != 0:
        logger.error("FFmpeg not found. Install: winget install Gyan.FFmpeg")
        sys.exit(1)


def _get_audio_duration(audio_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        ffprobe = str(Path(FFMPEG_WINGET_PATH) / "ffprobe.exe")
    result = subprocess.run(
        [ffprobe, "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio_path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _trim_audio(src: Path, dst: Path, end_time: float) -> None:
    """Write a trimmed copy of audio from 0 to end_time seconds."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-t", f"{end_time:.3f}",
        "-c:a", "copy",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="ignore")[-500:])


def _compute_clip_durations(prompts: list[dict], audio_duration: float) -> list[float]:
    """Each image shows from its start to the NEXT image's start (preserving gaps).
    First image starts at 0 (even if prompt.start > 0).
    Last image holds until audio ends.
    """
    durations = []
    for i, item in enumerate(prompts):
        if i < len(prompts) - 1:
            next_start = prompts[i + 1]["start"]
            dur = next_start - item["start"]
        else:
            dur = audio_duration - item["start"]
        # First image covers from 0 to its next sibling start
        if i == 0:
            dur = (prompts[1]["start"] if len(prompts) > 1 else audio_duration)
        durations.append(max(dur, 0.1))
    return durations


def _timestamps_to_srt(timestamps: list[dict], srt_path: Path) -> None:
    lines = []
    for seg in timestamps:
        start = _seconds_to_srt_time(seg["start"])
        end = _seconds_to_srt_time(seg["end"])
        lines.append(f"{seg['index']}\n{start} --> {end}\n{seg['text'].strip()}\n")
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def _seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def run(video_id: str, subtitles: bool = False) -> None:
    _check_ffmpeg()

    video_dir = Path(config.OUTPUT_DIR) / video_id
    prompts_path = video_dir / "image_prompts.json"
    audio_path = video_dir / "audio.mp3"
    images_dir = video_dir / "images"
    output_path = video_dir / "final.mp4"

    # Check for trimmed audio from demo mode
    trimmed_audio_path = video_dir / "audio_trimmed.mp3"
    if trimmed_audio_path.exists():
        audio_path = trimmed_audio_path

    for path, name in [(prompts_path, "image_prompts.json"), (audio_path, "audio.mp3")]:
        if not path.exists():
            logger.error("{} not found: {}", name, path)
            sys.exit(1)

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    n = len(prompts)

    missing = [p["index"] for p in prompts if not (images_dir / f"img_{p['index']:03d}.png").exists()]
    if missing:
        logger.error("Missing {} images: {}", len(missing), missing[:10])
        sys.exit(1)

    audio_duration = _get_audio_duration(audio_path)
    logger.info("Audio duration: {:.2f}s", audio_duration)

    clip_durations = _compute_clip_durations(prompts, audio_duration)
    total_video_dur = sum(clip_durations)
    logger.info("Rendering video: {} images, total {:.2f}s → {}", n, total_video_dur, output_path)

    # Generate .srt sidecar (always — useful for YouTube caption upload)
    timestamps_path = video_dir / "timestamps.json"
    srt_path = video_dir / "subtitles.srt"
    if timestamps_path.exists():
        # Only include timestamps that fall within our prompt range
        all_ts = json.loads(timestamps_path.read_text(encoding="utf-8"))
        last_end = prompts[-1]["end"]
        demo_ts = [t for t in all_ts if t["start"] <= last_end]
        _timestamps_to_srt(demo_ts, srt_path)
        logger.info("Subtitles: {} → {}", len(demo_ts), srt_path.name)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        clip_list_path = tmpdir_path / "clips.txt"
        clip_lines = []

        logger.info("Converting {} images to clips...", n)
        for i, (item, duration) in enumerate(zip(prompts, clip_durations)):
            img_file = images_dir / f"img_{item['index']:03d}.png"
            clip_path = tmpdir_path / f"clip_{i:04d}.mp4"

            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-t", f"{duration:.3f}", "-i", str(img_file),
                "-vf", f"scale={config.VIDEO_WIDTH}:{config.VIDEO_HEIGHT}:flags=lanczos",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-r", str(config.VIDEO_FPS),
                str(clip_path),
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                logger.error("Failed clip {}: {}", i, result.stderr[-500:].decode(errors="ignore"))
                sys.exit(1)

            clip_lines.append(f"file '{clip_path}'")

            if (i + 1) % 20 == 0:
                logger.info("  {}/{} clips done", i + 1, n)

        clip_list_path.write_text("\n".join(clip_lines), encoding="utf-8")

        logger.info("Concatenating clips and muxing audio...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(clip_list_path),
            "-i", str(audio_path),
            "-c:v", "libx264", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-b:v", config.VIDEO_BITRATE,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-shortest",
            "-r", str(config.VIDEO_FPS),
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("FFmpeg concat failed:\n{}", result.stderr[-3000:])
            sys.exit(1)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Video rendered: {} ({:.1f} MB)", output_path, file_size_mb)
