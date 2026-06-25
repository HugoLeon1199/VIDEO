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

ICONS_DIR = Path("assets/icons")
SFX_DIR = Path("assets/sfx")
SFX_LIBRARY_PATH = Path("assets/sfx/library.json")


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
        # First image: duration from its own start to next image's start
        if i == 0:
            dur = (prompts[1]["start"] - prompts[0]["start"]) if len(prompts) > 1 else audio_duration
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


def _apply_overlays(img_path: Path, scene: dict, tmp_dir: Path) -> Path:
    """Composite icon overlays onto image using FFmpeg. Returns path to composited PNG."""
    icon_overlays = scene.get("icon_overlays", [])
    if not icon_overlays:
        return img_path

    # Filter to icons that actually exist on disk
    valid = []
    for ov in icon_overlays:
        icon_name = ov.get("icon", "")
        icon_path = ICONS_DIR / f"{icon_name}.png"
        if icon_path.exists():
            valid.append((icon_path, ov))
        else:
            logger.warning("Icon not found, skipping: {}", icon_path)

    if not valid:
        return img_path

    out_path = tmp_dir / f"overlay_{img_path.stem}.png"

    # Build FFmpeg filter_complex: stack icons vertically centered on image
    # Each icon: 200x200, spaced 120px apart, centered horizontally
    icon_size = 200
    icon_gap = 120
    n = len(valid)
    total_h = n * icon_size + (n - 1) * icon_gap

    inputs = ["-i", str(img_path)]
    for icon_path, _ in valid:
        inputs += ["-i", str(icon_path)]

    # Build overlay chain: [0][1]overlay=x:y[tmp1]; [tmp1][2]overlay=x:y[tmp2]; ...
    filter_parts = []
    prev = "0"
    for j, (icon_path, _) in enumerate(valid):
        label_in = str(j + 1)
        label_out = f"tmp{j}"
        y_offset = (config.VIDEO_HEIGHT - total_h) // 2 + j * (icon_size + icon_gap)
        x_offset = f"(W-{icon_size})/2"
        overlay = (
            f"[{prev}][{label_in}]overlay="
            f"x={x_offset}:y={y_offset}"
        )
        if j < len(valid) - 1:
            overlay += f"[{label_out}]"
            prev = label_out
        filter_parts.append(overlay)

    filter_complex = "; ".join(filter_parts)

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + ["-filter_complex", filter_complex, "-frames:v", "1", str(out_path)]
    )
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning(
            "Icon overlay failed for {}: {}",
            img_path.name,
            result.stderr[-300:].decode(errors="ignore"),
        )
        return img_path

    return out_path


def _mix_sfx_audio(
    voice_path: Path,
    prompts: list[dict],
    soundscape: list[dict],
    sfx_dir: Path,
    library: list[dict],
    out_path: Path,
) -> Path:
    """Mix SFX events from soundscape.json into voice audio via FFmpeg.

    Returns out_path on success, or voice_path unchanged if no SFX / error.
    """
    tag_map = {item["tag"]: item for item in library}
    scene_map = {p["index"]: p for p in prompts}

    # Flatten all events with resolved absolute start time and file path
    entries = []
    for scene_entry in soundscape:
        scene_idx = scene_entry.get("scene_index")
        scene = scene_map.get(scene_idx)
        if scene is None:
            logger.warning("SFX: scene_index {} not found in prompts — skipping", scene_idx)
            continue
        for ev in scene_entry.get("events", []):
            tag = ev.get("tag", "")
            lib_item = tag_map.get(tag)
            if lib_item is None:
                logger.warning("SFX: unknown tag '{}' — skipping", tag)
                continue
            sfx_file = sfx_dir / lib_item["file"]
            if not sfx_file.exists():
                logger.warning("SFX: file not found for tag '{}': {} — skipping", tag, sfx_file)
                continue
            abs_start_ms = int((scene["start"] + ev.get("offset", 0.0)) * 1000)
            duration_mode = ev.get("duration_mode", lib_item["duration_mode"])
            scene_duration = scene["end"] - scene["start"]
            entries.append({
                "path": sfx_file,
                "delay_ms": abs_start_ms,
                "volume": ev.get("volume", lib_item["default_volume"]),
                "duration_mode": duration_mode,
                "scene_duration": scene_duration,
                "fade_in": ev.get("fade_in", 0.0),
                "fade_out": ev.get("fade_out", 0.0),
            })

    if not entries:
        logger.info("SFX: no valid events — skipping mix")
        return voice_path

    # Build FFmpeg filter_complex
    inputs = ["-i", str(voice_path)]
    filter_parts = []
    mix_labels = ["0:a"]

    for j, entry in enumerate(entries):
        inputs += ["-i", str(entry["path"])]
        src = str(j + 1)
        label = f"s{j}"
        delay = f"{entry['delay_ms']}|{entry['delay_ms']}"
        flt = f"[{src}]adelay={delay},volume={entry['volume']}"

        mode = entry["duration_mode"]
        dur = entry["scene_duration"]
        if mode in ("scene", "loop") and dur > 0:
            if mode == "loop":
                # loop enough times then trim
                flt += f",aloop=loop=99:size=2e+09,atrim=duration={dur:.3f}"
            else:
                flt += f",atrim=duration={dur:.3f}"

        if entry["fade_in"] > 0:
            flt += f",afade=t=in:st=0:d={entry['fade_in']:.2f}"
        if entry["fade_out"] > 0 and mode in ("scene", "loop") and dur > 0:
            fade_start = max(0.0, dur - entry["fade_out"])
            flt += f",afade=t=out:st={fade_start:.2f}:d={entry['fade_out']:.2f}"

        flt += f"[{label}]"
        filter_parts.append(flt)
        mix_labels.append(label)

    n_inputs = len(mix_labels)
    mix_in = "".join(f"[{l}]" for l in mix_labels)
    filter_parts.append(f"{mix_in}amix=inputs={n_inputs}:normalize=0[sfxout]")
    filter_complex = "; ".join(filter_parts)

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + [
            "-filter_complex", filter_complex,
            "-map", "[sfxout]",
            "-c:a", "libmp3lame", "-qscale:a", "0",
            str(out_path),
        ]
    )
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning(
            "SFX mix failed (using voice only): {}",
            result.stderr[-400:].decode(errors="ignore"),
        )
        return voice_path

    logger.info("SFX mixed: {} events → {}", len(entries), out_path)
    return out_path


def run(video_id: str, subtitles: bool = False) -> None:
    _check_ffmpeg()

    video_dir = Path(config.OUTPUT_DIR) / video_id
    prompts_path = video_dir / "image_prompts.json"
    audio_path = video_dir / "audio.mp3"
    images_dir = video_dir / "images"
    # Fall back to track-specific subdirs if no rendered PNGs in default dir
    if not (images_dir / "img_001.png").exists():
        for candidate in ["images_flat2d", "images_en", "images_vi"]:
            candidate_dir = video_dir / candidate
            if (candidate_dir / "img_001.png").exists():
                images_dir = candidate_dir
                break
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
        demo_ts = [dict(t) for t in all_ts if t["start"] <= last_end]
        for t in demo_ts:
            t["end"] = min(t["end"], audio_duration)
        _timestamps_to_srt(demo_ts, srt_path)
        logger.info("Subtitles: {} → {}", len(demo_ts), srt_path.name)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Mix SFX into audio if soundscape.json + library present
        soundscape_path = video_dir / "soundscape.json"
        if soundscape_path.exists() and SFX_LIBRARY_PATH.exists():
            soundscape = json.loads(soundscape_path.read_text(encoding="utf-8"))
            library = json.loads(SFX_LIBRARY_PATH.read_text(encoding="utf-8"))
            sfx_out = tmpdir_path / "audio_with_sfx.mp3"
            audio_path = _mix_sfx_audio(audio_path, prompts, soundscape, SFX_DIR, library, sfx_out)

        clip_list_path = tmpdir_path / "clips.txt"
        clip_lines = []

        logger.info("Converting {} images to clips...", n)
        for i, (item, duration) in enumerate(zip(prompts, clip_durations)):
            img_file = images_dir / f"img_{item['index']:03d}.png"
            img_file = _apply_overlays(img_file, item, tmpdir_path)
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
            "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2",
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
