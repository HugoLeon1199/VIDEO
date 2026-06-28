"""Step 7: Render final video with timeline-safe effects composition."""

from __future__ import annotations

import json
import math
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


def _probe_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        ffprobe = str(Path(FFMPEG_WINGET_PATH) / "ffprobe.exe")
    result = subprocess.run(
        [ffprobe, "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _trim_audio(src: Path, dst: Path, end_time: float) -> None:
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
    if prompts and "display_start" in prompts[0] and "display_end" in prompts[0]:
        return [max(0.1, float(item["display_end"]) - float(item["display_start"])) for item in prompts]
    durations = []
    for i, item in enumerate(prompts):
        if i < len(prompts) - 1:
            next_start = prompts[i + 1]["start"]
            dur = next_start - item["start"]
        else:
            dur = audio_duration - item["start"]
        if i == 0:
            dur = (prompts[1]["start"] - prompts[0]["start"]) if len(prompts) > 1 else audio_duration
        durations.append(max(dur, 0.1))
    return durations


def _resolve_images_dir(video_dir: Path) -> Path:
    images_dir = video_dir / "images"
    if (images_dir / "img_001.png").exists():
        return images_dir
    for candidate in ["images_flat2d", "images_en", "images_vi"]:
        candidate_dir = video_dir / candidate
        if (candidate_dir / "img_001.png").exists():
            return candidate_dir
    return images_dir


def _apply_overlays(img_path: Path, scene: dict, tmp_dir: Path) -> Path:
    icon_overlays = scene.get("icon_overlays", [])
    if not icon_overlays:
        return img_path

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
    icon_size = 200
    icon_gap = 120
    n = len(valid)
    total_h = n * icon_size + (n - 1) * icon_gap
    inputs = ["-i", str(img_path)]
    for icon_path, _ in valid:
        inputs += ["-i", str(icon_path)]
    filter_parts = []
    prev = "0"
    for j, (_icon_path, _ov) in enumerate(valid):
        label_in = str(j + 1)
        label_out = f"tmp{j}"
        y_offset = (config.VIDEO_HEIGHT - total_h) // 2 + j * (icon_size + icon_gap)
        overlay = f"[{prev}][{label_in}]overlay=x=(W-{icon_size})/2:y={y_offset}"
        if j < len(valid) - 1:
            overlay += f"[{label_out}]"
            prev = label_out
        filter_parts.append(overlay)
    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + ["-filter_complex", "; ".join(filter_parts), "-frames:v", "1", str(out_path)]
    )
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning("Icon overlay failed for {}: {}", img_path.name, result.stderr[-300:].decode(errors="ignore"))
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
    tag_map = {item["tag"]: item for item in library}
    scene_map = {p["index"]: p for p in prompts}
    entries = []
    for scene_entry in soundscape:
        scene_idx = scene_entry.get("scene_index")
        scene = scene_map.get(scene_idx)
        if scene is None:
            continue
        for ev in scene_entry.get("events", []):
            tag = ev.get("tag", "")
            lib_item = tag_map.get(tag)
            if lib_item is None:
                continue
            sfx_file = sfx_dir / lib_item["file"]
            if not sfx_file.exists():
                continue
            abs_start_ms = int((scene["start"] + ev.get("offset", 0.0)) * 1000)
            duration_mode = ev.get("duration_mode", lib_item["duration_mode"])
            scene_duration = max(0.0, float(scene["end"]) - float(scene["start"]))
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

    with tempfile.TemporaryDirectory(prefix="sfxmix_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        script_path = tmpdir_path / "audio_filters.txt"
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
        mix_in = "".join(f"[{label}]" for label in mix_labels)
        filter_parts.append(f"{mix_in}amix=inputs={len(mix_labels)}:normalize=0[aout]")
        script_path.write_text(";\n".join(filter_parts), encoding="utf-8")
        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + [
                "-filter_complex_script", str(script_path),
                "-map", "[aout]",
                "-c:a", "pcm_s16le",
                "-ar", "48000",
                "-ac", "2",
                str(out_path),
            ]
        )
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.warning("SFX mix failed (using voice only): {}", result.stderr[-400:].decode(errors="ignore"))
            return voice_path
    return out_path


def _build_static_effects(prompts: list[dict], audio_duration: float) -> dict:
    scenes = []
    for idx, prompt in enumerate(prompts):
        display_start = 0.0 if idx == 0 else float(prompt["start"])
        display_end = float(prompts[idx + 1]["start"]) if idx + 1 < len(prompts) else audio_duration
        scenes.append(
            {
                "scene_index": int(prompt["index"]),
                "source_sentence_index": int(prompt.get("source_sentence_index", prompt["index"])),
                "source_start": float(prompt["start"]),
                "source_end": float(prompt["end"]),
                "display_start": round(display_start, 3),
                "display_end": round(display_end, 3),
                "motion": {
                    "type": "hold",
                    "start_scale": 1.0,
                    "end_scale": 1.0,
                    "focus_x": 0.5,
                    "focus_y": 0.45,
                    "easing": "ease_in_out",
                },
                "transition_out": {"type": "hard_cut", "duration": 0.0},
            }
        )
    return {
        "version": "cinematic-documentary-v1",
        "global_look": {
            "preset": config.EFFECTS_LOOK_PRESET,
            "grade": config.EFFECTS_DEFAULT_GRADE,
            "grain": 0.0,
            "vignette": 0.0,
            "enabled": False,
        },
        "effects_enabled": False,
        "scenes": scenes,
    }


def _load_effects_plan(video_dir: Path, prompts: list[dict], audio_duration: float) -> dict:
    effects_path = video_dir / "effects_plan.json"
    if effects_path.exists():
        plan = json.loads(effects_path.read_text(encoding="utf-8"))
        if plan.get("effects_enabled", True) is False:
            return plan
        return plan
    return _build_static_effects(prompts, audio_duration)


def _effect_stream_duration(scene: dict) -> float:
    transition = scene.get("transition_out", {})
    tail = float(transition.get("duration", 0.0)) if transition.get("type") != "hard_cut" else 0.0
    return max(0.1, (float(scene["display_end"]) - float(scene["display_start"])) + tail)


def _zoom_expr(motion: dict, frames: int) -> str:
    start_scale = float(motion["start_scale"])
    end_scale = float(motion["end_scale"])
    if frames <= 1 or abs(end_scale - start_scale) < 1e-6:
        return f"{start_scale:.5f}"
    delta = (end_scale - start_scale) / max(1, frames - 1)
    return f"if(eq(on,1),{start_scale:.5f},zoom+{delta:.7f})"


def _x_expr(motion: dict) -> str:
    motion_type = motion["type"]
    focus_x = float(motion.get("focus_x", 0.5))
    if motion_type == "pan_left_to_right":
        return "((on-1)/max(1\\,d-1))*(iw-iw/zoom)"
    if motion_type == "pan_right_to_left":
        return "(1-(on-1)/max(1\\,d-1))*(iw-iw/zoom)"
    return f"max(0,min(iw-iw/zoom,{focus_x:.5f}*iw-iw/zoom/2))"


def _y_expr(motion: dict) -> str:
    focus_y = float(motion.get("focus_y", 0.45))
    return f"max(0,min(ih-ih/zoom,{focus_y:.5f}*ih-ih/zoom/2))"


def _build_scene_filter(scene: dict, stream_index: int, fps: int) -> tuple[str, str]:
    duration = _effect_stream_duration(scene)
    frames = max(1, round(duration * fps))
    label = f"scene{stream_index}"
    motion = scene["motion"]
    fade_parts: list[str] = []
    incoming = float(scene.get("transition_in_duration", 0.0))
    outgoing = float(scene["transition_out"].get("duration", 0.0))
    if incoming > 0:
        fade_parts.append(f"fade=t=in:st=0:d={incoming:.3f}:alpha=1")
    if outgoing > 0 and scene["transition_out"]["type"] in {"crossfade", "dip_to_black"}:
        relative_boundary = float(scene["display_end"]) - float(scene["display_start"])
        fade_parts.append(f"fade=t=out:st={relative_boundary:.3f}:d={outgoing:.3f}:alpha=1")
    filter_chain = [
        f"[{stream_index}:v]format=rgba",
        f"zoompan=z='{_zoom_expr(motion, frames)}':x='{_x_expr(motion)}':y='{_y_expr(motion)}':d={frames}:s={config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}:fps={fps}",
        f"trim=duration={duration:.3f}",
        *fade_parts,
        f"setpts=PTS-STARTPTS+{float(scene['display_start']):.3f}/TB[{label}]",
    ]
    return label, ",".join(filter_chain)


def _build_black_overlay_filters(scenes: list[dict]) -> list[tuple[str, str]]:
    filters: list[tuple[str, str]] = []
    for idx, scene in enumerate(scenes):
        transition = scene["transition_out"]
        if transition["type"] != "dip_to_black" or transition["duration"] <= 0:
            continue
        label = f"blk{idx}"
        duration = float(transition["duration"])
        start = float(scene["display_end"])
        chain = (
            f"color=c=black@1.0:s={config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}:r={config.VIDEO_FPS}:d={duration:.3f},"
            f"format=rgba,fade=t=out:st=0:d={duration:.3f}:alpha=1,"
            f"setpts=PTS-STARTPTS+{start:.3f}/TB[{label}]"
        )
        filters.append((label, chain))
    return filters


def _build_video_filter_script(
    scenes: list[dict],
    audio_duration: float,
    preview_window: tuple[float, float] | None = None,
) -> tuple[str, str]:
    fps = config.VIDEO_FPS
    filter_parts = [f"color=c=black:s={config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}:r={fps}:d={audio_duration:.3f}[base0]"]
    scene_labels = []
    for stream_index, scene in enumerate(scenes):
        label, chain = _build_scene_filter(scene, stream_index, fps)
        scene_labels.append(label)
        filter_parts.append(chain)
    overlay_base = "base0"
    for label in scene_labels:
        out_label = f"ov_{label}"
        filter_parts.append(f"[{overlay_base}][{label}]overlay=eof_action=pass:repeatlast=0[{out_label}]")
        overlay_base = out_label
    for label, chain in _build_black_overlay_filters(scenes):
        filter_parts.append(chain)
        out_label = f"ov_{label}"
        filter_parts.append(f"[{overlay_base}][{label}]overlay=eof_action=pass:repeatlast=0[{out_label}]")
        overlay_base = out_label
    final_label = "vout"
    look_enabled = bool(config.EFFECTS_LOOK_ENABLED and scenes and scenes[0].get("plan_look_enabled", True))
    look_parts = []
    if look_enabled:
        look_parts.extend(
            [
                "eq=contrast=1.02:brightness=0.01:saturation=1.03:gamma=1.01",
                f"noise=alls={max(0, round(config.EFFECTS_DEFAULT_GRAIN * 100))}:allf=t+u",
                "vignette=PI/5",
            ]
        )
    if preview_window is not None:
        start, end = preview_window
        look_parts.extend([f"trim=start={start:.3f}:end={end:.3f}", "setpts=PTS-STARTPTS"])
    if look_parts:
        filter_parts.append(f"[{overlay_base}]{','.join(look_parts)}[{final_label}]")
    else:
        filter_parts.append(f"[{overlay_base}]null[{final_label}]")
    return ";\n".join(filter_parts), final_label


def _build_audio_filter_script(audio_input_index: int, preview_window: tuple[float, float] | None = None) -> tuple[str | None, str]:
    if preview_window is None:
        return None, f"{audio_input_index}:a"
    start, end = preview_window
    return f"[{audio_input_index}:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[aout]", "aout"


def _prepare_effect_scenes(prompts: list[dict], effects_plan: dict) -> list[dict]:
    scenes = []
    plan_scenes = effects_plan.get("scenes", [])
    for idx, prompt in enumerate(prompts):
        plan_scene = plan_scenes[idx] if idx < len(plan_scenes) else None
        if plan_scene is None:
            raise RuntimeError("effects_plan must contain one entry per prompt scene")
        entry = dict(plan_scene)
        previous = plan_scenes[idx - 1] if idx > 0 else None
        incoming_duration = 0.0
        if previous is not None and previous.get("transition_out", {}).get("type") in {"crossfade", "dip_to_black"}:
            incoming_duration = float(previous["transition_out"].get("duration", 0.0))
        entry["transition_in_duration"] = incoming_duration
        entry["plan_look_enabled"] = bool(effects_plan.get("global_look", {}).get("enabled", False))
        scenes.append(entry)
    return scenes


def _prepare_image_inputs(video_dir: Path, prompts: list[dict], tmpdir_path: Path) -> list[Path]:
    images_dir = _resolve_images_dir(video_dir)
    prepared: list[Path] = []
    for item in prompts:
        img_file = images_dir / f"img_{int(item['index']):03d}.png"
        if not img_file.exists():
            raise FileNotFoundError(f"Missing image for scene {item['index']}: {img_file}")
        overlaid = _apply_overlays(img_file, item, tmpdir_path)
        out_path = tmpdir_path / f"i{int(item['index']):03d}.png"
        if overlaid != out_path:
            shutil.copy2(overlaid, out_path)
        prepared.append(out_path)
    return prepared


def _render_composition(
    video_id: str,
    output_path: Path,
    *,
    preview_window: tuple[float, float] | None = None,
) -> Path:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    prompts_path = video_dir / "image_prompts.json"
    audio_path = video_dir / "audio.mp3"
    if not prompts_path.exists() or not audio_path.exists():
        raise FileNotFoundError("image_prompts.json and audio.mp3 are required for render")
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    audio_duration = _get_audio_duration(audio_path)
    effects_plan = _load_effects_plan(video_dir, prompts, audio_duration)
    scenes = _prepare_effect_scenes(prompts, effects_plan)
    with tempfile.TemporaryDirectory(prefix="renderfx_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        prepared_images = _prepare_image_inputs(video_dir, prompts, tmpdir_path)
        working_audio = audio_path
        soundscape_path = video_dir / "soundscape.json"
        if soundscape_path.exists() and SFX_LIBRARY_PATH.exists():
            soundscape = json.loads(soundscape_path.read_text(encoding="utf-8"))
            library = json.loads(SFX_LIBRARY_PATH.read_text(encoding="utf-8"))
            sfx_out = tmpdir_path / "audio_with_sfx.wav"
            working_audio = _mix_sfx_audio(audio_path, prompts, soundscape, SFX_DIR, library, sfx_out)
        video_script, video_label = _build_video_filter_script(scenes, audio_duration, preview_window=preview_window)
        audio_script, audio_label = _build_audio_filter_script(len(prepared_images), preview_window=preview_window)
        script_path = tmpdir_path / "filter_graph.txt"
        script_body = video_script
        if audio_script:
            script_body += ";\n" + audio_script
        script_path.write_text(script_body, encoding="utf-8")
        cmd = ["ffmpeg", "-y"]
        for image_path in prepared_images:
            cmd.extend(["-loop", "1", "-i", str(image_path)])
        cmd.extend(["-i", str(working_audio)])
        cmd.extend(
            [
                "-filter_complex_script", str(script_path),
                "-map", f"[{video_label}]",
                "-map", f"[{audio_label}]",
                "-c:v", "libx264",
                "-preset", config.VIDEO_PRESET,
                "-crf", str(config.VIDEO_CRF),
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "320k",
                "-ar", "48000",
                "-ac", "2",
                "-shortest",
                "-r", str(config.VIDEO_FPS),
                str(output_path),
            ]
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("FFmpeg render failed:\n{}", result.stderr[-4000:])
            sys.exit(1)
    target_duration = audio_duration if preview_window is None else preview_window[1] - preview_window[0]
    actual_duration = _probe_duration(output_path)
    if actual_duration <= 0:
        logger.warning("Could not verify rendered duration for {}", output_path)
    elif abs(actual_duration - target_duration) > (1.0 / max(1, config.VIDEO_FPS)) + 0.02:
        raise RuntimeError(f"Rendered duration drifted from target timeline ({actual_duration:.3f}s vs {target_duration:.3f}s)")
    return output_path


def run(video_id: str, subtitles: bool = False) -> None:
    _check_ffmpeg()
    video_dir = Path(config.OUTPUT_DIR) / video_id
    output_path = video_dir / "final.mp4"
    _render_composition(video_id, output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Video rendered: {} ({:.1f} MB)", output_path, file_size_mb)
    if subtitles:
        from steps import subtitles as subtitle_step

        logger.info("Generating subtitle artifacts for burned output...")
        subtitle_result = subtitle_step.generate(video_id, style=config.SUBTITLE_DEFAULT_STYLE)
        final_subbed_path = video_dir / "final_subbed.mp4"
        subtitle_step.burn_subtitles(output_path, subtitle_result["ass_path"], final_subbed_path)
        logger.info("Subtitle-burned video rendered: {}", final_subbed_path)


def render_preview(video_id: str, seconds: float = 45.0) -> Path:
    _check_ffmpeg()
    video_dir = Path(config.OUTPUT_DIR) / video_id
    audio_duration = _get_audio_duration(video_dir / "audio.mp3")
    preview_end = min(audio_duration, max(0.1, float(seconds)))
    output_path = video_dir / "effects_preview.mp4"
    return _render_composition(video_id, output_path, preview_window=(0.0, preview_end))
