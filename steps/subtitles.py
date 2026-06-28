from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

import config
from steps import render_video as render_step
from steps import transcribe as transcribe_step
from steps.text_units import load_sentence_units

STYLE_CINEMATIC_CLEAN = "cinematic_clean"
STYLE_CINEMATIC_ACCENT = "cinematic_accent"
VALID_STYLES = {STYLE_CINEMATIC_CLEAN, STYLE_CINEMATIC_ACCENT}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fh:
        fh.write(text)
        tmp_path = Path(fh.name)
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _audio_duration(path: Path) -> float:
    if path.suffix.lower() == ".wav":
        try:
            import soundfile as sf

            info = sf.info(str(path))
            return info.frames / float(info.samplerate)
        except Exception:
            pass
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        ffprobe = str(Path(render_step.FFMPEG_WINGET_PATH) / "ffprobe.exe")
    result = subprocess.run(
        [ffprobe, "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}")
    return float(result.stdout.strip())


def _preferred_audio_path(video_dir: Path) -> Path:
    master = video_dir / "audio_master.wav"
    if master.exists():
        return master
    mp3 = video_dir / "audio.mp3"
    if mp3.exists():
        return mp3
    raise FileNotFoundError(f"No audio file found in {video_dir}")


def _canonical_script_words(script_path: Path) -> list[dict]:
    words: list[dict] = []
    global_word_index = 1
    for sentence in load_sentence_units(script_path):
        word_index = 1
        for raw_token in sentence.text.split():
            normalized = transcribe_step._normalize_token(raw_token)
            if not normalized:
                continue
            words.append(
                {
                    "sentence_index": sentence.sentence_index,
                    "word_index": word_index,
                    "global_word_index": global_word_index,
                    "text": raw_token,
                    "normalized": normalized,
                }
            )
            word_index += 1
            global_word_index += 1
    return words


def _load_word_assets(video_dir: Path) -> tuple[list[dict], dict, Path]:
    diagnostics_path = video_dir / transcribe_step.WORD_DIAGNOSTICS_NAME
    words_path = video_dir / transcribe_step.WORD_TIMESTAMPS_NAME
    if not diagnostics_path.exists():
        raise RuntimeError(
            f"Missing {diagnostics_path.name}. Run step 3 again before generating subtitles."
        )
    diagnostics = _load_json(diagnostics_path)
    if not diagnostics.get("subtitle_ready"):
        raise RuntimeError(
            f"Subtitles are not ready: {diagnostics.get('reason', 'unknown')}. "
            f"Re-run step 3 for this video to regenerate exact word alignment."
        )
    if not words_path.exists():
        raise RuntimeError(
            f"Missing {words_path.name} even though subtitle_ready=true. Re-run step 3."
        )
    return _load_json(words_path), diagnostics, words_path


def _wrap_tokens(tokens: list[str], max_chars_per_line: int | None = None) -> list[str]:
    if max_chars_per_line is None:
        max_chars_per_line = config.SUBTITLE_MAX_CHARS_PER_LINE
    lines: list[str] = []
    current: list[str] = []
    for token in tokens:
        candidate = " ".join(current + [token]).strip()
        if current and len(candidate) > max_chars_per_line:
            lines.append(" ".join(current).strip())
            current = [token]
        else:
            current.append(token)
    if current:
        lines.append(" ".join(current).strip())
    return [line for line in lines if line]


def _format_cue_text(tokens: list[str]) -> tuple[str, int]:
    lines = _wrap_tokens(tokens, max_chars_per_line=config.SUBTITLE_MAX_CHARS_PER_LINE)
    if len(lines) > 2:
        raise ValueError("Cue exceeds 2 lines")
    return "\n".join(lines), len(lines)


def _cue_gap_after(words: list[dict], end_index: int) -> float:
    if end_index + 1 >= len(words):
        return 0.0
    return max(0.0, float(words[end_index + 1]["start"]) - float(words[end_index]["end"]))


def _is_numeric_token(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _is_acronym_token(text: str) -> bool:
    letters = "".join(ch for ch in text if ch.isalpha())
    return bool(letters) and letters.isupper()


def _pick_cue_end(words: list[dict], start_index: int) -> int:
    max_end = min(len(words) - 1, start_index + config.SUBTITLE_MAX_WORDS - 1)
    best_end = start_index
    best_score = -10_000.0
    for end_index in range(start_index, max_end + 1):
        tokens = [item["text"] for item in words[start_index : end_index + 1]]
        word_count = len(tokens)
        duration = float(words[end_index]["end"]) - float(words[start_index]["start"])
        if word_count < config.SUBTITLE_MIN_WORDS and end_index != len(words) - 1:
            continue
        try:
            lines = _wrap_tokens(tokens)
        except Exception:
            continue
        if len(lines) > 2:
            continue
        score = 0.0
        if config.SUBTITLE_TARGET_MIN_SECONDS <= duration <= config.SUBTITLE_TARGET_MAX_SECONDS:
            score += 100.0
        else:
            score -= abs(duration - min(max(duration, config.SUBTITLE_TARGET_MIN_SECONDS), config.SUBTITLE_TARGET_MAX_SECONDS)) * 35.0
        score -= abs(word_count - 5) * 6.0
        gap_after = _cue_gap_after(words, end_index)
        score += min(gap_after, 0.8) * 20.0
        if tokens[-1][-1:] in {",", ".", "?", "!", ";", ":"}:
            score += 12.0
        if len(lines) == 1:
            score += 4.0
        if end_index == len(words) - 1:
            score += 2.0
        if score > best_score:
            best_score = score
            best_end = end_index
    return best_end


def _merge_short_cues(cues: list[dict]) -> list[dict]:
    merged: list[dict] = []
    idx = 0
    while idx < len(cues):
        cue = dict(cues[idx])
        duration = cue["end"] - cue["start"]
        word_count = cue["word_end"] - cue["word_start"] + 1
        if merged and (duration < config.SUBTITLE_TARGET_MIN_SECONDS or word_count < config.SUBTITLE_MIN_WORDS):
            previous = dict(merged[-1])
            combined_tokens = previous["_tokens"] + cue["_tokens"]
            if len(combined_tokens) <= config.SUBTITLE_MAX_WORDS:
                text, line_count = _format_cue_text(combined_tokens)
                previous["end"] = cue["end"]
                previous["text"] = text
                previous["line_count"] = line_count
                previous["word_end"] = cue["word_end"]
                previous["_tokens"] = combined_tokens
                merged[-1] = previous
                idx += 1
                continue
        merged.append(cue)
        idx += 1
    return merged


def _validate_word_stream(script_path: Path, word_timestamps: list[dict]) -> None:
    canonical = _canonical_script_words(script_path)
    if len(canonical) != len(word_timestamps):
        raise RuntimeError("Canonical word count does not match word_timestamps.json")
    for expected, actual in zip(canonical, word_timestamps):
        if expected["text"] != actual["text"]:
            raise RuntimeError(f"Word mismatch: expected '{expected['text']}' got '{actual['text']}'")
        if expected["normalized"] != actual["normalized"]:
            raise RuntimeError(f"Normalized word mismatch for '{expected['text']}'")


def build_subtitle_cues(
    script_path: Path,
    word_timestamps: list[dict],
    audio_duration: float,
    sentence_timestamps: list[dict] | None = None,
) -> list[dict]:
    _validate_word_stream(script_path, word_timestamps)
    sentence_timestamps_by_index = {item["index"]: item for item in sentence_timestamps or []}
    cues: list[dict] = []
    cursor = 0
    cue_index = 1
    previous_end = 0.0
    while cursor < len(word_timestamps):
        end_index = _pick_cue_end(word_timestamps, cursor)
        sentence_index = int(word_timestamps[cursor]["sentence_index"])
        sentence_end_index = cursor
        while (
            sentence_end_index + 1 < len(word_timestamps)
            and int(word_timestamps[sentence_end_index + 1]["sentence_index"]) == sentence_index
        ):
            sentence_end_index += 1
        if end_index < sentence_end_index:
            sentence_tail = word_timestamps[end_index + 1 : sentence_end_index + 1]
            if sentence_tail and all(float(item["end"]) <= float(item["start"]) for item in sentence_tail):
                end_index = sentence_end_index
        tokens = [item["text"] for item in word_timestamps[cursor : end_index + 1]]
        text, line_count = _format_cue_text(tokens)
        start = round(float(word_timestamps[cursor]["start"]), 3)
        end = round(min(float(word_timestamps[end_index]["end"]), audio_duration), 3)
        if end <= start:
            if sentence_timestamps_by_index:
                start_sentence = int(word_timestamps[cursor]["sentence_index"])
                end_sentence = int(word_timestamps[end_index]["sentence_index"])
                start = round(float(sentence_timestamps_by_index.get(start_sentence, {}).get("start", start)), 3)
                end = round(min(float(sentence_timestamps_by_index.get(end_sentence, {}).get("end", end)), audio_duration), 3)
            if end <= start:
                raise RuntimeError("Cue has non-positive duration")
        if start < previous_end:
            start = round(previous_end, 3)
        cues.append(
            {
                "index": cue_index,
                "start": start,
                "end": end,
                "text": text,
                "line_count": line_count,
                "word_start": cursor + 1,
                "word_end": end_index + 1,
                "_tokens": tokens,
            }
        )
        previous_end = end
        cue_index += 1
        cursor = end_index + 1

    cues = _merge_short_cues(cues)
    for idx, cue in enumerate(cues, start=1):
        cue["index"] = idx
    _validate_cues(word_timestamps, cues, audio_duration)
    return cues


def _validate_cues(word_timestamps: list[dict], cues: list[dict], audio_duration: float) -> None:
    reconstructed: list[str] = []
    overlap_count = 0
    previous_end = 0.0
    for cue in cues:
        if cue["end"] <= cue["start"]:
            raise RuntimeError("Cue end must be greater than start")
        if cue["start"] < previous_end:
            overlap_count += 1
        if cue["end"] > audio_duration + 0.001:
            raise RuntimeError("Cue exceeds audio duration")
        if cue["line_count"] > 2:
            raise RuntimeError("Cue exceeds 2 lines")
        reconstructed.extend(cue["_tokens"])
        previous_end = cue["end"]
    expected = [item["text"] for item in word_timestamps]
    if reconstructed != expected:
        raise RuntimeError("Cue word stream differs from canonical script")
    if overlap_count:
        raise RuntimeError("Cue overlap detected")


def _seconds_to_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    if millis == 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _seconds_to_ass(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:d}:{minutes:02d}:{secs:05.2f}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def _highlight_token(tokens: list[str]) -> int | None:
    for idx, token in enumerate(tokens):
        if _is_numeric_token(token):
            return idx
    for idx, token in enumerate(tokens):
        if _is_acronym_token(token):
            return idx
    return None


def _ass_dialogue_text(cue: dict, style: str) -> str:
    tokens = list(cue["_tokens"])
    if style == STYLE_CINEMATIC_ACCENT:
        highlight_index = _highlight_token(tokens)
        escaped_tokens = [_escape_ass(token) for token in tokens]
        if highlight_index is not None:
            escaped_tokens[highlight_index] = r"{\1c&H66C7F2&}" + escaped_tokens[highlight_index] + r"{\rDefault}"
            line_text = " ".join(escaped_tokens)
            return line_text.replace("\n", r"\N")
    return _escape_ass(cue["text"])


def _ass_header(style_name: str) -> str:
    outline = config.SUBTITLE_OUTLINE
    shadow = config.SUBTITLE_SHADOW
    margin_v = config.SUBTITLE_MARGIN_V
    font_size = config.SUBTITLE_FONT_SIZE
    font_name = config.SUBTITLE_FONT_FAMILY
    primary = config.SUBTITLE_PRIMARY_COLOR_ASS
    secondary = config.SUBTITLE_SECONDARY_COLOR_ASS
    outline_color = config.SUBTITLE_OUTLINE_COLOR_ASS
    shadow_color = config.SUBTITLE_SHADOW_COLOR_ASS
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {config.VIDEO_WIDTH}\n"
        f"PlayResY: {config.VIDEO_HEIGHT}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,"
        "ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"Style: Default,{font_name},{font_size},{primary},{secondary},{outline_color},{shadow_color},1,0,0,0,100,100,0,0,1,{outline},{shadow},2,120,120,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )


def _serialize_srt(cues: list[dict]) -> str:
    lines = []
    for cue in cues:
        lines.append(
            f"{cue['index']}\n{_seconds_to_srt(cue['start'])} --> {_seconds_to_srt(cue['end'])}\n{cue['text']}\n"
        )
    return "\n".join(lines)


def _serialize_ass(cues: list[dict], style: str) -> str:
    output = [_ass_header(style)]
    for cue in cues:
        output.append(
            f"Dialogue: 0,{_seconds_to_ass(cue['start'])},{_seconds_to_ass(cue['end'])},Default,,0,0,0,,"
            f"{{\\fad({config.SUBTITLE_FADE_MS},{config.SUBTITLE_FADE_MS})}}{_ass_dialogue_text(cue, style)}\n"
        )
    return "".join(output)


def _build_diagnostics(script_path: Path, word_timestamps: list[dict], cues: list[dict], audio_duration: float) -> dict:
    reconstructed = [token for cue in cues for token in cue["_tokens"]]
    expected = [item["text"] for item in word_timestamps]
    missing_word_count = max(0, len(expected) - len(reconstructed))
    repeated_word_count = max(0, len(reconstructed) - len(expected))
    overlap_count = sum(1 for left, right in zip(cues, cues[1:]) if left["end"] > right["start"])
    max_lines = max((cue["line_count"] for cue in cues), default=0)
    return {
        "sentence_count": len(load_sentence_units(script_path)),
        "word_count": len(word_timestamps),
        "cue_count": len(cues),
        "missing_word_count": missing_word_count,
        "repeated_word_count": repeated_word_count,
        "overlap_count": overlap_count,
        "max_lines": max_lines,
        "audio_duration": round(audio_duration, 3),
        "last_cue_end": round(cues[-1]["end"], 3) if cues else 0.0,
        "validation_passed": missing_word_count == 0 and repeated_word_count == 0 and overlap_count == 0 and max_lines <= 2,
    }


def generate(video_id: str, style: str = STYLE_CINEMATIC_CLEAN) -> dict:
    if style not in VALID_STYLES:
        raise ValueError(f"Unknown subtitle style: {style}")
    video_dir = Path(config.OUTPUT_DIR) / video_id
    script_path = video_dir / "script.txt"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script.txt for {video_id}")
    word_timestamps, _diagnostics, _words_path = _load_word_assets(video_dir)
    audio_path = _preferred_audio_path(video_dir)
    audio_duration = _audio_duration(audio_path)
    sentence_timestamps_path = video_dir / "timestamps.json"
    sentence_timestamps = _load_json(sentence_timestamps_path) if sentence_timestamps_path.exists() else None
    cues = build_subtitle_cues(script_path, word_timestamps, audio_duration, sentence_timestamps=sentence_timestamps)
    diagnostics = _build_diagnostics(script_path, word_timestamps, cues, audio_duration)
    if not diagnostics["validation_passed"]:
        raise RuntimeError("Subtitle validation failed")

    cues_payload = [{key: value for key, value in cue.items() if not key.startswith("_")} for cue in cues]
    _atomic_write_json(video_dir / "subtitle_cues.json", cues_payload)
    _atomic_write_text(video_dir / "subtitles.srt", _serialize_srt(cues))
    _atomic_write_text(video_dir / "subtitles.ass", _serialize_ass(cues, style))
    _atomic_write_json(video_dir / "subtitle_diagnostics.json", diagnostics)
    return {
        "cues": cues,
        "diagnostics": diagnostics,
        "audio_path": audio_path,
        "audio_duration": audio_duration,
        "ass_path": video_dir / "subtitles.ass",
    }


def _pick_preview_window(cues: list[dict], preview_seconds: float) -> tuple[float, float]:
    for start_index in range(len(cues)):
        window_start = cues[start_index]["start"]
        window_end = min(window_start + preview_seconds, cues[-1]["end"])
        window = [cue for cue in cues if cue["start"] < window_end and cue["end"] > window_start]
        if not window:
            continue
        lengths = [cue["word_end"] - cue["word_start"] + 1 for cue in window]
        flat_text = " ".join(cue["text"] for cue in window)
        has_short = any(length <= 4 for length in lengths)
        has_long = any(length >= 6 for length in lengths)
        has_feature = any(ch.isdigit() for ch in flat_text) or any(mark in flat_text for mark in [",", "?", ";"])
        if has_short and has_long and has_feature:
            return window_start, window_end
    first_start = cues[0]["start"]
    return first_start, min(first_start + preview_seconds, cues[-1]["end"])


def _rebase_preview_cues(cues: list[dict], preview_start: float, preview_end: float) -> list[dict]:
    preview_cues = []
    for cue in cues:
        if cue["start"] >= preview_end or cue["end"] <= preview_start:
            continue
        rebased = dict(cue)
        rebased["start"] = round(max(0.0, cue["start"] - preview_start), 3)
        rebased["end"] = round(min(preview_end, cue["end"]) - preview_start, 3)
        preview_cues.append(rebased)
    return preview_cues


def _trim_audio_segment(src: Path, dst: Path, start: float, duration: float) -> None:
    render_step._ensure_ffmpeg_path()
    codec_args = ["-c:a", "copy"]
    if dst.suffix.lower() == ".mp3":
        codec_args = ["-c:a", "libmp3lame", "-q:a", "3"]
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src), *codec_args, str(dst)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1000:])


def _render_preview_base_video(video_dir: Path, preview_start: float, preview_end: float, output_path: Path) -> None:
    prompts_path = video_dir / "image_prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(f"Missing image_prompts.json for preview: {prompts_path}")
    prompts = _load_json(prompts_path)
    audio_path = _preferred_audio_path(video_dir)
    full_audio_duration = _audio_duration(audio_path)
    clip_durations = render_step._compute_clip_durations(prompts, full_audio_duration)
    images_dir = render_step._resolve_images_dir(video_dir)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        trimmed_audio = tmpdir_path / "preview_audio.mp3"
        _trim_audio_segment(audio_path, trimmed_audio, preview_start, preview_end - preview_start)
        clip_lines = []
        clip_index = 0
        for prompt, full_duration in zip(prompts, clip_durations):
            clip_global_start = float(prompt["start"])
            clip_global_end = clip_global_start + float(full_duration)
            local_start = max(clip_global_start, preview_start)
            local_end = min(clip_global_end, preview_end)
            if local_end <= local_start:
                continue
            duration = local_end - local_start
            img_file = images_dir / f"img_{prompt['index']:03d}.png"
            clip_path = tmpdir_path / f"clip_{clip_index:04d}.mp4"
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loop",
                    "1",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    str(img_file),
                    "-vf",
                    f"scale={config.VIDEO_WIDTH}:{config.VIDEO_HEIGHT}:flags=lanczos",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    str(config.VIDEO_FPS),
                    str(clip_path),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr[-1000:])
            clip_lines.append(f"file '{clip_path.as_posix()}'")
            clip_index += 1

        clip_list_path = tmpdir_path / "clips.txt"
        clip_list_path.write_text("\n".join(clip_lines), encoding="utf-8")
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(clip_list_path),
                "-i",
                str(trimmed_audio),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-b:v",
                config.VIDEO_BITRATE,
                "-c:a",
                "aac",
                "-b:a",
                "320k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-shortest",
                "-r",
                str(config.VIDEO_FPS),
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])


def build_burn_command(input_video: Path, output_video: Path, ass_path: Path, working_dir: Path) -> list[str]:
    safe_ass = working_dir / "subtitles.ass"
    shutil.copy2(ass_path, safe_ass)
    render_step._ensure_ffmpeg_path()
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video.resolve()),
        "-vf",
        "subtitles=subtitles.ass",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(output_video.resolve()),
    ]


def burn_subtitles(input_video: Path, ass_path: Path, output_video: Path) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        cmd = build_burn_command(input_video, output_video, ass_path, tmpdir_path)
        result = subprocess.run(cmd, cwd=tmpdir_path, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])


def render_preview(video_id: str, style: str, preview_seconds: float) -> Path:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    subtitle_result = generate(video_id, style=style)
    cues = subtitle_result["cues"]
    preview_start, preview_end = _pick_preview_window(cues, preview_seconds)
    preview_cues = _rebase_preview_cues(cues, preview_start, preview_end)
    preview_style_name = "clean" if style == STYLE_CINEMATIC_CLEAN else "accent"
    output_path = video_dir / f"subtitle_preview_{preview_style_name}.mp4"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        preview_base = tmpdir_path / "preview_base.mp4"
        preview_ass = tmpdir_path / "subtitles.ass"
        _render_preview_base_video(video_dir, preview_start, preview_end, preview_base)
        _atomic_write_text(preview_ass, _serialize_ass(preview_cues, style))
        burn_subtitles(preview_base, preview_ass, output_path)
    return output_path
