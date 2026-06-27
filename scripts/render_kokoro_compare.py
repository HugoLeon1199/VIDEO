"""Render compare-only English Kokoro audio without touching production audio.mp3.

Usage:
    <codex-python> scripts/render_kokoro_compare.py --video-id <slug> [--video-id <slug>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from steps import tts as tts_step


FFMPEG_FALLBACK = Path(
    r"C:\Users\LEON_RM\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
)
FFPROBE_FALLBACK = FFMPEG_FALLBACK.with_name("ffprobe.exe")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _ffmpeg_bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    fallback = FFMPEG_FALLBACK if name == "ffmpeg" else FFPROBE_FALLBACK
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError(f"{name} not found in PATH or fallback location")


def _mp3_duration(path: Path) -> float:
    proc = subprocess.run(
        [
            _ffmpeg_bin("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(proc.stdout.strip())


def _extract_mp3_excerpt(src: Path, dst: Path, start: float, duration: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _ffmpeg_bin("ffmpeg"),
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{max(0.1, duration):.3f}",
            "-i",
            str(src),
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "2",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )


def _load_sentences(script_path: Path) -> list[str]:
    return tts_step._split_script_sentences(script_path)


def _build_excerpt_plan(duration: float, timestamps: list[dict]) -> list[dict]:
    if timestamps:
        mid_ts = timestamps[len(timestamps) // 2]
        near_end_ts = timestamps[max(0, int(len(timestamps) * 0.85) - 1)]
        middle_start = max(0.0, float(mid_ts.get("start", duration * 0.45)) - 8.0)
        near_end_start = max(0.0, float(near_end_ts.get("start", duration * 0.75)) - 8.0)
    else:
        middle_start = max(0.0, duration * 0.45)
        near_end_start = max(0.0, duration * 0.75)

    return [
        {"slug": "first_30s", "start": 0.0, "duration": min(30.0, duration), "label": "Opening sample"},
        {"slug": "middle_30s", "start": middle_start, "duration": min(30.0, max(0.1, duration - middle_start)), "label": "Middle multi-sentence sample"},
        {"slug": "near_end_30s", "start": near_end_start, "duration": min(30.0, max(0.1, duration - near_end_start)), "label": "Near-end sample"},
        {"slug": "final_30s", "start": max(0.0, duration - 30.0), "duration": min(30.0, duration), "label": "Ending sample"},
    ]


def _render_video_compare(video_id: str) -> dict:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    script_path = video_dir / "script.txt"
    timestamps_path = video_dir / "timestamps.json"
    compare_dir = video_dir / "compare_kokoro"
    output_path = compare_dir / "full_script_kokoro.mp3"
    diagnostics_path = compare_dir / "tts_diagnostics.json"
    excerpt_dir = compare_dir / "excerpts"

    compare_dir.mkdir(parents=True, exist_ok=True)
    script_text = script_path.read_text(encoding="utf-8").strip()
    sentences = _load_sentences(script_path)
    timestamps = []
    if timestamps_path.exists():
        timestamps = json.loads(timestamps_path.read_text(encoding="utf-8"))

    diagnostics = {
        "video_id": video_id,
        "engine": "kokoro",
        "voice": config.TTS_VOICE,
        "speed": config.TTS_SPEED,
        "script_chars": len(script_text),
        "script_sentence_count": len(sentences),
        "fallback_used": False,
        "success": False,
        "output": _display_path(output_path),
    }

    try:
        tts_step._kokoro_tts(script_text, output_path)
        duration = _mp3_duration(output_path)
        diagnostics["success"] = True
        diagnostics["duration_sec"] = round(duration, 3)

        excerpt_plan = _build_excerpt_plan(duration, timestamps)
        for excerpt in excerpt_plan:
            _extract_mp3_excerpt(
                output_path,
                excerpt_dir / f"{excerpt['slug']}.mp3",
                float(excerpt["start"]),
                float(excerpt["duration"]),
            )
        diagnostics["excerpts"] = excerpt_plan
    except Exception as exc:
        diagnostics["error"] = str(exc)

    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return diagnostics


def _write_summary(results: list[dict]) -> None:
    summary_path = Path(config.OUTPUT_DIR) / "_kokoro_compare_summary.json"
    summary_md_path = Path(config.OUTPUT_DIR) / "_kokoro_compare_summary.md"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Kokoro English Compare Summary",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result['video_id']}",
                f"- Success: `{result['success']}`",
                f"- Voice: `{result['voice']}`",
                f"- Speed: `{result['speed']}`",
                f"- Duration: `{result.get('duration_sec', 'n/a')}`",
                f"- Output: `{result['output']}`",
                f"- Fallback used: `{result['fallback_used']}`",
                "",
            ]
        )
        if not result["success"]:
            lines.append(f"- Error: `{result.get('error', 'unknown')}`")
            lines.append("")
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", action="append", required=True)
    args = parser.parse_args()

    results = []
    for video_id in args.video_id:
        results.append(_render_video_compare(video_id))
    _write_summary(results)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
