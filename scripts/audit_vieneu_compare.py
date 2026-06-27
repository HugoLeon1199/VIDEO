"""Audit VieNeu behavior by comparing paragraph/full-script/current outputs.

Usage:
    <codex-python> scripts/audit_vieneu_compare.py --video-id <slug>
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import unicodedata
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

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


def _slugify_voice(voice: str) -> str:
    normalized = unicodedata.normalize("NFKD", voice)
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    ascii_text = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")
    return ascii_text.lower() or "voice"


def _ffmpeg_bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    fallback = FFMPEG_FALLBACK if name == "ffmpeg" else FFPROBE_FALLBACK
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError(f"{name} not found in PATH or fallback location")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=False, **kwargs)


def _load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    ffmpeg = _ffmpeg_bin("ffmpeg")
    proc = _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-",
        ]
    )
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio, 48000


def _mp3_duration(path: Path) -> float:
    ffprobe = _ffmpeg_bin("ffprobe")
    proc = subprocess.run(
        [
            ffprobe,
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


def _analyze_silence(path: Path, threshold: float = 0.01, frame_ms: int = 50) -> dict:
    audio, sample_rate = _load_audio_mono(path)
    if audio.size == 0:
        return {
            "duration_sec": 0.0,
            "silence_ratio": 0.0,
            "silent_duration_sec": 0.0,
            "longest_silence_sec": 0.0,
            "silence_blocks_over_1_5s": 0,
        }

    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    frame_count = math.ceil(len(audio) / frame_len)
    pad = frame_count * frame_len - len(audio)
    if pad:
        audio = np.pad(audio, (0, pad))
    frames = audio.reshape(frame_count, frame_len)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    silent = rms <= threshold

    blocks = []
    start = None
    for idx, is_silent in enumerate(silent):
        if is_silent and start is None:
            start = idx
        elif not is_silent and start is not None:
            blocks.append((start, idx))
            start = None
    if start is not None:
        blocks.append((start, len(silent)))

    silent_frames = int(silent.sum())
    silent_duration = silent_frames * frame_ms / 1000
    block_durations = [(end - begin) * frame_ms / 1000 for begin, end in blocks]
    return {
        "duration_sec": round(len(audio) / sample_rate, 3),
        "silence_ratio": round(silent_duration / (len(audio) / sample_rate), 4),
        "silent_duration_sec": round(silent_duration, 3),
        "longest_silence_sec": round(max(block_durations, default=0.0), 3),
        "silence_blocks_over_1_5s": sum(1 for dur in block_durations if dur > 1.5),
    }


def _extract_mp3_excerpt(src: Path, dst: Path, start: float, duration: float) -> None:
    ffmpeg = _ffmpeg_bin("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
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


def _parse_helper_params(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    def _grab(pattern: str, cast=str):
        match = re.search(pattern, text)
        if not match:
            return None
        return cast(match.group(1))

    return {
        "path": _display_path(path),
        "voice": _grab(r'voice(?:_full)?\s*=\s*"([^"]+)"'),
        "temperature": _grab(r"temperature\s*=\s*([0-9.]+)", float),
        "top_k": _grab(r"top_k\s*=\s*([0-9.]+)", float),
        "top_p": _grab(r"top_p\s*=\s*([0-9.]+)", float),
        "repetition_penalty": _grab(r"repetition_penalty\s*=\s*([0-9.]+)", float),
        "max_chars": _grab(r"max_chars\s*=\s*([0-9.]+)", float),
        "crossfade_p": _grab(r"crossfade_p\s*=\s*([0-9.]+)", float),
        "silence_p": _grab(r"silence_p\s*=\s*([0-9.]+)", float),
        "uses_ref_codes": "ref_codes=" in text,
        "uses_full_script": "full script" in text.lower() or "tts.infer(\n    text" in text,
    }


def _write_mp3_from_float(audio: np.ndarray, sample_rate: int, output_path: Path) -> None:
    import soundfile as sf

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        sf.write(str(wav_path), audio, sample_rate)
        subprocess.run(
            [
                _ffmpeg_bin("ffmpeg"),
                "-y",
                "-i",
                str(wav_path),
                "-codec:a",
                "libmp3lame",
                "-qscale:a",
                "0",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        wav_path.unlink(missing_ok=True)


def _run_full_script_variant(script_text: str, output_path: Path, voice: str) -> dict:
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from vieneu import Vieneu

    params = {
        "voice": voice,
        "temperature": 0.5,
        "top_k": 20,
        "top_p": 0.90,
        "repetition_penalty": 1.2,
        "max_chars": 256,
        "crossfade_p": 0.1,
        "silence_p": 0.12,
        "apply_watermark": False,
    }
    tts = Vieneu()
    audio = tts.infer(script_text, **params)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    _write_mp3_from_float(audio, tts.sample_rate, output_path)
    return {
        "mode": "full_script",
        "voice": voice,
        "sample_rate": tts.sample_rate,
        "infer_kwargs": params,
    }


def _build_excerpts(
    timestamps: list[dict],
    diagnostics: dict,
    current_duration: float,
    compare_dir: Path,
    outputs: dict[str, Path],
) -> list[dict]:
    excerpts: list[dict] = []
    excerpts.append(
        {
            "slug": "first_30s",
            "label": "First 30s",
            "start": 0.0,
            "duration": min(30.0, current_duration),
            "note": "Opening sample",
        }
    )

    comma_sentence = next((ts for ts in timestamps if "," in ts.get("text", "")), None)
    if comma_sentence:
        excerpts.append(
            {
                "slug": "comma_sentence",
                "label": "Sentence with comma",
                "start": max(0.0, float(comma_sentence["start"]) - 1.0),
                "duration": min(18.0, current_duration - max(0.0, float(comma_sentence["start"]) - 1.0)),
                "note": comma_sentence["text"][:120],
            }
        )

    plain_sentence = next((ts for ts in timestamps if "," not in ts.get("text", "")), None)
    if plain_sentence:
        excerpts.append(
            {
                "slug": "plain_sentence",
                "label": "Sentence without comma",
                "start": max(0.0, float(plain_sentence["start"]) - 1.0),
                "duration": min(18.0, current_duration - max(0.0, float(plain_sentence["start"]) - 1.0)),
                "note": plain_sentence["text"][:120],
            }
        )

    transition = next((chunk for chunk in diagnostics.get("chunks", []) if not chunk.get("is_sentence_end")), None)
    if transition:
        transition_start = max(0.0, float(transition["speech_end"]) - 2.0)
        excerpts.append(
            {
                "slug": "chunk_transition",
                "label": "Chunk boundary",
                "start": transition_start,
                "duration": min(12.0, current_duration - transition_start),
                "note": transition["text"][:120],
            }
        )

    excerpts.append(
        {
            "slug": "final_30s",
            "label": "Final 30s",
            "start": max(0.0, current_duration - 30.0),
            "duration": min(30.0, current_duration),
            "note": "Ending sample",
        }
    )

    excerpt_dir = compare_dir / "excerpts"
    for excerpt in excerpts:
        for run_slug, output_path in outputs.items():
            duration = _mp3_duration(output_path)
            start = excerpt["start"]
            if excerpt["slug"] == "final_30s":
                start = max(0.0, duration - 30.0)
            clip_duration = min(float(excerpt["duration"]), max(0.1, duration - start))
            target = excerpt_dir / run_slug / f"{excerpt['slug']}.mp3"
            _extract_mp3_excerpt(output_path, target, start, clip_duration)
    return excerpts


def _write_report(
    report_path: Path,
    summary_path: Path,
    report: dict,
) -> None:
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# VieNeu Compare Audit",
        "",
        f"- Video: `{report['video_id']}`",
        f"- Active `steps/tts.py` path: `{report['active_path']}`",
        f"- Compare dir: `{report['compare_dir']}`",
        "",
        "## Run Parameters",
    ]
    for slug, run_info in report["runs"].items():
        params = run_info["config"]["infer_kwargs"]
        lines.extend(
            [
                f"### {slug}",
                f"- Output: `{run_info['output']}`",
                f"- Duration: {run_info['metrics']['duration_sec']}s",
                f"- Silence ratio: {run_info['metrics']['silence_ratio']}",
                f"- Longest silence: {run_info['metrics']['longest_silence_sec']}s",
                f"- Silence blocks >1.5s: {run_info['metrics']['silence_blocks_over_1_5s']}",
                f"- Params: `{json.dumps(params, ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Timing Checks",
            f"- Current pipeline timestamp entries: {report['timing_checks']['timestamp_count']}",
            f"- Current pipeline script sentence count: {report['timing_checks']['script_sentence_count']}",
            f"- Current pipeline final timestamp end: {report['timing_checks']['last_timestamp_end']}",
            f"- Current pipeline audio duration: {report['timing_checks']['current_audio_duration']}",
            "",
            "## Helper Script Inventory",
        ]
    )
    for helper in report["helper_inventory"]:
        lines.append(f"- `{helper['path']}`: `{json.dumps(helper, ensure_ascii=False, sort_keys=True)}`")

    lines.extend(
        [
            "",
            "## Excerpts To Review",
        ]
    )
    for excerpt in report["excerpts"]:
        lines.append(f"- `{excerpt['slug']}` ({excerpt['label']}): {excerpt['note']}")

    lines.extend(
        [
            "",
            "## Manual Listening Checklist",
            "- Repeated words or phrases",
            "- Duplicated clause openings",
            "- Abrupt pitch or prosody reset",
            "- Unnatural slowdown or stretched ending",
            "- Audible join artifact between chunks",
            "- Voice identity drift",
            "",
            "## Conclusion",
            f"- Auto summary: {report['auto_conclusion']}",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--voice", default="Thái Sơn")
    parser.add_argument("--voices", nargs="+", help="Optional list of voices to compare; overrides --voice and the default extra voice")
    parser.add_argument("--include-full-script", action="store_true", help="Also render the helper-style full-script reference")
    args = parser.parse_args()

    video_dir = Path(config.OUTPUT_DIR) / args.video_id
    script_path = video_dir / "script.txt"
    if not script_path.exists():
        raise SystemExit(f"script.txt not found: {script_path}")

    compare_dir = video_dir / "compare_vieneu"
    compare_dir.mkdir(parents=True, exist_ok=True)

    script_text = script_path.read_text(encoding="utf-8").strip()
    helper_inventory = [
        _parse_helper_params(Path("scripts/_gen_voice_compare.py")),
        _parse_helper_params(Path("scripts/_duc_tri_gen.py")),
    ]
    active_path = "_run_vieneu_sentence_mode" if 'if tts_engine == "vieneu"' in inspect.getsource(tts_step.run) else "unknown"

    voice_list = args.voices or [args.voice, "Bình An"]
    voice_list = list(dict.fromkeys(voice_list))
    primary_voice = voice_list[0]
    primary_slug = _slugify_voice(primary_voice)

    outputs: dict[str, Path] = {}
    runs: dict[str, dict] = {}

    print("Generating paragraph whole-block output...", flush=True)
    paragraph_output = compare_dir / f"paragraph_whole_blocks__{primary_slug}.mp3"
    paragraph_diag_path = compare_dir / f"paragraph_whole_blocks__{primary_slug}_diagnostics.json"
    if not paragraph_output.exists() or not paragraph_diag_path.exists():
        tts_step._run_vieneu_paragraph_mode(
            script_path=script_path,
            output_path=paragraph_output,
            voice=primary_voice,
            diagnostics_path=paragraph_diag_path,
        )
    outputs["paragraph_whole_blocks"] = paragraph_output
    runs["paragraph_whole_blocks"] = {
        "output": _display_path(paragraph_output),
        "config": json.loads(paragraph_diag_path.read_text(encoding="utf-8")),
        "metrics": _analyze_silence(paragraph_output),
    }

    for voice in voice_list:
        slug = _slugify_voice(voice)
        print(f"Generating fixed current pipeline output for voice={voice}...", flush=True)
        current_output = compare_dir / f"current_pipeline_fixed__{slug}.mp3"
        current_ts_path = compare_dir / f"current_pipeline_fixed__{slug}_timestamps.json"
        current_diag_path = compare_dir / f"current_pipeline_fixed__{slug}_diagnostics.json"
        if not current_output.exists() or not current_ts_path.exists() or not current_diag_path.exists():
            tts_step._run_vieneu_sentence_mode(
                script_path=script_path,
                output_path=current_output,
                timestamps_path=current_ts_path,
                voice=voice,
                diagnostics_path=current_diag_path,
            )
        key = f"current_pipeline_fixed__{slug}"
        outputs[key] = current_output
        runs[key] = {
            "output": _display_path(current_output),
            "config": json.loads(current_diag_path.read_text(encoding="utf-8")),
            "metrics": _analyze_silence(current_output),
        }
        if voice == primary_voice:
            outputs["current_pipeline_fixed"] = current_output
            primary_timestamps_path = current_ts_path
            primary_diag_key = key

    if args.include_full_script:
        print("Generating full-script reference...", flush=True)
        full_output = compare_dir / f"full_script__{primary_slug}.mp3"
        full_config = _run_full_script_variant(script_text, full_output, primary_voice)
        outputs["full_script"] = full_output
        runs["full_script"] = {
            "output": _display_path(full_output),
            "config": full_config,
            "metrics": _analyze_silence(full_output),
        }

    timestamps = json.loads(primary_timestamps_path.read_text(encoding="utf-8"))
    diagnostics = runs[primary_diag_key]["config"]
    current_duration = _mp3_duration(outputs["current_pipeline_fixed"])
    excerpts = _build_excerpts(
        timestamps,
        diagnostics,
        current_duration,
        compare_dir,
        {"current_pipeline_fixed": outputs["current_pipeline_fixed"]},
    )

    script_sentence_count = len(tts_step._split_script_sentences(script_path))
    last_timestamp_end = float(timestamps[-1]["end"]) if timestamps else 0.0
    timing_checks = {
        "timestamp_count": len(timestamps),
        "script_sentence_count": script_sentence_count,
        "last_timestamp_end": round(last_timestamp_end, 3),
        "current_audio_duration": round(current_duration, 3),
        "timestamp_matches_sentence_count": len(timestamps) == script_sentence_count,
        "timestamp_end_delta_sec": round(abs(last_timestamp_end - current_duration), 3),
    }

    paragraph_metrics = runs["paragraph_whole_blocks"]["metrics"]
    current_metrics = runs[primary_diag_key]["metrics"]
    auto_conclusion = []
    if paragraph_metrics["longest_silence_sec"] > current_metrics["longest_silence_sec"] + 2.0:
        auto_conclusion.append("Paragraph whole-block mode has materially longer silence than the current chunked pipeline.")
    if paragraph_metrics["silence_ratio"] < current_metrics["silence_ratio"] - 0.01:
        auto_conclusion.append("Paragraph whole-block mode reduces aggregate silence, so it may preserve prosody better if listening also sounds cleaner.")
    if timing_checks["timestamp_matches_sentence_count"]:
        auto_conclusion.append("Primary voice timestamps match script sentence count.")
    if timing_checks["timestamp_end_delta_sec"] <= 0.5:
        auto_conclusion.append("Primary voice timestamps end close to rendered audio duration.")
    if args.include_full_script:
        full_metrics = runs["full_script"]["metrics"]
        if full_metrics["longest_silence_sec"] > current_metrics["longest_silence_sec"] + 2.0:
            auto_conclusion.append("Fixed current pipeline also remains materially better than full-script VieNeu for silence control.")
    if not auto_conclusion:
        auto_conclusion.append("Silence metrics alone are inconclusive; use the excerpt clips for listening review.")

    report = {
        "video_id": args.video_id,
        "voice": args.voice,
        "compare_dir": _display_path(compare_dir),
        "active_path": active_path,
        "helper_inventory": helper_inventory,
        "runs": runs,
        "timing_checks": timing_checks,
        "excerpts": excerpts,
        "auto_conclusion": " ".join(auto_conclusion),
    }
    _write_report(compare_dir / "compare_report.json", compare_dir / "compare_report.md", report)

    print(json.dumps(report["timing_checks"], ensure_ascii=False, indent=2), flush=True)
    print(report["auto_conclusion"], flush=True)


if __name__ == "__main__":
    main()
