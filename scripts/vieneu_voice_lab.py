"""Independent VieNeu voice lab for Vietnamese voice selection and continuity review."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steps import tts as tts_step  # noqa: E402

LAB_VERSION = 1
DEFAULT_SEED = 20260628
DEFAULT_OUTPUT_DIR = Path("output") / "vieneu_voice_lab"
ROUND_SEQUENCE = ["base", "topic", "style", "final"]
ROUND_LIMITS = {
    "topic": 5,
    "style": 3,
    "final": 2,
}
DECISION_PRIORITY = {
    "Keep": 0,
    "Maybe": 1,
    "Reject": 2,
    "": 3,
}
DECISIONS_FIELDNAMES = [
    "blind_id",
    "round",
    "decision",
    "revealed",
    "notes",
    "revealed_voice",
    "revealed_preset",
    "revealed_source",
    "revealed_params",
]

BASE_SAMPLE = (
    "Bạn có tin rằng cùng một giọng nói có thể giữ được sự tự nhiên khi phải đọc ngày 27 tháng 6 năm 2026, "
    "các con số 31.000, 2,7 tỷ, 18,5%, rồi chuyển sang AI, DNA, NASA, YouTube mà vẫn mượt không? "
    "Hôm nay chúng ta còn phải đi qua Nguyễn Trãi, Điện Biên Phủ, Neanderthal, Çatalhöyük, và một câu dài có dấu phẩy, dấu chấm phẩy; "
    "vì Leon cần nghe xem giọng này có đủ rõ, đủ êm, và đủ vững để kể lịch sử theo cách khiến người nghe ở lại đến cuối hay không."
)

TOPIC_REEL_TEXT = (
    "Lịch sử bắt đầu khi một dấu vết nhỏ trên đá buộc bạn phải nhìn lại cả một thời đại. "
    "Rồi bí ẩn xuất hiện, vì thứ tưởng như rõ ràng nhất thường lại che mất câu trả lời thật. "
    "Khoa học và AI bước vào sau đó, không để làm câu chuyện lạnh đi, mà để kiểm tra xem dữ liệu và trực giác gặp nhau ở đâu. "
    "Tài chính cũng có nhịp điệu riêng, bởi mỗi con số đều là một quyết định bị trì hoãn hoặc bị ép phải chọn ngay. "
    "Tâm lý và đời sống mới là phần giữ người nghe lại, vì ai cũng muốn biết một câu chuyện lớn chạm vào mình như thế nào. "
    "Cuối cùng là pronunciation stress: Nguyễn Trãi, Điện Biên Phủ, Neanderthal, Çatalhöyük, AI, DNA, NASA, YouTube."
)

STYLE_SAMPLE = (
    "Bạn có nghe thấy sự khác biệt rất nhẹ giữa một giọng chỉ đúng kỹ thuật và một giọng thật sự dễ nghe không? "
    "Ở đoạn này, Leon không cần kịch tính quá mức; Leon cần sự điềm tĩnh, độ rõ, và cảm giác tự nhiên đủ lâu để người nghe tin rằng người kể đang hiểu điều mình nói."
)

FINAL_SCRIPT = (
    "Bạn hãy tưởng tượng một đêm rất cũ, gió đi qua cửa hang, lửa cháy thấp, và một nhóm người ngồi lại đủ gần để nghe thấy nhau thở. "
    "Câu chuyện bắt đầu không phải bằng một phát hiện lớn, mà bằng một chi tiết nhỏ tưởng như vô hại: một dấu cắt, một mảnh xương, một vệt khoáng chất bám lên vách đá từ hàng chục nghìn năm trước. "
    "Nhưng lịch sử thường mở ra như thế, âm thầm trước khi trở nên chấn động. "
    "Khi bạn nhìn kỹ hơn, mọi thứ bắt đầu dịch chuyển. "
    "Một con số như 31.000 không còn chỉ là dữ liệu. "
    "Nó trở thành khoảng cách giữa bạn và một con người đã từng sống, từng đau, từng chờ được giúp, từng nghe người khác nói với mình trong bóng tối. "
    "Đó là lúc câu chuyện chuyển từ khảo cổ sang đời sống thật. "
    "Bởi vì nếu một cộng đồng có thể giữ ai đó sống sót qua thương tích, qua bệnh tật, qua thời gian hồi phục kéo dài, thì điều còn lại không chỉ là kỹ thuật. "
    "Điều còn lại là sự kiên nhẫn, ký ức, và khả năng phối hợp giữa những người biết rằng một sai lầm nhỏ có thể đổi bằng cả mạng sống. "
    "Ngày nay, AI, DNA, NASA hay YouTube làm cho thế giới của bạn có vẻ hiện đại hơn rất nhiều; nhưng cái lõi khiến một câu chuyện được tin vẫn cũ như thế. "
    "Bạn vẫn cần một giọng đủ rõ để giữ dữ kiện, đủ êm để người nghe không mệt, và đủ ổn định để khi câu chuyện chạm tới phần con người nhất, nó không bị vỡ ra thành trình diễn. "
    "Vì thế bài test cuối cùng không phải để xem giọng này có đẹp ở từng câu riêng lẻ hay không. "
    "Bài test cuối cùng là continuity: giọng đó có thể đi xuyên qua nhiều block, qua những chỗ ngắt, qua những đoạn thông tin dày đặc, mà vẫn giữ được cảm giác liền mạch, có người kể thật ở phía sau, và đủ tự nhiên để Leon tin rằng đây là giọng nên dùng lâu dài."
)

STYLE_PRESETS = {
    "production_default": {},
    "natural_calm": {
        "temperature": 0.38,
        "top_p": 0.90,
        "repetition_penalty": 1.15,
        "silence_p": 0.12,
    },
}


@dataclasses.dataclass
class LabArtifact:
    blind_id: str
    kind: str
    round: str
    round_order: int
    source_voice: str
    preset: str
    effective_infer_params: dict[str, Any]
    duration_seconds: float
    audio_wav: str
    audio_mp3: str
    source_ref: str
    sample_id: str
    metadata: dict[str, Any]


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _ffmpeg_path() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        r"C:\Users\LEON_RM\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("ffmpeg not found")


def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _write_wav(audio: np.ndarray, sample_rate: int, wav_path: Path) -> None:
    import soundfile as sf

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), audio, sample_rate)


def _wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    ffmpeg = _ffmpeg_path()
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-q:a", "3", str(mp3_path)],
        check=True,
        capture_output=True,
    )


def _wav_duration_seconds(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _blank_decision_row() -> dict[str, str]:
    return {field: "" for field in DECISIONS_FIELDNAMES}


def _load_decisions(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            blind_id = (row.get("blind_id") or "").strip()
            if not blind_id:
                continue
            normalized = _blank_decision_row()
            for field in DECISIONS_FIELDNAMES:
                normalized[field] = (row.get(field) or "").strip()
            rows[blind_id] = normalized
    return rows


def _write_decisions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DECISIONS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in DECISIONS_FIELDNAMES})


def _resolve_decisions_path(out_dir: Path, decisions: Path | None) -> Path:
    if decisions is not None:
        return decisions
    return out_dir / "decisions.csv"


def _generate_blind_ids(count: int, seed: int) -> list[str]:
    ids = [f"V{i:03d}" for i in range(1, max(count, 1) + 1)]
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[:count]


def _allocate_blind_id(existing_ids: set[str], seed: int) -> str:
    for candidate in _generate_blind_ids(999, seed):
        if candidate not in existing_ids:
            return candidate
    raise RuntimeError("Ran out of blind ids")


def _decision_sort_key(artifact: LabArtifact, decisions: dict[str, dict[str, str]]) -> tuple[int, int, str]:
    row = decisions.get(artifact.blind_id, {})
    return (DECISION_PRIORITY.get(row.get("decision", ""), 3), artifact.round_order, artifact.blind_id)


def _load_manifest(out_dir: Path) -> dict[str, Any]:
    return _load_json(
        out_dir / "manifest.json",
        default={
            "version": LAB_VERSION,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "active_round": None,
            "artifacts": [],
            "round_counts": {},
            "round_configs": {},
        },
    )


def _load_all_artifacts(manifest: dict[str, Any]) -> list[LabArtifact]:
    return [LabArtifact(**item) for item in manifest.get("artifacts", [])]


def _artifacts_for_round(manifest: dict[str, Any], round_name: str) -> list[LabArtifact]:
    artifacts = [artifact for artifact in _load_all_artifacts(manifest) if artifact.round == round_name]
    return sorted(artifacts, key=lambda item: (item.round_order, item.blind_id))


def _round_counts(artifacts: list[LabArtifact]) -> dict[str, int]:
    return {round_name: sum(1 for artifact in artifacts if artifact.round == round_name) for round_name in ROUND_SEQUENCE}


def _save_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    _save_json(out_dir / "manifest.json", manifest)


def _update_manifest(
    out_dir: Path,
    manifest: dict[str, Any],
    round_name: str,
    artifacts: list[LabArtifact],
    round_config: dict[str, Any],
) -> dict[str, Any]:
    existing = [artifact for artifact in _load_all_artifacts(manifest) if artifact.round != round_name]
    combined = existing + artifacts
    combined.sort(key=lambda item: (ROUND_SEQUENCE.index(item.round), item.round_order, item.blind_id))
    round_configs = dict(manifest.get("round_configs", {}))
    round_configs[round_name] = round_config
    manifest.update(
        {
            "version": LAB_VERSION,
            "updated_at": _now_iso(),
            "active_round": round_name,
            "artifacts": [dataclasses.asdict(item) for item in combined],
            "round_counts": _round_counts(combined),
            "round_configs": round_configs,
        }
    )
    _save_manifest(out_dir, manifest)
    return manifest


def _merge_decisions(path: Path, artifacts: list[LabArtifact]) -> None:
    existing = _load_decisions(path)
    for artifact in artifacts:
        row = existing.get(artifact.blind_id, _blank_decision_row())
        row["blind_id"] = artifact.blind_id
        row["round"] = artifact.round
        row["revealed_voice"] = artifact.source_voice
        row["revealed_preset"] = artifact.preset
        row["revealed_source"] = artifact.source_ref
        row["revealed_params"] = json.dumps(artifact.effective_infer_params, ensure_ascii=False, sort_keys=True)
        existing[artifact.blind_id] = row
    rows = list(existing.values())
    rows.sort(key=lambda item: (ROUND_SEQUENCE.index(item["round"]) if item["round"] in ROUND_SEQUENCE else 999, item["blind_id"]))
    _write_decisions(path, rows)


def _next_round_order(manifest: dict[str, Any], round_name: str) -> int:
    current = [artifact.round_order for artifact in _artifacts_for_round(manifest, round_name)]
    return max(current, default=0) + 1


def _validate_duration(
    label: str,
    duration: float,
    target_min: float,
    target_max: float,
    acceptable_min: float,
    acceptable_max: float,
) -> None:
    if duration < acceptable_min or duration > acceptable_max:
        raise RuntimeError(
            f"{label} duration {duration:.3f}s is outside acceptable range {acceptable_min:.1f}-{acceptable_max:.1f}s"
        )
    if duration < target_min or duration > target_max:
        warnings.warn(
            f"{label} duration {duration:.3f}s is outside target range {target_min:.1f}-{target_max:.1f}s",
            stacklevel=2,
        )


def _validate_final_duration(label: str, duration: float) -> None:
    if duration < 90.0 or duration > 120.0:
        raise RuntimeError(f"{label} duration {duration:.3f}s must be within 90.0-120.0s")


def _effective_infer_params(voice: str, preset: str) -> dict[str, Any]:
    params = tts_step._merge_dict(tts_step.VIENEU_INFER_DEFAULTS, STYLE_PRESETS[preset])
    params["voice"] = voice
    return params


def discover_vieneu_voices() -> list[dict[str, str]]:
    from vieneu import Vieneu

    voices = []
    tts = Vieneu()
    for item in tts.list_preset_voices():
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            voices.append({"display_name": str(item[0]), "voice_name": str(item[1])})
        elif isinstance(item, dict):
            display = str(item.get("display_name") or item.get("label") or item.get("name") or "")
            voice_name = str(item.get("voice_name") or item.get("name") or "")
            if voice_name:
                voices.append({"display_name": display or voice_name, "voice_name": voice_name})
    deduped = {entry["voice_name"]: entry for entry in voices}
    return [deduped[name] for name in sorted(deduped)]


def _render_direct_sample(
    out_dir: Path,
    relative_dir: str,
    blind_id: str,
    text: str,
    source_voice: str,
    preset: str,
) -> tuple[str, str, float, dict[str, Any]]:
    runtime = tts_step.VieNeuRuntime(
        voice=source_voice,
        block_config=tts_step._merge_dict(tts_step.VIENEU_BLOCK_DEFAULTS, None),
        infer_overrides=STYLE_PRESETS[preset],
    )
    sample_rate = runtime.sample_rate
    gap_ms = int(runtime.block_config["fallback_sentence_gap_ms"])
    gap = np.zeros(int(gap_ms / 1000 * sample_rate), dtype=np.float32)
    parts: list[np.ndarray] = []
    sentences = _split_sentences(text)
    if not sentences:
        raise RuntimeError("No sentences found in sample text")
    effective_params = _effective_infer_params(source_voice, preset)
    for offset, sentence in enumerate(sentences):
        audio, _sample_rate, _infer_params = runtime.synthesize(sentence)
        audio = tts_step._trim_trailing_silence(
            audio,
            sample_rate,
            threshold=runtime.block_config["trim_trailing_threshold"],
            keep_ms=runtime.block_config["trim_trailing_keep_ms"],
        )
        parts.append(audio)
        if offset != len(sentences) - 1:
            parts.append(gap)
    combined = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    wav_path = out_dir / relative_dir / f"{blind_id}.wav"
    mp3_path = out_dir / relative_dir / f"{blind_id}.mp3"
    _write_wav(combined, sample_rate, wav_path)
    _wav_to_mp3(wav_path, mp3_path)
    duration = _wav_duration_seconds(wav_path)
    return _relative(wav_path, out_dir), _relative(mp3_path, out_dir), duration, effective_params


def _select_finalists(
    manifest: dict[str, Any],
    decisions: dict[str, dict[str, str]],
    previous_round: str,
    limit: int,
) -> list[LabArtifact]:
    ranked = sorted(_artifacts_for_round(manifest, previous_round), key=lambda artifact: _decision_sort_key(artifact, decisions))
    ranked = [artifact for artifact in ranked if decisions.get(artifact.blind_id, {}).get("decision", "")]
    if not ranked:
        raise RuntimeError(f"No reviewed finalists found for round '{previous_round}'")
    return ranked[:limit]


def _base_round(out_dir: Path, seed: int, manifest: dict[str, Any]) -> tuple[dict[str, Any], list[LabArtifact]]:
    voices = discover_vieneu_voices()
    if not voices:
        raise RuntimeError("No VieNeu voices discovered from runtime")
    existing_ids = {artifact.blind_id for artifact in _load_all_artifacts(manifest)}
    next_order = _next_round_order(manifest, "base")
    artifacts: list[LabArtifact] = []
    for index, voice in enumerate(voices):
        blind_id = _allocate_blind_id(existing_ids, seed + index)
        existing_ids.add(blind_id)
        wav_rel, mp3_rel, duration, effective = _render_direct_sample(
            out_dir,
            "audio/base",
            blind_id,
            BASE_SAMPLE,
            voice["voice_name"],
            "production_default",
        )
        _validate_duration(f"base:{blind_id}", duration, 20.0, 25.0, 18.0, 30.0)
        artifacts.append(
            LabArtifact(
                blind_id=blind_id,
                kind="base",
                round="base",
                round_order=next_order + index,
                source_voice=voice["voice_name"],
                preset="production_default",
                effective_infer_params=effective,
                duration_seconds=duration,
                audio_wav=wav_rel,
                audio_mp3=mp3_rel,
                source_ref=voice["display_name"],
                sample_id="base_sample",
                metadata={"display_name": voice["display_name"]},
            )
        )
    round_config = {
        "target_seconds": [20, 25],
        "acceptable_seconds": [18, 30],
        "voice_count": len(voices),
        "finalist_limit": ROUND_LIMITS["topic"],
    }
    manifest = _update_manifest(out_dir, manifest, "base", artifacts, round_config)
    return manifest, artifacts


def _topic_round(out_dir: Path, seed: int, manifest: dict[str, Any], decisions_path: Path) -> tuple[dict[str, Any], list[LabArtifact]]:
    decisions = _load_decisions(decisions_path)
    finalists = _select_finalists(manifest, decisions, "base", ROUND_LIMITS["topic"])
    existing_ids = {artifact.blind_id for artifact in _load_all_artifacts(manifest)}
    next_order = _next_round_order(manifest, "topic")
    artifacts: list[LabArtifact] = []
    for index, source in enumerate(finalists):
        blind_id = _allocate_blind_id(existing_ids, seed + 100 + index)
        existing_ids.add(blind_id)
        wav_rel, mp3_rel, duration, effective = _render_direct_sample(
            out_dir,
            "audio/topic",
            blind_id,
            TOPIC_REEL_TEXT,
            source.source_voice,
            "production_default",
        )
        if duration < 45.0 or duration > 60.0:
            raise RuntimeError(f"topic:{blind_id} duration {duration:.3f}s must be within 45.0-60.0s")
        artifacts.append(
            LabArtifact(
                blind_id=blind_id,
                kind="topic",
                round="topic",
                round_order=next_order + index,
                source_voice=source.source_voice,
                preset="production_default",
                effective_infer_params=effective,
                duration_seconds=duration,
                audio_wav=wav_rel,
                audio_mp3=mp3_rel,
                source_ref=source.blind_id,
                sample_id="topic_reel",
                metadata={"source_blind_id": source.blind_id},
            )
        )
    round_config = {
        "finalists_from": "base",
        "finalist_limit": ROUND_LIMITS["topic"],
        "target_seconds": [45, 60],
    }
    manifest = _update_manifest(out_dir, manifest, "topic", artifacts, round_config)
    return manifest, artifacts


def _style_round(out_dir: Path, seed: int, manifest: dict[str, Any], decisions_path: Path) -> tuple[dict[str, Any], list[LabArtifact]]:
    decisions = _load_decisions(decisions_path)
    finalists = _select_finalists(manifest, decisions, "topic", ROUND_LIMITS["style"])
    existing_ids = {artifact.blind_id for artifact in _load_all_artifacts(manifest)}
    next_order = _next_round_order(manifest, "style")
    artifacts: list[LabArtifact] = []
    order = 0
    for source in finalists:
        for preset in ["production_default", "natural_calm"]:
            blind_id = _allocate_blind_id(existing_ids, seed + 200 + order)
            existing_ids.add(blind_id)
            wav_rel, mp3_rel, duration, effective = _render_direct_sample(
                out_dir,
                "audio/style",
                blind_id,
                STYLE_SAMPLE,
                source.source_voice,
                preset,
            )
            _validate_duration(f"style:{blind_id}", duration, 25.0, 30.0, 22.0, 35.0)
            artifacts.append(
                LabArtifact(
                    blind_id=blind_id,
                    kind="style",
                    round="style",
                    round_order=next_order + order,
                    source_voice=source.source_voice,
                    preset=preset,
                    effective_infer_params=effective,
                    duration_seconds=duration,
                    audio_wav=wav_rel,
                    audio_mp3=mp3_rel,
                    source_ref=source.blind_id,
                    sample_id="style_sample",
                    metadata={"source_blind_id": source.blind_id},
                )
            )
            order += 1
    round_config = {
        "finalists_from": "topic",
        "finalist_limit": ROUND_LIMITS["style"],
        "presets": ["production_default", "natural_calm"],
        "target_seconds": [25, 30],
        "acceptable_seconds": [22, 35],
    }
    manifest = _update_manifest(out_dir, manifest, "style", artifacts, round_config)
    return manifest, artifacts


def _analyze_block_metrics(wav_path: Path, threshold: float) -> dict[str, float]:
    import soundfile as sf

    audio, sample_rate = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    trailing = tts_step._measure_trailing_silence(audio, sample_rate, threshold=threshold)
    return {
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "trailing_silence_seconds": round(trailing, 3),
        "duration_seconds": round(len(audio) / sample_rate, 3) if audio.size else 0.0,
    }


def _export_boundary_clip(src_wav: Path, dst_mp3: Path, start: float, duration: float) -> None:
    dst_mp3.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _ffmpeg_path(),
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{max(0.1, duration):.3f}",
            "-i",
            str(src_wav),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(dst_mp3),
        ],
        check=True,
        capture_output=True,
    )


def _rank_suspicious_boundaries(video_dir: Path, manifest: dict[str, Any], max_count: int = 5) -> list[dict[str, Any]]:
    blocks = manifest["blocks"]
    if len(blocks) < 2:
        return []
    threshold = float(manifest["block_config"]["trim_trailing_threshold"])
    expected_gap = float(manifest["block_config"]["gap_after_ms"]) / 1000.0
    metrics = {}
    for block in blocks:
        metrics[block["block_index"]] = _analyze_block_metrics(video_dir / block["wav_path"], threshold)
    ranked = []
    for left, right in zip(blocks, blocks[1:]):
        left_m = metrics[left["block_index"]]
        right_m = metrics[right["block_index"]]
        gap = max(0.0, float(right.get("audio_start", right["audio_end"])) - float(left["audio_end"]))
        short_penalty = 0.0
        if left_m["duration_seconds"] < 2.5:
            short_penalty += (2.5 - left_m["duration_seconds"]) * 8.0
        if right_m["duration_seconds"] < 2.5:
            short_penalty += (2.5 - right_m["duration_seconds"]) * 8.0
        score = (
            abs(left_m["rms"] - right_m["rms"]) * 100.0
            + abs(left_m["peak"] - right_m["peak"]) * 40.0
            + short_penalty
            + max(left_m["trailing_silence_seconds"], right_m["trailing_silence_seconds"]) * 10.0
            + abs(gap - expected_gap) * 25.0
        )
        reasons = []
        if abs(left_m["rms"] - right_m["rms"]) > 0.03:
            reasons.append("rms_delta")
        if abs(left_m["peak"] - right_m["peak"]) > 0.08:
            reasons.append("peak_delta")
        if left_m["duration_seconds"] < 2.5 or right_m["duration_seconds"] < 2.5:
            reasons.append("short_block")
        if max(left_m["trailing_silence_seconds"], right_m["trailing_silence_seconds"]) > 0.7:
            reasons.append("trailing_silence")
        if abs(gap - expected_gap) > 0.12:
            reasons.append("gap_anomaly")
        ranked.append(
            {
                "boundary_index": left["block_index"],
                "left_block": left["block_index"],
                "right_block": right["block_index"],
                "start": max(0.0, float(left["audio_end"]) - 2.2),
                "duration": 4.8,
                "score": round(score, 3),
                "gap_seconds": round(gap, 3),
                "left_metrics": left_m,
                "right_metrics": right_m,
                "reasons": reasons,
                "clip": f"boundary_{left['block_index']:03d}_to_{right['block_index']:03d}.mp3",
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:max_count]


def _render_final_artifact(
    out_dir: Path,
    blind_id: str,
    source_voice: str,
    preset: str,
) -> tuple[str, str, float, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    video_dir = out_dir / "final" / blind_id
    video_dir.mkdir(parents=True, exist_ok=True)
    script_path = video_dir / "script.txt"
    script_path.write_text(FINAL_SCRIPT, encoding="utf-8")
    tts_cfg = {
        "engine": "vieneu",
        "voice": source_voice,
        "mode": "block",
        "infer_params": STYLE_PRESETS[preset],
    }
    tts_step._run_block_mode(video_dir, script_path, "vieneu", source_voice, tts_cfg)
    manifest = json.loads((video_dir / "tts_blocks" / "blocks.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((video_dir / "tts_blocks" / "diagnostics.json").read_text(encoding="utf-8"))
    boundary_items = _rank_suspicious_boundaries(video_dir, manifest, max_count=5)
    for item in boundary_items:
        _export_boundary_clip(video_dir / "audio_master.wav", video_dir / "boundary_clips" / item["clip"], item["start"], item["duration"])
    blocks_payload = {
        "blocks": manifest["blocks"],
        "block_count": manifest["block_count"],
        "sentence_count": manifest["sentence_count"],
    }
    _save_json(video_dir / "blocks.json", blocks_payload)
    diagnostics_payload = {
        **diagnostics,
        "suspicious_boundaries": boundary_items,
    }
    _save_json(video_dir / "diagnostics.json", diagnostics_payload)
    duration = _wav_duration_seconds(video_dir / "audio_master.wav")
    effective = _effective_infer_params(source_voice, preset)
    return (
        _relative(video_dir / "audio_master.wav", out_dir),
        _relative(video_dir / "audio.mp3", out_dir),
        duration,
        effective,
        boundary_items,
        diagnostics_payload,
    )


def _final_round(out_dir: Path, seed: int, manifest: dict[str, Any], decisions_path: Path) -> tuple[dict[str, Any], list[LabArtifact]]:
    decisions = _load_decisions(decisions_path)
    finalists = _select_finalists(manifest, decisions, "style", ROUND_LIMITS["final"])
    existing_ids = {artifact.blind_id for artifact in _load_all_artifacts(manifest)}
    next_order = _next_round_order(manifest, "final")
    artifacts: list[LabArtifact] = []
    for index, source in enumerate(finalists):
        blind_id = _allocate_blind_id(existing_ids, seed + 300 + index)
        existing_ids.add(blind_id)
        wav_rel, mp3_rel, duration, effective, boundary_items, diagnostics = _render_final_artifact(
            out_dir, blind_id, source.source_voice, source.preset
        )
        _validate_final_duration(f"final:{blind_id}", duration)
        artifacts.append(
            LabArtifact(
                blind_id=blind_id,
                kind="final",
                round="final",
                round_order=next_order + index,
                source_voice=source.source_voice,
                preset=source.preset,
                effective_infer_params=effective,
                duration_seconds=duration,
                audio_wav=wav_rel,
                audio_mp3=mp3_rel,
                source_ref=source.blind_id,
                sample_id="final_block_mode",
                metadata={
                    "source_blind_id": source.blind_id,
                    "boundary_clip_count": len(boundary_items),
                    "diagnostics_path": _relative(out_dir / "final" / blind_id / "diagnostics.json", out_dir),
                    "blocks_path": _relative(out_dir / "final" / blind_id / "blocks.json", out_dir),
                    "production_diagnostics": diagnostics,
                },
            )
        )
    round_config = {
        "finalists_from": "style",
        "finalist_limit": ROUND_LIMITS["final"],
        "required_seconds": [90, 120],
    }
    manifest = _update_manifest(out_dir, manifest, "final", artifacts, round_config)
    return manifest, artifacts


def _report_rows(manifest: dict[str, Any], decisions: dict[str, dict[str, str]], active_round: str) -> list[dict[str, Any]]:
    rows = []
    for artifact in _artifacts_for_round(manifest, active_round):
        decision = decisions.get(artifact.blind_id, _blank_decision_row())
        rows.append(
            {
                "blind_id": artifact.blind_id,
                "round": artifact.round,
                "kind": artifact.kind,
                "round_order": artifact.round_order,
                "audio_mp3": artifact.audio_mp3,
                "audio_wav": artifact.audio_wav,
                "source_voice": artifact.source_voice,
                "preset": artifact.preset,
                "effective_infer_params": artifact.effective_infer_params,
                "duration_seconds": artifact.duration_seconds,
                "decision": decision.get("decision", ""),
                "revealed": decision.get("revealed", ""),
                "notes": decision.get("notes", ""),
            }
        )
    rows.sort(key=lambda item: (DECISION_PRIORITY.get(item["decision"], 3), item["round_order"], item["blind_id"]))
    return rows


def _build_review_html(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    manifest_summary = {
        "active_round": manifest.get("active_round"),
        "round_counts": manifest.get("round_counts", {}),
        "updated_at": manifest.get("updated_at"),
    }
    public_rows = []
    reveal_rows = {}
    for row in rows:
        public_rows.append(
            {
                "blind_id": row["blind_id"],
                "round": row["round"],
                "kind": row["kind"],
                "round_order": row["round_order"],
                "audio_mp3": row["audio_mp3"],
                "audio_wav": row["audio_wav"],
                "duration_seconds": row["duration_seconds"],
                "decision": row["decision"],
                "revealed": row["revealed"],
                "notes": row["notes"],
            }
        )
        reveal_rows[row["blind_id"]] = {
            "source_voice": row["source_voice"],
            "preset": row["preset"],
            "effective_infer_params": row["effective_infer_params"],
            "source_kind": row["kind"],
        }
    manifest_json = json.dumps(manifest_summary, ensure_ascii=False)
    rows_json = json.dumps(public_rows, ensure_ascii=False)
    reveal_json = json.dumps(reveal_rows, ensure_ascii=False).encode("utf-8").hex()
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VieNeu Voice Lab</title>
  <style>
    :root {{
      --bg: #0f1418;
      --panel: #151d23;
      --panel-2: #1b252d;
      --text: #edf3f7;
      --muted: #93a6b6;
      --line: #2a3945;
    }}
    body {{ margin: 0; font-family: Segoe UI, system-ui, sans-serif; background: linear-gradient(180deg, #0d1115, #121920); color: var(--text); }}
    header {{ padding: 24px 28px 16px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .sub {{ color: var(--muted); font-size: 14px; }}
    .toolbar {{ display: flex; gap: 12px; flex-wrap: wrap; padding: 16px 28px; position: sticky; top: 0; background: rgba(13,17,21,0.92); backdrop-filter: blur(8px); border-bottom: 1px solid var(--line); z-index: 5; }}
    .toolbar input, .toolbar select, .toolbar button {{ background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; }}
    .toolbar button {{ cursor: pointer; }}
    main {{ padding: 20px 28px 36px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 12px 8px; vertical-align: top; }}
    th {{ text-align: left; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    tr.card td {{ background: var(--panel); }}
    tr.card:nth-child(even) td {{ background: var(--panel-2); }}
    .blind {{ font-weight: 700; font-size: 16px; }}
    .muted {{ color: var(--muted); }}
    .hiddenMeta {{ color: var(--muted); font-size: 12px; min-height: 38px; }}
    .decisionRow {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .decisionRow label {{ display: inline-flex; align-items: center; gap: 5px; }}
    .notes {{ width: 240px; min-height: 48px; background: #10161b; color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 8px; }}
    audio {{ width: 240px; }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 999px; background: #202c36; font-size: 12px; }}
    .revealBtn[disabled] {{ opacity: .5; cursor: not-allowed; }}
  </style>
</head>
<body>
  <header>
    <h1>VieNeu Voice Lab</h1>
    <div class="sub">Blind review cho round hiện tại. Voice và preset chỉ hiện sau khi Leon chọn Keep, Maybe hoặc Reject.</div>
  </header>
  <div class="toolbar">
    <input id="filterInput" placeholder="Filter by blind ID or round" />
    <select id="sortSelect">
      <option value="round_order">Sort by round order</option>
      <option value="decision">Sort by decision</option>
      <option value="blind_id">Sort by blind ID</option>
    </select>
    <button id="exportBtn">Export decisions.csv</button>
    <button id="clearBtn">Clear local review</button>
  </div>
  <main>
    <table id="labTable">
      <thead>
        <tr>
          <th>Blind</th>
          <th>Round</th>
          <th>Audio</th>
          <th>Decision</th>
          <th>Reveal</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </main>
  <script>
    window.VOICE_LAB_MANIFEST = {manifest_json};
    window.VOICE_LAB_ROWS = {rows_json};
    window.VOICE_LAB_REVEAL_HEX = {json.dumps(reveal_json)};
    const LOCAL_KEY = 'vieneu_voice_lab_review';
    const rows = window.VOICE_LAB_ROWS || [];
    const revealMap = JSON.parse(
      new TextDecoder().decode(
        Uint8Array.from(
          (window.VOICE_LAB_REVEAL_HEX || '').match(/../g) || [],
          (pair) => parseInt(pair, 16),
        ),
      ) || '{{}}'
    );
    const saved = JSON.parse(localStorage.getItem(LOCAL_KEY) || '[]');
    const savedMap = new Map(saved.map((row) => [row.blind_id, row]));
    const tbody = document.querySelector('#labTable tbody');
    const filterInput = document.querySelector('#filterInput');
    const sortSelect = document.querySelector('#sortSelect');
    const exportBtn = document.querySelector('#exportBtn');
    const clearBtn = document.querySelector('#clearBtn');

    function restoreRow(blindId) {{
      return savedMap.get(blindId) || {{}};
    }}

    function collectRows() {{
      const out = [];
      tbody.querySelectorAll('tr').forEach((tr) => {{
        const decision = tr.querySelector('input[type="radio"]:checked')?.value || '';
        const revealed = tr.querySelector('.revealBtn').dataset.revealed || '';
        const notes = tr.querySelector('.notes').value.trim();
        const item = revealMap[tr.dataset.blind] || {{}};
        out.push({{
          blind_id: tr.dataset.blind,
          round: tr.dataset.round,
          decision,
          revealed,
          notes,
          revealed_voice: revealed ? (item.source_voice || '') : '',
          revealed_preset: revealed ? (item.preset || '') : '',
          revealed_source: revealed ? (item.source_kind || '') : '',
          revealed_params: revealed ? JSON.stringify(item.effective_infer_params || {{}}) : '',
        }});
      }});
      return out;
    }}

    function saveLocal() {{
      localStorage.setItem(LOCAL_KEY, JSON.stringify(collectRows()));
    }}

    function applyReveal(tr) {{
      const item = revealMap[tr.dataset.blind] || {{}};
      const meta = tr.querySelector('.hiddenMeta');
      meta.textContent = [item.source_voice, item.preset, JSON.stringify(item.effective_infer_params || {{}})]
        .filter(Boolean)
        .join(' | ') || 'No reveal metadata';
      const button = tr.querySelector('.revealBtn');
      button.dataset.revealed = 'yes';
      saveLocal();
    }}

    function updateRevealState(tr) {{
      const selected = tr.querySelector('input[type="radio"]:checked');
      const button = tr.querySelector('.revealBtn');
      button.disabled = !selected;
    }}

    function makeRow(item) {{
      const tr = document.createElement('tr');
      tr.className = 'card';
      tr.dataset.blind = item.blind_id;
      tr.dataset.round = item.round;
      const restored = restoreRow(item.blind_id);
      tr.innerHTML = `
        <td><div class="blind">${{item.blind_id}}</div><div class="muted">${{item.kind}}</div></td>
        <td><span class="pill">${{item.round}}</span></td>
        <td><audio controls preload="none" src="${{item.audio_mp3 || item.audio_wav}}"></audio></td>
        <td>
          <div class="decisionRow">
            <label><input type="radio" name="decision_${{item.blind_id}}" value="Keep">Keep</label>
            <label><input type="radio" name="decision_${{item.blind_id}}" value="Maybe">Maybe</label>
            <label><input type="radio" name="decision_${{item.blind_id}}" value="Reject">Reject</label>
          </div>
        </td>
        <td>
          <button class="revealBtn" disabled data-revealed="">Reveal</button>
          <div class="hiddenMeta">Hidden until reveal</div>
        </td>
        <td><textarea class="notes" placeholder="Notes"></textarea></td>
      `;
      if (restored.decision) {{
        const input = tr.querySelector(`input[value="${{restored.decision}}"]`);
        if (input) input.checked = true;
      }}
      tr.querySelector('.notes').value = restored.notes || '';
      tr.querySelectorAll('input[type="radio"]').forEach((input) => {{
        input.addEventListener('change', () => {{
          updateRevealState(tr);
          saveLocal();
        }});
      }});
      tr.querySelector('.notes').addEventListener('input', saveLocal);
      tr.querySelector('.revealBtn').addEventListener('click', () => applyReveal(tr));
      updateRevealState(tr);
      if (restored.revealed === 'yes') {{
        applyReveal(tr);
      }}
      return tr;
    }}

    function render() {{
      const filter = filterInput.value.trim().toLowerCase();
      const sortBy = sortSelect.value;
      let items = rows.slice();
      if (filter) {{
        items = items.filter((item) => `${{item.blind_id}} ${{item.round}} ${{item.kind}}`.toLowerCase().includes(filter));
      }}
      items.sort((a, b) => {{
        const ra = restoreRow(a.blind_id);
        const rb = restoreRow(b.blind_id);
        if (sortBy === 'decision') {{
          const da = {{Keep: 0, Maybe: 1, Reject: 2}}[ra.decision] ?? 3;
          const db = {{Keep: 0, Maybe: 1, Reject: 2}}[rb.decision] ?? 3;
          return da - db || a.round_order - b.round_order || a.blind_id.localeCompare(b.blind_id);
        }}
        if (sortBy === 'blind_id') {{
          return a.blind_id.localeCompare(b.blind_id);
        }}
        return a.round_order - b.round_order || a.blind_id.localeCompare(b.blind_id);
      }});
      tbody.innerHTML = '';
      items.forEach((item) => tbody.appendChild(makeRow(item)));
    }}

    function exportCsv() {{
      const currentRows = collectRows();
      const header = {json.dumps(DECISIONS_FIELDNAMES)};
      const lines = [header.join(',')];
      for (const row of currentRows) {{
        lines.push(header.map((key) => JSON.stringify(row[key] || '')).join(','));
      }}
      const blob = new Blob([lines.join('\\n')], {{type: 'text/csv;charset=utf-8;'}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'decisions.csv';
      a.click();
      URL.revokeObjectURL(url);
      saveLocal();
    }}

    function clearLocal() {{
      localStorage.removeItem(LOCAL_KEY);
      location.reload();
    }}

    filterInput.addEventListener('input', render);
    sortSelect.addEventListener('change', render);
    exportBtn.addEventListener('click', exportCsv);
    clearBtn.addEventListener('click', clearLocal);
    render();
  </script>
</body>
</html>
"""


def _report(out_dir: Path, manifest: dict[str, Any], decisions_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    active_round = manifest.get("active_round")
    if not active_round:
        raise RuntimeError("Manifest has no active_round")
    decisions = _load_decisions(decisions_path)
    rows = _report_rows(manifest, decisions, active_round)
    leaderboard_path = out_dir / "leaderboard.md"
    lines = [
        "# VieNeu Voice Lab Leaderboard",
        "",
        f"- Active round: `{active_round}`",
        f"- Updated: `{_now_iso()}`",
        "",
        "| Blind | Decision | Preset | Duration | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['blind_id']} | {row['decision'] or 'pending'} | {row['preset']} | {row['duration_seconds']:.2f}s | {row['notes'] or ''} |"
        )
    leaderboard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    html_path = out_dir / "review.html"
    html_path.write_text(_build_review_html(manifest, rows), encoding="utf-8")
    summary = {
        "active_round": active_round,
        "artifact_count": len(rows),
        "reviewed_count": sum(1 for row in rows if row["decision"]),
        "leaderboard_path": _relative(leaderboard_path, out_dir),
        "html_path": _relative(html_path, out_dir),
        "decisions_path": str(decisions_path),
    }
    _save_json(out_dir / "report.json", summary)
    return summary, rows


def cmd_base(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(out_dir)
    decisions_path = _resolve_decisions_path(out_dir, args.decisions)
    manifest, artifacts = _base_round(out_dir, args.seed, manifest)
    _merge_decisions(decisions_path, artifacts)
    print(f"Generated {len(artifacts)} VieNeu base samples in {out_dir}")
    return 0


def cmd_topic(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(out_dir)
    decisions_path = _resolve_decisions_path(out_dir, args.decisions)
    manifest, artifacts = _topic_round(out_dir, args.seed, manifest, decisions_path)
    _merge_decisions(decisions_path, artifacts)
    print(f"Generated {len(artifacts)} VieNeu topic reels in {out_dir}")
    return 0


def cmd_style(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(out_dir)
    decisions_path = _resolve_decisions_path(out_dir, args.decisions)
    manifest, artifacts = _style_round(out_dir, args.seed, manifest, decisions_path)
    _merge_decisions(decisions_path, artifacts)
    print(f"Generated {len(artifacts)} VieNeu style samples in {out_dir}")
    return 0


def cmd_final(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(out_dir)
    decisions_path = _resolve_decisions_path(out_dir, args.decisions)
    manifest, artifacts = _final_round(out_dir, args.seed, manifest, decisions_path)
    _merge_decisions(decisions_path, artifacts)
    print(f"Generated {len(artifacts)} VieNeu final artifacts in {out_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(out_dir)
    decisions_path = _resolve_decisions_path(out_dir, args.decisions)
    summary, _rows = _report(out_dir, manifest, decisions_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Leaderboard written to {out_dir / 'leaderboard.md'}")
    print(f"Review HTML written to {out_dir / 'review.html'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent VieNeu voice lab")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--decisions", type=Path, default=None, help="Path to decisions.csv (default: <output-dir>/decisions.csv)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_base = sub.add_parser("base", help="Generate one base sample per discovered VieNeu voice")
    p_base.set_defaults(func=cmd_base)

    p_topic = sub.add_parser("topic", help="Generate topic reels for reviewed base finalists")
    p_topic.set_defaults(func=cmd_topic)

    p_style = sub.add_parser("style", help="Generate style comparisons for reviewed topic finalists")
    p_style.set_defaults(func=cmd_style)

    p_final = sub.add_parser("final", help="Generate block-mode finals for reviewed style finalists")
    p_final.set_defaults(func=cmd_final)

    p_report = sub.add_parser("report", help="Build review HTML and leaderboard from decisions.csv")
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
