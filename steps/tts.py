"""Step 2: Text-to-Speech with block-aware production paths for VieNeu and Kokoro."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

import config
from steps.text_units import SentenceUnit, load_script_text, load_sentence_units, split_paragraph_texts


BLOCK_CACHE_SCHEMA_VERSION = "block-tts-v2"

VIENEU_BLOCK_DEFAULTS = {
    "block_target_min_seconds": 10.0,
    "block_target_max_seconds": 16.0,
    "block_soft_max_seconds": 18.0,
    "block_hard_max_seconds": 20.0,
    "initial_chars_per_second": 14.0,
    "block_soft_max_normalized_chars": 240,
    "block_hard_max_normalized_chars": 280,
    "max_chars": 384,
    "max_new_frames": 300,
    "gap_after_ms": 300,
    "fallback_sentence_gap_ms": 220,
    "trim_trailing_threshold": 0.01,
    "trim_trailing_keep_ms": 120,
    "abnormal_trailing_silence_seconds": 1.2,
}

VIENEU_INFER_DEFAULTS = {
    "temperature": 0.45,
    "top_k": 25,
    "top_p": 0.92,
    "repetition_penalty": 1.18,
    "crossfade_p": 0.0,
    "silence_p": 0.15,
    "max_new_frames": 300,
    "max_chars": 384,
    "apply_watermark": False,
}

KOKORO_BLOCK_DEFAULTS = {
    "block_soft_max_phoneme_chars": 420,
    "block_hard_max_phoneme_chars": 500,
    "official_phoneme_cap": 510,
    "block_target_min_seconds": 12.0,
    "block_target_max_seconds": 22.0,
    "block_hard_max_seconds": 25.0,
    "gap_after_ms": 300,
    "fallback_sentence_gap_ms": 220,
}


def _get_audio_duration(audio_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError):
        return 0.0


def _merge_dict(base: dict, override: dict | None) -> dict:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _json_fingerprint(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _trim_trailing_silence(
    audio,
    sample_rate: int,
    threshold: float = 0.01,
    keep_ms: int = 120,
):
    import numpy as np

    if audio.size == 0:
        return audio
    loud = np.abs(audio) > threshold
    if not loud.any():
        return audio[:0]
    last = len(loud) - int(np.argmax(loud[::-1]))
    keep = int(keep_ms / 1000 * sample_rate)
    end = min(len(audio), last + keep)
    return audio[:end]


def _measure_trailing_silence(audio, sample_rate: int, threshold: float = 0.01) -> float:
    import numpy as np

    if audio.size == 0:
        return 0.0
    loud = np.abs(audio) > threshold
    if not loud.any():
        return len(audio) / sample_rate
    last = len(loud) - int(np.argmax(loud[::-1]))
    return max(0.0, (len(audio) - last) / sample_rate)


def _write_wav(audio, sample_rate: int, output_path: Path) -> None:
    import soundfile as sf

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate)


def _wav_to_mp3(input_wav: Path, output_mp3: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "0",
            str(output_mp3),
        ],
        check=True,
        capture_output=True,
    )


def _load_tts_config(video_dir: Path) -> dict:
    tts_config_path = video_dir / "tts_config.json"
    if not tts_config_path.exists():
        return {}
    return json.loads(tts_config_path.read_text(encoding="utf-8-sig"))


def _invalidate_timestamps(video_dir: Path) -> None:
    timestamps_path = video_dir / "timestamps.json"
    stale_path = video_dir / "timestamps.stale.json"
    if timestamps_path.exists():
        shutil.move(str(timestamps_path), str(stale_path))
        logger.info("Renamed stale timestamps.json -> {}", stale_path.name)
    marker = {
        "reason": "block_tts_generated",
        "audio": "audio_master.wav",
        "manifest": "tts_blocks/blocks.json",
    }
    (video_dir / "tts_blocks" / "needs_alignment.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_manifest(video_dir: Path, manifest: dict) -> None:
    manifest_path = video_dir / "tts_blocks" / "blocks.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_manifest(video_dir: Path) -> dict:
    manifest_path = video_dir / "tts_blocks" / "blocks.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_existing_manifest(video_dir: Path) -> dict | None:
    manifest_path = video_dir / "tts_blocks" / "blocks.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _validate_reusable_wav(video_dir: Path, wav_path: str, expected_sample_rate: int) -> bool:
    import soundfile as sf

    candidate = video_dir / wav_path
    if not candidate.exists() or candidate.stat().st_size <= 0:
        return False
    try:
        info = sf.info(str(candidate))
    except Exception:
        return False
    if info.samplerate != expected_sample_rate:
        return False
    if info.frames <= 0 or (info.frames / info.samplerate) <= 0:
        return False
    return True


def _sentence_indices_are_contiguous(blocks: list[dict], expected_count: int) -> None:
    actual = [idx for block in blocks for idx in block["sentence_indices"]]
    expected = list(range(1, expected_count + 1))
    if actual != expected:
        raise RuntimeError(f"Sentence coverage mismatch in blocks manifest: expected {expected[:5]}... got {actual[:5]}...")


@dataclass
class BlockSpec:
    sentence_units: list[SentenceUnit]
    text: str
    sentence_indices: list[int]
    sentence_texts: list[str]
    raw_chars: int
    estimated_seconds: float
    normalized_chars: int | None = None
    phoneme_chars: int | None = None
    block_hash: str | None = None


class VieNeuRuntime:
    package_name = "vieneu"

    def __init__(self, voice: str, block_config: dict, infer_overrides: dict | None = None, speed: float | None = None):
        import numpy as np  # noqa: F401
        from vieneu import Vieneu
        from vieneu_utils.phonemize_text import PuncNormalizer

        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        self.voice = voice
        self.speed = speed
        self.block_config = block_config
        self.normalizer = PuncNormalizer()
        self.tts = Vieneu()
        self.sample_rate = self.tts.sample_rate
        self.package_version = _package_version(self.package_name)
        self.infer_params = _merge_dict(VIENEU_INFER_DEFAULTS, infer_overrides)
        self.infer_params["voice"] = voice
        self.infer_params["max_new_frames"] = block_config["max_new_frames"]
        self.infer_params["max_chars"] = block_config["max_chars"]

    def normalized_chars(self, text: str) -> int:
        normalized_text = self.normalizer.normalize(text, punc_norm=True)
        return len(normalized_text)

    def estimate_seconds(self, text: str) -> float:
        return self.normalized_chars(text) / self.block_config["initial_chars_per_second"]

    def synthesize(self, text: str, infer_params: dict | None = None):
        import numpy as np

        params = dict(self.infer_params)
        if infer_params:
            params.update(infer_params)
        audio = self.tts.infer(text, **params)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), self.sample_rate, params


class KokoroRuntime:
    package_name = "kokoro"

    def __init__(self, voice: str | None = None, speed: float | None = None):
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code="a")
        self.sample_rate = 24000
        self.voice = voice or config.TTS_VOICE
        self.speed = config.TTS_SPEED if speed is None else speed
        self.package_version = _package_version(self.package_name)

    def phoneme_chars(self, text: str) -> int:
        phoneme_text, _tokens = self.pipeline.g2p(text, preprocess=True)
        return len(phoneme_text)

    def estimate_seconds(self, text: str) -> float:
        word_count = max(1, len(text.split()))
        return word_count / 2.6

    def synthesize(self, text: str):
        import numpy as np

        chunks = [audio for _gs, _ps, audio in self.pipeline(text, voice=self.voice, speed=self.speed)]
        if not chunks:
            raise RuntimeError("Kokoro produced no audio output")
        if len(chunks) > 1:
            raise RuntimeError("Kokoro internal-split block unexpectedly; lower phoneme limit or split block")
        return np.concatenate(chunks), self.sample_rate, {
            "voice": self.voice,
            "speed": self.speed,
        }


def _effective_kokoro_infer_params(runtime: KokoroRuntime) -> dict:
    return {
        "voice": runtime.voice,
        "speed": runtime.speed,
    }


def _compute_block_hash(
    *,
    engine: str,
    voice: str | None,
    speed: float | None,
    block_config: dict,
    infer_params: dict,
    block_text: str,
    engine_version: str,
) -> str:
    return _json_fingerprint(
        {
            "cache_schema_version": BLOCK_CACHE_SCHEMA_VERSION,
            "engine": engine,
            "voice": voice,
            "speed": speed,
            "block_config": block_config,
            "infer_params": infer_params,
            "block_text": block_text,
            "engine_version": engine_version,
        }
    )


def _build_vieneu_blocks(sentence_units: list[SentenceUnit], runtime: VieNeuRuntime) -> list[BlockSpec]:
    cfg = runtime.block_config
    blocks: list[BlockSpec] = []
    current: list[SentenceUnit] = []

    def finalize(units: list[SentenceUnit]) -> BlockSpec:
        text = " ".join(unit.text for unit in units)
        return BlockSpec(
            sentence_units=list(units),
            text=text,
            sentence_indices=[unit.sentence_index for unit in units],
            sentence_texts=[unit.text for unit in units],
            raw_chars=len(text),
            normalized_chars=runtime.normalized_chars(text),
            estimated_seconds=runtime.estimate_seconds(text),
        )

    for unit in sentence_units:
        single_norm = runtime.normalized_chars(unit.text)
        single_seconds = runtime.estimate_seconds(unit.text)
        if single_norm > cfg["block_hard_max_normalized_chars"] or single_seconds > cfg["block_hard_max_seconds"]:
            raise RuntimeError(
                f"Sentence {unit.sentence_index} exceeds VieNeu block limit "
                f"({single_norm} normalized chars, {single_seconds:.2f}s estimated)"
            )

        if current and unit.paragraph_index != current[-1].paragraph_index:
            current_spec = finalize(current)
            if current_spec.estimated_seconds >= 8.0:
                blocks.append(current_spec)
                current = []

        candidate = current + [unit]
        candidate_spec = finalize(candidate)
        if current and (
            candidate_spec.normalized_chars > cfg["block_soft_max_normalized_chars"]
            or candidate_spec.estimated_seconds > cfg["block_target_max_seconds"]
        ):
            blocks.append(finalize(current))
            current = [unit]
        else:
            current = candidate

    if current:
        blocks.append(finalize(current))
    return blocks


def _build_kokoro_blocks(sentence_units: list[SentenceUnit], runtime: KokoroRuntime, block_config: dict) -> list[BlockSpec]:
    blocks: list[BlockSpec] = []
    current: list[SentenceUnit] = []

    def finalize(units: list[SentenceUnit]) -> BlockSpec:
        text = " ".join(unit.text for unit in units)
        return BlockSpec(
            sentence_units=list(units),
            text=text,
            sentence_indices=[unit.sentence_index for unit in units],
            sentence_texts=[unit.text for unit in units],
            raw_chars=len(text),
            phoneme_chars=runtime.phoneme_chars(text),
            estimated_seconds=runtime.estimate_seconds(text),
        )

    for unit in sentence_units:
        single_phonemes = runtime.phoneme_chars(unit.text)
        if single_phonemes > block_config["block_hard_max_phoneme_chars"]:
            raise RuntimeError(
                f"Sentence {unit.sentence_index} exceeds Kokoro block limit "
                f"({single_phonemes} phoneme chars)"
            )

        if not current:
            current = [unit]
            continue

        candidate = current + [unit]
        candidate_spec = finalize(candidate)
        if (
            candidate_spec.phoneme_chars > block_config["block_soft_max_phoneme_chars"]
            or candidate_spec.phoneme_chars > block_config["block_hard_max_phoneme_chars"]
        ):
            blocks.append(finalize(current))
            current = [unit]
        else:
            current = candidate

    if current:
        blocks.append(finalize(current))
    return blocks


def _split_block_midpoint(block: BlockSpec) -> list[BlockSpec]:
    if len(block.sentence_units) <= 1:
        raise RuntimeError(f"Cannot split block with only one sentence: {block.sentence_indices}")
    mid = len(block.sentence_units) // 2
    left = block.sentence_units[:mid]
    right = block.sentence_units[mid:]
    return [
        BlockSpec(
            sentence_units=left,
            text=" ".join(unit.text for unit in left),
            sentence_indices=[unit.sentence_index for unit in left],
            sentence_texts=[unit.text for unit in left],
            raw_chars=len(" ".join(unit.text for unit in left)),
            estimated_seconds=0.0,
        ),
        BlockSpec(
            sentence_units=right,
            text=" ".join(unit.text for unit in right),
            sentence_indices=[unit.sentence_index for unit in right],
            sentence_texts=[unit.text for unit in right],
            raw_chars=len(" ".join(unit.text for unit in right)),
            estimated_seconds=0.0,
        ),
    ]


def _recompute_block_spec(block: BlockSpec, engine: str, runtime, block_config: dict) -> BlockSpec:
    if engine == "vieneu":
        normalized_chars = runtime.normalized_chars(block.text)
        return BlockSpec(
            sentence_units=block.sentence_units,
            text=block.text,
            sentence_indices=block.sentence_indices,
            sentence_texts=block.sentence_texts,
            raw_chars=block.raw_chars,
            estimated_seconds=runtime.estimate_seconds(block.text),
            normalized_chars=normalized_chars,
            block_hash=block.block_hash,
        )
    phoneme_chars = runtime.phoneme_chars(block.text)
    return BlockSpec(
        sentence_units=block.sentence_units,
        text=block.text,
        sentence_indices=block.sentence_indices,
        sentence_texts=block.sentence_texts,
        raw_chars=block.raw_chars,
        estimated_seconds=runtime.estimate_seconds(block.text),
        phoneme_chars=phoneme_chars,
        block_hash=block.block_hash,
    )


def _attach_block_hashes(
    blocks: list[BlockSpec],
    *,
    engine: str,
    voice: str | None,
    speed: float | None,
    block_config: dict,
    infer_params: dict,
    engine_version: str,
) -> list[BlockSpec]:
    hashed_blocks: list[BlockSpec] = []
    for block in blocks:
        hashed_blocks.append(
            BlockSpec(
                sentence_units=block.sentence_units,
                text=block.text,
                sentence_indices=block.sentence_indices,
                sentence_texts=block.sentence_texts,
                raw_chars=block.raw_chars,
                estimated_seconds=block.estimated_seconds,
                normalized_chars=block.normalized_chars,
                phoneme_chars=block.phoneme_chars,
                block_hash=_compute_block_hash(
                    engine=engine,
                    voice=voice,
                    speed=speed,
                    block_config=block_config,
                    infer_params=infer_params,
                    block_text=block.text,
                    engine_version=engine_version,
                ),
            )
        )
    return hashed_blocks


def _existing_blocks_by_hash(existing_manifest: dict | None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    if not existing_manifest:
        return grouped
    for block in existing_manifest.get("blocks", []):
        block_hash = block.get("block_hash")
        if not block_hash:
            continue
        grouped.setdefault(block_hash, []).append(block)
    return grouped


def _render_sentence_fallback_block(
    video_dir: Path,
    block_index: int,
    block: BlockSpec,
    engine: str,
    runtime,
    block_config: dict,
    infer_retry_params: dict | None = None,
) -> dict:
    import numpy as np

    tts_blocks_dir = video_dir / "tts_blocks"
    gap_ms = int(block_config.get("fallback_sentence_gap_ms", 220))
    sample_rate = runtime.sample_rate
    sentence_gap = np.zeros(int(gap_ms / 1000 * sample_rate), dtype=np.float32)
    combined_parts: list[np.ndarray] = []
    fallback_segments: list[dict] = []
    cursor = 0.0

    for sentence_offset, sentence in enumerate(block.sentence_texts, start=1):
        if engine == "vieneu":
            audio, _sample_rate, infer_params = runtime.synthesize(sentence, infer_retry_params)
            audio = _trim_trailing_silence(
                audio,
                sample_rate,
                threshold=block_config["trim_trailing_threshold"],
                keep_ms=block_config["trim_trailing_keep_ms"],
            )
        else:
            audio, _sample_rate, infer_params = runtime.synthesize(sentence)
        sentence_path = tts_blocks_dir / f"block_{block_index:03d}_sentence_{sentence_offset:03d}.wav"
        _write_wav(audio, sample_rate, sentence_path)
        duration = len(audio) / sample_rate
        fallback_segments.append(
            {
                "sentence_index": block.sentence_indices[sentence_offset - 1],
                "text": sentence,
                "wav_path": str(sentence_path.relative_to(video_dir)).replace("\\", "/"),
                "actual_seconds": round(duration, 3),
                "start_in_block": round(cursor, 3),
                "end_in_block": round(cursor + duration, 3),
                "gap_after_ms": gap_ms if sentence_offset != len(block.sentence_texts) else 0,
                "infer_params": infer_params,
            }
        )
        combined_parts.append(audio)
        cursor += duration
        if sentence_offset != len(block.sentence_texts):
            combined_parts.append(sentence_gap)
            cursor += gap_ms / 1000

    combined_audio = np.concatenate(combined_parts) if combined_parts else np.zeros(0, dtype=np.float32)
    wav_path = tts_blocks_dir / f"block_{block_index:03d}.wav"
    _write_wav(combined_audio, sample_rate, wav_path)
    return {
        "wav_path": str(wav_path.relative_to(video_dir)).replace("\\", "/"),
        "sample_rate": sample_rate,
        "actual_seconds": round(len(combined_audio) / sample_rate, 3),
        "fallback_level": 2,
        "fallback_segments": fallback_segments,
    }


def _generate_vieneu_block_entry(
    video_dir: Path,
    block_index: int,
    block: BlockSpec,
    runtime: VieNeuRuntime,
    reuse_candidate: dict | None = None,
) -> tuple[list[BlockSpec], dict | None]:
    block = _recompute_block_spec(block, "vieneu", runtime, runtime.block_config)
    cfg = runtime.block_config
    if reuse_candidate and _validate_reusable_wav(video_dir, reuse_candidate["wav_path"], runtime.sample_rate):
        return [], {
            **reuse_candidate,
            "block_index": block_index,
            "sentence_indices": block.sentence_indices,
            "sentence_texts": block.sentence_texts,
            "text": block.text,
            "raw_chars": block.raw_chars,
            "normalized_chars": block.normalized_chars,
            "phoneme_chars": None,
            "estimated_seconds": round(block.estimated_seconds, 3),
            "gap_after_ms": cfg["gap_after_ms"],
            "infer_params": dict(runtime.infer_params),
            "block_hash": block.block_hash,
            "cache_schema_version": BLOCK_CACHE_SCHEMA_VERSION,
            "engine_version": runtime.package_version,
            "voice": runtime.voice,
            "speed": runtime.speed,
            "generation_status": "reused",
        }

    audio, sample_rate, infer_params = runtime.synthesize(block.text)
    trailing_silence = _measure_trailing_silence(audio, sample_rate, threshold=cfg["trim_trailing_threshold"])
    if trailing_silence > cfg["abnormal_trailing_silence_seconds"]:
        audio, sample_rate, infer_params = runtime.synthesize(block.text, {"temperature": 0.25})

    audio = _trim_trailing_silence(
        audio,
        sample_rate,
        threshold=cfg["trim_trailing_threshold"],
        keep_ms=cfg["trim_trailing_keep_ms"],
    )
    actual_seconds = len(audio) / sample_rate
    if actual_seconds >= 23.0 or (actual_seconds > cfg["block_hard_max_seconds"] and len(block.sentence_units) > 1):
        children = [_recompute_block_spec(child, "vieneu", runtime, cfg) for child in _split_block_midpoint(block)]
        return children, None
    if audio.size == 0:
        fallback = _render_sentence_fallback_block(
            video_dir, block_index, block, "vieneu", runtime, cfg, {"temperature": 0.25}
        )
        return [], {
            "block_index": block_index,
            "engine": "vieneu",
            "sentence_indices": block.sentence_indices,
            "sentence_texts": block.sentence_texts,
            "text": block.text,
            "raw_chars": block.raw_chars,
            "normalized_chars": block.normalized_chars,
            "phoneme_chars": None,
            "estimated_seconds": round(block.estimated_seconds, 3),
            "actual_seconds": fallback["actual_seconds"],
            "wav_path": fallback["wav_path"],
            "sample_rate": fallback["sample_rate"],
            "gap_after_ms": cfg["gap_after_ms"],
            "infer_params": infer_params,
            "fallback_level": fallback["fallback_level"],
            "fallback_segments": fallback["fallback_segments"],
            "block_hash": block.block_hash,
            "cache_schema_version": BLOCK_CACHE_SCHEMA_VERSION,
            "engine_version": runtime.package_version,
            "voice": runtime.voice,
            "speed": runtime.speed,
            "generation_status": "fallback",
        }

    wav_path = video_dir / "tts_blocks" / f"block_{block_index:03d}.wav"
    _write_wav(audio, sample_rate, wav_path)
    entry = {
        "block_index": block_index,
        "engine": "vieneu",
        "sentence_indices": block.sentence_indices,
        "sentence_texts": block.sentence_texts,
        "text": block.text,
        "raw_chars": block.raw_chars,
        "normalized_chars": block.normalized_chars,
        "phoneme_chars": None,
        "estimated_seconds": round(block.estimated_seconds, 3),
        "actual_seconds": round(actual_seconds, 3),
        "wav_path": str(wav_path.relative_to(video_dir)).replace("\\", "/"),
        "sample_rate": sample_rate,
        "gap_after_ms": cfg["gap_after_ms"],
        "infer_params": infer_params,
        "fallback_level": 0,
        "block_hash": block.block_hash,
        "cache_schema_version": BLOCK_CACHE_SCHEMA_VERSION,
        "engine_version": runtime.package_version,
        "voice": runtime.voice,
        "speed": runtime.speed,
        "generation_status": "regenerated",
    }
    return [], entry


def _generate_kokoro_block_entry(
    video_dir: Path,
    block_index: int,
    block: BlockSpec,
    runtime: KokoroRuntime,
    block_config: dict,
    reuse_candidate: dict | None = None,
) -> tuple[list[BlockSpec], dict | None]:
    block = _recompute_block_spec(block, "kokoro", runtime, block_config)
    if reuse_candidate and _validate_reusable_wav(video_dir, reuse_candidate["wav_path"], runtime.sample_rate):
        return [], {
            **reuse_candidate,
            "block_index": block_index,
            "sentence_indices": block.sentence_indices,
            "sentence_texts": block.sentence_texts,
            "text": block.text,
            "raw_chars": block.raw_chars,
            "normalized_chars": None,
            "phoneme_chars": block.phoneme_chars,
            "estimated_seconds": round(block.estimated_seconds, 3),
            "gap_after_ms": block_config["gap_after_ms"],
            "infer_params": _effective_kokoro_infer_params(runtime),
            "block_hash": block.block_hash,
            "cache_schema_version": BLOCK_CACHE_SCHEMA_VERSION,
            "engine_version": runtime.package_version,
            "voice": runtime.voice,
            "speed": runtime.speed,
            "generation_status": "reused",
        }

    audio, sample_rate, infer_params = runtime.synthesize(block.text)
    actual_seconds = len(audio) / sample_rate
    if actual_seconds > block_config["block_hard_max_seconds"] and len(block.sentence_units) > 1:
        children = [_recompute_block_spec(child, "kokoro", runtime, block_config) for child in _split_block_midpoint(block)]
        return children, None
    wav_path = video_dir / "tts_blocks" / f"block_{block_index:03d}.wav"
    _write_wav(audio, sample_rate, wav_path)
    entry = {
        "block_index": block_index,
        "engine": "kokoro",
        "sentence_indices": block.sentence_indices,
        "sentence_texts": block.sentence_texts,
        "text": block.text,
        "raw_chars": block.raw_chars,
        "normalized_chars": None,
        "phoneme_chars": block.phoneme_chars,
        "estimated_seconds": round(block.estimated_seconds, 3),
        "actual_seconds": round(actual_seconds, 3),
        "wav_path": str(wav_path.relative_to(video_dir)).replace("\\", "/"),
        "sample_rate": sample_rate,
        "gap_after_ms": block_config["gap_after_ms"],
        "infer_params": infer_params,
        "fallback_level": 0,
        "block_hash": block.block_hash,
        "cache_schema_version": BLOCK_CACHE_SCHEMA_VERSION,
        "engine_version": runtime.package_version,
        "voice": runtime.voice,
        "speed": runtime.speed,
        "generation_status": "regenerated",
    }
    return [], entry


def _rebuild_audio_from_manifest(video_dir: Path, manifest: dict) -> None:
    import numpy as np
    import soundfile as sf

    blocks = manifest["blocks"]
    if not blocks:
        raise RuntimeError("Cannot rebuild audio: manifest has no blocks")

    combined_parts: list[np.ndarray] = []
    cursor = 0.0
    for i, block in enumerate(blocks):
        wav_path = video_dir / block["wav_path"]
        audio, sample_rate = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        block["sample_rate"] = sample_rate
        block["actual_seconds"] = round(len(audio) / sample_rate, 3)
        block["audio_start"] = round(cursor, 3)
        block["audio_end"] = round(cursor + len(audio) / sample_rate, 3)
        combined_parts.append(audio)
        cursor += len(audio) / sample_rate
        gap_ms = int(block.get("gap_after_ms", 0)) if i != len(blocks) - 1 else 0
        if gap_ms > 0:
            gap = np.zeros(int(gap_ms / 1000 * sample_rate), dtype=np.float32)
            combined_parts.append(gap)
            cursor += gap_ms / 1000

    master_audio = np.concatenate(combined_parts)
    master_wav = video_dir / "audio_master.wav"
    _write_wav(master_audio, blocks[0]["sample_rate"], master_wav)
    _wav_to_mp3(master_wav, video_dir / "audio.mp3")


def _run_block_mode(video_dir: Path, script_path: Path, engine: str, voice: str, tts_cfg: dict) -> None:
    sentence_units = load_sentence_units(script_path)
    if not sentence_units:
        logger.error("No spoken sentences found in script.txt")
        sys.exit(1)

    tts_blocks_dir = video_dir / "tts_blocks"
    tts_blocks_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = _load_existing_manifest(video_dir)

    if engine == "vieneu":
        block_config = _merge_dict(VIENEU_BLOCK_DEFAULTS, tts_cfg.get("block_config"))
        runtime = VieNeuRuntime(
            voice=voice,
            block_config=block_config,
            infer_overrides=tts_cfg.get("infer_params"),
            speed=tts_cfg.get("speed"),
        )
        initial_blocks = _attach_block_hashes(
            _build_vieneu_blocks(sentence_units, runtime),
            engine=engine,
            voice=runtime.voice,
            speed=runtime.speed,
            block_config=block_config,
            infer_params=runtime.infer_params,
            engine_version=runtime.package_version,
        )
    else:
        block_config = _merge_dict(KOKORO_BLOCK_DEFAULTS, tts_cfg.get("block_config"))
        runtime = KokoroRuntime(voice=tts_cfg.get("voice"), speed=tts_cfg.get("speed"))
        initial_blocks = _attach_block_hashes(
            _build_kokoro_blocks(sentence_units, runtime, block_config),
            engine=engine,
            voice=runtime.voice,
            speed=runtime.speed,
            block_config=block_config,
            infer_params=_effective_kokoro_infer_params(runtime),
            engine_version=runtime.package_version,
        )

    queue = list(initial_blocks)
    final_blocks: list[dict] = []
    block_index = 1
    existing_by_hash = _existing_blocks_by_hash(existing_manifest)
    reused_count = 0
    regenerated_count = 0
    fallback_count = 0
    while queue:
        block = queue.pop(0)
        if not block.block_hash:
            block = _attach_block_hashes(
                [block],
                engine=engine,
                voice=runtime.voice,
                speed=runtime.speed,
                block_config=block_config,
                infer_params=runtime.infer_params if engine == "vieneu" else _effective_kokoro_infer_params(runtime),
                engine_version=runtime.package_version,
            )[0]
        reuse_candidate = None
        if block.block_hash:
            candidates = existing_by_hash.get(block.block_hash, [])
            while candidates:
                candidate = candidates.pop(0)
                if _validate_reusable_wav(video_dir, candidate["wav_path"], runtime.sample_rate):
                    reuse_candidate = candidate
                    break
        if engine == "vieneu":
            replacement_blocks, entry = _generate_vieneu_block_entry(
                video_dir, block_index, block, runtime, reuse_candidate=reuse_candidate
            )
        else:
            replacement_blocks, entry = _generate_kokoro_block_entry(
                video_dir, block_index, block, runtime, block_config, reuse_candidate=reuse_candidate
            )
        if replacement_blocks:
            queue = replacement_blocks + queue
            continue
        if entry is None:
            raise RuntimeError("Block generation returned neither replacement blocks nor manifest entry")
        final_blocks.append(entry)
        if entry.get("generation_status") == "reused":
            reused_count += 1
        else:
            regenerated_count += 1
        if entry.get("fallback_level") == 2:
            fallback_count += 1
        block_index += 1
        logger.info("  TTS block progress: {}/{}", len(final_blocks), len(final_blocks) + len(queue))

    _sentence_indices_are_contiguous(final_blocks, len(sentence_units))
    manifest = {
        "engine": engine,
        "mode": "block",
        "voice": voice if engine == "vieneu" else runtime.voice,
        "speed": runtime.speed,
        "sample_rate": final_blocks[0]["sample_rate"] if final_blocks else None,
        "sentence_count": len(sentence_units),
        "block_count": len(final_blocks),
        "block_config": block_config,
        "cache_schema_version": BLOCK_CACHE_SCHEMA_VERSION,
        "engine_version": runtime.package_version,
        "reused_block_count": reused_count,
        "regenerated_block_count": regenerated_count,
        "fallback_block_count": fallback_count,
        "blocks": final_blocks,
    }
    _rebuild_audio_from_manifest(video_dir, manifest)
    _write_manifest(video_dir, manifest)
    diagnostics = {
        "engine": engine,
        "mode": "block",
        "sentence_count": len(sentence_units),
        "block_count": len(final_blocks),
        "voice": manifest["voice"],
        "speed": runtime.speed,
        "block_config": block_config,
        "cache_schema_version": BLOCK_CACHE_SCHEMA_VERSION,
        "engine_version": runtime.package_version,
        "reused_block_count": reused_count,
        "regenerated_block_count": regenerated_count,
        "fallback_block_count": fallback_count,
    }
    (tts_blocks_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _invalidate_timestamps(video_dir)


async def _edge_tts_async(
    text: str,
    output_path: Path,
    voice: str,
    rate: str = "-5%",
    retries: int = 4,
    log: bool = True,
) -> None:
    import edge_tts

    if log:
        logger.info("Using edge-tts (voice: {})", voice)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
            await communicate.save(str(output_path))
            if output_path.exists() and output_path.stat().st_size > 0:
                if log:
                    logger.info("edge-tts complete -> {}", output_path)
                return
            raise RuntimeError("empty audio file")
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                wait = attempt * 1.5
                logger.warning(
                    "edge-tts attempt {}/{} failed ({}); retrying in {:.1f}s",
                    attempt,
                    retries,
                    str(exc)[:60],
                    wait,
                )
                await asyncio.sleep(wait)
    raise RuntimeError(f"edge-tts failed after {retries} attempts: {last_err}")


def _edge_tts(text: str, output_path: Path, voice: str = "en-US-GuyNeural", rate: str = "-5%") -> None:
    asyncio.run(_edge_tts_async(text, output_path, voice=voice, rate=rate))


def _kokoro_tts(text: str, output_path: Path) -> None:
    runtime = KokoroRuntime()
    audio, sample_rate, _infer_params = runtime.synthesize(text)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _write_wav(audio, sample_rate, tmp_path)
        _wav_to_mp3(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    logger.info("Kokoro TTS complete -> {}", output_path)


def _run_sentence_legacy_mode(
    sentences: list[str],
    output_path: Path,
    timestamps_path: Path,
    tts_engine: str,
    edge_voice: str,
    edge_rate: str,
) -> None:
    import numpy as np
    import soundfile as sf

    logger.info("Sentence legacy mode: {} sentences", len(sentences))
    tmp_dir = Path(tempfile.mkdtemp(prefix="tts_sentence_legacy_"))
    try:
        wavs: list[Path] = []
        vieneu_runtime = None
        kokoro_runtime = None
        if tts_engine == "vieneu":
            vieneu_runtime = VieNeuRuntime(
                voice=edge_voice,
                block_config=_merge_dict(VIENEU_BLOCK_DEFAULTS, None),
                infer_overrides=None,
            )
        elif tts_engine == "kokoro":
            kokoro_runtime = KokoroRuntime()
        for i, sentence in enumerate(sentences, start=1):
            wav_path = tmp_dir / f"sent_{i:04d}.wav"
            if tts_engine == "edge":
                mp3_path = tmp_dir / f"sent_{i:04d}.mp3"
                asyncio.run(_edge_tts_async(sentence, mp3_path, voice=edge_voice, rate=edge_rate, log=False))
                subprocess.run(["ffmpeg", "-y", "-i", str(mp3_path), str(wav_path)], check=True, capture_output=True)
            elif tts_engine == "vieneu":
                assert vieneu_runtime is not None
                audio, sample_rate, _ = vieneu_runtime.synthesize(sentence)
                audio = _trim_trailing_silence(
                    audio,
                    sample_rate,
                    threshold=vieneu_runtime.block_config["trim_trailing_threshold"],
                    keep_ms=vieneu_runtime.block_config["trim_trailing_keep_ms"],
                )
                _write_wav(audio, sample_rate, wav_path)
            else:
                assert kokoro_runtime is not None
                audio, sample_rate, _ = kokoro_runtime.synthesize(sentence)
                _write_wav(audio, sample_rate, wav_path)
            wavs.append(wav_path)

        audio_parts: list[np.ndarray] = []
        timestamps: list[dict] = []
        cursor = 0.0
        silence_ms = 300
        first_audio, sample_rate = sf.read(str(wavs[0]), dtype="float32")
        if first_audio.ndim > 1:
            first_audio = first_audio.mean(axis=1)
        silence = np.zeros(int(silence_ms / 1000 * sample_rate), dtype=np.float32)
        for i, wav_path in enumerate(wavs, start=1):
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            duration = len(audio) / sr
            timestamps.append(
                {
                    "index": i,
                    "start": round(cursor, 3),
                    "end": round(cursor + duration, 3),
                    "text": sentences[i - 1],
                }
            )
            audio_parts.append(audio)
            audio_parts.append(silence)
            cursor += duration + silence_ms / 1000

        combined = np.concatenate(audio_parts)
        combined_wav = tmp_dir / "combined.wav"
        _write_wav(combined, sample_rate, combined_wav)
        _wav_to_mp3(combined_wav, output_path)
        timestamps_path.write_text(json.dumps(timestamps, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_vieneu_paragraph_mode(
    script_path: Path,
    output_path: Path,
    voice: str,
    diagnostics_path: Path | None = None,
    silence_ms: int = 450,
    infer_overrides: dict | None = None,
    trim_paragraphs: bool = False,
    trim_keep_ms: int = 90,
) -> None:
    import numpy as np

    paragraphs = split_paragraph_texts(load_script_text(script_path))
    if not paragraphs:
        raise RuntimeError("No paragraphs found in script.txt")
    runtime = VieNeuRuntime(voice=voice, block_config=_merge_dict(VIENEU_BLOCK_DEFAULTS, None), infer_overrides=infer_overrides)
    gap = np.zeros(int(silence_ms / 1000 * runtime.sample_rate), dtype=np.float32)
    all_audio = []
    diagnostics = []
    cursor = 0.0
    for i, paragraph in enumerate(paragraphs, start=1):
        audio, sample_rate, infer_params = runtime.synthesize(paragraph)
        if trim_paragraphs:
            audio = _trim_trailing_silence(audio, sample_rate, keep_ms=trim_keep_ms)
        start = cursor
        all_audio.append(audio)
        cursor += len(audio) / sample_rate
        if i != len(paragraphs):
            all_audio.append(gap)
            cursor += silence_ms / 1000
        diagnostics.append(
            {
                "paragraph_index": i,
                "text": paragraph,
                "start": round(start, 3),
                "speech_end": round(start + len(audio) / sample_rate, 3),
                "end_with_gap": round(cursor, 3),
                "infer_params": infer_params,
            }
        )
    combined = np.concatenate(all_audio)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        _write_wav(combined, runtime.sample_rate, wav_path)
        _wav_to_mp3(wav_path, output_path)
    finally:
        wav_path.unlink(missing_ok=True)
    if diagnostics_path:
        diagnostics_path.write_text(
            json.dumps(
                {
                    "engine": "vieneu",
                    "mode": "paragraph_whole_blocks",
                    "voice": voice,
                    "sample_rate": runtime.sample_rate,
                    "silence_ms": silence_ms,
                    "paragraphs": diagnostics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def materialize_sentence_fallback_for_block(video_dir: Path, block_index: int) -> dict:
    manifest = _load_manifest(video_dir)
    block = next((item for item in manifest["blocks"] if item["block_index"] == block_index), None)
    if block is None:
        raise RuntimeError(f"Block {block_index} not found in manifest")
    if block.get("fallback_level") == 2 and block.get("fallback_segments"):
        return block

    tts_cfg = _load_tts_config(video_dir)
    engine = manifest["engine"]
    if engine == "vieneu":
        block_config = _merge_dict(VIENEU_BLOCK_DEFAULTS, tts_cfg.get("block_config"))
        runtime = VieNeuRuntime(
            voice=tts_cfg.get("voice", manifest.get("voice", "Bình An")),
            block_config=block_config,
            infer_overrides=tts_cfg.get("infer_params"),
            speed=tts_cfg.get("speed"),
        )
    else:
        block_config = _merge_dict(KOKORO_BLOCK_DEFAULTS, tts_cfg.get("block_config"))
        runtime = KokoroRuntime(voice=tts_cfg.get("voice", manifest.get("voice")), speed=tts_cfg.get("speed", manifest.get("speed")))

    block_spec = BlockSpec(
        sentence_units=[],
        text=block["text"],
        sentence_indices=block["sentence_indices"],
        sentence_texts=block["sentence_texts"],
        raw_chars=block["raw_chars"],
        estimated_seconds=block["estimated_seconds"],
        normalized_chars=block.get("normalized_chars"),
        phoneme_chars=block.get("phoneme_chars"),
    )
    fallback = _render_sentence_fallback_block(video_dir, block_index, block_spec, engine, runtime, block_config, {"temperature": 0.25})
    block["wav_path"] = fallback["wav_path"]
    block["sample_rate"] = fallback["sample_rate"]
    block["actual_seconds"] = fallback["actual_seconds"]
    block["fallback_level"] = 2
    block["fallback_segments"] = fallback["fallback_segments"]
    block["generation_status"] = "fallback"
    _rebuild_audio_from_manifest(video_dir, manifest)
    manifest["fallback_block_count"] = sum(1 for item in manifest["blocks"] if item.get("fallback_level") == 2)
    _write_manifest(video_dir, manifest)
    diagnostics_path = video_dir / "tts_blocks" / "diagnostics.json"
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        diagnostics["fallback_block_count"] = manifest["fallback_block_count"]
        diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return block


def run(video_id: str) -> None:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    script_path = video_dir / "script.txt"
    output_path = video_dir / "audio.mp3"
    timestamps_path = video_dir / "timestamps.json"

    tts_cfg = _load_tts_config(video_dir)
    tts_engine = tts_cfg.get("engine", "kokoro")
    tts_mode = tts_cfg.get("mode")
    edge_voice = tts_cfg.get("voice", "en-US-GuyNeural")
    edge_rate = tts_cfg.get("rate", "-5%")
    clone_voice_id = tts_cfg.get("voice_id", "default")
    clone_ref_audio = tts_cfg.get("ref_audio")
    clone_speed = tts_cfg.get("speed", 1.0)
    clone_ref_text = tts_cfg.get("ref_text", "")

    if not script_path.exists():
        logger.error("script.txt not found: {}", script_path)
        sys.exit(1)

    text = load_script_text(script_path)
    if not text:
        logger.error("script.txt is empty")
        sys.exit(1)

    if tts_mode is None and tts_engine in {"vieneu", "kokoro"}:
        tts_mode = "block"
    elif tts_mode is None:
        tts_mode = "default"

    logger.info("TTS config: engine={} mode={} voice={} rate={}", tts_engine, tts_mode, edge_voice, edge_rate)

    if tts_mode == "paragraph_audit" and tts_engine == "vieneu":
        try:
            _run_vieneu_paragraph_mode(script_path=script_path, output_path=output_path, voice=edge_voice)
        except Exception as exc:
            logger.error("VieNeu paragraph audit TTS failed: {}", exc)
            sys.exit(1)
    elif tts_mode == "block" and tts_engine in {"vieneu", "kokoro"}:
        try:
            _run_block_mode(video_dir, script_path, tts_engine, edge_voice, tts_cfg)
        except Exception as exc:
            logger.error("Block TTS failed: {}", exc)
            sys.exit(1)
    elif tts_mode in {"sentence", "sentence_legacy"}:
        sentences = [unit.text for unit in load_sentence_units(script_path)]
        if not sentences:
            logger.error("No sentences found in script.txt")
            sys.exit(1)
        try:
            _run_sentence_legacy_mode(sentences, output_path, timestamps_path, tts_engine, edge_voice, edge_rate)
        except Exception as exc:
            logger.error("Sentence legacy TTS failed: {}", exc)
            sys.exit(1)
    elif tts_engine == "clone":
        try:
            from tts_generation.runpod_tts_client import clone_voice

            ref_path = None
            if clone_ref_audio:
                ref_candidate = Path(clone_ref_audio)
                ref_path = ref_candidate if ref_candidate.is_absolute() else video_dir / ref_candidate
            audio_bytes = clone_voice(
                text,
                voice_id=clone_voice_id,
                ref_audio_path=ref_path,
                ref_text=clone_ref_text,
                speed=clone_speed,
            )
            output_path.write_bytes(audio_bytes)
        except Exception as exc:
            logger.error("F5-TTS clone failed: {}", exc)
            sys.exit(1)
    elif tts_engine == "edge":
        try:
            _edge_tts(text, output_path, voice=edge_voice, rate=edge_rate)
        except Exception as exc:
            logger.error("edge-tts failed: {}", exc)
            sys.exit(1)
    else:
        try:
            _kokoro_tts(text, output_path)
        except Exception as exc:
            logger.warning("Kokoro TTS failed ({}), falling back to edge-tts", exc)
            try:
                _edge_tts(text, output_path, voice=edge_voice, rate=edge_rate)
            except Exception as edge_exc:
                logger.error("edge-tts also failed: {}", edge_exc)
                sys.exit(1)

    if output_path.exists():
        duration = _get_audio_duration(output_path)
        minutes = duration / 60
        logger.info("Audio duration: {:.1f} minutes ({:.0f}s)", minutes, duration)
        if minutes < 7:
            logger.warning("Audio is short ({:.1f} min). Target: 8-10 min.", minutes)
        elif minutes > 11:
            logger.warning("Audio is long ({:.1f} min). Target: 8-10 min.", minutes)
    else:
        logger.error("TTS output file not created: {}", output_path)
        sys.exit(1)
