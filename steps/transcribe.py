"""Step 3: Transcribe audio to sentence-level timestamps."""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from loguru import logger

import config
from steps import tts as tts_step
from steps.text_units import load_sentence_units

MAX_BLOCK_ALIGNMENT_RESTARTS = 2


def _normalize_token(text: str) -> str:
    lowered = text.lower()
    lowered = unicodedata.normalize("NFD", lowered)
    lowered = "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")
    lowered = re.sub(r"[^\w\s]", "", lowered)
    return lowered.strip()


def _tokenize_sentence(text: str) -> list[str]:
    return [token for token in (_normalize_token(part) for part in text.split()) if token]


def _audio_duration(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def _load_blocks_manifest(video_dir: Path) -> dict | None:
    manifest_path = video_dir / "tts_blocks" / "blocks.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _should_use_block_mode(video_dir: Path, manifest: dict | None) -> bool:
    if not manifest:
        return False
    if manifest.get("mode") != "block":
        return False
    if not manifest.get("blocks"):
        return False
    return (video_dir / "audio_master.wav").exists()


def _collect_aligned_words(result) -> list[dict]:
    words: list[dict] = []
    for seg in result.segments:
        for word in seg.words:
            if word.start is None or word.end is None:
                continue
            normalized = _normalize_token(word.word)
            if not normalized:
                continue
            words.append(
                {
                    "word": word.word,
                    "normalized": normalized,
                    "start": float(word.start),
                    "end": float(word.end),
                }
            )
    return words


def _map_aligned_words_to_sentences(
    sentence_texts: list[str],
    aligned_words: list[dict],
    audio_start: float,
    starting_index: int,
) -> tuple[list[dict], float, bool]:
    canonical_words: list[dict] = []
    sentence_spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in sentence_texts:
        tokens = _tokenize_sentence(sentence)
        start = cursor
        for token in tokens:
            canonical_words.append({"token": token})
            cursor += 1
        sentence_spans.append((start, cursor))

    aligned_tokens = [word["normalized"] for word in aligned_words]
    canonical_tokens = [word["token"] for word in canonical_words]
    matcher = SequenceMatcher(a=canonical_tokens, b=aligned_tokens, autojunk=False)

    matched: dict[int, dict] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(i2 - i1):
            matched[i1 + offset] = aligned_words[j1 + offset]

    coverage = len(matched) / max(1, len(canonical_words))
    timestamps: list[dict] = []
    all_sentences_matched = True
    for sentence_offset, sentence in enumerate(sentence_texts):
        span_start, span_end = sentence_spans[sentence_offset]
        matched_words = [matched[idx] for idx in range(span_start, span_end) if idx in matched]
        if not matched_words:
            all_sentences_matched = False
            continue
        start = round(audio_start + matched_words[0]["start"], 3)
        end = round(audio_start + matched_words[-1]["end"], 3)
        if end <= start:
            end = round(start + 0.1, 3)
        timestamps.append(
            {
                "index": starting_index + sentence_offset,
                "start": start,
                "end": end,
                "text": sentence,
            }
        )

    return timestamps, coverage, all_sentences_matched


def _timestamps_from_fallback_segments(block: dict, starting_index: int) -> list[dict]:
    timestamps = []
    audio_start = float(block["audio_start"])
    for sentence_offset, segment in enumerate(block.get("fallback_segments", [])):
        timestamps.append(
            {
                "index": starting_index + sentence_offset,
                "start": round(audio_start + float(segment["start_in_block"]), 3),
                "end": round(audio_start + float(segment["end_in_block"]), 3),
                "text": segment["text"],
            }
        )
    return timestamps


def _run_stable_ts_blocks(
    video_dir: Path,
    model_name: str,
    language: str,
    device: str,
) -> list[dict]:
    import stable_whisper

    manifest_path = video_dir / "tts_blocks" / "blocks.json"
    logger.info("Block-aware alignment active via {}", manifest_path)
    model = stable_whisper.load_model(model_name, device=device)
    audio_master_path = video_dir / "audio_master.wav"
    audio_duration = _audio_duration(audio_master_path)
    for restart_count in range(MAX_BLOCK_ALIGNMENT_RESTARTS + 1):
        manifest = _load_blocks_manifest(video_dir)
        if not _should_use_block_mode(video_dir, manifest):
            logger.error("Block-aware stable-ts requested but block artifacts are incomplete")
            sys.exit(1)

        all_timestamps: list[dict] = []
        cursor_index = 1
        block_diagnostics: list[dict] = []
        restart_needed = False

        for block in manifest["blocks"]:
            if block.get("fallback_level") == 2 and block.get("fallback_segments"):
                timestamps = _timestamps_from_fallback_segments(block, cursor_index)
                if not timestamps or any(item["end"] <= item["start"] for item in timestamps):
                    logger.error("Block {} is already fallback_level=2 but fallback timestamps are invalid", block["block_index"])
                    sys.exit(1)
                all_timestamps.extend(timestamps)
                block_diagnostics.append(
                    {
                        "block_index": block["block_index"],
                        "coverage": 1.0,
                        "used_fallback_segments": True,
                    }
                )
                cursor_index += len(block["sentence_texts"])
                continue

            block_wav = video_dir / block["wav_path"]
            canonical_text = " ".join(block["sentence_texts"])
            result = model.align(str(block_wav), canonical_text, language=language)
            aligned_words = _collect_aligned_words(result)
            timestamps, coverage, all_matched = _map_aligned_words_to_sentences(
                block["sentence_texts"],
                aligned_words,
                float(block["audio_start"]),
                cursor_index,
            )

            if coverage < 0.90 or not all_matched:
                if block.get("fallback_level") == 2:
                    logger.error(
                        "Block {} is already fallback_level=2 but still failed stable-ts validation ({:.2%}, all_matched={})",
                        block["block_index"],
                        coverage,
                        all_matched,
                    )
                    sys.exit(1)
                if restart_count >= MAX_BLOCK_ALIGNMENT_RESTARTS:
                    logger.error(
                        "Block {} exceeded maximum stable-ts fallback restarts ({})",
                        block["block_index"],
                        MAX_BLOCK_ALIGNMENT_RESTARTS,
                    )
                    sys.exit(1)
                logger.warning(
                    "Block {} alignment coverage {:.2%} (all_matched={}) -> sentence fallback and restart",
                    block["block_index"],
                    coverage,
                    all_matched,
                )
                tts_step.materialize_sentence_fallback_for_block(video_dir, block["block_index"])
                restart_needed = True
                break

            all_timestamps.extend(timestamps)
            block_diagnostics.append(
                {
                    "block_index": block["block_index"],
                    "coverage": round(coverage, 4),
                    "used_fallback_segments": False,
                }
            )
            cursor_index += len(block["sentence_texts"])

        if restart_needed:
            logger.info(
                "Restarting stable-ts block alignment after fallback rebuild ({}/{})",
                restart_count + 1,
                MAX_BLOCK_ALIGNMENT_RESTARTS,
            )
            continue

        if not all_timestamps:
            logger.error("No timestamps produced in block-aware stable-ts mode")
            sys.exit(1)

        if len(all_timestamps) != manifest["sentence_count"]:
            logger.error(
                "Timestamp count {} does not match manifest sentence count {}",
                len(all_timestamps),
                manifest["sentence_count"],
            )
            sys.exit(1)

        prev_end = 0.0
        for entry in all_timestamps:
            entry["start"] = round(max(entry["start"], prev_end), 3)
            entry["end"] = round(max(entry["end"], entry["start"] + 0.1), 3)
            prev_end = entry["end"]

        if audio_duration and abs(all_timestamps[-1]["end"] - audio_duration) > 2.5:
            logger.error(
                "Final timestamp {:.3f}s drifts too far from audio_master duration {:.3f}s",
                all_timestamps[-1]["end"],
                audio_duration,
            )
            sys.exit(1)

        diagnostics_path = video_dir / "tts_blocks" / "alignment_diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(
                {
                    "engine": "stable_ts",
                    "mode": "block",
                    "audio_duration": round(audio_duration, 3),
                    "restart_count": restart_count,
                    "blocks": block_diagnostics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return all_timestamps

    logger.error("Stable-ts block alignment exhausted restart guard unexpectedly")
    sys.exit(1)


def _run_faster_whisper_blocks(
    video_dir: Path,
    model_name: str,
    language: str | None,
) -> list[dict]:
    from faster_whisper import WhisperModel

    logger.info("Block-aware faster-whisper active via {}", video_dir / "tts_blocks" / "blocks.json")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    audio_master_path = video_dir / "audio_master.wav"
    audio_duration = _audio_duration(audio_master_path)
    for restart_count in range(MAX_BLOCK_ALIGNMENT_RESTARTS + 1):
        manifest = _load_blocks_manifest(video_dir)
        if not _should_use_block_mode(video_dir, manifest):
            logger.error("Block-aware faster-whisper requested but block artifacts are incomplete")
            sys.exit(1)

        all_timestamps: list[dict] = []
        cursor_index = 1
        block_diagnostics: list[dict] = []
        restart_needed = False

        for block in manifest["blocks"]:
            if block.get("fallback_level") == 2 and block.get("fallback_segments"):
                timestamps = _timestamps_from_fallback_segments(block, cursor_index)
                if not timestamps or any(item["end"] <= item["start"] for item in timestamps):
                    logger.error("Block {} is already fallback_level=2 but fallback timestamps are invalid", block["block_index"])
                    sys.exit(1)
                all_timestamps.extend(timestamps)
                block_diagnostics.append(
                    {
                        "block_index": block["block_index"],
                        "coverage": 1.0,
                        "used_fallback_segments": True,
                    }
                )
                cursor_index += len(block["sentence_texts"])
                continue

            block_wav = video_dir / block["wav_path"]
            kwargs = {"word_timestamps": True}
            if language:
                kwargs["language"] = language
            segments_gen, _info = model.transcribe(str(block_wav), **kwargs)
            aligned_words = []
            for seg in segments_gen:
                if not seg.words:
                    continue
                for word in seg.words:
                    normalized = _normalize_token(word.word)
                    if not normalized or word.start is None or word.end is None:
                        continue
                    aligned_words.append(
                        {
                            "word": word.word,
                            "normalized": normalized,
                            "start": float(word.start),
                            "end": float(word.end),
                        }
                    )

            timestamps, coverage, all_matched = _map_aligned_words_to_sentences(
                block["sentence_texts"],
                aligned_words,
                float(block["audio_start"]),
                cursor_index,
            )
            if coverage < 0.90 or not all_matched:
                if block.get("fallback_level") == 2:
                    logger.error(
                        "Block {} is already fallback_level=2 but still failed faster-whisper validation ({:.2%}, all_matched={})",
                        block["block_index"],
                        coverage,
                        all_matched,
                    )
                    sys.exit(1)
                if restart_count >= MAX_BLOCK_ALIGNMENT_RESTARTS:
                    logger.error(
                        "Block {} exceeded maximum faster-whisper fallback restarts ({})",
                        block["block_index"],
                        MAX_BLOCK_ALIGNMENT_RESTARTS,
                    )
                    sys.exit(1)
                logger.warning(
                    "Block {} faster-whisper coverage {:.2%} (all_matched={}) -> sentence fallback and restart",
                    block["block_index"],
                    coverage,
                    all_matched,
                )
                tts_step.materialize_sentence_fallback_for_block(video_dir, block["block_index"])
                restart_needed = True
                break

            all_timestamps.extend(timestamps)
            block_diagnostics.append(
                {
                    "block_index": block["block_index"],
                    "coverage": round(coverage, 4),
                    "used_fallback_segments": False,
                }
            )
            cursor_index += len(block["sentence_texts"])

        if restart_needed:
            logger.info(
                "Restarting faster-whisper block alignment after fallback rebuild ({}/{})",
                restart_count + 1,
                MAX_BLOCK_ALIGNMENT_RESTARTS,
            )
            continue

        if len(all_timestamps) != manifest["sentence_count"]:
            logger.error(
                "Timestamp count {} does not match manifest sentence count {}",
                len(all_timestamps),
                manifest["sentence_count"],
            )
            sys.exit(1)

        prev_end = 0.0
        for entry in all_timestamps:
            entry["start"] = round(max(entry["start"], prev_end), 3)
            entry["end"] = round(max(entry["end"], entry["start"] + 0.1), 3)
            prev_end = entry["end"]

        if audio_duration and abs(all_timestamps[-1]["end"] - audio_duration) > 2.5:
            logger.error(
                "Final timestamp {:.3f}s drifts too far from audio_master duration {:.3f}s",
                all_timestamps[-1]["end"],
                audio_duration,
            )
            sys.exit(1)

        diagnostics_path = video_dir / "tts_blocks" / "alignment_diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(
                {
                    "engine": "faster_whisper",
                    "mode": "block",
                    "audio_duration": round(audio_duration, 3),
                    "restart_count": restart_count,
                    "blocks": block_diagnostics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return all_timestamps

    logger.error("Faster-whisper block alignment exhausted restart guard unexpectedly")
    sys.exit(1)


def _run_stable_ts_full(
    audio_path: Path,
    script_path: Path,
    model_name: str,
    language: str,
    device: str = "cpu",
) -> list[dict]:
    import stable_whisper

    sentences = [unit.text for unit in load_sentence_units(script_path)]
    if not sentences:
        logger.error("No sentences found in script.txt")
        sys.exit(1)

    canonical_text = " ".join(sentences)
    model = stable_whisper.load_model(model_name, device=device)
    result = model.align(str(audio_path), canonical_text, language=language)
    aligned_words = _collect_aligned_words(result)
    timestamps, coverage, all_matched = _map_aligned_words_to_sentences(sentences, aligned_words, 0.0, 1)
    if coverage < 0.90 or not all_matched:
        logger.error(
            "Full-file stable-ts coverage too low ({:.2%}, all_matched={})",
            coverage,
            all_matched,
        )
        sys.exit(1)
    return timestamps


def _align_sentences_to_words(sentences: list[str], whisper_words: list[dict]) -> list[dict]:
    result = []
    word_index = 0
    total_words = len(whisper_words)
    for i, sentence in enumerate(sentences, start=1):
        sentence_tokens = _tokenize_sentence(sentence)
        token_count = len(sentence_tokens)
        if token_count == 0:
            continue
        if word_index >= total_words:
            prev_end = result[-1]["end"] if result else 0.0
            result.append(
                {
                    "index": i,
                    "start": round(prev_end, 3),
                    "end": round(prev_end + 1.0, 3),
                    "text": sentence,
                }
            )
            continue
        end_index = min(word_index + token_count, total_words) - 1
        result.append(
            {
                "index": i,
                "start": round(whisper_words[word_index]["start"], 3),
                "end": round(whisper_words[end_index]["end"], 3),
                "text": sentence,
            }
        )
        word_index = end_index + 1
    return result


def _run_faster_whisper(
    audio_path: Path,
    script_path: Path | None,
    model_name: str,
    language: str | None,
    align_mode: bool,
) -> list[dict]:
    from faster_whisper import WhisperModel

    logger.info("Loading faster-whisper model ({}, CPU, int8)...", model_name)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    kwargs = {"word_timestamps": True}
    if language:
        kwargs["language"] = language

    segments_gen, info = model.transcribe(str(audio_path), **kwargs)
    segments = list(segments_gen)
    logger.info("Detected language: {} (probability {:.2f})", info.language, info.language_probability)

    if align_mode and script_path and script_path.exists():
        whisper_words = []
        for seg in segments:
            if not seg.words:
                continue
            for word in seg.words:
                whisper_words.append({"word": word.word, "start": word.start, "end": word.end})
        sentences = [unit.text for unit in load_sentence_units(script_path)]
        return _align_sentences_to_words(sentences, whisper_words)

    result = []
    current_words: list[str] = []
    current_start = None
    current_end = None
    index = 1
    for seg in segments:
        for word in seg.words:
            if current_start is None:
                current_start = word.start
            current_words.append(word.word)
            current_end = word.end
            text_so_far = "".join(current_words).strip()
            if text_so_far.endswith((".", "!", "?", "...")) or (current_end - current_start) >= 4.5:
                result.append(
                    {
                        "index": index,
                        "start": round(current_start, 3),
                        "end": round(current_end, 3),
                        "text": text_so_far,
                    }
                )
                index += 1
                current_words = []
                current_start = None
                current_end = None

    if current_words and current_start is not None and current_end is not None:
        result.append(
            {
                "index": index,
                "start": round(current_start, 3),
                "end": round(current_end, 3),
                "text": "".join(current_words).strip(),
            }
        )
    return result


def run(video_id: str) -> None:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    audio_path = video_dir / "audio.mp3"
    output_path = video_dir / "timestamps.json"
    script_path = video_dir / "script.txt"

    if not audio_path.exists():
        logger.error("audio.mp3 not found: {}", audio_path)
        sys.exit(1)

    engine = "faster_whisper"
    whisper_model = "medium.en"
    whisper_language: str | None = None
    align_mode = False
    device = "cpu"

    cfg_path = video_dir / "transcribe_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        engine = cfg.get("engine", engine)
        whisper_model = cfg.get("model", whisper_model)
        whisper_language = cfg.get("language", whisper_language)
        align_mode = cfg.get("mode", "") == "align"
        device = cfg.get("device", device)
        logger.info(
            "Transcribe config: engine={} model={} language={} mode={} device={}",
            engine,
            whisper_model,
            whisper_language,
            "align" if align_mode else "default",
            device,
        )

    manifest = _load_blocks_manifest(video_dir)
    use_block_mode = _should_use_block_mode(video_dir, manifest)

    if use_block_mode and engine == "stable_ts":
        result = _run_stable_ts_blocks(
            video_dir,
            model_name=whisper_model,
            language=whisper_language or "vi",
            device=device,
        )
    elif use_block_mode:
        result = _run_faster_whisper_blocks(
            video_dir,
            model_name=whisper_model,
            language=whisper_language,
        )
    elif engine == "stable_ts":
        if not script_path.exists():
            logger.error("stable_ts engine requires script.txt: {}", script_path)
            sys.exit(1)
        result = _run_stable_ts_full(
            audio_path,
            script_path,
            model_name=whisper_model,
            language=whisper_language or "vi",
            device=device,
        )
    else:
        result = _run_faster_whisper(
            audio_path,
            script_path,
            model_name=whisper_model,
            language=whisper_language,
            align_mode=align_mode,
        )

    expected_sentence_count = len(load_sentence_units(script_path)) if script_path.exists() else None
    if expected_sentence_count is not None and len(result) != expected_sentence_count:
        logger.error(
            "timestamps.json count {} does not match script sentence count {}",
            len(result),
            expected_sentence_count,
        )
        sys.exit(1)

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    total_duration = result[-1]["end"] if result else 0.0
    logger.info("Transcription complete: {} segments, {:.1f}s total -> {}", len(result), total_duration, output_path)
