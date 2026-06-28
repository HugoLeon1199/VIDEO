from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import config
from steps import transcribe


@dataclass(frozen=True)
class WordSpan:
    sentence_index: int
    word_index: int
    text: str
    normalized: str
    start: float
    end: float


@dataclass(frozen=True)
class SentenceSpan:
    sentence_index: int
    text: str
    start: float
    end: float
    word_start: int | None
    word_end: int | None


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8").lstrip("\ufeff"))


def load_sentence_spans(video_dir: Path) -> list[SentenceSpan]:
    timestamps = _load_json(video_dir / "timestamps.json")
    word_spans_by_sentence: dict[int, list[WordSpan]] = {}
    for word in load_exact_word_spans(video_dir):
        word_spans_by_sentence.setdefault(word.sentence_index, []).append(word)
    spans: list[SentenceSpan] = []
    for item in timestamps:
        sentence_index = int(item["index"])
        words = word_spans_by_sentence.get(sentence_index, [])
        spans.append(
            SentenceSpan(
                sentence_index=sentence_index,
                text=str(item["text"]).strip(),
                start=float(item["start"]),
                end=float(item["end"]),
                word_start=words[0].word_index if words else None,
                word_end=words[-1].word_index if words else None,
            )
        )
    return spans


def load_word_diagnostics(video_dir: Path) -> dict:
    path = video_dir / transcribe.WORD_DIAGNOSTICS_NAME
    if not path.exists():
        return {
            "subtitle_ready": False,
            "reason": "missing_word_timestamps_diagnostics",
            "affected_blocks": [],
            "alignment_coverage": 0.0,
        }
    return _load_json(path)


def load_exact_word_spans(video_dir: Path) -> list[WordSpan]:
    diagnostics = load_word_diagnostics(video_dir)
    if not diagnostics.get("subtitle_ready"):
        return []
    path = video_dir / transcribe.WORD_TIMESTAMPS_NAME
    if not path.exists():
        return []
    payload = _load_json(path)
    return [
        WordSpan(
            sentence_index=int(item["sentence_index"]),
            word_index=int(item["word_index"]),
            text=str(item["text"]),
            normalized=str(item["normalized"]),
            start=float(item["start"]),
            end=float(item["end"]),
        )
        for item in payload
    ]


def exact_word_timing_ready(video_dir: Path) -> bool:
    diagnostics = load_word_diagnostics(video_dir)
    if not diagnostics.get("subtitle_ready"):
        return False
    return (video_dir / transcribe.WORD_TIMESTAMPS_NAME).exists()


def build_fallback_sentence_beats(sentence_spans: list[SentenceSpan]) -> list[dict]:
    beats: list[dict] = []
    for span in sentence_spans:
        beats.append(
            {
                "index": len(beats) + 1,
                "source_sentence_index": span.sentence_index,
                "beat_index": 1,
                "word_start": span.word_start,
                "word_end": span.word_end,
                "start": round(span.start, 3),
                "end": round(span.end, 3),
                "scene_text": span.text,
                "visual_intent": span.text,
            }
        )
    return beats


def derive_beat_timings(
    beat_plan: list[dict],
    sentence_spans: list[SentenceSpan],
    word_spans: list[WordSpan],
) -> list[dict]:
    sentence_map = {item.sentence_index: item for item in sentence_spans}
    words_by_sentence: dict[int, list[WordSpan]] = {}
    for word in word_spans:
        words_by_sentence.setdefault(word.sentence_index, []).append(word)

    normalized: list[dict] = []
    sentence_cursors: dict[int, int] = {}
    for raw in beat_plan:
        sentence_index = int(raw["source_sentence_index"])
        sentence = sentence_map[sentence_index]
        sentence_words = words_by_sentence.get(sentence_index, [])
        if not sentence_words:
            raise ValueError(f"Missing exact word timings for sentence {sentence_index}")
        start_word = int(raw["word_start"])
        end_word = int(raw["word_end"])
        if start_word > end_word:
            raise ValueError(f"Invalid word range {start_word}>{end_word} in sentence {sentence_index}")
        if start_word < sentence_words[0].word_index or end_word > sentence_words[-1].word_index:
            raise ValueError(f"Word range {start_word}-{end_word} is outside sentence {sentence_index}")
        expected_start = sentence_cursors.get(sentence_index, sentence_words[0].word_index)
        if start_word != expected_start:
            raise ValueError(
                f"Sentence {sentence_index} beat boundary gap/overlap: expected {expected_start}, got {start_word}"
            )
        if end_word >= sentence_words[-1].word_index:
            sentence_cursors[sentence_index] = sentence_words[-1].word_index + 1
        else:
            sentence_cursors[sentence_index] = end_word + 1
        word_lookup = {word.word_index: word for word in sentence_words}
        start = word_lookup[start_word].start
        end = word_lookup[end_word].end
        normalized.append(
            {
                "index": len(normalized) + 1,
                "source_sentence_index": sentence_index,
                "beat_index": int(raw["beat_index"]),
                "word_start": start_word,
                "word_end": end_word,
                "start": round(start, 3),
                "end": round(end, 3),
                "scene_text": str(raw["scene_text"]).strip() or sentence.text,
                "visual_intent": str(raw.get("visual_intent", raw["scene_text"])).strip() or sentence.text,
            }
        )
    validate_visual_beats(normalized, sentence_spans)
    return normalize_visual_beats(normalized, sentence_spans)


def normalize_visual_beats(beats: list[dict], sentence_spans: list[SentenceSpan]) -> list[dict]:
    sentence_map = {item.sentence_index: item for item in sentence_spans}
    merged: list[dict] = []
    for beat in beats:
        duration = float(beat["end"]) - float(beat["start"])
        if duration >= 2.3 or not merged:
            merged.append(dict(beat))
            continue
        prev = merged[-1]
        if int(prev["source_sentence_index"]) != int(beat["source_sentence_index"]):
            merged.append(dict(beat))
            continue
        prev["word_end"] = beat["word_end"]
        prev["end"] = beat["end"]
        prev["scene_text"] = sentence_map[int(prev["source_sentence_index"])].text
        prev["visual_intent"] = f"{prev['visual_intent']} / {beat['visual_intent']}".strip(" /")
    for idx, beat in enumerate(merged, start=1):
        beat["index"] = idx
    validate_visual_beats(merged, sentence_spans)
    return merged


def validate_visual_beats(beats: list[dict], sentence_spans: list[SentenceSpan]) -> None:
    sentence_map = {item.sentence_index: item for item in sentence_spans}
    previous_end = None
    grouped: dict[int, list[dict]] = {}
    for beat in beats:
        sentence_index = int(beat["source_sentence_index"])
        grouped.setdefault(sentence_index, []).append(beat)
        start = float(beat["start"])
        end = float(beat["end"])
        if end <= start:
            raise ValueError(f"Beat {beat['index']} has non-positive duration")
        if previous_end is not None and start < previous_end - 0.001:
            raise ValueError(f"Beat {beat['index']} overlaps previous beat")
        previous_end = end
        sentence = sentence_map[sentence_index]
        if start < sentence.start - 0.001 or end > sentence.end + 0.001:
            raise ValueError(f"Beat {beat['index']} escapes sentence {sentence_index} timing bounds")
    for sentence_index, items in grouped.items():
        if len(items) > 3:
            raise ValueError(f"Sentence {sentence_index} exceeds max 3 beats")
        previous_word_end = None
        for item in items:
            word_start = item.get("word_start")
            word_end = item.get("word_end")
            if word_start is not None and word_end is not None:
                if previous_word_end is not None and int(word_start) != previous_word_end + 1:
                    raise ValueError(f"Sentence {sentence_index} has non-contiguous word ranges")
                previous_word_end = int(word_end)


def prompt_template_metadata(language: str) -> dict:
    prompt_path = Path(config.PROMPTS_DIR) / f"image_prompt_{language}.txt"
    text = prompt_path.read_text(encoding="utf-8")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            break
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return {"path": prompt_path, "text": text, "fields": fields}
