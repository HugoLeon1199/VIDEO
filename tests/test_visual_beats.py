from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
from steps import visual_beats
from steps import transcribe


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.fixture(autouse=True)
def _patch_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))


def _make_video_dir(tmp_path: Path, subtitle_ready: bool = True) -> Path:
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    _write_json(
        video_dir / "timestamps.json",
        [
            {"index": 1, "start": 0.0, "end": 4.2, "text": "A short clear line."},
            {"index": 2, "start": 4.2, "end": 11.0, "text": "A longer sentence with two ideas, then a reveal at the end."},
        ],
    )
    _write_json(
        video_dir / transcribe.WORD_DIAGNOSTICS_NAME,
        {
            "subtitle_ready": subtitle_ready,
            "reason": "" if subtitle_ready else "exact_canonical_word_timing_unavailable",
            "affected_blocks": [] if subtitle_ready else [1],
            "alignment_coverage": 1.0 if subtitle_ready else 0.72,
        },
    )
    if subtitle_ready:
        _write_json(
            video_dir / transcribe.WORD_TIMESTAMPS_NAME,
            [
                {"sentence_index": 1, "word_index": 1, "text": "A", "normalized": "a", "start": 0.0, "end": 0.4, "timing_source": "stable_ts"},
                {"sentence_index": 1, "word_index": 2, "text": "short", "normalized": "short", "start": 0.45, "end": 1.0, "timing_source": "stable_ts"},
                {"sentence_index": 1, "word_index": 3, "text": "clear", "normalized": "clear", "start": 1.05, "end": 1.9, "timing_source": "stable_ts"},
                {"sentence_index": 1, "word_index": 4, "text": "line.", "normalized": "line", "start": 1.95, "end": 4.2, "timing_source": "stable_ts"},
                {"sentence_index": 2, "word_index": 1, "text": "A", "normalized": "a", "start": 4.2, "end": 4.5, "timing_source": "stable_ts"},
                {"sentence_index": 2, "word_index": 2, "text": "longer", "normalized": "longer", "start": 4.55, "end": 5.2, "timing_source": "stable_ts"},
                {"sentence_index": 2, "word_index": 3, "text": "sentence", "normalized": "sentence", "start": 5.25, "end": 6.0, "timing_source": "stable_ts"},
                {"sentence_index": 2, "word_index": 4, "text": "with", "normalized": "with", "start": 6.05, "end": 6.4, "timing_source": "stable_ts"},
                {"sentence_index": 2, "word_index": 5, "text": "two", "normalized": "two", "start": 6.45, "end": 6.9, "timing_source": "stable_ts"},
                {"sentence_index": 2, "word_index": 6, "text": "ideas,", "normalized": "ideas", "start": 6.95, "end": 7.6, "timing_source": "stable_ts"},
                {"sentence_index": 2, "word_index": 7, "text": "then", "normalized": "then", "start": 8.5, "end": 8.9, "timing_source": "stable_ts"},
                {"sentence_index": 2, "word_index": 8, "text": "a", "normalized": "a", "start": 8.95, "end": 9.1, "timing_source": "stable_ts"},
                {"sentence_index": 2, "word_index": 9, "text": "reveal", "normalized": "reveal", "start": 9.15, "end": 10.0, "timing_source": "stable_ts"},
                {"sentence_index": 2, "word_index": 10, "text": "at", "normalized": "at", "start": 10.05, "end": 10.3, "timing_source": "stable_ts"},
                {"sentence_index": 2, "word_index": 11, "text": "the", "normalized": "the", "start": 10.35, "end": 10.55, "timing_source": "stable_ts"},
                {"sentence_index": 2, "word_index": 12, "text": "end.", "normalized": "end", "start": 10.6, "end": 11.0, "timing_source": "stable_ts"},
            ],
        )
    return video_dir


def test_sentence_fallback_used_when_exact_words_unavailable(tmp_path: Path):
    video_dir = _make_video_dir(tmp_path, subtitle_ready=False)
    spans = visual_beats.load_sentence_spans(video_dir)
    beats = visual_beats.build_fallback_sentence_beats(spans)
    assert len(beats) == 2
    assert beats[0]["source_sentence_index"] == 1
    assert beats[1]["source_sentence_index"] == 2


def test_multi_clause_sentence_uses_exact_word_boundaries(tmp_path: Path):
    video_dir = _make_video_dir(tmp_path, subtitle_ready=True)
    spans = visual_beats.load_sentence_spans(video_dir)
    words = visual_beats.load_exact_word_spans(video_dir)
    beats = visual_beats.derive_beat_timings(
        [
            {"source_sentence_index": 1, "beat_index": 1, "word_start": 1, "word_end": 4, "scene_text": "A short clear line.", "visual_intent": "single scene"},
            {"source_sentence_index": 2, "beat_index": 1, "word_start": 1, "word_end": 6, "scene_text": "A longer sentence with two ideas,", "visual_intent": "set up the first idea"},
            {"source_sentence_index": 2, "beat_index": 2, "word_start": 7, "word_end": 12, "scene_text": "then a reveal at the end.", "visual_intent": "the reveal lands after the pause"},
        ],
        spans,
        words,
    )
    assert len(beats) == 3
    assert beats[1]["start"] == 4.2
    assert beats[1]["end"] == 7.6
    assert beats[2]["start"] == 8.5
    assert beats[2]["end"] == 11.0


def test_short_beat_is_merged(tmp_path: Path):
    video_dir = _make_video_dir(tmp_path, subtitle_ready=True)
    spans = visual_beats.load_sentence_spans(video_dir)
    merged = visual_beats.normalize_visual_beats(
        [
            {"index": 1, "source_sentence_index": 2, "beat_index": 1, "word_start": 1, "word_end": 8, "start": 4.2, "end": 9.1, "scene_text": "part 1", "visual_intent": "part 1"},
            {"index": 2, "source_sentence_index": 2, "beat_index": 2, "word_start": 9, "word_end": 12, "start": 9.15, "end": 11.0, "scene_text": "part 2", "visual_intent": "part 2"},
        ],
        spans,
    )
    assert len(merged) == 1
    assert merged[0]["word_end"] == 12


def test_visual_beats_reject_gaps_or_overlaps(tmp_path: Path):
    video_dir = _make_video_dir(tmp_path, subtitle_ready=True)
    spans = visual_beats.load_sentence_spans(video_dir)
    with pytest.raises(ValueError, match="non-contiguous"):
        visual_beats.validate_visual_beats(
            [
                {"index": 1, "source_sentence_index": 2, "beat_index": 1, "word_start": 1, "word_end": 4, "start": 4.2, "end": 6.4, "scene_text": "a", "visual_intent": "a"},
                {"index": 2, "source_sentence_index": 2, "beat_index": 2, "word_start": 6, "word_end": 12, "start": 6.95, "end": 11.0, "scene_text": "b", "visual_intent": "b"},
            ],
            spans,
        )


def test_zero_width_word_alignment_falls_back_to_sentence_window(tmp_path: Path):
    video_dir = _make_video_dir(tmp_path, subtitle_ready=True)
    _write_json(
        video_dir / transcribe.WORD_TIMESTAMPS_NAME,
        [
            {"sentence_index": 1, "word_index": 1, "text": "A", "normalized": "a", "start": 0.0, "end": 0.0, "timing_source": "stable_ts"},
            {"sentence_index": 1, "word_index": 2, "text": "short", "normalized": "short", "start": 0.0, "end": 0.0, "timing_source": "stable_ts"},
            {"sentence_index": 1, "word_index": 3, "text": "clear", "normalized": "clear", "start": 0.0, "end": 0.0, "timing_source": "stable_ts"},
            {"sentence_index": 1, "word_index": 4, "text": "line.", "normalized": "line", "start": 0.0, "end": 0.0, "timing_source": "stable_ts"},
            {"sentence_index": 2, "word_index": 1, "text": "A", "normalized": "a", "start": 4.2, "end": 4.5, "timing_source": "stable_ts"},
            {"sentence_index": 2, "word_index": 2, "text": "longer", "normalized": "longer", "start": 4.55, "end": 5.2, "timing_source": "stable_ts"},
            {"sentence_index": 2, "word_index": 3, "text": "sentence", "normalized": "sentence", "start": 5.25, "end": 6.0, "timing_source": "stable_ts"},
            {"sentence_index": 2, "word_index": 4, "text": "with", "normalized": "with", "start": 6.05, "end": 6.4, "timing_source": "stable_ts"},
            {"sentence_index": 2, "word_index": 5, "text": "two", "normalized": "two", "start": 6.45, "end": 6.9, "timing_source": "stable_ts"},
            {"sentence_index": 2, "word_index": 6, "text": "ideas,", "normalized": "ideas", "start": 6.95, "end": 7.6, "timing_source": "stable_ts"},
            {"sentence_index": 2, "word_index": 7, "text": "then", "normalized": "then", "start": 8.5, "end": 8.9, "timing_source": "stable_ts"},
            {"sentence_index": 2, "word_index": 8, "text": "a", "normalized": "a", "start": 8.95, "end": 9.1, "timing_source": "stable_ts"},
            {"sentence_index": 2, "word_index": 9, "text": "reveal", "normalized": "reveal", "start": 9.15, "end": 10.0, "timing_source": "stable_ts"},
            {"sentence_index": 2, "word_index": 10, "text": "at", "normalized": "at", "start": 10.05, "end": 10.3, "timing_source": "stable_ts"},
            {"sentence_index": 2, "word_index": 11, "text": "the", "normalized": "the", "start": 10.35, "end": 10.55, "timing_source": "stable_ts"},
            {"sentence_index": 2, "word_index": 12, "text": "end.", "normalized": "end", "start": 10.6, "end": 11.0, "timing_source": "stable_ts"},
        ],
    )
    spans = visual_beats.load_sentence_spans(video_dir)
    words = visual_beats.load_exact_word_spans(video_dir)
    beats = visual_beats.derive_beat_timings(
        [
            {"source_sentence_index": 1, "beat_index": 1, "word_start": 1, "word_end": 4, "scene_text": "A short clear line.", "visual_intent": "single scene"},
            {"source_sentence_index": 2, "beat_index": 1, "word_start": 1, "word_end": 12, "scene_text": "A longer sentence with two ideas, then a reveal at the end.", "visual_intent": "single scene"},
        ],
        spans,
        words,
    )
    assert beats[0]["start"] == 0.0
    assert beats[0]["end"] == 4.2


def test_prompt_template_metadata_reads_real_template():
    meta = visual_beats.prompt_template_metadata("vi")
    assert meta["fields"]["model"] == "black-forest-labs/FLUX.1-dev"
    assert "fully clothed" in meta["text"]
