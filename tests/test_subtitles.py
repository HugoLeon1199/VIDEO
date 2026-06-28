from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
from steps import render_video
from steps import subtitles
from steps import transcribe


@pytest.fixture(autouse=True)
def _patch_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(subtitles.config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(transcribe.config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(render_video.config, "OUTPUT_DIR", str(tmp_path))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_script(video_dir: Path, text: str) -> None:
    (video_dir / "script.txt").write_text(text, encoding="utf-8")


def _make_word_assets(video_dir: Path, words: list[dict], ready: bool = True, reason: str = "") -> None:
    _write_json(
        video_dir / transcribe.WORD_DIAGNOSTICS_NAME,
        {
            "subtitle_ready": ready,
            "reason": reason,
            "affected_blocks": [2] if not ready else [],
            "alignment_coverage": 1.0 if ready else 0.82,
            "timing_source": "stable_ts",
            "word_count": len(words) if ready else 0,
            "block_diagnostics": [],
        },
    )
    if ready:
        _write_json(video_dir / transcribe.WORD_TIMESTAMPS_NAME, words)


def _sample_words() -> list[dict]:
    raw = [
        ("Một", "mot", 0.10, 0.40, 1, 1),
        ("con", "con", 0.45, 0.65, 1, 2),
        ("số", "so", 0.70, 0.95, 1, 3),
        ("như", "nhu", 1.00, 1.20, 1, 4),
        ("31.000,", "31000", 1.25, 1.60, 1, 5),
        ("bạn", "ban", 1.90, 2.20, 1, 6),
        ("có", "co", 2.25, 2.45, 1, 7),
        ("thấy", "thay", 2.50, 2.85, 1, 8),
        ("không?", "khong", 2.90, 3.30, 1, 9),
    ]
    return [
        {
            "sentence_index": sentence_index,
            "word_index": word_index,
            "text": text,
            "normalized": normalized,
            "start": start,
            "end": end,
            "timing_source": "stable_ts",
        }
        for text, normalized, start, end, sentence_index, word_index in raw
    ]


def test_transcribe_step_preserves_timestamps_and_writes_word_diagnostics(tmp_path, monkeypatch):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    _make_script(video_dir, "Hello world.")
    (video_dir / "audio.mp3").write_bytes(b"fake")

    expected = [{"index": 1, "start": 0.0, "end": 1.0, "text": "Hello world."}]
    monkeypatch.setattr(transcribe, "_should_use_block_mode", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(transcribe, "_load_blocks_manifest", lambda *_args, **_kwargs: {"mode": "block", "blocks": [1]})

    def fake_block(*_args, **_kwargs):
        transcribe._set_subtitle_export(
            transcribe._subtitle_success(
                [
                    {
                        "sentence_index": 1,
                        "word_index": 1,
                        "text": "Hello",
                        "normalized": "hello",
                        "start": 0.0,
                        "end": 0.4,
                        "timing_source": "stable_ts",
                    },
                    {
                        "sentence_index": 1,
                        "word_index": 2,
                        "text": "world.",
                        "normalized": "world",
                        "start": 0.5,
                        "end": 1.0,
                        "timing_source": "stable_ts",
                    },
                ],
                timing_source="stable_ts",
                alignment_coverage=1.0,
            )
        )
        return expected

    (video_dir / "transcribe_config.json").write_text('{"engine":"stable_ts"}', encoding="utf-8")
    monkeypatch.setattr(transcribe, "_run_stable_ts_blocks", fake_block)

    transcribe.run("video")

    assert json.loads((video_dir / "timestamps.json").read_text(encoding="utf-8")) == expected
    diagnostics = json.loads((video_dir / transcribe.WORD_DIAGNOSTICS_NAME).read_text(encoding="utf-8"))
    assert diagnostics["subtitle_ready"] is True


def test_block_fallback_marks_subtitle_not_ready(tmp_path):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    _make_script(video_dir, "Một câu. Hai câu.")
    _make_word_assets(video_dir, [], ready=False, reason="sentence_fallback_without_exact_word_timestamps")

    with pytest.raises(RuntimeError, match="Subtitles are not ready"):
        subtitles.generate("video")


def test_generate_subtitle_cues_keeps_canonical_word_order_and_unicode(tmp_path):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    _make_script(video_dir, "Một con số như 31.000, bạn có thấy không?")
    _make_word_assets(video_dir, _sample_words(), ready=True)
    (video_dir / "audio_master.wav").write_bytes(b"RIFFfake")

    from steps import subtitles as subtitle_step

    original_audio_duration = subtitle_step._audio_duration
    subtitle_step._audio_duration = lambda _path: 3.5
    try:
        result = subtitle_step.generate("video", style=subtitle_step.STYLE_CINEMATIC_CLEAN)
    finally:
        subtitle_step._audio_duration = original_audio_duration

    cues = json.loads((video_dir / "subtitle_cues.json").read_text(encoding="utf-8"))
    reconstructed = " ".join(cue["text"].replace("\n", " ") for cue in cues).split()
    assert reconstructed == [item["text"] for item in _sample_words()]
    assert result["diagnostics"]["validation_passed"] is True
    assert "Một" in (video_dir / "subtitles.srt").read_text(encoding="utf-8")


def test_ass_escape_and_two_line_wrap(tmp_path):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    text = "AI {DNA} NASA YouTube Nguyễn Trãi Điện Biên Phủ."
    _make_script(video_dir, text)
    words = []
    for idx, token in enumerate(subtitles._canonical_script_words(video_dir / "script.txt"), start=1):
        words.append(
            {
                "sentence_index": 1,
                "word_index": idx,
                "text": token["text"],
                "normalized": token["normalized"],
                "start": 0.2 * idx,
                "end": 0.2 * idx + 0.15,
                "timing_source": "stable_ts",
            }
        )
    _make_word_assets(video_dir, words, ready=True)
    (video_dir / "audio_master.wav").write_bytes(b"RIFFfake")

    monkeypatch_audio = subtitles._audio_duration
    subtitles._audio_duration = lambda _path: 4.0
    try:
        subtitles.generate("video", style=subtitles.STYLE_CINEMATIC_ACCENT)
    finally:
        subtitles._audio_duration = monkeypatch_audio

    ass_text = (video_dir / "subtitles.ass").read_text(encoding="utf-8")
    assert subtitles._escape_ass("{DNA}") == r"\{DNA\}"
    assert r"\{" in ass_text
    cues = json.loads((video_dir / "subtitle_cues.json").read_text(encoding="utf-8"))
    assert max(cue["line_count"] for cue in cues) <= 2


def test_preview_timestamps_rebased_near_zero():
    cues = [
        {"index": 1, "start": 12.0, "end": 13.4, "text": "Một cue", "line_count": 1, "word_start": 1, "word_end": 2, "_tokens": ["Một", "cue"]},
        {"index": 2, "start": 13.8, "end": 15.2, "text": "Dài hơn một chút", "line_count": 1, "word_start": 3, "word_end": 6, "_tokens": ["Dài", "hơn", "một", "chút"]},
    ]
    rebased = subtitles._rebase_preview_cues(cues, 12.0, 20.0)
    assert rebased[0]["start"] == 0.0
    assert rebased[1]["start"] < 2.0


def test_windows_ass_path_is_safe(tmp_path):
    input_video = tmp_path / "final.mp4"
    ass_path = tmp_path / "nested path" / "subtitles.ass"
    ass_path.parent.mkdir(parents=True)
    input_video.write_bytes(b"x")
    ass_path.write_text("ass", encoding="utf-8")
    working_dir = tmp_path / "work"
    working_dir.mkdir()

    cmd = subtitles.build_burn_command(input_video, tmp_path / "final_subbed.mp4", ass_path, working_dir)

    assert "subtitles=subtitles.ass" in cmd
    assert str(ass_path) not in " ".join(cmd)


def test_failed_validation_leaves_no_partial_files(tmp_path):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    _make_script(video_dir, "Một con số như 31.000, bạn có thấy không?")
    _make_word_assets(video_dir, _sample_words(), ready=True)
    (video_dir / "audio_master.wav").write_bytes(b"RIFFfake")
    stale = video_dir / "subtitle_cues.json"
    stale.write_text('{"old": true}', encoding="utf-8")

    original_builder = subtitles.build_subtitle_cues
    subtitles.build_subtitle_cues = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken"))
    original_audio_duration = subtitles._audio_duration
    subtitles._audio_duration = lambda _path: 3.5
    try:
        with pytest.raises(RuntimeError, match="broken"):
            subtitles.generate("video")
    finally:
        subtitles.build_subtitle_cues = original_builder
        subtitles._audio_duration = original_audio_duration

    assert stale.read_text(encoding="utf-8") == '{"old": true}'


def test_timestamps_json_regression_shape_unchanged(tmp_path, monkeypatch):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    _make_script(video_dir, "Hello world.")
    (video_dir / "audio.mp3").write_bytes(b"x")
    monkeypatch.setattr(transcribe, "_should_use_block_mode", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        transcribe,
        "_run_faster_whisper",
        lambda *_args, **_kwargs: [{"index": 1, "start": 0.0, "end": 1.0, "text": "Hello world."}],
    )

    transcribe.run("video")

    assert (video_dir / "timestamps.json").read_text(encoding="utf-8") == json.dumps(
        [{"index": 1, "start": 0.0, "end": 1.0, "text": "Hello world."}],
        ensure_ascii=False,
        indent=2,
    )


def test_render_step_with_subtitles_creates_final_subbed_without_overwriting_final(tmp_path, monkeypatch):
    video_id = "video"
    video_dir = tmp_path / video_id
    images_dir = video_dir / "images"
    images_dir.mkdir(parents=True)
    _write_json(video_dir / "image_prompts.json", [{"index": 1, "start": 0.0, "end": 1.0, "prompt": "x"}])
    (video_dir / "audio.mp3").write_bytes(b"audio")
    (images_dir / "img_001.png").write_bytes(b"png")

    monkeypatch.setattr(render_video.config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(render_video, "_check_ffmpeg", lambda: None)
    monkeypatch.setattr(render_video, "_get_audio_duration", lambda _path: 1.0)

    def fake_run(cmd, *args, **kwargs):
        target = Path(cmd[-1])
        if target.suffix == ".mp4":
            target.write_bytes(b"video-bytes")
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(render_video.subprocess, "run", fake_run)
    monkeypatch.setattr(subtitles, "generate", lambda _video_id, style: {"ass_path": video_dir / "subtitles.ass"})
    monkeypatch.setattr(subtitles, "burn_subtitles", lambda _in, _ass, out: out.write_bytes(b"subbed"))

    render_video.run(video_id, subtitles=True)

    assert (video_dir / "final.mp4").read_bytes() == b"video-bytes"
    assert (video_dir / "final_subbed.mp4").read_bytes() == b"subbed"


def test_zero_duration_word_timings_remain_exact_and_ready():
    sentences = ["Xin chào."]
    aligned_words = [
        {"word": " Xin", "normalized": "xin", "start": 0.10, "end": 0.32},
        {"word": " chào.", "normalized": "chao", "start": 0.32, "end": 0.32},
    ]

    word_timestamps, coverage, exact_words = transcribe._build_exact_word_timestamps(
        sentences,
        aligned_words,
        audio_start=0.0,
        starting_sentence_index=1,
        starting_global_word_index=1,
        timing_source="stable_ts",
    )

    assert coverage == 1.0
    assert exact_words is True
    assert word_timestamps == [
        {
            "sentence_index": 1,
            "word_index": 1,
            "text": "Xin",
            "normalized": "xin",
            "start": 0.1,
            "end": 0.32,
            "timing_source": "stable_ts",
        },
        {
            "sentence_index": 1,
            "word_index": 2,
            "text": "chào.",
            "normalized": "chao",
            "start": 0.32,
            "end": 0.32,
            "timing_source": "stable_ts",
        },
    ]


def test_zero_duration_cue_uses_sentence_bounds_fallback(tmp_path):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    script = "Xin chào. Cụm từ này cần né plateau."
    _make_script(video_dir, script)
    _write_json(
        video_dir / "timestamps.json",
        [
            {"index": 1, "start": 0.0, "end": 0.9, "text": "Xin chào."},
            {"index": 2, "start": 1.0, "end": 3.5, "text": "Cụm từ này cần né plateau."},
        ],
    )
    word_timestamps = [
        {"sentence_index": 1, "word_index": 1, "text": "Xin", "normalized": "xin", "start": 0.0, "end": 0.3, "timing_source": "stable_ts"},
        {"sentence_index": 1, "word_index": 2, "text": "chào.", "normalized": "chao", "start": 0.3, "end": 0.9, "timing_source": "stable_ts"},
        {"sentence_index": 2, "word_index": 1, "text": "Cụm", "normalized": "cum", "start": 1.0, "end": 1.0, "timing_source": "stable_ts"},
        {"sentence_index": 2, "word_index": 2, "text": "từ", "normalized": "tu", "start": 1.0, "end": 1.0, "timing_source": "stable_ts"},
        {"sentence_index": 2, "word_index": 3, "text": "này", "normalized": "nay", "start": 1.0, "end": 1.0, "timing_source": "stable_ts"},
        {"sentence_index": 2, "word_index": 4, "text": "cần", "normalized": "can", "start": 1.0, "end": 1.0, "timing_source": "stable_ts"},
        {"sentence_index": 2, "word_index": 5, "text": "né", "normalized": "ne", "start": 1.0, "end": 1.0, "timing_source": "stable_ts"},
        {"sentence_index": 2, "word_index": 6, "text": "plateau.", "normalized": "plateau", "start": 1.0, "end": 1.0, "timing_source": "stable_ts"},
    ]

    cues = subtitles.build_subtitle_cues(video_dir / "script.txt", word_timestamps, 3.5, sentence_timestamps=json.loads((video_dir / "timestamps.json").read_text(encoding="utf-8")))

    assert cues[1]["start"] == 1.0
    assert cues[1]["end"] == 3.5
