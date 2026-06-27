import json
from pathlib import Path

import numpy as np

from steps import tts
from steps import transcribe
from steps.text_units import load_sentence_units, split_sentence_units, strip_script_metadata


class StubVieNeuRuntime:
    def __init__(self):
        self.block_config = {
            "block_target_max_seconds": 16.0,
            "block_hard_max_seconds": 20.0,
            "block_soft_max_normalized_chars": 240,
            "block_hard_max_normalized_chars": 280,
            "initial_chars_per_second": 14.0,
        }

    def normalized_chars(self, text: str) -> int:
        return len(text)

    def estimate_seconds(self, text: str) -> float:
        return len(text) / self.block_config["initial_chars_per_second"]


class StubKokoroRuntime:
    def __init__(self, voice: str = "voice", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self.sample_rate = 24000
        self.package_version = "test-kokoro"

    def phoneme_chars(self, text: str) -> int:
        return len(text)

    def estimate_seconds(self, text: str) -> float:
        return len(text.split()) / 2.6

    def synthesize(self, text: str):
        return np.ones(2400, dtype=np.float32), self.sample_rate, {"voice": self.voice, "speed": self.speed}


def test_strip_script_metadata():
    raw = "Line one.\n\nCOMMENT SEED: ignore me\nRESEARCH NOTES:\n- hidden"
    assert strip_script_metadata(raw) == "Line one."


def test_split_sentence_units_preserves_paragraph_index():
    script = "Title\n\nFirst sentence. Second sentence.\n\nThird sentence?"
    units = split_sentence_units(script)
    assert [unit.sentence_index for unit in units] == [1, 2, 3]
    assert [unit.paragraph_index for unit in units] == [1, 1, 2]
    assert units[0].text == "First sentence."


def test_vieneu_block_builder_respects_soft_limit():
    sentence_units = split_sentence_units(
        "A" * 120 + ". " + "B" * 120 + ". " + "C" * 120 + "."
    )
    blocks = tts._build_vieneu_blocks(sentence_units, StubVieNeuRuntime())
    assert len(blocks) == 3
    assert blocks[0].sentence_indices == [1]
    assert blocks[1].sentence_indices == [2]


def test_kokoro_block_builder_uses_phoneme_soft_limit():
    sentence_units = split_sentence_units(
        ("alpha " * 20).strip() + ". " + ("beta " * 20).strip() + ". " + ("gamma " * 20).strip() + "."
    )
    runtime = StubKokoroRuntime()
    cfg = {
        "block_soft_max_phoneme_chars": 120,
        "block_hard_max_phoneme_chars": 500,
    }
    blocks = tts._build_kokoro_blocks(sentence_units, runtime, cfg)
    assert len(blocks) >= 2
    assert [idx for block in blocks for idx in block.sentence_indices] == [1, 2, 3]


def test_kokoro_candidate_overflow_starts_new_block():
    sentence_units = split_sentence_units("A" * 280 + ". " + "B" * 280 + ".")
    runtime = StubKokoroRuntime()
    cfg = {
        "block_soft_max_phoneme_chars": 420,
        "block_hard_max_phoneme_chars": 500,
    }
    blocks = tts._build_kokoro_blocks(sentence_units, runtime, cfg)
    assert len(blocks) == 2
    assert blocks[0].sentence_indices == [1]
    assert blocks[1].sentence_indices == [2]


def test_load_sentence_units_from_fixture():
    script_path = Path("output") / "to-tien-ban-lam-gi-ca-ngay-vi" / "script.txt"
    units = load_sentence_units(script_path)
    assert units
    assert units[0].sentence_index == 1


def test_materialize_fallback_rebuilds_following_offsets(tmp_path, monkeypatch):
    video_dir = tmp_path / "video"
    tts_blocks = video_dir / "tts_blocks"
    tts_blocks.mkdir(parents=True)
    (video_dir / "tts_config.json").write_text(
        json.dumps({"engine": "kokoro", "voice": "per-video", "speed": 1.1}),
        encoding="utf-8",
    )
    old_block_1 = tts_blocks / "block_001.wav"
    old_block_2 = tts_blocks / "block_002.wav"
    tts._write_wav(np.ones(24000, dtype=np.float32), 24000, old_block_1)
    tts._write_wav(np.ones(24000, dtype=np.float32), 24000, old_block_2)
    manifest = {
        "engine": "kokoro",
        "mode": "block",
        "voice": "per-video",
        "speed": 1.1,
        "sentence_count": 2,
        "block_count": 2,
        "blocks": [
            {
                "block_index": 1,
                "sentence_indices": [1],
                "sentence_texts": ["One."],
                "text": "One.",
                "raw_chars": 4,
                "estimated_seconds": 1.0,
                "phoneme_chars": 4,
                "wav_path": "tts_blocks/block_001.wav",
                "sample_rate": 24000,
                "gap_after_ms": 300,
                "fallback_level": 0,
                "audio_start": 0.0,
                "audio_end": 1.0,
            },
            {
                "block_index": 2,
                "sentence_indices": [2],
                "sentence_texts": ["Two."],
                "text": "Two.",
                "raw_chars": 4,
                "estimated_seconds": 1.0,
                "phoneme_chars": 4,
                "wav_path": "tts_blocks/block_002.wav",
                "sample_rate": 24000,
                "gap_after_ms": 0,
                "fallback_level": 0,
                "audio_start": 1.3,
                "audio_end": 2.3,
            },
        ],
    }
    (tts_blocks / "blocks.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tts_blocks / "diagnostics.json").write_text(json.dumps({"fallback_block_count": 0}), encoding="utf-8")

    def fake_render(video_dir_arg, block_index, block, engine, runtime, block_config, infer_retry_params=None):
        assert engine == "kokoro"
        audio = np.ones(48000, dtype=np.float32)
        wav_path = video_dir_arg / "tts_blocks" / f"block_{block_index:03d}.wav"
        tts._write_wav(audio, 24000, wav_path)
        return {
            "wav_path": str(wav_path.relative_to(video_dir_arg)).replace("\\", "/"),
            "sample_rate": 24000,
            "actual_seconds": 2.0,
            "fallback_level": 2,
            "fallback_segments": [
                {
                    "sentence_index": 1,
                    "text": "One.",
                    "wav_path": "tts_blocks/block_001_sentence_001.wav",
                    "actual_seconds": 2.0,
                    "start_in_block": 0.0,
                    "end_in_block": 2.0,
                    "gap_after_ms": 0,
                    "infer_params": {"voice": "per-video", "speed": 1.1},
                }
            ],
        }

    monkeypatch.setattr(tts, "_render_sentence_fallback_block", fake_render)
    patched = tts.materialize_sentence_fallback_for_block(video_dir, 1)
    updated_manifest = json.loads((tts_blocks / "blocks.json").read_text(encoding="utf-8"))
    assert patched["fallback_level"] == 2
    assert updated_manifest["blocks"][1]["audio_start"] == 2.3


def test_transcribe_run_uses_block_mode_twice(tmp_path, monkeypatch):
    video_dir = tmp_path / "video"
    tts_blocks = video_dir / "tts_blocks"
    tts_blocks.mkdir(parents=True)
    (video_dir / "audio.mp3").write_bytes(b"x")
    (video_dir / "audio_master.wav").write_bytes(b"x")
    (video_dir / "script.txt").write_text("Hello world.", encoding="utf-8")
    (video_dir / "transcribe_config.json").write_text(
        json.dumps({"engine": "stable_ts", "model": "medium", "language": "vi", "mode": "align", "device": "cpu"}),
        encoding="utf-8",
    )
    (tts_blocks / "blocks.json").write_text(
        json.dumps(
            {
                "mode": "block",
                "engine": "vieneu",
                "sentence_count": 1,
                "blocks": [{"block_index": 1, "sentence_texts": ["Hello world."]}],
            }
        ),
        encoding="utf-8",
    )

    calls = {"block": 0, "full": 0}

    def fake_block(video_dir_arg, model_name, language, device):
        calls["block"] += 1
        return [{"index": 1, "start": 0.0, "end": 1.0, "text": "Hello world."}]

    def fake_full(*args, **kwargs):
        calls["full"] += 1
        return [{"index": 1, "start": 0.0, "end": 1.0, "text": "Hello world."}]

    monkeypatch.setattr(transcribe, "_run_stable_ts_blocks", fake_block)
    monkeypatch.setattr(transcribe, "_run_stable_ts_full", fake_full)
    monkeypatch.setattr(transcribe.config, "OUTPUT_DIR", str(tmp_path))

    transcribe.run("video")
    transcribe.run("video")

    assert calls["block"] == 2
    assert calls["full"] == 0


def test_sentence_legacy_vieneu_does_not_use_kokoro(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "KokoroRuntime", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("kokoro should not be used")))

    class FakeVieNeuRuntime:
        def __init__(self, voice, block_config, infer_overrides=None, speed=None):
            self.block_config = {"trim_trailing_threshold": 0.01, "trim_trailing_keep_ms": 120}

        def synthesize(self, sentence, infer_params=None):
            return np.ones(2400, dtype=np.float32), 24000, {}

    monkeypatch.setattr(tts, "VieNeuRuntime", FakeVieNeuRuntime)
    out_mp3 = tmp_path / "audio.mp3"
    out_ts = tmp_path / "timestamps.json"
    tts._run_sentence_legacy_mode(["One.", "Two."], out_mp3, out_ts, "vieneu", "Voice", "-5%")
    assert out_mp3.exists()
    assert out_ts.exists()


def test_kokoro_runtime_uses_per_video_voice_speed(tmp_path, monkeypatch):
    script_path = tmp_path / "script.txt"
    video_dir = tmp_path
    script_path.write_text("One. Two.", encoding="utf-8")

    captured = {}

    class FakeKokoroRuntime(StubKokoroRuntime):
        def __init__(self, voice=None, speed=None):
            super().__init__(voice=voice, speed=speed)
            captured["voice"] = voice
            captured["speed"] = speed
            self.package_version = "test-kokoro"

    monkeypatch.setattr(tts, "KokoroRuntime", FakeKokoroRuntime)
    tts._run_block_mode(
        video_dir,
        script_path,
        "kokoro",
        "voice-from-config",
        {"voice": "voice-from-config", "speed": 1.23, "block_config": {"block_soft_max_phoneme_chars": 999, "block_hard_max_phoneme_chars": 999}},
    )
    diagnostics = json.loads((video_dir / "tts_blocks" / "diagnostics.json").read_text(encoding="utf-8"))
    assert captured == {"voice": "voice-from-config", "speed": 1.23}
    assert diagnostics["voice"] == "voice-from-config"
    assert diagnostics["speed"] == 1.23


def test_block_mode_rerun_reuses_all_then_regenerates_one_block(tmp_path, monkeypatch):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    script_path = video_dir / "script.txt"
    script_path.write_text("Alpha one. Beta two. Gamma three.", encoding="utf-8")

    class FakeVieNeuRuntime:
        package_version = "test-vieneu"

        def __init__(self, voice, block_config, infer_overrides=None, speed=None):
            self.voice = voice
            self.speed = speed
            self.block_config = tts._merge_dict(tts.VIENEU_BLOCK_DEFAULTS, block_config)
            self.sample_rate = 48000
            self.package_version = "test-vieneu"
            self.infer_params = tts._merge_dict(tts.VIENEU_INFER_DEFAULTS, infer_overrides)
            self.infer_params["voice"] = voice

        def normalized_chars(self, text):
            return len(text)

        def estimate_seconds(self, text):
            return len(text) / 14.0

        def synthesize(self, text, infer_params=None):
            return np.ones(4800 + len(text), dtype=np.float32), self.sample_rate, dict(self.infer_params)

    monkeypatch.setattr(tts, "VieNeuRuntime", FakeVieNeuRuntime)

    base_cfg = {"voice": "Voice A", "block_config": {"block_soft_max_normalized_chars": 20, "block_hard_max_normalized_chars": 999}}
    tts._run_block_mode(video_dir, script_path, "vieneu", "Voice A", base_cfg)
    first_diag = json.loads((video_dir / "tts_blocks" / "diagnostics.json").read_text(encoding="utf-8"))
    assert first_diag["regenerated_block_count"] > 0

    tts._run_block_mode(video_dir, script_path, "vieneu", "Voice A", base_cfg)
    second_diag = json.loads((video_dir / "tts_blocks" / "diagnostics.json").read_text(encoding="utf-8"))
    assert second_diag["reused_block_count"] == second_diag["block_count"]
    assert second_diag["regenerated_block_count"] == 0

    script_path.write_text("Alpha one. Beta too. Gamma three.", encoding="utf-8")
    tts._run_block_mode(video_dir, script_path, "vieneu", "Voice A", base_cfg)
    third_diag = json.loads((video_dir / "tts_blocks" / "diagnostics.json").read_text(encoding="utf-8"))
    assert third_diag["regenerated_block_count"] == 1
