from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
from image_generation.flux_prompting import normalize_prompt_text
from steps import image_prompts


@pytest.fixture(autouse=True)
def _patch_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(image_prompts.config, "OUTPUT_DIR", str(tmp_path))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _setup_video(video_dir: Path) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "script.txt").write_text("Một câu ngắn.\n\nMột câu dài hơn, nhưng vẫn rõ ràng.", encoding="utf-8")
    (video_dir / "audio.mp3").write_bytes(b"audio")
    _write_json(
        video_dir / "timestamps.json",
        [
            {"index": 1, "start": 0.0, "end": 2.0, "text": "Một câu ngắn."},
            {"index": 2, "start": 2.0, "end": 6.0, "text": "Một câu dài hơn, nhưng vẫn rõ ràng."},
        ],
    )


def test_scene_prompts_emit_clip_and_t5_diagnostics(tmp_path: Path, monkeypatch) -> None:
    video_dir = tmp_path / "demo"
    _setup_video(video_dir)
    monkeypatch.setattr(image_prompts, "_get_audio_duration", lambda _path: 6.0)
    monkeypatch.setattr(
        image_prompts,
        "token_count_for_model",
        lambda _model_id, _text, subfolder, _revision=None: 21 if subfolder == "tokenizer" else 84,
    )

    image_prompts.run("demo")

    prompts = json.loads((video_dir / "image_prompts.json").read_text(encoding="utf-8"))
    first = prompts[0]
    assert first["clip_prompt"].startswith("cinematic")
    assert first["clip_token_count"] == 21
    assert first["clip_limit"] == config.FLUX_CLIP_TOKEN_LIMIT
    assert first["t5_token_count"] == 84
    assert first["unicode_valid"] is True
    assert first["template_version"]
    assert first["scene_text"] == "Một câu ngắn."


def test_clip_prompt_unicode_normalization_and_rejection() -> None:
    assert normalize_prompt_text("Cafe\u0301") == "Café"
    with pytest.raises(ValueError, match="U\\+FFFD"):
        normalize_prompt_text("bad\ufffdtext")


def test_clip_prompt_is_condensed_for_long_scene_text() -> None:
    long_text = " ".join(f"word{i}" for i in range(1, 60))
    clip_prompt = image_prompts._build_clip_prompt("vi", long_text)
    assert "word24" in clip_prompt
    assert "word25" not in clip_prompt


def test_scene_prompt_generation_fails_when_clip_prompt_overflows(tmp_path: Path, monkeypatch) -> None:
    video_dir = tmp_path / "demo"
    _setup_video(video_dir)
    monkeypatch.setattr(image_prompts, "_get_audio_duration", lambda _path: 6.0)
    monkeypatch.setattr(
        image_prompts,
        "token_count_for_model",
        lambda _model_id, _text, subfolder, _revision=None: config.FLUX_CLIP_TOKEN_LIMIT + 1 if subfolder == "tokenizer" else 84,
    )

    with pytest.raises(SystemExit):
        image_prompts.run("demo")

    assert not (video_dir / "image_prompts.json").exists()


def test_scene_prompt_generation_falls_back_when_tokenizer_access_is_gated(tmp_path: Path, monkeypatch) -> None:
    video_dir = tmp_path / "demo"
    _setup_video(video_dir)
    monkeypatch.setattr(image_prompts, "_get_audio_duration", lambda _path: 6.0)

    def _raise_gated(_model_id, _text, _subfolder, _revision=None):
        raise RuntimeError("401 Client Error: gated repo")

    monkeypatch.setattr(image_prompts, "token_count_for_model", _raise_gated)

    image_prompts.run("demo")

    prompts = json.loads((video_dir / "image_prompts.json").read_text(encoding="utf-8"))
    first = prompts[0]
    assert first["clip_token_count"] > 0
    assert first["t5_token_count"] > 0
    assert first["clip_token_count_mode"] == "heuristic"
    assert first["t5_token_count_mode"] == "heuristic"
