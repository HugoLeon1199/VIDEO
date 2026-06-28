from __future__ import annotations

import json
from pathlib import Path

import config
from steps import render_video


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_build_audio_mix_uses_lossless_wav(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")
    sfx_dir = tmp_path / "sfx"
    sfx_dir.mkdir()
    (sfx_dir / "birds.wav").write_bytes(b"wav")
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"wav")
        return type("Result", (), {"returncode": 0, "stderr": b"", "stdout": b""})()

    monkeypatch.setattr(render_video.subprocess, "run", fake_run)
    prompts = [{"index": 1, "start": 0.0, "end": 2.0}]
    soundscape = [{"scene_index": 1, "events": [{"tag": "birds", "offset": 0.0, "duration_mode": "scene", "volume": 0.2}]}]
    library = [{"tag": "birds", "file": "birds.wav", "duration_mode": "scene", "default_volume": 0.2}]

    out = render_video._mix_sfx_audio(audio_path, prompts, soundscape, sfx_dir, library, tmp_path / "audio_with_sfx.wav")

    assert out.suffix == ".wav"
    assert "-c:a" in captured["cmd"]
    assert "pcm_s16le" in captured["cmd"]


def test_preview_uses_same_renderer(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(render_video.config, "OUTPUT_DIR", str(tmp_path))
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    (video_dir / "audio.mp3").write_bytes(b"audio")
    called = []

    def fake_render(video_id, output_path, preview_window=None):
        called.append((video_id, output_path.name, preview_window))
        output_path.write_bytes(b"preview")
        return output_path

    monkeypatch.setattr(render_video, "_get_audio_duration", lambda _path: 12.0)
    monkeypatch.setattr(render_video, "_render_composition", fake_render)
    output = render_video.render_preview("video", seconds=4.0)

    assert output.name == "effects_preview.mp4"
    assert called == [("video", "effects_preview.mp4", (0.0, 4.0))]


def test_build_audio_filter_preview_trims_after_composition() -> None:
    script, label = render_video._build_audio_filter_script(3, preview_window=(12.0, 45.0))

    assert "[3:a]" in script
    assert "atrim=start=12.000:end=45.000" in script
    assert label == "aout"
