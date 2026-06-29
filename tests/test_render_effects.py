from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

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


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe unavailable")
def test_real_ffmpeg_render_composition_supports_pan_scene(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(render_video.config, "OUTPUT_DIR", str(tmp_path))
    video_dir = tmp_path / "video"
    images_dir = video_dir / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (1024, 1024), (180, 120, 70)).save(images_dir / "img_001.png")
    Image.new("RGB", (1024, 1024), (90, 120, 180)).save(images_dir / "img_002.png")
    _write_json(
        video_dir / "image_prompts.json",
        [
            {"index": 1, "start": 0.0, "end": 1.0, "prompt": "journey movement", "scene_text": "journey movement", "source_sentence_index": 1},
            {"index": 2, "start": 1.0, "end": 2.0, "prompt": "detail", "scene_text": "detail", "source_sentence_index": 2},
        ],
    )
    _write_json(
        video_dir / "effects_plan.json",
        {
            "version": "cinematic-documentary-v1",
            "global_look": {"grade": "warm_documentary", "grain": 0.0, "vignette": 0.0, "enabled": False},
            "effects_enabled": True,
            "scenes": [
                {
                    "scene_index": 1,
                    "source_sentence_index": 1,
                    "source_start": 0.0,
                    "source_end": 1.0,
                    "display_start": 0.0,
                    "display_end": 1.0,
                    "motion": {"type": "pan_left_to_right", "start_scale": 1.05, "end_scale": 1.05, "focus_x": 0.5, "focus_y": 0.45, "easing": "ease_in_out"},
                    "transition_out": {"type": "hard_cut", "duration": 0.0},
                },
                {
                    "scene_index": 2,
                    "source_sentence_index": 2,
                    "source_start": 1.0,
                    "source_end": 2.0,
                    "display_start": 1.0,
                    "display_end": 2.0,
                    "motion": {"type": "hold", "start_scale": 1.0, "end_scale": 1.0, "focus_x": 0.5, "focus_y": 0.45, "easing": "ease_in_out"},
                    "transition_out": {"type": "hard_cut", "duration": 0.0},
                },
            ],
        },
    )
    audio_path = video_dir / "audio.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-q:a", "2", str(audio_path)],
        check=True,
        capture_output=True,
    )

    output = render_video._render_composition("video", video_dir / "final.mp4")

    assert output.exists()
    assert render_video._probe_duration(output) > 0


def test_runtime_effects_disabled_overrides_stale_motion_plan(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "EFFECTS_ENABLED", False)
    monkeypatch.setattr(render_video.config, "EFFECTS_ENABLED", False)
    video_dir = tmp_path / "video"
    video_dir.mkdir(parents=True)
    prompts = [{"index": 1, "start": 0.0, "end": 1.0}]
    _write_json(
        video_dir / "effects_plan.json",
        {
            "version": "cinematic-documentary-v1",
            "global_look": {"grade": "warm_documentary", "grain": 0.2, "vignette": 0.2, "enabled": True},
            "effects_enabled": True,
            "scenes": [
                {
                    "scene_index": 1,
                    "source_sentence_index": 1,
                    "source_start": 0.0,
                    "source_end": 1.0,
                    "display_start": 0.0,
                    "display_end": 1.0,
                    "motion": {"type": "pan_left_to_right", "start_scale": 1.05, "end_scale": 1.05, "focus_x": 0.5, "focus_y": 0.45, "easing": "ease_in_out"},
                    "transition_out": {"type": "crossfade", "duration": 0.25},
                }
            ],
        },
    )

    plan = render_video._load_effects_plan(video_dir, prompts, 1.0)

    assert plan["effects_enabled"] is False
    assert plan["global_look"]["enabled"] is False
    assert plan["scenes"][0]["motion"]["type"] == "hold"
    assert plan["scenes"][0]["transition_out"]["type"] == "hard_cut"
