from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import config
from steps import autopilot


@pytest.fixture(autouse=True)
def _patch_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(autopilot.config, "OUTPUT_DIR", str(tmp_path))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _stub_pipeline(monkeypatch, *, fail_stage: str | None = None):
    from steps import design_effects, design_soundscape, generate_images, image_prompts, metadata, render_video, thumbnails, transcribe, tts
    from image_generation import production

    def _write_tts(video_id: str):
        video_dir = Path(config.OUTPUT_DIR) / video_id
        (video_dir / "audio.mp3").write_bytes(b"audio")

    def _write_transcribe(video_id: str):
        video_dir = Path(config.OUTPUT_DIR) / video_id
        (video_dir / "timestamps.json").write_text(
            json.dumps([{"index": 1, "start": 0.0, "end": 4.0, "text": "Sentence one."}], indent=2),
            encoding="utf-8",
        )
        (video_dir / "word_timestamps_diagnostics.json").write_text(
            json.dumps({"subtitle_ready": True, "alignment_coverage": 1.0}, indent=2),
            encoding="utf-8",
        )

    def _write_prompts(video_id: str):
        video_dir = Path(config.OUTPUT_DIR) / video_id
        (video_dir / "image_prompts.json").write_text(
            json.dumps([{"index": 1, "start": 0.0, "end": 4.0, "prompt": "scene prompt"}], indent=2),
            encoding="utf-8",
        )
        publishing = video_dir / config.PUBLISHING_DIRNAME
        publishing.mkdir(parents=True, exist_ok=True)
        (publishing / "thumbnail_prompts.json").write_text(
            json.dumps(
                [
                    {"concept_id": 1, "type": "human_closeup", "image_prompt": "one", "thumbnail_text": "ONE", "text_side": "right"},
                    {"concept_id": 2, "type": "mystery_reveal", "image_prompt": "two", "thumbnail_text": "TWO", "text_side": "left"},
                    {"concept_id": 3, "type": "scale_or_danger", "image_prompt": "three", "thumbnail_text": "THREE", "text_side": "right"},
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_images(video_id: str, n_override=None, **_kwargs):
        video_dir = Path(config.OUTPUT_DIR) / video_id / "images"
        video_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (120, 80, 40)).save(video_dir / "img_001.png")

    def _write_thumbnails(video_id: str, **_kwargs):
        publishing = Path(config.OUTPUT_DIR) / video_id / config.PUBLISHING_DIRNAME
        thumbs_dir = publishing / "thumbnails"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        for concept_id in (1, 2, 3):
            Image.new("RGB", (320, 180), (concept_id * 30, 70, 60)).save(thumbs_dir / f"thumbnail_{concept_id:02d}_background.png")
            Image.new("RGB", (320, 180), (concept_id * 30, 70, 60)).save(thumbs_dir / f"thumbnail_{concept_id:02d}.jpg")
        Image.new("RGB", (640, 360), (20, 20, 20)).save(publishing / "thumbnail_contact_sheet.jpg")
        return {
            "thumbnail_prompt_count": 3,
            "thumbnail_generated_count": 3,
            "thumbnail_failed_ids": [],
            "validation_passed": True,
            "expected_thumbnail_count": 3,
            "contact_sheet_path": str(publishing / "thumbnail_contact_sheet.jpg"),
        }

    def _write_soundscape(video_id: str):
        video_dir = Path(config.OUTPUT_DIR) / video_id
        (video_dir / "soundscape.json").write_text("[]", encoding="utf-8")

    def _write_effects(video_id: str):
        video_dir = Path(config.OUTPUT_DIR) / video_id
        (video_dir / "effects_plan.json").write_text(
            json.dumps(
                {
                    "version": "cinematic-documentary-v1",
                    "global_look": {"grade": "warm_documentary", "grain": 0.0, "vignette": 0.0, "enabled": False},
                    "effects_enabled": False,
                    "scenes": [
                        {
                            "scene_index": 1,
                            "source_sentence_index": 1,
                            "source_start": 0.0,
                            "source_end": 4.0,
                            "display_start": 0.0,
                            "display_end": 4.0,
                            "motion": {"type": "hold", "start_scale": 1.0, "end_scale": 1.0, "focus_x": 0.5, "focus_y": 0.45, "easing": "ease_in_out"},
                            "transition_out": {"type": "hard_cut", "duration": 0.0},
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (video_dir / "effects_diagnostics.json").write_text(
            json.dumps({"scene_count": 1, "motion_distribution": {"hold": 1}, "transition_distribution": {}, "max_scale": 1.0, "max_pan": 0.0, "duration_drift": 0.0, "validation_passed": True, "warnings": []}, indent=2),
            encoding="utf-8",
        )

    def _write_render(video_id: str, subtitles: bool = False):
        video_dir = Path(config.OUTPUT_DIR) / video_id
        (video_dir / "final.mp4").write_bytes(b"final")
        if subtitles:
            (video_dir / "final_subbed.mp4").write_bytes(b"subbed")

    def _write_metadata(video_id: str, allow_stale_package: bool = False):
        if fail_stage == "publishing":
            raise RuntimeError("publishing failed")
        publishing = Path(config.OUTPUT_DIR) / video_id / config.PUBLISHING_DIRNAME
        publishing.mkdir(parents=True, exist_ok=True)
        (publishing / "title_options.txt").write_text("Title\n", encoding="utf-8")
        (publishing / "description.txt").write_text("Description\n", encoding="utf-8")
        (publishing / "chapters.txt").write_text("00:00 Start\n", encoding="utf-8")

    monkeypatch.setattr(tts, "run", _write_tts)
    monkeypatch.setattr(transcribe, "run", _write_transcribe)
    monkeypatch.setattr(image_prompts, "run", _write_prompts)
    monkeypatch.setattr(generate_images, "run", _write_images)
    monkeypatch.setattr(thumbnails, "generate_thumbnail_backgrounds", lambda *args, **kwargs: {"background_generated_count": 0, "thumbnail_failed_ids": []})
    monkeypatch.setattr(thumbnails, "generate_thumbnail_assets", _write_thumbnails)
    monkeypatch.setattr(design_soundscape, "run", _write_soundscape)
    monkeypatch.setattr(design_effects, "run", _write_effects)
    monkeypatch.setattr(render_video, "run", _write_render)
    monkeypatch.setattr(metadata, "run", _write_metadata)
    monkeypatch.setattr(
        production.VastSession,
        "__enter__",
        lambda self: setattr(self, "backend", object()) or self,
    )
    monkeypatch.setattr(production.VastSession, "__exit__", lambda self, exc_type, exc, tb: False)


def test_safe_video_id_and_script_normalization():
    assert autopilot.safe_video_id("Hello Ancient World!\n\nBody") == "hello-ancient-world"
    assert "—" not in autopilot.normalize_script_text("Một — hai")


def test_autopilot_generates_vi_configs_and_summary(tmp_path, monkeypatch):
    _stub_pipeline(monkeypatch)
    script_path = tmp_path / "input.txt"
    script_path.write_text("Bạn đã thấy điều này chưa?\n\nĐây là một câu đầy đủ.", encoding="utf-8")

    summary = autopilot.run("video-vi", str(script_path))
    video_dir = tmp_path / "video-vi"

    assert _read_json(video_dir / "tts_config.json")["voice"] == "Thái Sơn"
    assert _read_json(video_dir / "transcribe_config.json")["language"] == "vi"
    assert summary["tts_engine"] == "vieneu"
    assert (video_dir / "autopilot_summary.json").exists()


def test_autopilot_generates_en_configs_and_manual_package_bypass(tmp_path, monkeypatch):
    _stub_pipeline(monkeypatch)
    script_path = tmp_path / "input.txt"
    script_path.write_text("Did you ever wonder what happened?\n\nThis is a full sentence.", encoding="utf-8")
    video_dir = tmp_path / "video-en"
    video_dir.mkdir()
    (video_dir / "creative_package.json").write_text(
        json.dumps(
            {
                "package_version": "creative-package-v1",
                "language": "en",
                "core_promise": "promise",
                "target_viewer": "viewer",
                "primary_hook": "hook",
                "title_options": [
                    {"id": "title_1", "angle": "curiosity", "text": "Title One"},
                    {"id": "title_2", "angle": "discovery", "text": "Title Two"},
                    {"id": "title_3", "angle": "emotion", "text": "Title Three"},
                ],
                "description_draft": "Description draft",
                "search_keywords": ["ancient", "history"],
                "chapter_plan": [{"sentence_index": 1, "label": "Intro"}],
                "thumbnail_concepts": [
                    {"id": 1, "type": "human_closeup", "visual_hook": "face", "emotional_goal": "shock", "thumbnail_text": "WHO DID THIS", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_1"], "must_show": [], "must_avoid": []},
                    {"id": 2, "type": "mystery_reveal", "visual_hook": "cave", "emotional_goal": "mystery", "thumbnail_text": "INSIDE CAVE", "subject_side": "right", "text_side": "left", "paired_title_ids": ["title_2"], "must_show": [], "must_avoid": []},
                    {"id": 3, "type": "scale_or_danger", "visual_hook": "danger", "emotional_goal": "danger", "thumbnail_text": "TOO BIG", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_3"], "must_show": [], "must_avoid": []},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = autopilot.run("video-en", str(script_path))
    assert _read_json(video_dir / "tts_config.json")["voice"] == "am_fenrir"
    assert summary["tts_engine"] == "kokoro"


def test_resume_rejects_stale_hash(tmp_path, monkeypatch):
    _stub_pipeline(monkeypatch)
    script_path = tmp_path / "input.txt"
    script_path.write_text("Hello world.\n\nNext sentence.", encoding="utf-8")
    autopilot.run("video", str(script_path))
    script_path.write_text("Changed script.\n\nNext sentence.", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash changed"):
        autopilot.run("video", str(script_path), resume=True)


def test_downstream_failure_preserves_previous_artifacts(tmp_path, monkeypatch):
    _stub_pipeline(monkeypatch, fail_stage="publishing")
    script_path = tmp_path / "input.txt"
    script_path.write_text("Hello world.\n\nNext sentence.", encoding="utf-8")
    with pytest.raises(RuntimeError, match="publishing failed"):
        autopilot.run("video", str(script_path))
    video_dir = tmp_path / "video"
    assert (video_dir / "final.mp4").exists()
    state = _read_json(video_dir / "autopilot_state.json")
    assert state["stages"]["publishing"]["status"] == "failed"


def test_render_failure_marks_autopilot_stage_failed_with_original_error(tmp_path, monkeypatch):
    _stub_pipeline(monkeypatch)
    from steps import render_video

    script_path = tmp_path / "input.txt"
    script_path.write_text("Hello world.\n\nNext sentence.", encoding="utf-8")

    def _fail_render(video_id: str, subtitles: bool = False):
        raise render_video.RenderPipelineError("ffmpeg broke")

    monkeypatch.setattr(render_video, "run", _fail_render)

    with pytest.raises(render_video.RenderPipelineError, match="ffmpeg broke"):
        autopilot.run("video", str(script_path))

    state = _read_json(tmp_path / "video" / "autopilot_state.json")
    assert state["stages"]["render"]["status"] == "failed"
    assert state["stages"]["render"]["error"] == "ffmpeg broke"
