from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import config
from image_generation import production
from steps import autopilot


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _patch_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(autopilot.config, "OUTPUT_DIR", str(tmp_path))


def test_vast_session_verifies_destroy_before_confirmation(monkeypatch):
    lifecycle = production.VastLifecycle()
    session = production.VastSession(lifecycle=lifecycle)
    session.managed = True
    session.teardown = lambda: None
    session.owned_instance_id = 123
    monkeypatch.setattr(session, "verify_destroyed", lambda: False)

    with pytest.raises(RuntimeError, match="could not be destroy-verified"):
        session.__exit__(None, None, None)

    assert lifecycle.teardown_attempt_count == 1
    assert lifecycle.teardown_verified_count == 0
    assert lifecycle.vast_teardown_confirmed is False


def test_autopilot_summary_uses_real_lifecycle_counters(tmp_path, monkeypatch):
    from steps import design_effects, design_soundscape, generate_images, image_prompts, metadata, render_video, thumbnails, transcribe, tts

    script_path = tmp_path / "input.txt"
    script_path.write_text("Hello world.\n\nNext sentence.", encoding="utf-8")

    def _write_tts(video_id: str):
        (tmp_path / video_id / "audio.mp3").write_bytes(b"audio")

    def _write_transcribe(video_id: str):
        video_dir = tmp_path / video_id
        (video_dir / "timestamps.json").write_text(json.dumps([{"index": 1, "start": 0.0, "end": 4.0, "text": "Hello world."}], indent=2), encoding="utf-8")
        (video_dir / "word_timestamps_diagnostics.json").write_text(json.dumps({"subtitle_ready": True}, indent=2), encoding="utf-8")

    def _write_prompts(video_id: str, **_kwargs):
        video_dir = tmp_path / video_id
        (video_dir / "image_prompts.json").write_text(json.dumps([{"index": 1, "start": 0.0, "end": 4.0, "prompt": "scene"}], indent=2), encoding="utf-8")
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

    def _write_images(video_id: str, **_kwargs):
        img_dir = tmp_path / video_id / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (110, 80, 50)).save(img_dir / "img_001.png")

    def _write_thumbs(video_id: str, **_kwargs):
        publishing = tmp_path / video_id / config.PUBLISHING_DIRNAME
        thumbs_dir = publishing / "thumbnails"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        for concept_id in (1, 2, 3):
            Image.new("RGB", (320, 180), (concept_id * 20, 70, 50)).save(thumbs_dir / f"thumbnail_{concept_id:02d}_background.png")
            Image.new("RGB", (320, 180), (concept_id * 20, 70, 50)).save(thumbs_dir / f"thumbnail_{concept_id:02d}.jpg")
        Image.new("RGB", (640, 360), (10, 10, 10)).save(publishing / "thumbnail_contact_sheet.jpg")
        return {
            "thumbnail_generated_count": 3,
            "thumbnail_failed_ids": [],
            "validation_passed": True,
            "expected_thumbnail_count": 3,
            "contact_sheet_path": str(publishing / "thumbnail_contact_sheet.jpg"),
        }

    def _write_sound(video_id: str):
        (tmp_path / video_id / "soundscape.json").write_text("[]", encoding="utf-8")

    def _write_effects(video_id: str):
        video_dir = tmp_path / video_id
        (video_dir / "effects_plan.json").write_text(json.dumps({"version": "cinematic-documentary-v1", "global_look": {"enabled": False}, "effects_enabled": False, "scenes": [{"scene_index": 1, "source_sentence_index": 1, "source_start": 0.0, "source_end": 4.0, "display_start": 0.0, "display_end": 4.0, "motion": {"type": "hold", "start_scale": 1.0, "end_scale": 1.0, "focus_x": 0.5, "focus_y": 0.45, "easing": "ease_in_out"}, "transition_out": {"type": "hard_cut", "duration": 0.0}}]}, indent=2), encoding="utf-8")
        (video_dir / "effects_diagnostics.json").write_text(json.dumps({"validation_passed": True}, indent=2), encoding="utf-8")

    def _write_render(video_id: str, subtitles: bool = False):
        video_dir = tmp_path / video_id
        (video_dir / "final.mp4").write_bytes(b"final")
        if subtitles:
            (video_dir / "final_subbed.mp4").write_bytes(b"subbed")

    def _write_metadata(video_id: str, **_kwargs):
        publishing = tmp_path / video_id / config.PUBLISHING_DIRNAME
        publishing.mkdir(parents=True, exist_ok=True)
        (publishing / "title_options.txt").write_text("Title\n", encoding="utf-8")
        (publishing / "description.txt").write_text("Description\n", encoding="utf-8")

    monkeypatch.setattr(tts, "run", _write_tts)
    monkeypatch.setattr(transcribe, "run", _write_transcribe)
    monkeypatch.setattr(image_prompts, "run", _write_prompts)
    monkeypatch.setattr(generate_images, "run", _write_images)
    monkeypatch.setattr(thumbnails, "generate_thumbnail_backgrounds", lambda *args, **kwargs: {"background_generated_count": 0, "thumbnail_failed_ids": []})
    monkeypatch.setattr(thumbnails, "generate_thumbnail_assets", _write_thumbs)
    monkeypatch.setattr(design_soundscape, "run", _write_sound)
    monkeypatch.setattr(design_effects, "run", _write_effects)
    monkeypatch.setattr(render_video, "run", _write_render)
    monkeypatch.setattr(metadata, "run", _write_metadata)

    def fake_enter(self):
        self.lifecycle.vast_session_count += 1
        self.lifecycle.backend_create_count += 1
        self.lifecycle.rent_count += 1
        self.lifecycle.worker_boot_id = "boot-1"
        self.lifecycle.worker_ready_count += 1
        self.lifecycle.model_load_count += 1
        self.backend = object()
        self.managed = True
        self.owned_instance_id = 123
        self.teardown = lambda: None
        return self

    monkeypatch.setattr(production.VastSession, "__enter__", fake_enter)
    monkeypatch.setattr(
        production.VastSession,
        "__exit__",
        lambda self, exc_type, exc, tb: setattr(self.lifecycle, "teardown_attempt_count", self.lifecycle.teardown_attempt_count + 1) or setattr(self.lifecycle, "teardown_verified_count", self.lifecycle.teardown_verified_count + 1) or setattr(self.lifecycle, "vast_teardown_confirmed", True) or False,
    )

    summary = autopilot.run("video", str(script_path))

    assert summary["vast_session_count"] == 1
    assert summary["rent_count"] == 1
    assert summary["backend_create_count"] == 1
    assert summary["worker_boot_id"] == "boot-1"
    assert summary["worker_ready_count"] == 1
    assert summary["model_load_count"] == 1
    assert summary["teardown_attempt_count"] == 1
    assert summary["teardown_verified_count"] == 1
    assert summary["vast_teardown_confirmed"] is True


def test_autopilot_reuses_same_backend_for_scenes_and_thumbnail_backgrounds(tmp_path, monkeypatch):
    from steps import design_effects, design_soundscape, generate_images, image_prompts, metadata, render_video, thumbnails, transcribe, tts

    script_path = tmp_path / "input.txt"
    script_path.write_text("Hello world.\n\nNext sentence.", encoding="utf-8")
    seen_backends = []

    def _write_tts(video_id: str):
        (tmp_path / video_id / "audio.mp3").write_bytes(b"audio")

    def _write_transcribe(video_id: str):
        video_dir = tmp_path / video_id
        (video_dir / "timestamps.json").write_text(json.dumps([{"index": 1, "start": 0.0, "end": 4.0, "text": "Hello world."}], indent=2), encoding="utf-8")
        (video_dir / "word_timestamps_diagnostics.json").write_text(json.dumps({"subtitle_ready": True}, indent=2), encoding="utf-8")

    def _write_prompts(video_id: str, **_kwargs):
        video_dir = tmp_path / video_id
        (video_dir / "image_prompts.json").write_text(json.dumps([{"index": 1, "start": 0.0, "end": 4.0, "prompt": "scene"}], indent=2), encoding="utf-8")
        publishing = video_dir / config.PUBLISHING_DIRNAME
        publishing.mkdir(parents=True, exist_ok=True)
        (publishing / "thumbnail_prompts.json").write_text(json.dumps([{"concept_id": 1, "type": "human_closeup", "image_prompt": "one", "thumbnail_text": "ONE", "text_side": "right"}, {"concept_id": 2, "type": "mystery_reveal", "image_prompt": "two", "thumbnail_text": "TWO", "text_side": "left"}, {"concept_id": 3, "type": "scale_or_danger", "image_prompt": "three", "thumbnail_text": "THREE", "text_side": "right"}], indent=2), encoding="utf-8")

    def _write_images(video_id: str, **kwargs):
        seen_backends.append(kwargs["backend_override"])
        img_dir = tmp_path / video_id / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (100, 70, 50)).save(img_dir / "img_001.png")

    def _write_thumb_backgrounds(video_id: str, **kwargs):
        seen_backends.append(kwargs["backend_override"])
        publishing = tmp_path / video_id / config.PUBLISHING_DIRNAME / "thumbnails"
        publishing.mkdir(parents=True, exist_ok=True)
        for concept_id in (1, 2, 3):
            Image.new("RGB", (320, 180), (concept_id * 20, 60, 40)).save(publishing / f"thumbnail_{concept_id:02d}_background.png")
        return {"background_generated_count": 3, "thumbnail_failed_ids": []}

    def _write_thumbs(video_id: str, **_kwargs):
        publishing = tmp_path / video_id / config.PUBLISHING_DIRNAME
        thumbs_dir = publishing / "thumbnails"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        for concept_id in (1, 2, 3):
            Image.new("RGB", (320, 180), (concept_id * 20, 70, 50)).save(thumbs_dir / f"thumbnail_{concept_id:02d}.jpg")
        Image.new("RGB", (640, 360), (15, 15, 15)).save(publishing / "thumbnail_contact_sheet.jpg")
        return {"thumbnail_generated_count": 3, "thumbnail_failed_ids": [], "validation_passed": True, "expected_thumbnail_count": 3, "contact_sheet_path": str(publishing / "thumbnail_contact_sheet.jpg")}

    def _write_sound(video_id: str):
        (tmp_path / video_id / "soundscape.json").write_text("[]", encoding="utf-8")

    def _write_effects(video_id: str):
        video_dir = tmp_path / video_id
        (video_dir / "effects_plan.json").write_text(json.dumps({"version": "cinematic-documentary-v1", "global_look": {"enabled": False}, "effects_enabled": False, "scenes": [{"scene_index": 1, "source_sentence_index": 1, "source_start": 0.0, "source_end": 4.0, "display_start": 0.0, "display_end": 4.0, "motion": {"type": "hold", "start_scale": 1.0, "end_scale": 1.0, "focus_x": 0.5, "focus_y": 0.45, "easing": "ease_in_out"}, "transition_out": {"type": "hard_cut", "duration": 0.0}}]}, indent=2), encoding="utf-8")
        (video_dir / "effects_diagnostics.json").write_text(json.dumps({"validation_passed": True}, indent=2), encoding="utf-8")

    def _write_render(video_id: str, subtitles: bool = False):
        video_dir = tmp_path / video_id
        (video_dir / "final.mp4").write_bytes(b"final")
        if subtitles:
            (video_dir / "final_subbed.mp4").write_bytes(b"subbed")

    def _write_metadata(video_id: str, **_kwargs):
        publishing = tmp_path / video_id / config.PUBLISHING_DIRNAME
        publishing.mkdir(parents=True, exist_ok=True)
        (publishing / "title_options.txt").write_text("Title\n", encoding="utf-8")
        (publishing / "description.txt").write_text("Description\n", encoding="utf-8")

    monkeypatch.setattr(tts, "run", _write_tts)
    monkeypatch.setattr(transcribe, "run", _write_transcribe)
    monkeypatch.setattr(image_prompts, "run", _write_prompts)
    monkeypatch.setattr(generate_images, "run", _write_images)
    monkeypatch.setattr(thumbnails, "generate_thumbnail_backgrounds", _write_thumb_backgrounds)
    monkeypatch.setattr(thumbnails, "generate_thumbnail_assets", _write_thumbs)
    monkeypatch.setattr(design_soundscape, "run", _write_sound)
    monkeypatch.setattr(design_effects, "run", _write_effects)
    monkeypatch.setattr(render_video, "run", _write_render)
    monkeypatch.setattr(metadata, "run", _write_metadata)

    def fake_enter(self):
        self.lifecycle.vast_session_count += 1
        self.lifecycle.backend_create_count += 1
        self.lifecycle.rent_count += 1
        self.backend = object()
        self.managed = True
        self.owned_instance_id = 55
        self.teardown = lambda: None
        return self

    def fake_exit(self, exc_type, exc, tb):
        self.lifecycle.teardown_attempt_count += 1
        self.lifecycle.teardown_verified_count += 1
        self.lifecycle.vast_teardown_confirmed = True
        return False

    monkeypatch.setattr(production.VastSession, "__enter__", fake_enter)
    monkeypatch.setattr(production.VastSession, "__exit__", fake_exit)

    summary = autopilot.run("video", str(script_path))

    assert len(seen_backends) == 2
    assert seen_backends[0] is seen_backends[1]
    assert summary["rent_count"] == 1
    assert summary["teardown_attempt_count"] == 1
    assert summary["teardown_verified_count"] == 1
