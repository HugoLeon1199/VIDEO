from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import config
from image_generation.schemas import CandidateResult, SceneResult
from steps import generate_images, thumbnails


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class _FakeBackend:
    def __init__(self, candidate_path: Path):
        self.calls: list[str] = []
        self.candidate_path = candidate_path

    def generate(self, request):
        self.calls.append(request.scene_id)
        return SceneResult(
            video_id=request.video_id,
            scene_id=request.scene_id,
            model="fake",
            mode="fake",
            duration_seconds=0.1,
            candidates=[
                CandidateResult(
                    candidate_index=0,
                    seed=11,
                    width=1024,
                    height=576,
                    sha256="sha",
                    generation_seconds=0.1,
                    mime_type="image/png",
                    local_path=str(self.candidate_path),
                )
            ],
        )


def test_step5_reuses_one_vast_backend_for_scenes_and_thumbnails(tmp_path: Path, monkeypatch) -> None:
    import image_generation.runpod_serverless_backend as runpod_backend

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(generate_images.config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(thumbnails.config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(generate_images.config, "IMAGE_BACKEND", "vast_instance")
    monkeypatch.setattr(config, "IMAGE_BACKEND", "vast_instance")
    monkeypatch.setattr(runpod_backend, "OUTPUT_ROOT", str(tmp_path))

    video_dir = tmp_path / "demo"
    video_dir.mkdir(parents=True, exist_ok=True)
    _write_json(video_dir / "image_prompts.json", [{"index": 1, "prompt": "full prompt", "clip_prompt": "short prompt", "negative_prompt": ""}])
    _write_json(video_dir / "generation_log.json", {})
    _write_json(video_dir / "publishing" / "thumbnail_prompts.json", [
        {
            "concept_id": 1,
            "type": "human_closeup",
            "clip_prompt": "YouTube thumbnail, human closeup, face",
            "image_prompt": "no text no logo no watermark",
            "negative_prompt": "",
            "thumbnail_text": "WHO WAS IT",
            "subject_side": "left",
            "text_side": "right",
            "paired_title_ids": ["title_1"],
        }
    ])
    _write_json(
        video_dir / "creative_package.json",
        {
            "package_version": "creative-package-v1",
            "language": "en",
            "core_promise": "promise",
            "target_viewer": "viewer",
            "primary_hook": "hook",
            "title_options": [
                {"id": "title_1", "angle": "curiosity", "text": "First Title"},
                {"id": "title_2", "angle": "discovery", "text": "Second Title"},
                {"id": "title_3", "angle": "emotion", "text": "Third Title"},
            ],
            "description_draft": "Description stays.",
            "search_keywords": ["ancient", "history"],
            "chapter_plan": [{"sentence_index": 1, "label": "Intro"}],
            "thumbnail_concepts": [
                {"id": 1, "type": "human_closeup", "visual_hook": "face", "emotional_goal": "shock", "thumbnail_text": "WHO WAS IT", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_1"], "must_show": [], "must_avoid": []},
                {"id": 2, "type": "mystery_reveal", "visual_hook": "cave", "emotional_goal": "mystery", "thumbnail_text": "INSIDE THE CAVE", "subject_side": "right", "text_side": "left", "paired_title_ids": ["title_2"], "must_show": [], "must_avoid": []},
                {"id": 3, "type": "scale_or_danger", "visual_hook": "danger", "emotional_goal": "fear", "thumbnail_text": "TOO BIG", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_3"], "must_show": [], "must_avoid": []},
            ],
        },
    )
    (video_dir / "script.txt").write_text("Sentence one.", encoding="utf-8")
    (video_dir / "audio.mp3").write_bytes(b"audio")

    candidate_path = tmp_path / "candidate.png"
    Image.new("RGB", (1024, 576), (120, 80, 40)).save(candidate_path)
    backend = _FakeBackend(candidate_path)
    teardown_called: list[bool] = []
    thumbnail_background_backends: list[object] = []
    thumbnail_finalize_calls: list[bool] = []

    def fake_build_backend():
        return backend, lambda: teardown_called.append(True)

    def fake_generate_thumbnail_backgrounds(video_id: str, **kwargs):
        thumbnail_background_backends.append(kwargs.get("backend_override"))
        return {"background_generated_count": 1, "thumbnail_failed_ids": []}

    def fake_generate_thumbnail_assets(video_id: str, **kwargs):
        thumbnail_finalize_calls.append(True)
        return {
            "thumbnail_prompt_count": 1,
            "thumbnail_generated_count": 1,
            "thumbnail_failed_ids": [],
            "validation_passed": True,
        }

    monkeypatch.setattr(generate_images, "_build_vast_backend", fake_build_backend)
    monkeypatch.setattr(thumbnails, "generate_thumbnail_backgrounds", fake_generate_thumbnail_backgrounds)
    monkeypatch.setattr(thumbnails, "generate_thumbnail_assets", fake_generate_thumbnail_assets)

    generate_images.run("demo")

    assert backend.calls == ["001"]
    assert thumbnail_background_backends == [backend]
    assert thumbnail_finalize_calls == [True]
    assert teardown_called == [True]
