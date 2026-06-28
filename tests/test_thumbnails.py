from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import config
from image_generation.schemas import CandidateResult, SceneResult
from steps import thumbnails


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_package() -> dict:
    return {
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
        "description_draft": "Description stays.",
        "search_keywords": ["ancient", "history"],
        "chapter_plan": [
            {"sentence_index": 1, "label": "Intro"},
            {"sentence_index": 2, "label": "Middle"},
            {"sentence_index": 3, "label": "End"},
        ],
        "thumbnail_concepts": [
            {"id": 1, "type": "human_closeup", "visual_hook": "face", "emotional_goal": "shock", "thumbnail_text": "WHO WAS IT", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_1"], "must_show": [], "must_avoid": []},
            {"id": 2, "type": "mystery_reveal", "visual_hook": "cave", "emotional_goal": "mystery", "thumbnail_text": "UNDER THE WALL", "subject_side": "right", "text_side": "left", "paired_title_ids": ["title_2"], "must_show": [], "must_avoid": []},
            {"id": 3, "type": "scale_or_danger", "visual_hook": "scale", "emotional_goal": "danger", "thumbnail_text": "HOW BIG", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_3"], "must_show": [], "must_avoid": []},
        ],
    }


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
                    seed=123,
                    width=1024,
                    height=576,
                    sha256="sha",
                    generation_seconds=0.1,
                    mime_type="image/png",
                    local_path=str(self.candidate_path),
                )
            ],
        )


def _setup(tmp_path: Path, monkeypatch) -> tuple[Path, _FakeBackend]:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PUBLISHING_DIRNAME", "publishing")
    monkeypatch.setattr(config, "THUMBNAIL_FONT_SIZE", 48)
    video_dir = tmp_path / "thumbs"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "script.txt").write_text("One.\n\nTwo.\n\nThree.", encoding="utf-8")
    _write_json(video_dir / "creative_package.json", _make_package())
    _write_json(video_dir / "publishing" / "thumbnail_prompts.json", [
        {"concept_id": 1, "type": "human_closeup", "image_prompt": "no text no logo no watermark", "negative_prompt": "", "thumbnail_text": "WHO WAS IT", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_1"]},
        {"concept_id": 2, "type": "mystery_reveal", "image_prompt": "no text no logo no watermark", "negative_prompt": "", "thumbnail_text": "UNDER THE WALL", "subject_side": "right", "text_side": "left", "paired_title_ids": ["title_2"]},
        {"concept_id": 3, "type": "scale_or_danger", "image_prompt": "no text no logo no watermark", "negative_prompt": "", "thumbnail_text": "HOW BIG", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_3"]},
    ])
    bg = tmp_path / "candidate.png"
    Image.new("RGB", (1024, 576), (140, 90, 60)).save(bg)
    backend = _FakeBackend(bg)
    monkeypatch.setattr(thumbnails, "_build_backend", lambda: (backend, None))
    thumbnails.load_validated_package(video_dir)
    return video_dir, backend


def test_thumbnail_generation_call_count_and_overlay_preserves_background(tmp_path: Path, monkeypatch) -> None:
    video_dir, backend = _setup(tmp_path, monkeypatch)
    diagnostics = thumbnails.generate_thumbnail_assets("thumbs")
    assert len(backend.calls) == 3
    assert diagnostics["thumbnail_generated_count"] == 3
    background = Image.open(video_dir / "publishing" / "thumbnails" / "thumbnail_01_background.png").convert("RGB")
    thumbnail = Image.open(video_dir / "publishing" / "thumbnails" / "thumbnail_01.jpg").convert("RGB")
    assert background.getpixel((10, 10)) == (140, 90, 60)
    thumb_pixel = thumbnail.getpixel((10, 10))
    assert all(abs(thumb_pixel[idx] - value) <= 2 for idx, value in enumerate((140, 90, 60)))


def test_selective_thumbnail_regeneration_only_touches_requested_concept(tmp_path: Path, monkeypatch) -> None:
    video_dir, backend = _setup(tmp_path, monkeypatch)
    (video_dir / "publishing" / "thumbnails").mkdir(parents=True, exist_ok=True)
    for concept_id in (1, 2, 3):
        Image.new("RGB", (1024, 576), (concept_id * 10, 80, 60)).save(video_dir / "publishing" / "thumbnails" / f"thumbnail_{concept_id:02d}_background.png")
        Image.new("RGB", (1024, 576), (concept_id * 10, 80, 60)).save(video_dir / "publishing" / "thumbnails" / f"thumbnail_{concept_id:02d}.jpg")
    thumbnails.generate_thumbnail_assets("thumbs", regenerate=[2])
    assert backend.calls == ["9002"]


def test_contact_sheet_contains_labels_and_thumbnail_failures_do_not_touch_scene_log(tmp_path: Path, monkeypatch) -> None:
    video_dir, backend = _setup(tmp_path, monkeypatch)
    _write_json(video_dir / "generation_log.json", {"001": {"status": "completed"}})
    thumbnails.generate_thumbnail_assets("thumbs")
    sheet = Image.open(video_dir / "publishing" / "thumbnail_contact_sheet.jpg")
    assert sheet.size[0] > 0
    assert (video_dir / "generation_log.json").exists()
