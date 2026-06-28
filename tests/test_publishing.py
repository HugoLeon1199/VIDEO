from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
from steps import creative_package as cp
from steps import metadata


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _creative_package() -> dict:
    return {
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
        "description_draft": "This description is reused directly.",
        "search_keywords": ["ancient", "cave", "history"],
        "chapter_plan": [
            {"sentence_index": 1, "label": "Hook"},
            {"sentence_index": 2, "label": "Evidence"},
            {"sentence_index": 4, "label": "Meaning"},
        ],
        "thumbnail_concepts": [
            {"id": 1, "type": "human_closeup", "visual_hook": "face", "emotional_goal": "shock", "thumbnail_text": "WHO DID THIS", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_1"], "must_show": [], "must_avoid": []},
            {"id": 2, "type": "mystery_reveal", "visual_hook": "cave", "emotional_goal": "mystery", "thumbnail_text": "INSIDE THE CAVE", "subject_side": "right", "text_side": "left", "paired_title_ids": ["title_2"], "must_show": [], "must_avoid": []},
            {"id": 3, "type": "scale_or_danger", "visual_hook": "danger", "emotional_goal": "fear", "thumbnail_text": "TOO BIG", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_3"], "must_show": [], "must_avoid": []},
        ],
    }


def _setup(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PUBLISHING_DIRNAME", "publishing")
    video_dir = tmp_path / "pub"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "script.txt").write_text("Title line.\n\nOne sentence.\n\nSecond sentence.\n\nThird sentence.\n\nFourth sentence.", encoding="utf-8")
    _write_json(video_dir / "creative_package.json", _creative_package())
    _write_json(video_dir / "timestamps.json", [
        {"index": 1, "start": 0.0, "end": 4.0, "text": "Title line."},
        {"index": 2, "start": 12.0, "end": 18.0, "text": "One sentence."},
        {"index": 3, "start": 28.0, "end": 32.0, "text": "Second sentence."},
        {"index": 4, "start": 45.0, "end": 52.0, "text": "Third sentence."},
        {"index": 5, "start": 66.0, "end": 72.0, "text": "Fourth sentence."},
    ])
    _write_json(video_dir / "publishing" / "thumbnail_prompts.json", [])
    cp.load_validated_package(video_dir)
    return video_dir


def test_publishing_reuses_description_and_maps_chapters_from_timestamps(tmp_path: Path, monkeypatch) -> None:
    video_dir = _setup(tmp_path, monkeypatch)
    metadata.run("pub")
    description = (video_dir / "publishing" / "description.txt").read_text(encoding="utf-8").strip()
    chapters = (video_dir / "publishing" / "chapters.txt").read_text(encoding="utf-8").strip().splitlines()
    package = json.loads((video_dir / "publishing" / "package.json").read_text(encoding="utf-8"))
    assert description == "This description is reused directly."
    assert chapters[0].startswith("00:00")
    assert any("00:45" in line for line in chapters)
    assert package["search_keywords"] == ["ancient", "cave", "history"]


def test_step8_blocks_stale_creative_package_without_override(tmp_path: Path, monkeypatch) -> None:
    video_dir = _setup(tmp_path, monkeypatch)
    (video_dir / "script.txt").write_text("Changed.\n\nBody new.", encoding="utf-8")
    with pytest.raises(SystemExit):
        metadata.run("pub")


def test_legacy_metadata_fallback_used_when_creative_package_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    video_dir = tmp_path / "legacy"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "script.txt").write_text("Legacy script.", encoding="utf-8")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(
        metadata,
        "_generate_legacy_metadata",
        lambda script: {"title": "Legacy Title", "description": "Legacy", "tags": ["a"], "chapters": ["0:00 Start"]},
    )
    metadata.run("legacy")
    payload = json.loads((video_dir / "metadata.json").read_text(encoding="utf-8"))
    assert payload["title"] == "Legacy Title"
