from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
from steps import creative_package as cp
from steps import image_prompts


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_package() -> dict:
    return {
        "package_version": "creative-package-v1",
        "language": "vi",
        "core_promise": "promise",
        "target_viewer": "viewer",
        "primary_hook": "hook",
        "title_options": [
            {"id": "title_1", "angle": "curiosity", "text": "Tieu de 1"},
            {"id": "title_2", "angle": "discovery", "text": "Tieu de 2"},
            {"id": "title_3", "angle": "emotion", "text": "Tieu de 3"},
        ],
        "description_draft": "Mo ta giu nguyen.",
        "search_keywords": ["lich su", "co dai"],
        "chapter_plan": [
            {"sentence_index": 1, "label": "Mo dau"},
            {"sentence_index": 3, "label": "Phat hien"},
            {"sentence_index": 5, "label": "Ket noi"},
        ],
        "thumbnail_concepts": [
            {
                "id": 1,
                "type": "human_closeup",
                "visual_hook": "face",
                "emotional_goal": "shock",
                "thumbnail_text": "Ai da lam",
                "subject_side": "left",
                "text_side": "right",
                "paired_title_ids": ["title_1"],
                "must_show": ["torch"],
                "must_avoid": ["blood"],
            },
            {
                "id": 2,
                "type": "human_closeup",
                "visual_hook": "eyes",
                "emotional_goal": "wonder",
                "thumbnail_text": "Ben trong hang",
                "subject_side": "right",
                "text_side": "left",
                "paired_title_ids": ["title_2"],
                "must_show": [],
                "must_avoid": [],
            },
            {
                "id": 3,
                "type": "mystery_reveal",
                "visual_hook": "painting",
                "emotional_goal": "mystery",
                "thumbnail_text": "Vet do nay",
                "subject_side": "left",
                "text_side": "right",
                "paired_title_ids": ["title_1"],
                "must_show": [],
                "must_avoid": [],
            },
            {
                "id": 4,
                "type": "mystery_reveal",
                "visual_hook": "wall",
                "emotional_goal": "discovery",
                "thumbnail_text": "Bi mat cu",
                "subject_side": "right",
                "text_side": "left",
                "paired_title_ids": ["title_2"],
                "must_show": [],
                "must_avoid": [],
            },
            {
                "id": 5,
                "type": "scale_or_danger",
                "visual_hook": "scale",
                "emotional_goal": "danger",
                "thumbnail_text": "Qua lon sao",
                "subject_side": "left",
                "text_side": "right",
                "paired_title_ids": ["title_3"],
                "must_show": [],
                "must_avoid": [],
            },
        ],
    }


def _setup_video_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PUBLISHING_DIRNAME", "publishing")
    video_dir = tmp_path / "demo"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "script.txt").write_text("Tieu de.\n\nDoan mot.\n\nDoan hai.\n\nDoan ba.\n\nDoan bon.", encoding="utf-8")
    _write_json(video_dir / "creative_package.json", _base_package())
    return video_dir


def test_creative_package_validation_and_script_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_dir = _setup_video_dir(tmp_path, monkeypatch)
    validated = cp.load_validated_package(video_dir)
    expected_hash = cp.compute_script_sha256(video_dir / "script.txt")
    assert validated["script_sha256"] == expected_hash
    validated_copy = json.loads((video_dir / "publishing" / "creative_package.validated.json").read_text(encoding="utf-8"))
    assert validated_copy["script_sha256"] == expected_hash


def test_stale_script_detection_blocks_without_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_dir = _setup_video_dir(tmp_path, monkeypatch)
    cp.load_validated_package(video_dir)
    (video_dir / "script.txt").write_text("Noi dung moi.\n\nDoan khac.", encoding="utf-8")
    with pytest.raises(cp.CreativePackageError):
        cp.load_validated_package(video_dir)
    validated = cp.load_validated_package(video_dir, allow_stale_package=True)
    assert validated["script_sha256"] == cp.compute_script_sha256(video_dir / "script.txt")


def test_creative_package_requires_valid_distribution_and_title_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_dir = _setup_video_dir(tmp_path, monkeypatch)
    package = _base_package()
    package["title_options"][1]["text"] = package["title_options"][0]["text"]
    package["thumbnail_concepts"][0]["paired_title_ids"] = ["title_9"]
    package["thumbnail_concepts"][0]["thumbnail_text"] = "mot hai ba bon nam"
    package["thumbnail_concepts"] = package["thumbnail_concepts"][:3]
    package["thumbnail_concepts"][1]["type"] = "human_closeup"
    _write_json(video_dir / "creative_package.json", package)
    with pytest.raises(cp.CreativePackageError):
        cp.load_validated_package(video_dir)


def test_script_prompt_requires_separate_creative_package_sections() -> None:
    prompt_text = Path("D:/CODE/VIDEO/YOUTUBE/prompts/script_prompt.txt").read_text(encoding="utf-8")
    assert "SCRIPT" in prompt_text
    assert "CREATIVE_PACKAGE_JSON" in prompt_text
    assert "script.txt" in prompt_text
    assert "creative_package.json" in prompt_text


def test_step4_preserves_scene_prompts_when_thumbnail_generation_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_dir = _setup_video_dir(tmp_path, monkeypatch)
    (video_dir / "audio.mp3").write_bytes(b"fake")
    _write_json(video_dir / "timestamps.json", [
        {"index": 1, "start": 0.0, "end": 1.0, "text": "Tieu de."},
        {"index": 2, "start": 1.0, "end": 2.0, "text": "Doan mot."},
        {"index": 3, "start": 2.0, "end": 3.0, "text": "Doan hai."},
        {"index": 4, "start": 3.0, "end": 4.0, "text": "Doan ba."},
        {"index": 5, "start": 4.0, "end": 5.0, "text": "Doan bon."},
    ])
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test")
    monkeypatch.setattr(image_prompts, "_get_audio_duration", lambda _: 5.0)
    monkeypatch.setattr(
        image_prompts,
        "_call_gemini",
        lambda script_text, sentences, system_prompt: [
            {"index": idx, "sentences": [idx], "scene_text": sentence, "prompt": f"scene {idx}", "icon_overlays": [], "text_overlays": []}
            for idx, sentence in enumerate(sentences, start=1)
        ],
    )

    def _fail_thumbnail(video_id: str, validated_package: dict, use_claude: bool) -> None:
        cp._atomic_write_json(
            video_dir / "publishing" / "thumbnail_prompt_diagnostics.json",
            {"validation_passed": False, "warnings": ["thumb failed"], "thumbnail_prompt_count": 0},
        )

    monkeypatch.setattr(image_prompts, "_generate_thumbnail_prompt_payload", _fail_thumbnail)
    image_prompts.run("demo")
    prompts = json.loads((video_dir / "image_prompts.json").read_text(encoding="utf-8"))
    assert len(prompts) == 5
    diagnostics = json.loads((video_dir / "publishing" / "thumbnail_prompt_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["validation_passed"] is False
