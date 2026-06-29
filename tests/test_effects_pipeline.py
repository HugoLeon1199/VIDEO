from __future__ import annotations

import json
from pathlib import Path

import config
from steps import design_effects, render_video
from steps.creative_package import CreativePackageError


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
            {"id": "title_1", "angle": "curiosity", "text": "One"},
            {"id": "title_2", "angle": "discovery", "text": "Two"},
            {"id": "title_3", "angle": "emotion", "text": "Three"},
        ],
        "description_draft": "desc",
        "search_keywords": ["ancient"],
        "chapter_plan": [
            {"sentence_index": 3, "label": "Chapter 2"},
        ],
        "thumbnail_concepts": [
            {"id": 1, "type": "human_closeup", "visual_hook": "face", "emotional_goal": "shock", "thumbnail_text": "WHO WAS IT", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_1"], "must_show": [], "must_avoid": []},
            {"id": 2, "type": "mystery_reveal", "visual_hook": "cave", "emotional_goal": "mystery", "thumbnail_text": "IN THE CAVE", "subject_side": "right", "text_side": "left", "paired_title_ids": ["title_2"], "must_show": [], "must_avoid": []},
            {"id": 3, "type": "scale_or_danger", "visual_hook": "size", "emotional_goal": "danger", "thumbnail_text": "TOO BIG", "subject_side": "left", "text_side": "right", "paired_title_ids": ["title_3"], "must_show": [], "must_avoid": []},
        ],
    }


def test_effects_plan_uses_display_timing_and_final_hold(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(design_effects.config, "OUTPUT_DIR", str(tmp_path))
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    _write_json(
        video_dir / "image_prompts.json",
        [
            {"index": 1, "start": 1.5, "end": 3.0, "scene_text": "face", "source_sentence_index": 1},
            {"index": 2, "start": 4.0, "end": 6.0, "scene_text": "landscape", "source_sentence_index": 2},
            {"index": 3, "start": 8.0, "end": 9.0, "scene_text": "detail", "source_sentence_index": 3},
        ],
    )
    _write_json(video_dir / "creative_package.json", _creative_package())
    monkeypatch.setattr(design_effects, "_audio_duration", lambda _video_dir: 12.0)

    plan, diagnostics = design_effects.build_effects_plan("video")

    assert plan["scenes"][0]["display_start"] == 0.0
    assert plan["scenes"][0]["display_end"] == 4.0
    assert plan["scenes"][1]["display_end"] == 8.0
    assert plan["scenes"][2]["display_end"] == 12.0
    assert diagnostics["scene_count"] == 3


def test_effects_plan_maps_chapter_sentence_to_visual_scene(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(design_effects.config, "OUTPUT_DIR", str(tmp_path))
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    _write_json(
        video_dir / "image_prompts.json",
        [
            {"index": 1, "start": 0.0, "end": 2.0, "scene_text": "wide", "source_sentence_index": 1},
            {"index": 2, "start": 2.0, "end": 4.0, "scene_text": "wide", "source_sentence_index": 2},
            {"index": 3, "start": 4.0, "end": 6.0, "scene_text": "wide", "source_sentence_index": 4},
            {"index": 4, "start": 6.0, "end": 8.0, "scene_text": "wide", "source_sentence_index": 5},
        ],
    )
    _write_json(video_dir / "creative_package.json", _creative_package())
    monkeypatch.setattr(design_effects, "_audio_duration", lambda _video_dir: 10.0)

    plan, _diagnostics = design_effects.build_effects_plan("video")

    assert plan["scenes"][1]["transition_out"]["type"] == "hard_cut"


def test_pan_has_overscan_and_pullout_never_below_one(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(design_effects.config, "OUTPUT_DIR", str(tmp_path))
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    _write_json(
        video_dir / "image_prompts.json",
        [
            {"index": 1, "start": 0.0, "end": 7.0, "scene_text": "journey movement", "source_sentence_index": 1},
            {"index": 2, "start": 7.0, "end": 14.0, "scene_text": "environment landscape", "source_sentence_index": 2},
        ],
    )
    monkeypatch.setattr(design_effects, "_audio_duration", lambda _video_dir: 16.0)

    plan, _diagnostics = design_effects.build_effects_plan("video")
    pan_motion = plan["scenes"][0]["motion"]
    pull_motion = plan["scenes"][1]["motion"]

    assert pan_motion["type"].startswith("pan_")
    assert pan_motion["start_scale"] >= 1.05
    assert pull_motion["type"] == "slow_pull_out"
    assert pull_motion["start_scale"] >= 1.0
    assert pull_motion["end_scale"] >= 1.0


def test_transition_distribution_small_fixture_warns_not_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(design_effects.config, "OUTPUT_DIR", str(tmp_path))
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    _write_json(
        video_dir / "image_prompts.json",
        [
            {"index": 1, "start": 0.0, "end": 2.0, "scene_text": "detail", "source_sentence_index": 1},
            {"index": 2, "start": 2.0, "end": 4.0, "scene_text": "detail", "source_sentence_index": 2},
            {"index": 3, "start": 4.0, "end": 6.0, "scene_text": "detail", "source_sentence_index": 3},
        ],
    )
    monkeypatch.setattr(design_effects, "_audio_duration", lambda _video_dir: 7.0)

    _plan, diagnostics = design_effects.build_effects_plan("video")

    assert diagnostics["validation_passed"] is True
    assert diagnostics["warnings"]


def test_render_helpers_preserve_pause_coverage_and_effects_disabled_static() -> None:
    prompts = [
        {"index": 1, "start": 1.0, "end": 2.0},
        {"index": 2, "start": 4.0, "end": 5.0},
    ]
    static_plan = render_video._build_static_effects(prompts, 8.0)

    assert static_plan["scenes"][0]["display_start"] == 0.0
    assert static_plan["scenes"][0]["display_end"] == 4.0
    assert static_plan["scenes"][1]["display_end"] == 8.0
    assert static_plan["effects_enabled"] is False
    assert static_plan["scenes"][0]["motion"]["type"] == "hold"


def test_invalid_creative_package_disables_chapter_dips(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(design_effects.config, "OUTPUT_DIR", str(tmp_path))
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    _write_json(
        video_dir / "image_prompts.json",
        [
            {"index": 1, "start": 0.0, "end": 2.0, "scene_text": "wide", "source_sentence_index": 1},
            {"index": 2, "start": 2.0, "end": 4.0, "scene_text": "wide", "source_sentence_index": 3},
        ],
    )
    _write_json(video_dir / "creative_package.json", _creative_package())
    monkeypatch.setattr(design_effects, "_audio_duration", lambda _video_dir: 5.0)
    monkeypatch.setattr(design_effects, "load_validated_package", lambda *args, **kwargs: (_ for _ in ()).throw(CreativePackageError("stale")))

    plan, _diagnostics = design_effects.build_effects_plan("video")

    assert plan["scenes"][0]["transition_out"]["type"] == "hard_cut"


def test_transition_falls_back_to_hard_cut_when_too_short(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(design_effects.config, "OUTPUT_DIR", str(tmp_path))
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    _write_json(
        video_dir / "image_prompts.json",
        [
            {"index": 1, "start": 0.0, "end": 0.5, "scene_text": "soft memory", "source_sentence_index": 1},
            {"index": 2, "start": 1.0, "end": 1.5, "scene_text": "detail", "source_sentence_index": 2},
            {"index": 3, "start": 2.0, "end": 2.5, "scene_text": "detail", "source_sentence_index": 3},
            {"index": 4, "start": 3.0, "end": 3.5, "scene_text": "detail", "source_sentence_index": 4},
            {"index": 5, "start": 4.0, "end": 4.5, "scene_text": "detail", "source_sentence_index": 5},
            {"index": 6, "start": 5.0, "end": 5.5, "scene_text": "detail", "source_sentence_index": 6},
        ],
    )
    monkeypatch.setattr(design_effects, "_audio_duration", lambda _video_dir: 6.0)

    plan, _diagnostics = design_effects.build_effects_plan("video")

    assert plan["scenes"][0]["transition_out"]["type"] == "hard_cut"
