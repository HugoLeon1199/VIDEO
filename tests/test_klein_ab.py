"""Tests for Klein-9B A/B experimental pipeline.

Covers:
- scene_classifier: character vs env/obj/diagram routing
- reference_selector: master seed lookup + chain depth + low confidence reset
- KleinBackend: mode selection (img2img vs t2i) per scene type
- ab_test_klein: scene planning logic (character detection, arm_b_mode assignment)
- Config: Klein keys present and sane defaults
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from image_generation.character_bible import detect_characters_in_text, get_character_block
from image_generation.scene_classifier import classify_scene, SceneClassification
from image_generation.reference_selector import select_reference, ReferenceDecision


# ---------------------------------------------------------------------------
# scene_classifier
# ---------------------------------------------------------------------------

def test_character_male_closeup_classified():
    cls = classify_scene("Close-up of Karo's weathered face by firelight")
    assert cls.scene_type == "character_closeup_male"
    assert cls.reference_key == "character_male"
    assert cls.confidence >= 0.75


def test_character_female_closeup_classified():
    # "stars" would trigger cosmic_sky; use a clearer character-only prompt
    cls = classify_scene("Close-up portrait of Luma, her face illuminated by firelight")
    assert cls.scene_type == "character_closeup_female"
    assert cls.reference_key == "character_female"


def test_object_macro_no_character():
    cls = classify_scene("A stone axe close-up resting on a rock")
    assert cls.scene_type == "object_macro"
    assert cls.expected_people == 0
    assert not cls.use_previous_image


def test_scientific_diagram_classified():
    cls = classify_scene("Diagram showing brain evolution across hominid species")
    assert cls.scene_type == "scientific_diagram"
    assert cls.expected_people == 0


def test_timeline_classified():
    cls = classify_scene("A timeline of early human migration from Africa")
    assert cls.scene_type == "timeline_cycle"


def test_night_fire_classified():
    cls = classify_scene("Campfire glowing in a dark cave, people gathered around")
    assert cls.scene_type == "night_fire"
    assert cls.expected_people >= 2


def test_day_wilderness_classified():
    cls = classify_scene("Wide shot of the African savanna at sunrise")
    assert cls.scene_type == "day_wilderness"
    assert cls.shot_type == "wide"


def test_same_scene_continuity_pose():
    prev = "Karo stands at the cave entrance"
    curr = "Karo raises his spear toward the horizon"
    cls = classify_scene(
        curr,
        previous_scene_text=prev,
        characters=["karo"],
        continuity_group="karo_cave",
        previous_continuity_group="karo_cave",
    )
    assert cls.use_previous_image is True
    assert cls.change_type == "pose"


def test_same_scene_continuity_angle():
    prev = "Karo and Luma walk through the valley"
    curr = "Close-up of Karo now showing his determined expression"
    cls = classify_scene(
        curr,
        previous_scene_text=prev,
        characters=["karo"],
        continuity_group="karo_valley",
        previous_continuity_group="karo_valley",
    )
    assert cls.use_previous_image is True
    assert cls.change_type == "angle"


# ---------------------------------------------------------------------------
# character_bible
# ---------------------------------------------------------------------------

def test_detect_karo_trigger():
    found = detect_characters_in_text("Karo walks toward the fire")
    assert "karo" in found


def test_detect_luma_trigger():
    found = detect_characters_in_text("the woman reaches out to the child")
    assert "luma" in found


def test_detect_no_character():
    found = detect_characters_in_text("Wide shot of an empty valley")
    assert found == []


def test_get_character_block_female_clothed():
    desc = get_character_block("luma")
    # Female characters must mention clothing — female clothing rule
    assert any(word in desc.lower() for word in ("dress", "tunic", "clothing", "cloth")), \
        f"Luma description must include clothing: {desc}"


# ---------------------------------------------------------------------------
# reference_selector
# ---------------------------------------------------------------------------

def test_reference_uses_master_seed(tmp_path):
    (tmp_path / "char_male.png").write_bytes(b"PNG")
    manifest = {"seeds": {"character_male": "char_male.png"}}
    cls = SceneClassification(
        scene_type="character_closeup_male",
        use_previous_image=False,
        reference_key="character_male",
        change_type="new",
        expected_people=1,
        shot_type="closeup",
        confidence=0.82,
    )
    decision = select_reference(cls, None, str(tmp_path), manifest, chain_depth=0)
    assert decision.mode == "img2img"
    assert decision.reference_source == "master_seed"
    assert decision.reference_key == "character_male"
    assert 0.5 <= decision.strength <= 0.75


def test_reference_resets_on_chain_depth_exceeded(tmp_path):
    (tmp_path / "char_male.png").write_bytes(b"PNG")
    manifest = {"seeds": {"character_male": "char_male.png"}}
    cls = SceneClassification(
        scene_type="same_scene_minor_pose",
        use_previous_image=True,
        reference_key="character_male",
        change_type="pose",
        expected_people=1,
        shot_type="medium",
        confidence=0.80,
    )
    decision = select_reference(
        cls,
        previous_scene_image=str(tmp_path / "char_male.png"),
        master_seed_dir=str(tmp_path),
        master_seed_manifest=manifest,
        chain_depth=4,  # >= MAX_CHAIN_DEPTH
    )
    assert decision.reset_reason is not None
    assert "chain_depth" in decision.reset_reason


def test_reference_low_confidence_falls_back_to_t2i(tmp_path):
    cls = SceneClassification(
        scene_type="new_context",
        use_previous_image=False,
        reference_key=None,
        change_type="new",
        expected_people=2,
        shot_type="medium",
        confidence=0.40,  # below 0.55 threshold
    )
    decision = select_reference(cls, None, None, None, chain_depth=0)
    assert decision.mode == "text_to_image"
    assert "low_confidence" in (decision.reset_reason or "")


def test_reference_env_scene_text_to_image():
    cls = SceneClassification(
        scene_type="day_wilderness",
        use_previous_image=False,
        reference_key="day_wilderness",
        change_type="new",
        expected_people=0,
        shot_type="wide",
        confidence=0.75,
    )
    # No master seed dir → falls back to t2i
    decision = select_reference(cls, None, None, None, chain_depth=0)
    assert decision.mode == "text_to_image"


def test_reference_previous_scene_image(tmp_path):
    prev_img = tmp_path / "prev.png"
    prev_img.write_bytes(b"PNG")
    cls = SceneClassification(
        scene_type="same_scene_minor_pose",
        use_previous_image=True,
        reference_key=None,
        change_type="pose",
        expected_people=1,
        shot_type="medium",
        confidence=0.80,
    )
    decision = select_reference(cls, str(prev_img), None, None, chain_depth=0)
    assert decision.mode == "img2img"
    assert decision.reference_source == "previous_scene"
    assert decision.strength == pytest.approx(0.55, abs=0.05)


# ---------------------------------------------------------------------------
# KleinBackend mode routing (unit — no real HTTP)
# ---------------------------------------------------------------------------

def test_klein_backend_selects_img2img_for_character(tmp_path, monkeypatch):
    """Character scene → KleinBackend injects img2img_base64."""
    from image_generation.klein_backend import KleinBackend
    from image_generation.schemas import SceneRequest, SceneResult, CandidateResult

    ref_file = tmp_path / "character_male.png"
    ref_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    manifest = {"seeds": {"character_male": "character_male.png"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    captured_requests = []

    class _FakeInner:
        def health_check(self): return True
        def generate(self, req):
            captured_requests.append(req)
            return SceneResult(
                video_id=req.video_id, scene_id=req.scene_id,
                model="fake", mode="fake", duration_seconds=0.01,
                candidates=[CandidateResult(
                    candidate_index=0, seed=11001, width=64, height=64,
                    sha256="abc", generation_seconds=0.01,
                    mime_type="image/png", local_path=None,
                )],
            )

    backend = KleinBackend.__new__(KleinBackend)
    backend._inner = _FakeInner()
    backend._master_seed_dir = str(tmp_path)
    backend._master_manifest = manifest
    backend._steps_t2i = 4
    backend._steps_img2img = 8
    backend._strength_character = 0.65
    backend._strength_continuity = 0.55
    backend._previous_image = None
    backend._previous_scene_text = None
    backend._previous_continuity_group = None
    backend._chain_depth = 0

    req = SceneRequest(
        video_id="test", scene_id="001",
        prompt="Close-up portrait of Karo by the fire",
        clip_prompt="Karo portrait closeup",
        width=64, height=64,
        candidate_seeds=[11001],
    )
    backend.generate(req)

    assert captured_requests, "inner.generate must have been called"
    sent = captured_requests[0]
    assert sent.img2img_base64 is not None, "character scene must inject img2img_base64"
    assert sent.steps == 8, f"img2img should use 8 steps, got {sent.steps}"


def test_klein_backend_selects_t2i_for_object_scene(tmp_path):
    """Object/env scene → KleinBackend uses text-to-image (no reference injected)."""
    from image_generation.klein_backend import KleinBackend
    from image_generation.schemas import SceneRequest, SceneResult

    captured_requests = []

    class _FakeInner:
        def health_check(self): return True
        def generate(self, req):
            captured_requests.append(req)
            return SceneResult(
                video_id=req.video_id, scene_id=req.scene_id,
                model="fake", mode="fake", duration_seconds=0.01, candidates=[],
            )

    backend = KleinBackend.__new__(KleinBackend)
    backend._inner = _FakeInner()
    backend._master_seed_dir = str(tmp_path)
    backend._master_manifest = {}  # no seeds → forces t2i
    backend._steps_t2i = 4
    backend._steps_img2img = 8
    backend._strength_character = 0.65
    backend._strength_continuity = 0.55
    backend._previous_image = None
    backend._previous_scene_text = None
    backend._previous_continuity_group = None
    backend._chain_depth = 0

    req = SceneRequest(
        video_id="test", scene_id="003",
        prompt="A stone axe close-up resting on a mossy log",
        clip_prompt="stone axe macro",
        width=64, height=64,
        candidate_seeds=[11001],
    )
    backend.generate(req)

    assert captured_requests
    sent = captured_requests[0]
    assert sent.img2img_base64 is None, "object scene must NOT inject reference image"
    assert sent.steps == 4, f"t2i should use 4 steps, got {sent.steps}"


# ---------------------------------------------------------------------------
# ab_test_klein: scene planning (dry-run)
# ---------------------------------------------------------------------------

def test_plan_scenes_character_flagged(tmp_path, monkeypatch):
    """_plan_scenes must mark character scenes as arm_b_mode=img2img."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.ab_test_klein import _plan_scenes

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    video_dir = tmp_path / "plantest"
    video_dir.mkdir()
    (video_dir / "image_prompts.json").write_text(json.dumps([
        {"index": 1, "prompt": "Close-up of Karo sharpening a flint knife",
         "clip_prompt": "karo closeup", "negative_prompt": ""},
        {"index": 2, "prompt": "Wide savanna at golden hour, no people",
         "clip_prompt": "savanna wide", "negative_prompt": ""},
        {"index": 3, "prompt": "Luma reaches into a clay pot",
         "clip_prompt": "luma pot", "negative_prompt": ""},
    ], ensure_ascii=False), encoding="utf-8")

    plan = _plan_scenes("plantest", count=3)

    char_scenes = [s for s in plan if s["has_character"]]
    env_scenes = [s for s in plan if not s["has_character"]]
    assert len(char_scenes) >= 1, "Karo and Luma scenes must be flagged as character"
    assert len(env_scenes) >= 1, "Savanna scene must NOT be flagged as character"
    for s in char_scenes:
        assert s["arm_b_mode"] == "img2img"
    for s in env_scenes:
        assert s["arm_b_mode"] == "text_to_image"


def test_plan_scenes_respects_count(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.ab_test_klein import _plan_scenes

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    video_dir = tmp_path / "counttest"
    video_dir.mkdir()
    (video_dir / "image_prompts.json").write_text(json.dumps([
        {"index": i + 1, "prompt": f"scene {i}", "clip_prompt": f"c{i}", "negative_prompt": ""}
        for i in range(30)
    ], ensure_ascii=False), encoding="utf-8")

    plan = _plan_scenes("counttest", count=5)
    assert len(plan) == 5


# ---------------------------------------------------------------------------
# Config: Klein keys present
# ---------------------------------------------------------------------------

def test_config_has_klein_keys():
    assert hasattr(config, "KLEIN_MODEL_ID")
    assert "klein" in config.KLEIN_MODEL_ID.lower() or "FLUX" in config.KLEIN_MODEL_ID
    assert hasattr(config, "KLEIN_STEPS_T2I")
    assert hasattr(config, "KLEIN_STEPS_IMG2IMG")
    assert hasattr(config, "KLEIN_IMG2IMG_STRENGTH_CHARACTER")
    assert hasattr(config, "KLEIN_IMG2IMG_STRENGTH_CONTINUITY")
    assert hasattr(config, "KLEIN_REFERENCE_DIR")
    assert hasattr(config, "KLEIN_AB_SCENE_COUNT")
    assert hasattr(config, "KLEIN_AB_CANDIDATE_SEEDS")
    assert isinstance(config.KLEIN_AB_CANDIDATE_SEEDS, list)
    assert len(config.KLEIN_AB_CANDIDATE_SEEDS) >= 1


def test_klein_strength_values_in_range():
    assert 0.50 <= config.KLEIN_IMG2IMG_STRENGTH_CHARACTER <= 0.75
    assert 0.45 <= config.KLEIN_IMG2IMG_STRENGTH_CONTINUITY <= 0.70


def test_production_defaults_unchanged():
    """Ensure production IMAGE_BACKEND default is still runpod_serverless, not klein."""
    assert config.IMAGE_BACKEND in ("runpod_serverless", "vast_instance"), \
        f"Production IMAGE_BACKEND must not default to 'klein', got: {config.IMAGE_BACKEND}"
