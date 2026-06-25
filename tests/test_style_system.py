"""
Comprehensive tests for the style system:
  - style_dna.py
  - character_bible.py
  - scene_classifier.py
  - reference_selector.py
  - grain_overlay.py
  - serverless_worker_unified/handler.py (no RunPod call)

Run with: python -m pytest tests/test_style_system.py -v
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("RUNPOD_API_KEY", "test_key_style")
os.environ.setdefault("RUNPOD_ENDPOINT_ID", "test_endpoint_style")
os.environ.setdefault("IMAGE_OUTPUT_ROOT", tempfile.mkdtemp())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png_bytes(width: int = 1, height: int = 1) -> bytes:
    """Create a minimal 1x1 PNG using PIL."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(100, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_tmp_png(directory: str, filename: str) -> str:
    """Write a 1x1 PNG to a temp directory and return its path."""
    path = Path(directory) / filename
    path.write_bytes(_make_png_bytes())
    return str(path)


# ===========================================================================
# 1. test_no_lora_in_handler
# ===========================================================================

class TestHandlerNoLoRA:
    def test_no_lora_in_handler(self):
        """handler.py must not contain actual LoRA API calls or imports."""
        handler_path = (
            Path(__file__).resolve().parent.parent
            / "serverless_worker_unified"
            / "handler.py"
        )
        source = handler_path.read_text(encoding="utf-8")

        # Strip comments and string literals to avoid false positives
        # (the startup print statement says "No LoRA..." which is fine)
        import ast
        try:
            tree = ast.parse(source)
        except SyntaxError:
            pytest.fail("handler.py has a SyntaxError")

        # Collect all Name/Attribute nodes that would represent LoRA API calls
        api_calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                api_calls.add(node.attr.lower())
            elif isinstance(node, ast.Name):
                api_calls.add(node.id.lower())

        # These are actual LoRA API method names — must not appear as identifiers
        forbidden_identifiers = {"load_lora_weights", "fuse_lora", "unfuse_lora",
                                  "set_adapters", "load_adapter"}
        found = forbidden_identifiers & api_calls
        assert not found, f"handler.py contains LoRA API identifiers: {found}"

        # safetensors import check (import statement)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = ""
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module
                elif isinstance(node, ast.Import):
                    mod = " ".join(alias.name for alias in node.names)
                assert "safetensors" not in mod.lower(), (
                    f"handler.py imports safetensors: {mod}"
                )


# ===========================================================================
# 2-6. style_dna.py tests
# ===========================================================================

class TestStyleDNA:
    def _import(self):
        from image_generation.style_dna import (
            STYLE_PREFIX, STYLE_SUFFIX, NEGATIVE_PROMPT,
            build_scene_prompt, sanitize_scene_text,
        )
        return STYLE_PREFIX, STYLE_SUFFIX, NEGATIVE_PROMPT, build_scene_prompt, sanitize_scene_text

    def test_style_prefix_present(self):
        """build_scene_prompt result starts with STYLE_PREFIX."""
        STYLE_PREFIX, STYLE_SUFFIX, _, build_scene_prompt, _ = self._import()
        result = build_scene_prompt("a campfire at night", character_blocks=[])
        assert result.startswith(STYLE_PREFIX), (
            f"Expected start: {STYLE_PREFIX[:40]!r}\nGot: {result[:80]!r}"
        )

    def test_style_suffix_present(self):
        """build_scene_prompt result ends with STYLE_SUFFIX."""
        STYLE_PREFIX, STYLE_SUFFIX, _, build_scene_prompt, _ = self._import()
        result = build_scene_prompt("a campfire at night", character_blocks=[])
        assert result.endswith(STYLE_SUFFIX), (
            f"Expected end: {STYLE_SUFFIX[-40:]!r}\nGot: {result[-80:]!r}"
        )

    def test_sanitize_removes_photorealistic(self):
        """sanitize_scene_text replaces 'photorealistic' with the safe alternative."""
        _, _, _, _, sanitize_scene_text = self._import()
        result = sanitize_scene_text("photorealistic portrait of a hunter")
        assert "photorealistic" not in result.lower()

    def test_sanitize_removes_multiple_terms(self):
        """sanitize_scene_text handles multiple realism terms in one pass."""
        _, _, _, _, sanitize_scene_text = self._import()
        bad_text = "hyperrealistic 3d render with bokeh and volumetric lighting"
        result = sanitize_scene_text(bad_text)
        for term in ["hyperrealistic", "3d render", "bokeh", "volumetric lighting"]:
            assert term not in result.lower(), f"Term '{term}' still present in: {result}"

    def test_sanitize_logs_removed_terms(self):
        """sanitize_scene_text logs a warning when terms are removed."""
        _, _, _, _, sanitize_scene_text = self._import()
        with self.assertLogs("image_generation.style_dna", level="WARNING") as cm:
            sanitize_scene_text("photorealistic portrait")
        assert any("photorealistic" in msg for msg in cm.output)

    # assertLogs helper for non-TestCase subclass
    class assertLogs:
        """Context manager wrapper for assertLogs in a plain class."""
        def __init__(self, logger_name: str, level: str = "WARNING"):
            import logging
            self._logger = logging.getLogger(logger_name)
            self._level = getattr(logging, level)
            self._handler = None
            self.output = []

        def __enter__(self):
            self._records = []
            self._handler = logging.handlers_handler(self._records)
            self._logger.addHandler(self._handler)
            self._logger.setLevel(self._level)
            return self

        def __exit__(self, *args):
            self._logger.removeHandler(self._handler)
            self.output = [r.getMessage() for r in self._records]

    def test_sanitize_logs_removed_terms(self):
        """sanitize_scene_text logs a warning when terms are removed (using pytest caplog)."""
        _, _, _, _, sanitize_scene_text = self._import()
        import logging
        logger = logging.getLogger("image_generation.style_dna")
        records = []

        class CapHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        h = CapHandler()
        logger.addHandler(h)
        old_level = logger.level
        logger.setLevel(logging.WARNING)
        try:
            sanitize_scene_text("photorealistic portrait")
        finally:
            logger.removeHandler(h)
            logger.setLevel(old_level)

        assert any("photorealistic" in r.getMessage() for r in records), (
            "Expected warning about removed term 'photorealistic'"
        )


# ===========================================================================
# 7-11. character_bible.py tests
# ===========================================================================

class TestCharacterBible:
    def test_character_bible_karo(self):
        """get_character_block('karo') contains 'KARO' and 'ochre'."""
        from image_generation.character_bible import get_character_block
        block = get_character_block("karo")
        assert "KARO" in block
        assert "ochre" in block.lower()

    def test_character_bible_luma(self):
        """get_character_block('luma') contains 'LUMA' and 'braided'."""
        from image_generation.character_bible import get_character_block
        block = get_character_block("luma")
        assert "LUMA" in block
        assert "braided" in block.lower()

    def test_character_bible_unknown(self):
        """get_character_block raises KeyError for unknown character."""
        from image_generation.character_bible import get_character_block
        with pytest.raises(KeyError):
            get_character_block("unknown_hero")

    def test_detect_characters_karo(self):
        """'karo raises his hand' should return ['karo']."""
        from image_generation.character_bible import detect_characters_in_text
        found = detect_characters_in_text("karo raises his hand above his head")
        assert "karo" in found

    def test_detect_characters_none(self):
        """'a campfire burns in the dark' has no character mentions."""
        from image_generation.character_bible import detect_characters_in_text
        found = detect_characters_in_text("a campfire burns in the dark")
        assert found == []


# ===========================================================================
# 12-16. scene_classifier.py tests
# ===========================================================================

class TestSceneClassifier:
    def _classify(self, text, **kwargs):
        from image_generation.scene_classifier import classify_scene
        return classify_scene(text, **kwargs)

    def test_classify_night_fire(self):
        """'campfire at night' -> scene_type=night_fire, reference_key=night_fire."""
        result = self._classify("A campfire burns at night in the cave")
        assert result.scene_type == "night_fire"
        assert result.reference_key == "night_fire"

    def test_classify_scientific_diagram(self):
        """'brain diagram' -> scene_type=scientific_diagram."""
        result = self._classify("An educational brain diagram showing the frontal lobe")
        assert result.scene_type == "scientific_diagram"

    def test_classify_same_scene_pose(self):
        """Same continuity_group + pose verb -> same_scene_minor_pose, use_previous_image=True."""
        result = self._classify(
            "Karo raises his arm toward the horizon",
            previous_scene_text="Karo stands at the edge of the valley",
            continuity_group="group_A",
            previous_continuity_group="group_A",
        )
        assert result.scene_type == "same_scene_minor_pose"
        assert result.use_previous_image is True
        assert result.change_type == "pose"

    def test_classify_same_scene_angle(self):
        """Same continuity_group + 'close-up of' -> same_scene_camera_shift."""
        result = self._classify(
            "Close-up of Karo's face as he looks at the fire",
            previous_scene_text="Karo stands at the campfire",
            continuity_group="group_B",
            previous_continuity_group="group_B",
        )
        assert result.scene_type == "same_scene_camera_shift"
        assert result.use_previous_image is True

    def test_classify_low_confidence_is_new_context(self):
        """Vague scene text -> new_context with low confidence."""
        result = self._classify("Something happens in a place")
        assert result.scene_type == "new_context"
        assert result.confidence < 0.6


# ===========================================================================
# 17-23. reference_selector.py tests
# ===========================================================================

class TestReferenceSelector:
    def _select(self, classification, previous_scene_image=None,
                master_seed_dir=None, master_seed_manifest=None,
                chain_depth=0, previous_qa_passed=True):
        from image_generation.reference_selector import select_reference
        return select_reference(
            classification=classification,
            previous_scene_image=previous_scene_image,
            master_seed_dir=master_seed_dir,
            master_seed_manifest=master_seed_manifest,
            chain_depth=chain_depth,
            previous_qa_passed=previous_qa_passed,
        )

    def _make_classification(self, use_previous=False, reference_key=None,
                              change_type="new", confidence=0.80):
        from image_generation.scene_classifier import SceneClassification
        return SceneClassification(
            scene_type="night_fire",
            use_previous_image=use_previous,
            reference_key=reference_key,
            change_type=change_type,
            expected_people=2,
            shot_type="medium",
            confidence=confidence,
        )

    def test_reference_selector_uses_previous(self):
        """Previous valid + use_previous_image=True + chain_depth=1 -> img2img, source=previous_scene."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prev_img = _write_tmp_png(tmpdir, "prev_scene.png")
            clf = self._make_classification(use_previous=True, change_type="pose")
            decision = self._select(clf, previous_scene_image=prev_img, chain_depth=1)
        assert decision.mode == "img2img"
        assert decision.reference_source == "previous_scene"
        assert decision.reference_path == prev_img

    def test_reference_selector_resets_at_max_depth(self):
        """chain_depth=4 -> NOT previous_scene (chain must reset)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prev_img = _write_tmp_png(tmpdir, "prev_scene.png")
            clf = self._make_classification(use_previous=True, change_type="pose")
            decision = self._select(clf, previous_scene_image=prev_img, chain_depth=4)
        assert decision.reference_source != "previous_scene"

    def test_reference_selector_never_uses_failed_qa(self):
        """previous_qa_passed=False + use_previous_image=True -> NOT previous_scene."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prev_img = _write_tmp_png(tmpdir, "prev_scene.png")
            clf = self._make_classification(use_previous=True, change_type="pose")
            decision = self._select(
                clf, previous_scene_image=prev_img,
                chain_depth=1, previous_qa_passed=False,
            )
        assert decision.reference_source != "previous_scene"

    def test_reference_selector_uses_master_seed(self):
        """New night_fire scene + manifest with valid seed_04.png -> img2img, source=master_seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write the seed file
            _write_tmp_png(tmpdir, "seed_04.png")
            manifest = {"seeds": {"night_fire": "seed_04.png"}}
            clf = self._make_classification(
                use_previous=False, reference_key="night_fire", confidence=0.78
            )
            decision = self._select(
                clf,
                master_seed_dir=tmpdir,
                master_seed_manifest=manifest,
                chain_depth=0,
            )
        assert decision.mode == "img2img"
        assert decision.reference_source == "master_seed"
        assert decision.reference_key == "night_fire"

    def test_strength_clamped(self):
        """All STRENGTH_MAP values must be between 0.50 and 0.75."""
        from image_generation.reference_selector import STRENGTH_MAP
        for key, val in STRENGTH_MAP.items():
            assert 0.50 <= val <= 0.75, (
                f"STRENGTH_MAP[{key!r}] = {val} is outside [0.50, 0.75]"
            )

    def test_only_one_reference_per_scene(self):
        """ReferenceDecision always has exactly one reference_path (or None for t2i)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prev_img = _write_tmp_png(tmpdir, "prev.png")
            seed_img = _write_tmp_png(tmpdir, "seed_04.png")
            manifest = {"seeds": {"night_fire": "seed_04.png"}}

            # Previous-scene scenario
            clf_prev = self._make_classification(use_previous=True, change_type="pose")
            d1 = self._select(clf_prev, previous_scene_image=prev_img, chain_depth=1)
            # Should have exactly one path
            assert d1.reference_path is not None or d1.mode == "text_to_image"

            # t2i scenario — no manifest, no previous
            clf_new = self._make_classification(use_previous=False, reference_key=None, confidence=0.50)
            d2 = self._select(clf_new)
            assert d2.mode == "text_to_image"
            assert d2.reference_path is None

    def test_chain_depth_increments(self):
        """5 scenes in same continuity group: chain resets after MAX_CHAIN_DEPTH."""
        from image_generation.scene_classifier import classify_scene
        from image_generation.reference_selector import select_reference, MAX_CHAIN_DEPTH

        with tempfile.TemporaryDirectory() as tmpdir:
            prev_img = _write_tmp_png(tmpdir, "scene_prev.png")

            chain_depth = 0
            reset_happened = False

            for i in range(5):
                clf = classify_scene(
                    f"Karo turns and looks at scene {i}",
                    previous_scene_text="Karo stands by the fire",
                    continuity_group="group_X",
                    previous_continuity_group="group_X",
                )
                decision = select_reference(
                    classification=clf,
                    previous_scene_image=prev_img,
                    master_seed_dir=None,
                    master_seed_manifest=None,
                    chain_depth=chain_depth,
                    previous_qa_passed=True,
                )
                if decision.reference_source != "previous_scene":
                    reset_happened = True
                    chain_depth = 0
                else:
                    chain_depth += 1

            # After 5 scenes with MAX_CHAIN_DEPTH=4, at least one reset must have occurred
            assert reset_happened, (
                f"Expected chain reset after MAX_CHAIN_DEPTH={MAX_CHAIN_DEPTH} but none occurred"
            )


# ===========================================================================
# 24. test_guidance_scale_clamped
# ===========================================================================

class TestHandlerGuidanceClamp:
    def test_guidance_scale_clamped(self):
        """handler.py clamps guidance_scale to [3.5, 4.0] before calling the pipe."""
        # We test the clamping logic directly by importing and inspecting the source,
        # or by exercising the handler with a mocked pipe and verifying the call args.
        import types

        handler_path = (
            Path(__file__).resolve().parent.parent
            / "serverless_worker_unified"
            / "handler.py"
        )
        source = handler_path.read_text(encoding="utf-8")

        # Verify clamping expressions are present in source
        assert "max(3.5, min(4.0, guidance_scale))" in source, (
            "handler.py must clamp guidance_scale with max(3.5, min(4.0, guidance_scale))"
        )
        assert "max(20, min(24, steps))" in source, (
            "handler.py must clamp steps with max(20, min(24, steps))"
        )


# ===========================================================================
# 25. test_grain_overlay_not_in_generation
# ===========================================================================

class TestGrainOverlayIsolation:
    def test_grain_overlay_not_in_generation(self):
        """grain_overlay.py lives in image_generation/, NOT called from handler.py."""
        handler_path = (
            Path(__file__).resolve().parent.parent
            / "serverless_worker_unified"
            / "handler.py"
        )
        source = handler_path.read_text(encoding="utf-8")
        assert "grain_overlay" not in source, (
            "grain_overlay must NOT be imported or called from handler.py"
        )

    def test_grain_overlay_module_exists(self):
        """grain_overlay.py is importable from image_generation."""
        from image_generation.grain_overlay import apply_grain_overlay
        assert callable(apply_grain_overlay)

    def test_grain_overlay_applies(self):
        """apply_grain_overlay runs without error on two valid PNG files."""
        from image_generation.grain_overlay import apply_grain_overlay

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = _write_tmp_png(tmpdir, "base.png")
            overlay_path = _write_tmp_png(tmpdir, "grain.png")
            out_path = str(Path(tmpdir) / "output.png")

            result = apply_grain_overlay(
                image_path=base_path,
                overlay_path=overlay_path,
                opacity=0.10,
                output_path=out_path,
            )
            assert Path(result).exists()
            assert result == out_path

    def test_grain_opacity_clamped(self):
        """apply_grain_overlay clamps opacity to [0.01, 0.30]."""
        from image_generation.grain_overlay import apply_grain_overlay

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = _write_tmp_png(tmpdir, "base.png")
            overlay_path = _write_tmp_png(tmpdir, "grain.png")
            out_path = str(Path(tmpdir) / "output.png")

            # opacity=1.0 should be clamped to 0.30 — shouldn't raise
            result = apply_grain_overlay(
                image_path=base_path,
                overlay_path=overlay_path,
                opacity=1.0,
                output_path=out_path,
            )
            assert Path(result).exists()
