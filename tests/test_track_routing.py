"""Tests for dual-track image generation routing (VI=12B, EN=9B)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ---------------------------------------------------------------------------
# Track config routing tests
# ---------------------------------------------------------------------------

def test_vi_track_steps_and_guidance():
    """VI track must use 20 steps and 3.5 guidance (FLUX.1-dev 12B)."""
    tc = config.TRACK_CONFIG["vi"]
    assert tc["steps"] == 20
    assert tc["guidance_scale"] == 3.5


def test_en_track_steps_and_guidance():
    """EN track must use 20 steps and 3.5 guidance (FLUX.1-dev 12B unified)."""
    tc = config.TRACK_CONFIG["en"]
    assert tc["steps"] == 20
    assert tc["guidance_scale"] == 3.5


def test_vi_uses_unified_endpoint_env():
    """VI track must read endpoint from RUNPOD_ENDPOINT_ID (unified endpoint)."""
    tc = config.TRACK_CONFIG["vi"]
    assert tc["endpoint_id_env"] == "RUNPOD_ENDPOINT_ID"


def test_en_uses_unified_endpoint_env():
    """EN track must read endpoint from RUNPOD_ENDPOINT_ID (unified endpoint)."""
    tc = config.TRACK_CONFIG["en"]
    assert tc["endpoint_id_env"] == "RUNPOD_ENDPOINT_ID"


def test_vi_output_subdir_is_images_vi():
    """VI track images must go to images_vi/ to avoid overwriting EN track."""
    tc = config.TRACK_CONFIG["vi"]
    assert tc["output_subdir"] == "images_vi"


def test_en_output_subdir_is_images_en():
    """EN track images must go to images_en/ to avoid overwriting VI track."""
    tc = config.TRACK_CONFIG["en"]
    assert tc["output_subdir"] == "images_en"


def test_vi_and_en_output_dirs_are_different():
    """VI and EN output subdirs must be distinct — no cross-track overwrite possible."""
    vi_subdir = config.TRACK_CONFIG["vi"]["output_subdir"]
    en_subdir = config.TRACK_CONFIG["en"]["output_subdir"]
    assert vi_subdir != en_subdir


# ---------------------------------------------------------------------------
# Prompt template content tests
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def test_vi_prompt_template_exists():
    assert (PROMPTS_DIR / "image_prompt_vi.txt").exists(), \
        "prompts/image_prompt_vi.txt not found"


def test_en_prompt_template_exists():
    assert (PROMPTS_DIR / "image_prompt_en.txt").exists(), \
        "prompts/image_prompt_en.txt not found"


def test_vi_prompt_contains_style_lock():
    content = (PROMPTS_DIR / "image_prompt_vi.txt").read_text(encoding="utf-8")
    assert "2D painted documentary" in content
    assert "vi-2d-documentary-v1" in content
    assert "RUNPOD_ENDPOINT_ID" in content


def test_en_prompt_contains_style_lock():
    content = (PROMPTS_DIR / "image_prompt_en.txt").read_text(encoding="utf-8")
    assert "Ink sketch illustration on aged parchment paper" in content
    assert "no extra limbs" in content
    assert "RUNPOD_ENDPOINT_ID" in content


def test_vi_prompt_no_cave_painting_leak():
    """VI template must not use the old cave-painting style (from system_prompt.txt)."""
    content = (PROMPTS_DIR / "image_prompt_vi.txt").read_text(encoding="utf-8").lower()
    assert "cave painting" not in content
    assert "rock art" not in content or "paleo art" in content  # paleo art OK, cave painting not


def test_en_prompt_no_cave_painting_leak():
    """EN template must not use the old cave-painting style (from system_prompt.txt)."""
    content = (PROMPTS_DIR / "image_prompt_en.txt").read_text(encoding="utf-8").lower()
    assert "cave painting" not in content


def test_en_prompt_no_written_labels():
    """EN style guide must explicitly forbid written labels/letters/numbers."""
    content = (PROMPTS_DIR / "image_prompt_en.txt").read_text(encoding="utf-8")
    assert "no written labels" in content or "NO written labels" in content


def test_en_prompt_allows_arrows_not_labels():
    """EN style guide must allow arrows/symbols but not letters or numbers."""
    content = (PROMPTS_DIR / "image_prompt_en.txt").read_text(encoding="utf-8")
    assert "arrows" in content.lower() or "arrow" in content.lower()
