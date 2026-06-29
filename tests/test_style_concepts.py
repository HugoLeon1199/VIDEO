"""Tests for Klein 9B style concept generation scripts.

Covers:
- Klein port/config alignment
- Numeric scene IDs (VastInstanceBackend compatibility)
- Health-check model validation (rejects 12B or unloaded worker)
- 10 concepts all have distinct style_variant
- All concepts share the same control_scene
- Karo + Luma appear in every unique_scene
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import config
from scripts.gen_style_concepts import (
    CONCEPTS,
    _CONTROL_SCENE,
    _VARIANTS,
    _build_full_prompt,
    _numeric_scene_id,
    _require_klein_worker_ready,
    _select_concepts,
)


# ---------------------------------------------------------------------------
# Config alignment
# ---------------------------------------------------------------------------

def test_klein_worker_port_is_8081():
    assert config.KLEIN_WORKER_PORT == 8081


def test_klein_steps_t2i_is_4():
    assert config.KLEIN_STEPS_T2I == 4


def test_klein_guidance_scale_is_1():
    assert config.KLEIN_GUIDANCE_SCALE == 1.0


def test_gen_script_default_port_matches_config():
    """build_parser() default port must equal config.KLEIN_WORKER_PORT."""
    from scripts.gen_style_concepts import build_parser
    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.vast_port == config.KLEIN_WORKER_PORT


def test_char_sheet_default_port_matches_config():
    from scripts.gen_character_sheets import build_parser
    parser = build_parser()
    defaults = parser.parse_args(["--concepts", "C01"])
    assert defaults.vast_port == config.KLEIN_WORKER_PORT


# ---------------------------------------------------------------------------
# Numeric scene IDs
# ---------------------------------------------------------------------------

def test_numeric_scene_id_format():
    sid = _numeric_scene_id("C01", "A")
    assert sid.isdigit(), f"scene_id must be numeric string, got {sid!r}"
    int(sid)  # must not raise


def test_numeric_scene_id_all_concepts_unique():
    seen = set()
    for concept in CONCEPTS:
        for _image_key, _scene_key, col in _VARIANTS:
            sid = _numeric_scene_id(concept["id"], col)
            assert sid not in seen, f"Duplicate scene_id {sid}"
            seen.add(sid)


def test_numeric_scene_id_values():
    # C01/A -> 11, C01/B -> 12, C10/A -> 101, C10/B -> 102
    assert _numeric_scene_id("C01", "A") == "11"
    assert _numeric_scene_id("C01", "B") == "12"
    assert _numeric_scene_id("C10", "A") == "101"
    assert _numeric_scene_id("C10", "B") == "102"


# ---------------------------------------------------------------------------
# Health check model validation
# ---------------------------------------------------------------------------

def _fake_backend(url: str = "http://fake-host:8081"):
    class _B:
        base_url = url
        _worker_headers: dict = {}
    return _B()


def _mock_health(monkeypatch, payload: dict):
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status = MagicMock()
    monkeypatch.setattr("scripts.gen_style_concepts.requests.get", lambda *a, **kw: mock_resp)


def test_require_klein_ready_rejects_unloaded(monkeypatch):
    """model_loaded=False must raise RuntimeError."""
    _mock_health(monkeypatch, {
        "model_id": config.KLEIN_MODEL_ID,
        "model_loaded": False,
        "device": "cuda",
    })
    with pytest.raises(RuntimeError, match="not ready"):
        _require_klein_worker_ready(_fake_backend())


def test_require_klein_ready_rejects_12b_model(monkeypatch):
    """Wrong model_id (12B) must raise RuntimeError."""
    _mock_health(monkeypatch, {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "model_loaded": True,
        "device": "cuda",
    })
    with pytest.raises(RuntimeError, match="mismatch"):
        _require_klein_worker_ready(_fake_backend())


def test_require_klein_ready_accepts_correct_model(monkeypatch):
    """Correct model_id + model_loaded=True must succeed and return payload."""
    _mock_health(monkeypatch, {
        "model_id": config.KLEIN_MODEL_ID,
        "model_loaded": True,
        "device": "cuda",
        "gpu_vram_gb": 24.0,
    })
    result = _require_klein_worker_ready(_fake_backend())
    assert result["model_loaded"] is True
    assert result["model_id"] == config.KLEIN_MODEL_ID


# ---------------------------------------------------------------------------
# Concepts: style_variant uniqueness
# ---------------------------------------------------------------------------

def test_all_concepts_have_distinct_style_variant():
    variants = [c["style_variant"] for c in CONCEPTS]
    assert len(variants) == len(set(variants)), "Every concept must have a unique style_variant"


def test_all_concepts_have_style_variant_non_empty():
    for c in CONCEPTS:
        assert c.get("style_variant", "").strip(), f"{c['id']} missing style_variant"


def test_ten_concepts_defined():
    assert len(CONCEPTS) == 10


# ---------------------------------------------------------------------------
# Concepts: shared control scene
# ---------------------------------------------------------------------------

def test_all_concepts_share_same_control_scene():
    for c in CONCEPTS:
        assert c["control_scene"] == _CONTROL_SCENE, \
            f"{c['id']} control_scene differs from _CONTROL_SCENE"


def test_control_scene_contains_karo_and_luma():
    assert "Karo" in _CONTROL_SCENE
    assert "Luma" in _CONTROL_SCENE


# ---------------------------------------------------------------------------
# Concepts: Karo + Luma in every unique scene
# ---------------------------------------------------------------------------

def test_all_unique_scenes_contain_karo():
    missing = [c["id"] for c in CONCEPTS if "Karo" not in c["unique_scene"]]
    assert not missing, f"unique_scene missing 'Karo': {missing}"


def test_all_unique_scenes_contain_luma():
    missing = [c["id"] for c in CONCEPTS if "Luma" not in c["unique_scene"]]
    assert not missing, f"unique_scene missing 'Luma': {missing}"


# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------

def test_full_prompt_contains_style_lock():
    from scripts.gen_style_concepts import _STYLE_LOCK
    prompt = _build_full_prompt(CONCEPTS[0], "control_scene")
    assert _STYLE_LOCK in prompt


def test_full_prompt_contains_style_variant():
    c = CONCEPTS[0]
    prompt = _build_full_prompt(c, "control_scene")
    assert c["style_variant"] in prompt


def test_full_prompt_contains_scene_text():
    c = CONCEPTS[0]
    prompt = _build_full_prompt(c, "control_scene")
    assert c["control_scene"] in prompt


# ---------------------------------------------------------------------------
# Smoke / concept selection
# ---------------------------------------------------------------------------

def test_select_concepts_smoke():
    result = _select_concepts("", smoke=True)
    assert len(result) == 1
    assert result[0]["id"] == "C01"


def test_select_concepts_subset():
    result = _select_concepts("C03,C07", smoke=False)
    ids = [c["id"] for c in result]
    assert "C03" in ids and "C07" in ids
    assert len(result) == 2


def test_select_concepts_all():
    result = _select_concepts("", smoke=False)
    assert len(result) == 10


def test_select_concepts_unknown_id_raises():
    with pytest.raises(RuntimeError, match="No matching"):
        _select_concepts("C99", smoke=False)
