from __future__ import annotations

from types import SimpleNamespace

import config
from scripts import gen_character_sheets, gen_style_concepts


def test_klein_port_and_config_defaults():
    style_args = gen_style_concepts.build_parser().parse_args([])
    char_args = gen_character_sheets.build_parser().parse_args(["--concepts", "C01"])
    assert style_args.vast_port == config.KLEIN_WORKER_PORT
    assert char_args.vast_port == config.KLEIN_WORKER_PORT
    assert config.KLEIN_STEPS_T2I == 4
    assert config.KLEIN_GUIDANCE_SCALE == 1.0


def test_numeric_scene_ids_for_style_and_character_requests():
    concept = gen_style_concepts.CONCEPTS[0]
    style_request = gen_style_concepts._build_request(concept, "A_control", "control_scene", "A")
    char_request = gen_character_sheets._scene_request(
        concept,
        "character_male",
        gen_character_sheets._character_prompt(concept, gen_character_sheets._KARO_BASE),
    )
    assert style_request.scene_id.isdigit()
    assert char_request.scene_id.isdigit()


def test_fail_when_health_reports_wrong_model(monkeypatch):
    backend = SimpleNamespace(base_url="http://worker", _worker_headers={})

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model_id": "black-forest-labs/FLUX.1-dev",
                "model_loaded": True,
            }

    monkeypatch.setattr(gen_style_concepts.requests, "get", lambda *args, **kwargs: _Resp())

    try:
        gen_style_concepts._require_klein_worker_ready(backend)
    except RuntimeError as exc:
        assert "mismatch" in str(exc).lower()
    else:
        raise AssertionError("Expected model mismatch failure")


def test_fail_when_health_reports_model_not_loaded(monkeypatch):
    backend = SimpleNamespace(base_url="http://worker", _worker_headers={})

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model_id": config.KLEIN_MODEL_ID,
                "model_loaded": False,
            }

    monkeypatch.setattr(gen_style_concepts.requests, "get", lambda *args, **kwargs: _Resp())

    try:
        gen_style_concepts._require_klein_worker_ready(backend)
    except RuntimeError as exc:
        assert "not ready" in str(exc).lower()
    else:
        raise AssertionError("Expected not-loaded failure")


def test_all_concepts_have_unique_style_variants():
    variants = [concept["style_variant"] for concept in gen_style_concepts.CONCEPTS]
    assert len(variants) == 10
    assert len(set(variants)) == 10


def test_all_concepts_share_same_control_scene():
    control_scenes = {concept["control_scene"] for concept in gen_style_concepts.CONCEPTS}
    assert control_scenes == {gen_style_concepts._CONTROL_SCENE}


def test_karo_and_luma_present_in_every_unique_scene():
    for concept in gen_style_concepts.CONCEPTS:
        scene = concept["unique_scene"].lower()
        assert "karo" in scene
        assert "luma" in scene
