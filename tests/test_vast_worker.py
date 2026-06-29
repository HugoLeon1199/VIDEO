"""Tests for the per-GPU FastAPI worker (gpu_worker.py).

The gateway (server.py) delegates all inference to gpu_worker subprocesses.
These tests validate the gpu_worker directly.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vast_worker import gpu_worker


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True):
        tokens = text.split()
        encoded = list(range(len(tokens)))
        if add_special_tokens:
            return [101, *encoded, 102]
        return encoded


class _FakePipe:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.tokenizer_2 = _FakeTokenizer()
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        img = Image.new("RGB", (16, 16), (120, 80, 40))
        return SimpleNamespace(images=[img])


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    monkeypatch.setenv("WORKER_API_TOKEN", "secret-token")
    monkeypatch.setenv("HF_MODEL_REVISION", "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")


def test_health_endpoint_is_minimal() -> None:
    client = TestClient(gpu_worker.app)
    payload = client.get("/health").json()
    assert set(payload) == {"status", "model_loaded", "load_error"}


def test_generate_rejects_missing_worker_token(monkeypatch) -> None:
    # Inject a loaded pipe so auth is the only gate
    monkeypatch.setattr(gpu_worker, "_pipe", _FakePipe())
    client = TestClient(gpu_worker.app)
    resp = client.post(
        "/generate",
        json={
            "video_id": "vid",
            "scene_id": "001",
            "prompt": "full prompt",
            "clip_prompt": "short prompt",
            "negative_prompt": "avoid",
            "width": 1024,
            "height": 576,
            "steps": 20,
            "guidance_scale": 3.5,
            "candidate_seeds": [11],
            "output_format": "WEBP",
            "quality": 92,
        },
    )
    assert resp.status_code == 401


def test_generate_uses_clip_and_full_prompt_contract(monkeypatch) -> None:
    pipe = _FakePipe()
    monkeypatch.setattr(gpu_worker, "_pipe", pipe)
    client = TestClient(gpu_worker.app)
    resp = client.post(
        "/generate",
        headers={"X-Worker-Token": "secret-token"},
        json={
            "video_id": "vid",
            "scene_id": "001",
            "prompt": "full detailed prompt",
            "clip_prompt": "short clip prompt",
            "negative_prompt": "bad lighting",
            "width": 1024,
            "height": 576,
            "steps": 20,
            "guidance_scale": 3.5,
            "candidate_seeds": [11],
            "output_format": "WEBP",
            "quality": 92,
        },
    )
    assert resp.status_code == 200
    assert pipe.calls
    call = pipe.calls[0]
    assert call["prompt"] == "short clip prompt"
    assert call["prompt_2"] == "full detailed prompt. Avoid: bad lighting"
    assert call["max_sequence_length"] == 512


def test_generate_rejects_overlong_clip_prompt(monkeypatch) -> None:
    pipe = _FakePipe()
    monkeypatch.setattr(gpu_worker, "_pipe", pipe)
    client = TestClient(gpu_worker.app)
    resp = client.post(
        "/generate",
        headers={"X-Worker-Token": "secret-token"},
        json={
            "video_id": "vid",
            "scene_id": "001",
            "prompt": "full prompt",
            "clip_prompt": " ".join(["word"] * 80),
            "negative_prompt": "",
            "width": 1024,
            "height": 576,
            "steps": 20,
            "guidance_scale": 3.5,
            "candidate_seeds": [11],
            "output_format": "WEBP",
            "quality": 92,
        },
    )
    assert resp.status_code == 400
    assert "clip_prompt exceeds 77-token limit" in resp.text


@pytest.mark.parametrize("revision", ["", "main"])
def test_model_revision_preflight_rejects_bad_values(monkeypatch, revision: str) -> None:
    monkeypatch.setenv("HF_MODEL_REVISION", revision)
    with pytest.raises(RuntimeError, match="pinned"):
        # Validate that the gateway still enforces this (gateway imports the validator too)
        from vast_worker.server import _validate_model_revision
        _validate_model_revision()
