from __future__ import annotations

from unittest.mock import MagicMock

import requests

from image_generation.schemas import SceneRequest
from image_generation.vast_backend import VastInstanceBackend


def test_vast_backend_sends_clip_prompt_and_worker_auth_headers(monkeypatch) -> None:
    backend = VastInstanceBackend("127.0.0.1", 8080, worker_token="secret-token")
    request = SceneRequest(
        video_id="vid",
        scene_id="001",
        prompt="full prompt",
        clip_prompt="short clip prompt",
        negative_prompt="avoid this",
        candidate_seeds=[11],
    )
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"images": [], "errors": []}

    captured = {}

    def fake_post(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return response

    monkeypatch.setattr(requests, "post", fake_post)

    result = backend.generate(request)

    assert result.candidates == []
    assert captured["kwargs"]["json"]["prompt"] == "full prompt"
    assert captured["kwargs"]["json"]["clip_prompt"] == "short clip prompt"
    assert captured["kwargs"]["headers"] == {
        "Authorization": "Bearer secret-token",
        "X-Worker-Token": "secret-token",
    }
