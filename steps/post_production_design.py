from __future__ import annotations

from steps import design_effects, design_soundscape


def run(video_id: str) -> None:
    design_soundscape.run(video_id)
    design_effects.run(video_id)
