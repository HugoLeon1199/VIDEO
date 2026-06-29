from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

import config
from steps.creative_package import CreativePackageError, load_validated_package

ALLOWED_MOTIONS = {
    "hold",
    "slow_push_in",
    "slow_pull_out",
    "pan_left_to_right",
    "pan_right_to_left",
}
ALLOWED_TRANSITIONS = {"hard_cut", "crossfade", "dip_to_black"}


def _video_dir(video_id: str) -> Path:
    return Path(config.OUTPUT_DIR) / video_id


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _audio_duration(video_dir: Path) -> float:
    from steps import render_video

    audio_path = video_dir / "audio.mp3"
    if not audio_path.exists():
        return 0.0
    return render_video._get_audio_duration(audio_path)


def _chapter_boundaries(video_dir: Path, prompts: list[dict]) -> set[int]:
    creative_path = video_dir / "creative_package.json"
    if not creative_path.exists():
        return set()
    try:
        package = load_validated_package(video_dir, allow_stale_package=False)
    except (CreativePackageError, FileNotFoundError):
        return set()
    boundaries: set[int] = set()
    for chapter in package.get("chapter_plan", []):
        sentence_index = int(chapter.get("sentence_index", 0))
        for prompt in prompts:
            if int(prompt.get("source_sentence_index", prompt["index"])) >= sentence_index:
                boundaries.add(int(prompt["index"]))
                break
    return boundaries


def _semantic_motion(scene: dict, duration: float) -> str:
    text = " ".join(
        str(scene.get(key, "")).lower()
        for key in ("scene_text", "visual_intent", "prompt")
    )
    if any(token in text for token in ("face", "emotion", "discover", "close-up", "close up", "gương mặt", "cảm xúc")):
        return "slow_push_in"
    if any(token in text for token in ("landscape", "environment", "wide", "forest", "mountain", "cave", "bối cảnh", "toàn cảnh")):
        return "slow_pull_out"
    if any(token in text for token in ("journey", "travel", "walk", "move", "migration", "hành trình", "di chuyển")):
        return "pan_left_to_right"
    if any(token in text for token in ("object", "detail", "artifact", "stat", "number", "tool", "chi tiết")):
        return "hold"
    if duration < 3.0:
        return "hold"
    if duration <= 6.0:
        return "slow_push_in"
    return "slow_pull_out"


def _motion_for_scene(scene: dict, duration: float, prior_motions: list[str]) -> dict[str, Any]:
    motion_type = _semantic_motion(scene, duration)
    if any(token in str(scene.get("scene_text", "")).lower() for token in ("face", "close", "emotion", "gương mặt")) and motion_type.startswith("pan_"):
        motion_type = "slow_push_in"
    if len(prior_motions) >= 3 and all(item == motion_type for item in prior_motions[-3:]):
        motion_type = "hold" if motion_type != "hold" else "slow_push_in"
    if len(prior_motions) >= 1 and motion_type.startswith("pan_") and prior_motions[-1].startswith("pan_") and prior_motions[-1] != motion_type:
        motion_type = "hold"

    focus_x = 0.5
    focus_y = 0.45
    if motion_type == "hold":
        start_scale = end_scale = 1.0 if duration < 3.0 else 1.01
    elif motion_type == "slow_push_in":
        if duration < 3.0:
            start_scale, end_scale = 1.0, 1.02
        elif duration <= 6.0:
            start_scale, end_scale = 1.0, 1.04
        else:
            start_scale, end_scale = 1.0, 1.06
    elif motion_type == "slow_pull_out":
        if duration < 3.0:
            start_scale, end_scale = 1.02, 1.0
        elif duration <= 6.0:
            start_scale, end_scale = 1.04, 1.0
        else:
            start_scale, end_scale = 1.06, 1.0
    else:
        start_scale, end_scale = 1.05, 1.05
    start_scale = max(1.0, min(config.EFFECTS_MAX_SCALE, round(start_scale, 4)))
    end_scale = max(1.0, min(config.EFFECTS_MAX_SCALE, round(end_scale, 4)))
    return {
        "type": motion_type,
        "start_scale": start_scale,
        "end_scale": end_scale,
        "focus_x": focus_x,
        "focus_y": focus_y,
        "easing": "ease_in_out",
    }


def _transition_for_scene(
    scene: dict,
    next_scene: dict | None,
    display_duration: float,
    next_display_duration: float | None,
    chapter_boundaries: set[int],
    crossfade_budget: int,
    dip_budget: int,
    used_crossfades: int,
    used_dips: int,
) -> dict[str, Any]:
    if next_scene is None:
        return {"type": "hard_cut", "duration": 0.0}
    next_duration = next_display_duration or display_duration
    allowed = round(min(display_duration, next_duration) * 0.2, 3)
    if int(next_scene["index"]) in chapter_boundaries and dip_budget > used_dips:
        return {"type": "hard_cut", "duration": 0.0}
    text = " ".join(str(scene.get(key, "")).lower() for key in ("scene_text", "visual_intent"))
    if crossfade_budget > used_crossfades and any(token in text for token in ("soft", "night", "memory", "reflect", "continue", "time", "dream", "calm")):
        if allowed < 0.25:
            return {"type": "hard_cut", "duration": 0.0}
        duration = min(0.35, allowed)
        duration = min(0.4, max(0.25, duration))
        return {"type": "crossfade", "duration": duration}
    return {"type": "hard_cut", "duration": 0.0}


def _validate_effects(prompts: list[dict], plan: dict, audio_duration: float) -> dict[str, Any]:
    warnings: list[str] = []
    scenes = plan.get("scenes", [])
    if len(scenes) != len(prompts):
        raise RuntimeError("effects_plan scene count must match image_prompts.json count")
    for idx, (prompt, scene) in enumerate(zip(prompts, scenes)):
        if int(scene["scene_index"]) != int(prompt["index"]):
            raise RuntimeError(f"effects scene_index mismatch for scene {prompt['index']}")
        if round(float(scene["source_start"]), 3) != round(float(prompt["start"]), 3):
            raise RuntimeError(f"effects source_start mismatch for scene {prompt['index']}")
        if round(float(scene["source_end"]), 3) != round(float(prompt["end"]), 3):
            raise RuntimeError(f"effects source_end mismatch for scene {prompt['index']}")
        motion = scene["motion"]
        if motion["type"] not in ALLOWED_MOTIONS:
            raise RuntimeError(f"Invalid motion type: {motion['type']}")
        if scene["transition_out"]["type"] not in ALLOWED_TRANSITIONS:
            raise RuntimeError(f"Invalid transition type: {scene['transition_out']['type']}")
        for key in ("focus_x", "focus_y"):
            if not 0.0 <= float(motion[key]) <= 1.0:
                raise RuntimeError(f"Invalid focal point {key} in scene {scene['scene_index']}")
        max_scale = max(float(motion["start_scale"]), float(motion["end_scale"]))
        min_scale = min(float(motion["start_scale"]), float(motion["end_scale"]))
        if max_scale > config.EFFECTS_MAX_SCALE:
            raise RuntimeError(f"Scale exceeds max for scene {scene['scene_index']}")
        if min_scale < 1.0:
            raise RuntimeError(f"Scale drops below 1.0 for scene {scene['scene_index']}")
        if motion["type"].startswith("pan_") and max_scale < 1.05:
            raise RuntimeError(f"Pan scene {scene['scene_index']} must use overscan scale")
        if float(scene["display_end"]) <= float(scene["display_start"]):
            raise RuntimeError(f"Invalid display range in scene {scene['scene_index']}")
        transition = scene["transition_out"]
        if transition["type"] != "hard_cut":
            if transition["duration"] > (scene["display_end"] - scene["display_start"]) * 0.2 + 1e-6:
                raise RuntimeError(f"Transition too long for scene {scene['scene_index']}")
            next_scene = scenes[idx + 1] if idx + 1 < len(scenes) else None
            if next_scene is not None and transition["duration"] > (float(next_scene["display_end"]) - float(next_scene["display_start"])) * 0.2 + 1e-6:
                raise RuntimeError(f"Transition too long for adjacent scene {next_scene['scene_index']}")
    if plan.get("effects_enabled", True) is False:
        if plan.get("global_look", {}).get("enabled", False):
            raise RuntimeError("effects disabled plan must also disable look")
        for scene in scenes:
            motion = scene["motion"]
            if motion["type"] != "hold" or float(motion["start_scale"]) != 1.0 or float(motion["end_scale"]) != 1.0:
                raise RuntimeError("effects disabled plan must be static")
            if scene["transition_out"]["type"] != "hard_cut" or float(scene["transition_out"]["duration"]) != 0.0:
                raise RuntimeError("effects disabled plan must use hard cuts only")
    duration_drift = round(float(scenes[-1]["display_end"]) - float(audio_duration), 6) if scenes else 0.0
    if abs(duration_drift) > config.EFFECTS_DURATION_TOLERANCE_SECONDS:
        raise RuntimeError(f"Display timeline drifts from audio by {duration_drift:.3f}s")
    motion_distribution = Counter(scene["motion"]["type"] for scene in scenes)
    transition_distribution = Counter(scene["transition_out"]["type"] for scene in scenes[:-1])
    total_transitions = max(1, len(scenes) - 1)
    hard_ratio = transition_distribution.get("hard_cut", 0) / total_transitions
    cross_ratio = transition_distribution.get("crossfade", 0) / total_transitions
    dip_ratio = transition_distribution.get("dip_to_black", 0) / total_transitions
    if not (0.75 <= hard_ratio <= 0.85):
        warnings.append("hard_cut ratio deviates from target distribution")
    if transition_distribution.get("crossfade", 0) and not (0.15 <= cross_ratio <= 0.20):
        warnings.append("crossfade ratio deviates from target distribution")
    if dip_ratio > 0.05:
        warnings.append("dip_to_black ratio exceeds target maximum")
    max_scale = max(max(float(scene["motion"]["start_scale"]), float(scene["motion"]["end_scale"])) for scene in scenes) if scenes else 1.0
    max_pan = config.EFFECTS_MAX_PAN if any(scene["motion"]["type"].startswith("pan_") for scene in scenes) else 0.0
    return {
        "scene_count": len(scenes),
        "motion_distribution": dict(motion_distribution),
        "transition_distribution": dict(transition_distribution),
        "max_scale": round(max_scale, 4),
        "max_pan": round(max_pan, 4),
        "duration_drift": duration_drift,
        "validation_passed": True,
        "warnings": warnings,
    }


def build_effects_plan(video_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    video_dir = _video_dir(video_id)
    prompts_path = video_dir / "image_prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(f"Missing image_prompts.json: {prompts_path}")
    prompts = _load_json(prompts_path)
    audio_duration = _audio_duration(video_dir)
    chapter_boundaries = _chapter_boundaries(video_dir, prompts)
    total_transitions = max(0, len(prompts) - 1)
    crossfade_budget = 0 if total_transitions < 5 else max(1, round(total_transitions * 0.18))
    dip_budget = 0
    scenes: list[dict[str, Any]] = []
    prior_motions: list[str] = []
    used_crossfades = 0
    used_dips = 0
    effects_enabled = bool(config.EFFECTS_ENABLED)
    for idx, prompt in enumerate(prompts):
        source_start = round(float(prompt["start"]), 3)
        source_end = round(float(prompt["end"]), 3)
        display_start = 0.0 if idx == 0 else source_start
        next_scene = prompts[idx + 1] if idx + 1 < len(prompts) else None
        display_end = round(float(next_scene["start"]), 3) if next_scene is not None else round(audio_duration, 3)
        display_duration = max(0.1, display_end - display_start)
        next_display_duration = None
        if next_scene is not None:
            next_display_duration = (
                round(float(prompts[idx + 2]["start"]), 3) if idx + 2 < len(prompts) else round(audio_duration, 3)
            ) - round(float(next_scene["start"]), 3)
        motion = _motion_for_scene(prompt, display_duration, prior_motions) if effects_enabled else {
            "type": "hold",
            "start_scale": 1.0,
            "end_scale": 1.0,
            "focus_x": 0.5,
            "focus_y": 0.45,
            "easing": "ease_in_out",
        }
        transition = _transition_for_scene(
            prompt,
            next_scene,
            display_duration,
            next_display_duration,
            chapter_boundaries,
            crossfade_budget,
            dip_budget,
            used_crossfades,
            used_dips,
        ) if effects_enabled else {"type": "hard_cut", "duration": 0.0}
        if transition["type"] == "crossfade":
            used_crossfades += 1
        elif transition["type"] == "dip_to_black":
            used_dips += 1
        scene_entry = {
            "scene_index": int(prompt["index"]),
            "source_sentence_index": int(prompt.get("source_sentence_index", prompt["index"])),
            "source_start": source_start,
            "source_end": source_end,
            "display_start": round(display_start, 3),
            "display_end": round(display_end, 3),
            "motion": motion,
            "transition_out": transition,
        }
        prior_motions.append(motion["type"])
        scenes.append(scene_entry)
    plan = {
        "version": "cinematic-documentary-v1",
        "global_look": {
            "preset": config.EFFECTS_LOOK_PRESET,
            "grade": config.EFFECTS_DEFAULT_GRADE,
            "grain": config.EFFECTS_DEFAULT_GRAIN if effects_enabled and bool(config.EFFECTS_LOOK_ENABLED) else 0.0,
            "vignette": config.EFFECTS_DEFAULT_VIGNETTE if effects_enabled and bool(config.EFFECTS_LOOK_ENABLED) else 0.0,
            "enabled": bool(effects_enabled and config.EFFECTS_LOOK_ENABLED),
        },
        "effects_enabled": effects_enabled,
        "scenes": scenes,
    }
    diagnostics = _validate_effects(prompts, plan, audio_duration)
    return plan, diagnostics


def run(video_id: str) -> None:
    video_dir = _video_dir(video_id)
    plan, diagnostics = build_effects_plan(video_id)
    _write_json(video_dir / "effects_plan.json", plan)
    _write_json(video_dir / "effects_diagnostics.json", diagnostics)
    logger.info("Effects plan saved: {} scenes -> {}", diagnostics["scene_count"], video_dir / "effects_plan.json")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m steps.design_effects <video-id>")
    run(sys.argv[1])
