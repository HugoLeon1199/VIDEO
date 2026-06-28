from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

import config
from steps.creative_package import _atomic_write_json, load_validated_package
from steps.text_units import load_sentence_units


AUTOPILOT_STAGES = [
    "ingest",
    "creative_package",
    "tts",
    "transcribe",
    "image_prompts",
    "images",
    "soundscape",
    "render",
    "publishing",
]


def _video_dir(video_id: str) -> Path:
    return Path(config.OUTPUT_DIR) / video_id


def _state_path(video_id: str) -> Path:
    return _video_dir(video_id) / "autopilot_state.json"


def _summary_path(video_id: str) -> Path:
    return _video_dir(video_id) / "autopilot_summary.json"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_script_text(text: str) -> str:
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\u00a0", " ")
    text = text.replace("—", ", ").replace("–", "-")
    lines = [re.sub(r"[ \t]+$", "", line) for line in text.splitlines()]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized.strip() + "\n"


def detect_language(script_text: str) -> str:
    lowered = script_text.lower()
    vi_chars = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    hits = sum(1 for ch in lowered if ch in vi_chars)
    letters = sum(1 for ch in lowered if ch.isalpha())
    if hits >= 5 or (letters > 0 and hits / letters >= 0.01):
        return "vi"
    return "en"


def safe_video_id(script_text: str) -> str:
    text = script_text.lstrip("\ufeff").strip().splitlines()[0] if script_text.strip() else "video"
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[_\s]+", "-", text, flags=re.UNICODE).strip("-")
    text = text[:80] or "video"
    return text


def _write_json(path: Path, payload: dict) -> None:
    _atomic_write_json(path, payload)


def _build_tts_config(language: str) -> dict:
    if language == "vi":
        return {"engine": "vieneu", "voice": "Thái Sơn", "mode": "block"}
    return {"engine": "kokoro", "voice": "am_fenrir", "speed": 0.95, "mode": "block"}


def _build_transcribe_config(language: str) -> dict:
    return {
        "engine": "stable_ts",
        "model": "medium" if language == "vi" else "base",
        "language": language,
        "mode": "align",
        "device": "cpu",
    }


def _keyword_candidates(script_text: str, language: str) -> list[str]:
    if language == "vi":
        stop_words = {"và", "là", "của", "những", "một", "bạn", "trong", "đó", "khi", "đã", "với"}
    else:
        stop_words = {"the", "and", "that", "with", "from", "into", "your", "this", "have", "were"}
    counts: dict[str, int] = {}
    for token in re.findall(r"[\wÀ-ỹ]+", script_text.lower()):
        if len(token) < 4 or token in stop_words:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [item[0] for item in ranked[:8]] or (["history", "ancient", "mystery"] if language == "en" else ["lich su", "co dai", "bi an"])


def generate_creative_package(script_text: str, language: str, sentence_count: int) -> dict:
    keywords = _keyword_candidates(script_text, language)
    lead = script_text.strip().splitlines()[0].strip().rstrip(".!?")
    if language == "vi":
        titles = [
            {"id": "title_1", "angle": "curiosity", "text": f"{lead}: điều gì thật sự đã xảy ra?"},
            {"id": "title_2", "angle": "discovery", "text": f"Sự thật ít người biết về {keywords[0]}"},
            {"id": "title_3", "angle": "emotion", "text": f"Nếu bạn sống trong {keywords[0]}, bạn sẽ ra sao?"},
        ]
        description = " ".join(script_text.strip().split()[:120]).strip()
        thumb_texts = ["AI ĐÃ LÀM", "BÊN TRONG HANG", "QUÁ LỚN SAO", "DẤU VẾT CŨ", "ĐIỀU BỊ GIẤU"]
    else:
        titles = [
            {"id": "title_1", "angle": "curiosity", "text": f"{lead}: what really happened?"},
            {"id": "title_2", "angle": "discovery", "text": f"The hidden truth about {keywords[0]}"},
            {"id": "title_3", "angle": "emotion", "text": f"If you lived through {keywords[0]}, what would break first?"},
        ]
        description = " ".join(script_text.strip().split()[:120]).strip()
        thumb_texts = ["WHO DID THIS", "INSIDE THE CAVE", "TOO BIG", "AN OLD TRACE", "WHAT THEY HID"]
    concept_types = ["human_closeup", "human_closeup", "mystery_reveal", "mystery_reveal", "scale_or_danger"]
    concepts = []
    for index, concept_type in enumerate(concept_types, start=1):
        concepts.append(
            {
                "id": index,
                "type": concept_type,
                "visual_hook": keywords[min(index - 1, len(keywords) - 1)],
                "emotional_goal": ["shock", "wonder", "mystery", "discovery", "danger"][index - 1],
                "thumbnail_text": thumb_texts[index - 1],
                "subject_side": "left" if index % 2 else "right",
                "text_side": "right" if index % 2 else "left",
                "paired_title_ids": [titles[min(index - 1, 2)]["id"]],
                "must_show": [],
                "must_avoid": ["text", "logo", "watermark"],
            }
        )
    chapter_points = sorted({1, max(2, sentence_count // 3), max(3, (sentence_count * 2) // 3), sentence_count})
    chapter_plan = [
        {"sentence_index": idx, "label": f"Chapter {i + 1}"}
        for i, idx in enumerate(chapter_points[:4])
    ]
    return {
        "package_version": "creative-package-v1",
        "language": language,
        "core_promise": lead,
        "target_viewer": "curious history viewers",
        "primary_hook": lead,
        "title_options": titles,
        "description_draft": description,
        "search_keywords": keywords[:6],
        "chapter_plan": chapter_plan,
        "thumbnail_concepts": concepts,
    }


def _state_template(video_id: str, script_sha256: str, language: str) -> dict:
    return {
        "video_id": video_id,
        "script_sha256": script_sha256,
        "language": language,
        "stages": {stage: {"status": "pending"} for stage in AUTOPILOT_STAGES},
        "warnings": [],
        "vast_teardown_confirmed": False,
    }


def _load_state(video_id: str) -> dict | None:
    path = _state_path(video_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_completed(state: dict, stage: str) -> bool:
    return state.get("stages", {}).get(stage, {}).get("status") == "completed"


def _update_stage(state: dict, video_id: str, stage: str, status: str, **extra) -> None:
    state["stages"][stage] = {"status": status, **extra}
    _write_json(_state_path(video_id), state)


def _probe_media(path: Path) -> bool:
    if not path.exists():
        return False
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _audio_duration(video_dir: Path) -> float:
    audio_path = video_dir / "audio.mp3"
    if not audio_path.exists():
        return 0.0
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _render_sizes(video_dir: Path) -> dict:
    result = {}
    for name in ("final.mp4", "final_subbed.mp4"):
        path = video_dir / name
        result[name] = path.stat().st_size if path.exists() else 0
    return result


def _validate_resume(video_id: str, script_sha256: str) -> dict:
    state = _load_state(video_id)
    if not state:
        raise RuntimeError(f"autopilot_state.json not found for resume: {video_id}")
    if state.get("script_sha256") != script_sha256:
        raise RuntimeError(f"Resume blocked: script hash changed for {video_id}")
    return state


def _ensure_output_slot(video_id: str, script_sha256: str, language: str, resume: bool) -> dict:
    video_dir = _video_dir(video_id)
    if resume:
        return _validate_resume(video_id, script_sha256)
    if video_dir.exists() and any(video_dir.iterdir()):
        state = _load_state(video_id)
        allowed_manual_inputs = {"creative_package.json"}
        existing_names = {path.name for path in video_dir.iterdir()}
        if state and state.get("script_sha256") == script_sha256:
            return state
        if existing_names.issubset(allowed_manual_inputs):
            state = _state_template(video_id, script_sha256, language)
            _write_json(_state_path(video_id), state)
            return state
        if not state or state.get("script_sha256") != script_sha256:
            raise RuntimeError(f"Refusing to overwrite existing video folder: {video_dir}")
    video_dir.mkdir(parents=True, exist_ok=True)
    state = _state_template(video_id, script_sha256, language)
    _write_json(_state_path(video_id), state)
    return state


def _run_image_generation_cli(video_id: str, language: str) -> None:
    python = sys.executable
    env = os.environ.copy()
    env["IMAGE_BACKEND"] = "vast_instance"
    track = language if language in {"vi", "en"} else "en"
    cmd = [
        python,
        "scripts/generate_images.py",
        "--video-id",
        video_id,
        "--backend",
        "vast_instance",
        "--workers",
        "1",
        "--output-root",
        str(config.OUTPUT_DIR),
    ]
    if track:
        cmd.extend(["--track", track])
    logger.info("Autopilot image generation command: {}", " ".join(cmd))
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])


def _promote_canonical_images(video_id: str) -> int:
    video_dir = _video_dir(video_id)
    canonical_dir = video_dir / "images"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    promoted = 0
    for track_dir in (video_dir / "images_vi", video_dir / "images_en"):
        if not track_dir.exists():
            continue
        for source in track_dir.glob("img_*.png"):
            target = canonical_dir / source.name
            if target.exists():
                continue
            shutil.copy2(source, target)
            promoted += 1
    return promoted


def run(video_id: str, script_file: str, resume: bool = False) -> dict:
    from steps import design_soundscape, image_prompts, metadata, render_video, thumbnails, transcribe, tts

    raw_script = Path(script_file).read_text(encoding="utf-8")
    normalized_script = normalize_script_text(raw_script)
    language = detect_language(normalized_script)
    script_sha256 = _sha256_text(normalized_script)
    state = _ensure_output_slot(video_id, script_sha256, language, resume=resume)
    video_dir = _video_dir(video_id)

    try:
        if resume:
            promoted = _promote_canonical_images(video_id)
            if promoted:
                logger.info("Promoted {} track images into canonical images/", promoted)

        if not _stage_completed(state, "ingest"):
            _update_stage(state, video_id, "ingest", "running")
            (video_dir / "script.txt").write_text(normalized_script, encoding="utf-8")
            _write_json(video_dir / "tts_config.json", _build_tts_config(language))
            _write_json(video_dir / "transcribe_config.json", _build_transcribe_config(language))
            _update_stage(state, video_id, "ingest", "completed")

        if not _stage_completed(state, "creative_package"):
            _update_stage(state, video_id, "creative_package", "running")
            creative_path = video_dir / "creative_package.json"
            if creative_path.exists():
                load_validated_package(video_dir, allow_stale_package=False)
            else:
                package = generate_creative_package(normalized_script, language, len(load_sentence_units(video_dir / "script.txt")))
                _write_json(creative_path, package)
                load_validated_package(video_dir, allow_stale_package=False)
            _update_stage(state, video_id, "creative_package", "completed")

        if not _stage_completed(state, "tts"):
            _update_stage(state, video_id, "tts", "running")
            tts.run(video_id)
            _update_stage(state, video_id, "tts", "completed")

        if not _stage_completed(state, "transcribe"):
            _update_stage(state, video_id, "transcribe", "running")
            transcribe.run(video_id)
            word_diag_path = video_dir / transcribe.WORD_DIAGNOSTICS_NAME
            if word_diag_path.exists():
                diagnostics = json.loads(word_diag_path.read_text(encoding="utf-8"))
                if not diagnostics.get("subtitle_ready"):
                    transcribe.run(video_id)
            _update_stage(state, video_id, "transcribe", "completed")

        if not _stage_completed(state, "image_prompts"):
            _update_stage(state, video_id, "image_prompts", "running")
            image_prompts.run(video_id)
            _update_stage(state, video_id, "image_prompts", "completed")

        if not _stage_completed(state, "images"):
            _update_stage(state, video_id, "images", "running")
            try:
                _run_image_generation_cli(video_id, language)
                promoted = _promote_canonical_images(video_id)
                if promoted:
                    logger.info("Promoted {} track images into canonical images/", promoted)
                old_backend = config.IMAGE_BACKEND
                try:
                    config.IMAGE_BACKEND = "vast_instance"
                    thumbnails.generate_thumbnail_assets(video_id)
                finally:
                    config.IMAGE_BACKEND = old_backend
            finally:
                state["vast_teardown_confirmed"] = True
                _write_json(_state_path(video_id), state)
            _update_stage(state, video_id, "images", "completed")

        if not _stage_completed(state, "soundscape"):
            _update_stage(state, video_id, "soundscape", "running")
            design_soundscape.run(video_id)
            _update_stage(state, video_id, "soundscape", "completed")

        if not _stage_completed(state, "render"):
            _update_stage(state, video_id, "render", "running")
            render_video.run(video_id, subtitles=True)
            _update_stage(state, video_id, "render", "completed")

        if not _stage_completed(state, "publishing"):
            _update_stage(state, video_id, "publishing", "running")
            metadata.run(video_id)
            _update_stage(state, video_id, "publishing", "completed")
    except Exception as exc:
        logger.error("Autopilot failed: {}", exc)
        failing_stage = next((stage for stage, entry in state["stages"].items() if entry["status"] == "running"), "unknown")
        _update_stage(state, video_id, failing_stage, "failed", error=str(exc))
        raise

    summary = {
        "video_id": video_id,
        "language": language,
        "tts_engine": _build_tts_config(language)["engine"],
        "voice": _build_tts_config(language)["voice"],
        "audio_duration": round(_audio_duration(video_dir), 3),
        "sentence_count": len(json.loads((video_dir / "timestamps.json").read_text(encoding="utf-8"))),
        "visual_scene_count": len(json.loads((video_dir / "image_prompts.json").read_text(encoding="utf-8"))),
        "image_count": len(list((video_dir / "images").glob("img_*.png"))),
        "image_regenerated_count": 0,
        "image_failed_count": 0,
        "subtitle_validation_status": json.loads((video_dir / transcribe.WORD_DIAGNOSTICS_NAME).read_text(encoding="utf-8")).get("subtitle_ready", False),
        "final_video_sizes": _render_sizes(video_dir),
        "thumbnail_path": str(video_dir / config.PUBLISHING_DIRNAME / "thumbnail_contact_sheet.jpg"),
        "publishing_title_path": str(video_dir / config.PUBLISHING_DIRNAME / "title_options.txt"),
        "publishing_description_path": str(video_dir / config.PUBLISHING_DIRNAME / "description.txt"),
        "vast_teardown_confirmed": state.get("vast_teardown_confirmed", False),
    }
    _write_json(_summary_path(video_id), summary)
    return summary
