from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

from loguru import logger

import config
from image_generation.flux_prompting import normalize_prompt_text, token_count_for_model
from steps.creative_package import CreativePackageError, _atomic_write_json, load_validated_package
from steps.text_units import load_script_text, load_sentence_units
from steps import visual_beats


def _load_prompt_template(language: str) -> dict:
    meta = visual_beats.prompt_template_metadata(language)
    text = meta["text"]
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "path": meta["path"],
        "text": text,
        "fields": meta["fields"],
        "sha256": sha,
    }


def _looks_vietnamese(text: str) -> bool:
    vi_chars = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    lowered = text.lower()
    hits = sum(1 for ch in lowered if ch in vi_chars)
    letters = sum(1 for ch in lowered if ch.isalpha())
    return hits >= 5 or (letters > 0 and hits / letters >= 0.01)


def _select_language(video_id: str, script_text: str) -> str:
    if video_id.lower().endswith("-vi") or _looks_vietnamese(script_text):
        return "vi"
    return "en"


def _get_audio_duration(audio_path: Path) -> float:
    import subprocess

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _parse_json_response(text: str):
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in response")
    return json.loads(text[start:end])


def _build_system_prompt(language: str, template_text: str) -> str:
    language_name = "Vietnamese" if language == "vi" else "English"
    return (
        f"You are a semantic storyboard planner for a {language_name} history video.\n"
        "Return a pure JSON array only.\n"
        "You may create 1 to 3 visual beats per sentence, but only when there is a real scene shift.\n"
        "Each item must contain: source_sentence_index, beat_index, word_start, word_end, scene_text, visual_intent, prompt.\n"
        "Rules:\n"
        "- word_start and word_end are word indices inside that single sentence only.\n"
        "- Beats for each sentence must be contiguous, ordered, and cover the whole sentence exactly once.\n"
        "- Do not create tiny filler beats.\n"
        "- Do not split only because the sentence is long.\n"
        "- prompt must follow the production image template below.\n\n"
        "PRODUCTION TEMPLATE SOURCE OF TRUTH:\n"
        f"{template_text}"
    )


def _call_claude(script_text: str, sentence_payload: list[dict], system_prompt: str) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": json.dumps({"script": script_text, "sentences": sentence_payload}, ensure_ascii=False, indent=2)}],
    )
    return _parse_json_response(response.content[0].text)


def _call_gemini(script_text: str, sentence_payload: list[dict], system_prompt: str) -> list[dict]:
    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_TEXT_MODEL,
        contents=f"{system_prompt}\n\n{json.dumps({'script': script_text, 'sentences': sentence_payload}, ensure_ascii=False, indent=2)}",
    )
    return _parse_json_response(response.text)


def _sentence_payload(sentence_spans: list[visual_beats.SentenceSpan]) -> list[dict]:
    payload: list[dict] = []
    for item in sentence_spans:
        payload.append(
            {
                "sentence_index": item.sentence_index,
                "text": item.text,
                "start": item.start,
                "end": item.end,
                "word_start": item.word_start,
                "word_end": item.word_end,
            }
        )
    return payload


def _deterministic_prompt(language: str, text: str) -> str:
    if language == "vi":
        return (
            "cinematic 2D painted documentary illustration, semi-realistic prehistoric humans, "
            "all visible people fully clothed in historically plausible full-coverage hide and fur clothing, "
            "warm earthy documentary lighting, anatomically coherent figures, no text, no logo, no watermark, "
            f"visualize this narration beat: {text}"
        )
    return (
        "Ink sketch illustration on aged parchment paper, cinematic illustrated history documentary, "
        "all visible people fully clothed in historically plausible full-coverage hide and fur clothing, "
        "correct human anatomy, no extra limbs, no written labels, no letters or numbers, no text, no logo, no watermark, "
        f"visualize this narration beat: {text}"
    )


def _clip_prompt_prefix(language: str) -> str:
    if language == "vi":
        return (
            "cinematic 2D painted documentary illustration, all visible people fully clothed in historically "
            "plausible full-coverage hide and fur clothing, no text, no logo, no watermark,"
        )
    return (
        "cinematic illustrated history documentary, all visible people fully clothed in historically plausible "
        "full-coverage hide and fur clothing, no text, no logo, no watermark,"
        )


def _condense_clip_scene_text(scene_text: str, *, max_words: int = 24, max_chars: int = 180) -> str:
    words = [part for part in re.split(r"\s+", scene_text.strip()) if part]
    condensed = " ".join(words[:max_words])
    if len(condensed) > max_chars:
        condensed = condensed[:max_chars].rsplit(" ", 1)[0].strip()
    return condensed or scene_text.strip()


def _build_clip_prompt(language: str, scene_text: str) -> str:
    condensed_scene_text = _condense_clip_scene_text(scene_text)
    clip_prompt = normalize_prompt_text(f"{_clip_prompt_prefix(language)} {condensed_scene_text}".strip())
    return clip_prompt


def _estimated_token_count(text: str) -> int:
    words = [part for part in re.split(r"\s+", text.strip()) if part]
    compact_len = len(re.sub(r"\s+", "", text))
    return max(len(words) + 4, max(1, (compact_len + 4) // 5))


def _should_fallback_token_count(exc: Exception) -> bool:
    message = str(exc).lower()
    fallback_markers = (
        "gated repo",
        "401",
        "403",
        "repository not found",
        "could not connect",
        "connection",
        "timeout",
        "temporary failure",
    )
    return any(marker in message for marker in fallback_markers)


def _count_tokens_with_fallback(
    model_id: str,
    text: str,
    subfolder: str,
    revision: str,
    *,
    scene_index: int,
) -> tuple[int, str]:
    try:
        return token_count_for_model(model_id, text, subfolder, revision), "tokenizer"
    except Exception as exc:
        if not _should_fallback_token_count(exc):
            raise
        estimate = _estimated_token_count(text)
        logger.warning(
            "Scene {} {} token count fell back to heuristic because tokenizer access failed: {}",
            scene_index,
            subfolder,
            exc,
        )
        return estimate, "heuristic"


def _deterministic_plan(sentence_spans: list[visual_beats.SentenceSpan], language: str) -> list[dict]:
    plan: list[dict] = []
    for span in sentence_spans:
        word_start = int(span.word_start or 1)
        word_end = int(span.word_end or word_start)
        plan.append(
            {
                "source_sentence_index": span.sentence_index,
                "beat_index": 1,
                "word_start": word_start,
                "word_end": word_end,
                "scene_text": span.text,
                "visual_intent": span.text,
                "prompt": _deterministic_prompt(language, span.text),
            }
        )
    return plan


def _normalize_model_beats(raw_items: list[dict], sentence_spans: list[visual_beats.SentenceSpan]) -> list[dict]:
    sentence_map = {item.sentence_index: item for item in sentence_spans}
    normalized: list[dict] = []
    per_sentence_counts: dict[int, int] = {}
    for item in raw_items:
        if "source_sentence_index" in item:
            sentence_index = int(item["source_sentence_index"])
            beat_index = int(item.get("beat_index", per_sentence_counts.get(sentence_index, 0) + 1))
            word_start = int(item["word_start"])
            word_end = int(item["word_end"])
            scene_text = str(item.get("scene_text", sentence_map[sentence_index].text)).strip()
            visual_intent = str(item.get("visual_intent", scene_text)).strip()
            prompt = str(item.get("prompt", visual_intent)).strip()
        else:
            sentence_indices = item.get("sentences", [])
            if len(sentence_indices) != 1:
                raise ValueError(f"Legacy scene entry must map to exactly one sentence: {item}")
            sentence_index = int(sentence_indices[0])
            sentence = sentence_map[sentence_index]
            beat_index = 1
            word_start = int(sentence.word_start or 1)
            word_end = int(sentence.word_end or word_start)
            scene_text = str(item.get("scene_text", sentence.text)).strip()
            visual_intent = scene_text
            prompt = str(item.get("prompt", scene_text)).strip()
        per_sentence_counts[sentence_index] = beat_index
        normalized.append(
            {
                "source_sentence_index": sentence_index,
                "beat_index": beat_index,
                "word_start": word_start,
                "word_end": word_end,
                "scene_text": scene_text,
                "visual_intent": visual_intent,
                "prompt": prompt,
            }
        )
    return normalized


def _attach_generation_fields(
    beats: list[dict],
    *,
    language: str,
    template: dict,
) -> list[dict]:
    track_cfg = config.TRACK_CONFIG[language]
    revision = config.require_pinned_hf_model_revision()
    model_id = track_cfg["model"]
    template_version = template["fields"].get("style_version", track_cfg["style_version"])
    entries: list[dict] = []
    for beat in beats:
        scene_index = int(beat["source_sentence_index"])
        scene_text = normalize_prompt_text(str(beat.get("scene_text", beat["visual_intent"]))).strip()
        final_prompt = normalize_prompt_text(str(beat.get("prompt", beat["visual_intent"])).strip())
        clip_prompt = _build_clip_prompt(language, scene_text)
        negative_prompt = (
            "Avoid: text, letters, logo, watermark, malformed anatomy, extra limbs, merged bodies, "
            "modern objects, sexualized body, bare torso, bare chest, exposed breasts."
        )
        clip_token_count, clip_token_count_mode = _count_tokens_with_fallback(
            model_id,
            clip_prompt,
            "tokenizer",
            revision,
            scene_index=scene_index,
        )
        t5_prompt = final_prompt
        if negative_prompt:
            t5_prompt = f"{t5_prompt}. {negative_prompt}"
        t5_token_count, t5_token_count_mode = _count_tokens_with_fallback(
            model_id,
            t5_prompt,
            "tokenizer_2",
            revision,
            scene_index=scene_index,
        )
        if clip_token_count > config.FLUX_CLIP_TOKEN_LIMIT:
            raise ValueError(
                f"clip_prompt exceeds {config.FLUX_CLIP_TOKEN_LIMIT} tokens ({clip_token_count}) for scene {beat['source_sentence_index']}"
            )
        if t5_token_count > config.FLUX_T5_TOKEN_LIMIT:
            raise ValueError(
                f"prompt_2 exceeds {config.FLUX_T5_TOKEN_LIMIT} tokens ({t5_token_count}) for scene {beat['source_sentence_index']}"
            )
        entry = dict(beat)
        entry.update(
            {
                "scene_text": scene_text,
                "prompt": final_prompt,
                "clip_prompt": clip_prompt,
                "negative_prompt": negative_prompt,
                "track": language,
                "template_sha256": template["sha256"],
                "style_version": track_cfg["style_version"],
                "template_version": template_version,
                "image_model": track_cfg["model"],
                "final_positive_prompt": final_prompt,
                "final_avoidance_prompt": negative_prompt,
                "clip_token_count": clip_token_count,
                "clip_token_count_mode": clip_token_count_mode,
                "clip_limit": config.FLUX_CLIP_TOKEN_LIMIT,
                "t5_token_count": t5_token_count,
                "t5_token_count_mode": t5_token_count_mode,
                "t5_limit": config.FLUX_T5_TOKEN_LIMIT,
                "unicode_valid": True,
                "steps": track_cfg["steps"],
                "guidance": track_cfg["guidance_scale"],
                "resolution": f"{config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT}",
                "width": config.IMAGE_WIDTH,
                "height": config.IMAGE_HEIGHT,
                "guidance_scale": track_cfg["guidance_scale"],
            }
        )
        entries.append(entry)
    return entries


def _generate_thumbnail_prompt_payload(video_id: str, validated_package: dict, use_claude: bool) -> None:
    publishing_dir = Path(config.OUTPUT_DIR) / video_id / config.PUBLISHING_DIRNAME
    output_path = publishing_dir / "thumbnail_prompts.json"
    diagnostics_path = publishing_dir / "thumbnail_prompt_diagnostics.json"
    system_prompt = (
        "You convert saved YouTube thumbnail strategy into technical image prompts. "
        "Keep the concept unchanged. Return strict JSON object with a single key "
        "`thumbnail_prompts` containing one entry per concept. Each entry must include "
        "concept_id, type, image_prompt, negative_prompt, thumbnail_text, subject_side, "
        "text_side, paired_title_ids. The image_prompt must enforce 16:9 composition, "
        "one strong focal subject, clean negative space on text_side, thumbnail-style "
        "contrast, no text, no letters, no logo, no watermark, no extra limbs, no "
        "malformed hands, and no merged bodies."
    )
    payload = {
        "language": validated_package["language"],
        "image_model": config.CLAUDE_MODEL if use_claude else config.GEMINI_TEXT_MODEL,
        "track_config": {
            "width": config.IMAGE_WIDTH,
            "height": config.IMAGE_HEIGHT,
            "steps": config.IMAGE_STEPS,
            "guidance_scale": config.IMAGE_GUIDANCE_SCALE,
        },
        "title_options": validated_package["title_options"],
        "thumbnail_concepts": validated_package["thumbnail_concepts"],
    }
    try:
        if use_claude:
            import anthropic

            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=6000,
                system=system_prompt,
                messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}],
            )
            raw = response.content[0].text
        else:
            from google import genai

            client = genai.Client(api_key=config.GEMINI_API_KEY)
            response = client.models.generate_content(
                model=config.GEMINI_TEXT_MODEL,
                contents=f"{system_prompt}\n\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
            )
            raw = response.text
        parsed = json.loads(re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip())
        entries = parsed.get("thumbnail_prompts", [])
        concepts = validated_package["thumbnail_concepts"]
        if not isinstance(entries, list) or len(entries) != len(concepts):
            raise ValueError("thumbnail_prompts must match the creative_package concept count")
        concept_map = {int(item["id"]): item for item in concepts}
        normalized: list[dict] = []
        for entry in entries:
            concept_id = int(entry["concept_id"])
            concept = concept_map.get(concept_id)
            if concept is None:
                raise ValueError(f"Unknown concept_id in thumbnail prompt output: {concept_id}")
            image_prompt = str(entry.get("image_prompt", "")).strip()
            clip_prompt = normalize_prompt_text(
                f"YouTube thumbnail, {concept['type'].replace('_', ' ')}, {concept['thumbnail_text']}"
            )
            prompt_lower = image_prompt.lower()
            for required in ("no text", "no logo", "no watermark"):
                if required not in prompt_lower:
                    raise ValueError(f"thumbnail prompt {concept_id} must contain '{required}'")
            normalized.append(
                {
                    "concept_id": concept_id,
                    "type": concept["type"],
                    "clip_prompt": clip_prompt,
                    "image_prompt": image_prompt,
                    "negative_prompt": str(entry.get("negative_prompt", "")).strip(),
                    "thumbnail_text": concept["thumbnail_text"],
                    "subject_side": concept["subject_side"],
                    "text_side": concept["text_side"],
                    "paired_title_ids": concept["paired_title_ids"],
                }
            )
        _atomic_write_json(output_path, normalized)
        _atomic_write_json(
            diagnostics_path,
            {
                "package_version": validated_package["package_version"],
                "script_sha256": validated_package["script_sha256"],
                "language": validated_package["language"],
                "concept_count": len(concepts),
                "thumbnail_prompt_count": len(normalized),
                "validation_passed": True,
                "warnings": [],
            },
        )
    except Exception as exc:
        _atomic_write_json(
            diagnostics_path,
            {
                "package_version": validated_package["package_version"],
                "script_sha256": validated_package["script_sha256"],
                "language": validated_package["language"],
                "concept_count": len(validated_package["thumbnail_concepts"]),
                "thumbnail_prompt_count": 0,
                "validation_passed": False,
                "warnings": [str(exc)],
            },
        )
        logger.warning("Thumbnail prompt generation failed: {}", exc)


def _generate_deterministic_thumbnail_prompts(video_id: str, validated_package: dict) -> None:
    publishing_dir = Path(config.OUTPUT_DIR) / video_id / config.PUBLISHING_DIRNAME
    output_path = publishing_dir / "thumbnail_prompts.json"
    diagnostics_path = publishing_dir / "thumbnail_prompt_diagnostics.json"
    entries: list[dict] = []
    for concept in validated_package["thumbnail_concepts"]:
        visual_hook = str(concept.get("visual_hook", "")).strip()
        must_show = ", ".join(str(item) for item in concept.get("must_show", []) if str(item).strip())
        must_avoid = ", ".join(str(item) for item in concept.get("must_avoid", []) if str(item).strip())
        clip_prompt = normalize_prompt_text(
            f"YouTube thumbnail, {concept['type'].replace('_', ' ')}, {concept['thumbnail_text']}"
        )
        image_prompt = (
            "YouTube thumbnail background, cinematic illustrated history documentary, 16:9 composition, "
            "one strong focal subject, dramatic readable silhouette, clean negative space on "
            f"{concept['text_side']}, no text, no letters, no logo, no watermark, no extra limbs, "
            "no malformed hands, no merged bodies, all visible people fully clothed in full-coverage hide and fur clothing, "
            f"concept: {visual_hook}"
        )
        if must_show:
            image_prompt += f", must show: {must_show}"
        negative_prompt = "text, letters, logo, watermark, extra limbs, malformed hands, merged bodies"
        if must_avoid:
            negative_prompt += f", {must_avoid}"
        entries.append(
            {
                "concept_id": int(concept["id"]),
                "type": concept["type"],
                "clip_prompt": clip_prompt,
                "image_prompt": image_prompt,
                "negative_prompt": negative_prompt,
                "thumbnail_text": concept["thumbnail_text"],
                "subject_side": concept["subject_side"],
                "text_side": concept["text_side"],
                "paired_title_ids": concept["paired_title_ids"],
            }
        )
    _atomic_write_json(output_path, entries)
    _atomic_write_json(
        diagnostics_path,
        {
            "package_version": validated_package["package_version"],
            "script_sha256": validated_package["script_sha256"],
            "language": validated_package["language"],
            "concept_count": len(validated_package["thumbnail_concepts"]),
            "thumbnail_prompt_count": len(entries),
            "validation_passed": True,
            "warnings": ["deterministic_fallback_no_text_model_key"],
        },
    )


def run(video_id: str, n_override: int | None = None, allow_stale_package: bool = False) -> None:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    script_path = video_dir / "script.txt"
    audio_path = video_dir / "audio.mp3"
    output_path = video_dir / "image_prompts.json"
    if not script_path.exists():
        logger.error("script.txt not found: {}", script_path)
        sys.exit(1)
    if not audio_path.exists():
        logger.error("audio.mp3 not found: {}", audio_path)
        sys.exit(1)

    script_text = load_script_text(script_path)
    sentence_units = load_sentence_units(script_path)
    sentence_spans = visual_beats.load_sentence_spans(video_dir)
    if n_override is not None:
        sentence_units = sentence_units[:n_override]
        sentence_spans = sentence_spans[:n_override]
    if len(sentence_units) != len(sentence_spans):
        logger.error("Script sentence count {} does not match timestamps {}", len(sentence_units), len(sentence_spans))
        sys.exit(1)

    language = _select_language(video_id, script_text)
    template = _load_prompt_template(language)
    use_claude = bool(config.ANTHROPIC_API_KEY)
    use_gemini = bool(config.GEMINI_API_KEY)
    if not use_claude and not use_gemini:
        logger.warning("No text-model API key found -> using deterministic production prompt fallback")

    audio_duration = _get_audio_duration(audio_path)
    if audio_duration <= 0:
        logger.error("Could not read audio duration from {}", audio_path)
        sys.exit(1)

    word_ready = visual_beats.exact_word_timing_ready(video_dir)
    if word_ready:
        logger.info("Exact word timings available -> semantic visual beats enabled")
        if use_claude or use_gemini:
            sentence_payload = _sentence_payload(sentence_spans)
            system_prompt = _build_system_prompt(language, template["text"])
            last_error = None
            raw_plan = None
            for attempt in range(1, config.CLAUDE_MAX_RETRIES + 2):
                try:
                    if use_claude:
                        raw_plan = _call_claude(script_text, sentence_payload, system_prompt)
                    else:
                        raw_plan = _call_gemini(script_text, sentence_payload, system_prompt)
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning("Attempt {}: scene planning failed -> {}", attempt, exc)
                    if attempt <= config.CLAUDE_MAX_RETRIES:
                        time.sleep(config.CLAUDE_RETRY_SLEEP)
            if raw_plan is None:
                logger.error("All visual beat attempts failed: {}", last_error)
                sys.exit(1)
            normalized_plan = _normalize_model_beats(raw_plan, sentence_spans)
        else:
            normalized_plan = _deterministic_plan(sentence_spans, language)
        prompt_plan = visual_beats.derive_beat_timings(
            normalized_plan,
            sentence_spans,
            visual_beats.load_exact_word_spans(video_dir),
        )
        for beat, source in zip(prompt_plan, normalized_plan):
            beat["prompt"] = source["prompt"]
    else:
        logger.warning("Exact word timing unavailable -> falling back to 1 sentence = 1 image planning")
        prompt_plan = visual_beats.build_fallback_sentence_beats(sentence_spans)
        for beat in prompt_plan:
            beat["prompt"] = beat["visual_intent"]

    try:
        prompts = _attach_generation_fields(prompt_plan, language=language, template=template)
    except Exception as exc:
        logger.error("Prompt validation failed: {}", exc)
        sys.exit(1)
    if prompts:
        prompts[-1]["end"] = round(audio_duration, 3)

    _atomic_write_json(output_path, prompts)
    logger.info("Saved {} visual scenes -> {}", len(prompts), output_path)

    creative_package_path = video_dir / "creative_package.json"
    if not creative_package_path.exists():
        logger.info("No creative_package.json found -> skipping thumbnail prompt generation")
        return
    try:
        validated_package = load_validated_package(video_dir, allow_stale_package=allow_stale_package)
    except CreativePackageError:
        logger.error("creative_package.json is invalid or stale for {}", video_id)
        sys.exit(1)
    if use_claude or use_gemini:
        _generate_thumbnail_prompt_payload(video_id, validated_package, use_claude)
    else:
        logger.warning("No text-model API key found -> using deterministic thumbnail prompt fallback")
        _generate_deterministic_thumbnail_prompts(video_id, validated_package)
