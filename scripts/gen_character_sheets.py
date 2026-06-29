"""Generate Karo + Luma character sheets for selected style concepts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

import config
from image_generation.schemas import SceneRequest
from scripts.gen_style_concepts import (
    CONCEPTS,
    _STYLE_LOCK,
    _build_backend,
    _numeric_scene_id,
    _require_klein_worker_ready,
    _save_result,
    build_parser as _build_style_parser,
)

_SEED = 11001

_KARO_BASE = (
    "Karo standing front view, full body, same prehistoric adult man in practical hide-and-cloth outfit, "
    "stone blade pouch at belt, relaxed neutral pose, plain off-white background"
)
_LUMA_BASE = (
    "Luma standing front view, full body, same prehistoric adult woman in fully clothed woven hide-and-cloth outfit, "
    "small utility basket at side, relaxed neutral pose, plain off-white background"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Klein 9B character sheets for selected concepts")
    parser.add_argument("--concepts", required=True, metavar="C01,C04,C07")
    parser.add_argument("--backend", choices=["runpod", "vast"], default="vast")
    parser.add_argument("--vast-host", default="", metavar="HOST")
    parser.add_argument("--vast-port", type=int, default=config.KLEIN_WORKER_PORT)
    parser.add_argument("--out-dir", default="reference_images/candidates", metavar="DIR")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _select_concepts(concepts_arg: str) -> list[dict]:
    requested = [item.strip().upper() for item in concepts_arg.split(",") if item.strip()]
    selected = [concept for concept in CONCEPTS if concept["id"] in requested]
    missing = [item for item in requested if item not in {concept["id"] for concept in selected}]
    if missing:
        raise RuntimeError(f"Unknown concept IDs: {missing}")
    return selected


def _character_prompt(concept: dict, base_prompt: str) -> str:
    return f"{_STYLE_LOCK}. {concept['style_variant']}. {base_prompt}"


def _scene_request(concept: dict, character_key: str, prompt: str) -> SceneRequest:
    column_label = "A" if character_key == "character_male" else "B"
    return SceneRequest(
        video_id="style_character_sheets_klein",
        scene_id=_numeric_scene_id(concept["id"], column_label),
        prompt=prompt,
        clip_prompt=f"{concept['style_variant']}, {character_key.replace('_', ' ')}",
        width=576,
        height=576,
        steps=config.KLEIN_STEPS_T2I,
        guidance_scale=config.KLEIN_GUIDANCE_SCALE,
        candidate_seeds=[_SEED],
        output_format="PNG",
        quality=100,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    selected = _select_concepts(args.concepts)
    out_dir = Path(args.out_dir)

    if args.dry_run:
        logger.info("DRY RUN - {} concepts x 2 character sheets", len(selected))
        for concept in selected:
            logger.info("")
            logger.info("  {} {}", concept["id"], concept["name"])
            logger.info("    style_variant: {}", concept["style_variant"])
            logger.info("    karo: {}", _character_prompt(concept, _KARO_BASE))
            logger.info("    luma: {}", _character_prompt(concept, _LUMA_BASE))
        return

    backend = _build_backend(args)
    health = _require_klein_worker_ready(backend)
    logger.info("Klein worker ready: model={} device={}", health.get("model_id"), health.get("device"))

    log_entries: list[dict] = []
    for concept in selected:
        prompt_pairs = (
            ("character_male", _character_prompt(concept, _KARO_BASE)),
            ("character_female", _character_prompt(concept, _LUMA_BASE)),
        )
        for character_key, prompt in prompt_pairs:
            dest = out_dir / concept["id"] / f"{character_key}.png"
            request = _scene_request(concept, character_key, prompt)
            t0 = time.time()
            error = None
            ok = False
            result = None
            try:
                result = backend.generate(request)
                ok = _save_result(result, dest)
                if not ok:
                    error = "; ".join(result.errors) or "no_image_returned"
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            elapsed = round(time.time() - t0, 2)
            status = "ok" if ok else f"FAIL ({error})"
            logger.info("{}/{}: {} in {:.2f}s", concept["id"], character_key, status, elapsed)
            log_entries.append(
                {
                    "concept_id": concept["id"],
                    "style_variant": concept["style_variant"],
                    "character_key": character_key,
                    "scene_id": int(request.scene_id),
                    "output_path": str(dest),
                    "model": getattr(result, "model", health.get("model_id")),
                    "mode": getattr(result, "mode", "vast_instance"),
                    "steps": request.steps,
                    "guidance": request.guidance_scale,
                    "seed": _SEED,
                    "elapsed": elapsed,
                    "error": error,
                    "ok": ok,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "generation_log.json"
    log_path.write_text(
        json.dumps(
            {
                "model_id": health.get("model_id"),
                "model_revision": health.get("model_revision"),
                "device": health.get("device"),
                "entries": log_entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Log: {}", log_path)

    ok_count = sum(1 for entry in log_entries if entry["ok"])
    total = len(log_entries)
    logger.info("Done: {}/{} character sheets generated", ok_count, total)
    if ok_count < total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
