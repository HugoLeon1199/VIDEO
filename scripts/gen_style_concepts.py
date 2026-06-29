"""Generate fair style concept pairs for Klein 9B style selection.

Each concept produces:
  - A: shared control scene with the same Karo + Luma composition
  - B: unique scene with the same Karo + Luma cast in a different setting

Outputs:
  reference_images/concepts/<concept_id>/A_control.png
  reference_images/concepts/<concept_id>/B_unique.png
  reference_images/concepts/generation_log.json
  reference_images/concepts/contact_sheet.jpg
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

import config
from image_generation.schemas import SceneRequest, SceneResult

_STYLE_LOCK = (
    "simple hand-drawn educational explainer illustration, "
    "warm off-white paper background, rough black ink outlines, "
    "flat muted earth colors, simple rounded characters, "
    "minimal background, clean silhouettes, readable staging, "
    "no text, no logo, no watermark"
)

_SEED = 11001
_CONTROL_SCENE = (
    "Karo and Luma kneel beside a small cave fire studying a carved stone map and a woven basket, "
    "medium-wide two-shot, both characters fully clothed, same pose and same props for style comparison"
)

CONCEPTS = [
    {
        "id": "C01",
        "name": "soft_ink_hatch",
        "style_variant": "soft graphite hatch shadows, feather-light watercolor wash, slightly rounded faces",
        "control_scene": _CONTROL_SCENE,
        "unique_scene": "Karo and Luma balance on a reed raft while checking a stone fishing hook at sunrise, medium-wide two-shot",
    },
    {
        "id": "C02",
        "name": "chalk_wash",
        "style_variant": "chalky dry-brush shading, pale ochre wash, simplified anatomy with soft edges",
        "control_scene": _CONTROL_SCENE,
        "unique_scene": "Karo and Luma shelter under a limestone overhang during warm rain, sharing one hide cloak, medium shot",
    },
    {
        "id": "C03",
        "name": "clean_contour",
        "style_variant": "clean contour lines, very sparse crosshatch, restrained terracotta accents",
        "control_scene": _CONTROL_SCENE,
        "unique_scene": "Karo and Luma cross an open savanna carrying water gourds at dawn, wide two-shot with long shadows",
    },
    {
        "id": "C04",
        "name": "storybook_earth",
        "style_variant": "storybook ink contours, dusty sienna fills, compact rounded hands and feet",
        "control_scene": _CONTROL_SCENE,
        "unique_scene": "Karo and Luma crouch together over a half-finished spear shaft beside a rock shelter, medium close two-shot",
    },
    {
        "id": "C05",
        "name": "paper_grain",
        "style_variant": "visible paper grain, broken ink strokes, olive and clay color blocks",
        "control_scene": _CONTROL_SCENE,
        "unique_scene": "Karo and Luma gather berries beside tangled roots while whispering to each other, medium-wide forest two-shot",
    },
    {
        "id": "C06",
        "name": "sepia_outline",
        "style_variant": "sepia-tinted outlines, faint smudged shading, broad simple clothing folds",
        "control_scene": _CONTROL_SCENE,
        "unique_scene": "Karo and Luma stand on a windy cliff watching seabirds circle over the water, wide coastal two-shot",
    },
    {
        "id": "C07",
        "name": "charcoal_note",
        "style_variant": "charcoal-like line weight changes, smoky wash, expressive brows and noses",
        "control_scene": _CONTROL_SCENE,
        "unique_scene": "Karo and Luma sketch animal tracks onto a cave wall with charcoal, warm torch-lit medium shot",
    },
    {
        "id": "C08",
        "name": "ember_pastel",
        "style_variant": "ember-red accent spots, pastel tan fills, thick playful silhouette lines",
        "control_scene": _CONTROL_SCENE,
        "unique_scene": "Karo and Luma sit back-to-back near a night camp ember sorting shells and beads, intimate medium shot",
    },
    {
        "id": "C09",
        "name": "museum_sketch",
        "style_variant": "museum-sketch precision, delicate hatch clusters, muted moss and rust palette",
        "control_scene": _CONTROL_SCENE,
        "unique_scene": "Karo and Luma inspect a freshly knapped stone blade on a flat work slab, tabletop medium shot",
    },
    {
        "id": "C10",
        "name": "rounded_field_guide",
        "style_variant": "rounded field-guide silhouettes, tidy brush fills, understated blue-grey shadows",
        "control_scene": _CONTROL_SCENE,
        "unique_scene": "Karo and Luma wade through a shallow tidal pool collecting shellfish with woven pouches, medium-wide two-shot",
    },
]

_VARIANTS = (
    ("A_control", "control_scene", "A"),
    ("B_unique", "unique_scene", "B"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fair Klein 9B style concept pairs")
    parser.add_argument("--backend", choices=["runpod", "vast"], default="vast")
    parser.add_argument("--vast-host", default="", metavar="HOST")
    parser.add_argument("--vast-port", type=int, default=config.KLEIN_WORKER_PORT)
    parser.add_argument("--out-dir", default="reference_images/concepts", metavar="DIR")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts only, no generation")
    parser.add_argument("--smoke", action="store_true", help="Generate only C01 (2 images)")
    parser.add_argument("--concepts", default="", metavar="C01,C02,...", help="Subset of concept IDs to generate")
    return parser


def _build_backend(args):
    if args.backend == "runpod":
        from image_generation.runpod_client import RunPodClient
        from image_generation.runpod_serverless_backend import RunPodServerlessBackend

        if not config.RUNPOD_API_KEY:
            raise RuntimeError("RUNPOD_API_KEY not set")
        client = RunPodClient(
            api_key=config.RUNPOD_API_KEY,
            endpoint_id=config.RUNPOD_ENDPOINT_ID,
            timeout=config.RUNPOD_REQUEST_TIMEOUT,
            poll_interval=config.RUNPOD_POLL_INTERVAL,
            max_retries=config.RUNPOD_MAX_RETRIES,
        )
        return RunPodServerlessBackend(client=client)

    if args.backend == "vast":
        from image_generation.vast_backend import VastInstanceBackend

        if not args.vast_host:
            raise RuntimeError("--vast-host required when --backend vast")
        return VastInstanceBackend(
            host=args.vast_host,
            port=args.vast_port,
            worker_token=config.WORKER_API_TOKEN or "",
        )

    raise ValueError(f"Unknown backend: {args.backend}")


def _select_concepts(concepts_arg: str, smoke: bool) -> list[dict]:
    if smoke:
        return [CONCEPTS[0]]
    if not concepts_arg:
        return list(CONCEPTS)
    ids = {item.strip().upper() for item in concepts_arg.split(",") if item.strip()}
    concepts = [concept for concept in CONCEPTS if concept["id"] in ids]
    if not concepts:
        raise RuntimeError(f"No matching concepts for: {concepts_arg}")
    return concepts


def _numeric_scene_id(concept_id: str, column_label: str) -> str:
    concept_num = int(concept_id[1:])
    column_offset = 1 if column_label == "A" else 2
    return str(concept_num * 10 + column_offset)


def _build_full_prompt(concept: dict, scene_key: str) -> str:
    scene_text = concept[scene_key]
    style_variant = concept["style_variant"]
    return f"{_STYLE_LOCK}. {style_variant}. {scene_text}"


def _build_clip_prompt(concept: dict, scene_key: str) -> str:
    if scene_key == "control_scene":
        return f"{concept['style_variant']}, Karo and Luma control scene"
    return f"{concept['style_variant']}, {concept['unique_scene'][:140]}"


def _save_result(result: SceneResult, dest: Path) -> bool:
    if not result.candidates:
        return False
    best = result.candidates[0]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if best.local_path and Path(best.local_path).exists():
        import shutil

        shutil.copy2(best.local_path, dest)
        return True

    if best.base64_data:
        from PIL import Image as PILImage

        raw = base64.b64decode(best.base64_data)
        with PILImage.open(io.BytesIO(raw)) as img:
            img.convert("RGB").save(dest, format="PNG")
        return True

    return False


def _require_klein_worker_ready(backend) -> dict:
    base_url = getattr(backend, "base_url", "")
    headers = getattr(backend, "_worker_headers", {})
    if not base_url:
        raise RuntimeError("Live Klein concept generation requires a direct worker health endpoint")
    response = requests.get(f"{base_url}/health", timeout=20, headers=headers)
    response.raise_for_status()
    payload = response.json()
    model_id = str(payload.get("model_id") or payload.get("model") or "").strip()
    if not payload.get("model_loaded"):
        raise RuntimeError(f"Klein worker not ready: model_loaded={payload.get('model_loaded')}")
    if model_id != config.KLEIN_MODEL_ID:
        raise RuntimeError(f"Klein worker model mismatch: expected {config.KLEIN_MODEL_ID}, got {model_id or 'unknown'}")
    if "12b" in model_id.lower():
        raise RuntimeError(f"Klein worker model mismatch: expected Klein 9B, got {model_id}")
    return payload


def _make_contact_sheet(concepts_dir: Path, concepts: list[dict]) -> Path:
    from PIL import Image as PILImage, ImageDraw, ImageFont

    cell_w = 480
    cell_h = 270
    margin = 20
    label_h = 36
    sheet_w = margin * 3 + cell_w * 2
    sheet_h = margin + len(concepts) * (cell_h + label_h + margin)
    sheet = PILImage.new("RGB", (sheet_w, sheet_h), (246, 240, 228))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for row, concept in enumerate(concepts):
        y0 = margin + row * (cell_h + label_h + margin)
        label = f"{concept['id']}  {concept['name']}  |  {concept['style_variant']}"
        draw.text((margin, y0), label, fill=(40, 34, 26), font=font)
        image_y = y0 + label_h
        for col, (image_key, _scene_key, column_label) in enumerate(_VARIANTS):
            x0 = margin + col * (cell_w + margin)
            image_path = concepts_dir / concept["id"] / f"{image_key}.png"
            if image_path.exists():
                with PILImage.open(image_path) as img:
                    img = img.convert("RGB").resize((cell_w, cell_h), PILImage.LANCZOS)
                    sheet.paste(img, (x0, image_y))
            else:
                draw.rectangle([x0, image_y, x0 + cell_w, image_y + cell_h], fill=(210, 205, 198))
                draw.text((x0 + 12, image_y + 12), "MISSING", fill=(100, 96, 88), font=font)
            draw.text((x0, image_y + cell_h + 4), f"{column_label}  {image_key}", fill=(58, 50, 42), font=font)

    out = concepts_dir / "contact_sheet.jpg"
    sheet.save(out, format="JPEG", quality=92)
    return out


def _build_request(concept: dict, image_key: str, scene_key: str, column_label: str) -> SceneRequest:
    return SceneRequest(
        video_id="style_concepts_klein",
        scene_id=_numeric_scene_id(concept["id"], column_label),
        prompt=_build_full_prompt(concept, scene_key),
        clip_prompt=_build_clip_prompt(concept, scene_key),
        width=1024,
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
    concepts = _select_concepts(args.concepts, args.smoke)
    out_dir = Path(args.out_dir)

    if args.dry_run:
        logger.info("DRY RUN - {} concepts, 2 images each", len(concepts))
        for concept in concepts:
            logger.info("")
            logger.info("  {} {}", concept["id"], concept["name"])
            logger.info("    style_variant: {}", concept["style_variant"])
            logger.info("    control: {}", concept["control_scene"])
            logger.info("    unique:  {}", concept["unique_scene"])
        logger.info("")
        logger.info("Output dir: {}", out_dir)
        return

    backend = _build_backend(args)
    health = _require_klein_worker_ready(backend)
    logger.info("Klein worker ready: model={} device={}", health.get("model_id"), health.get("device"))

    log_entries: list[dict] = []
    for concept in concepts:
        for image_key, scene_key, column_label in _VARIANTS:
            dest = out_dir / concept["id"] / f"{image_key}.png"
            request = _build_request(concept, image_key, scene_key, column_label)
            t0 = time.time()
            error = None
            ok = False
            result: SceneResult | None = None
            try:
                result = backend.generate(request)
                ok = _save_result(result, dest)
                if not ok:
                    error = "; ".join(result.errors) or "no_image_returned"
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            elapsed = round(time.time() - t0, 2)
            status = "ok" if ok else f"FAIL ({error})"
            logger.info("{}/{}: {} in {:.2f}s", concept["id"], image_key, status, elapsed)
            log_entries.append(
                {
                    "concept_id": concept["id"],
                    "style_variant": concept["style_variant"],
                    "image_key": image_key,
                    "scene_id": int(request.scene_id),
                    "scene_mode": "control" if image_key == "A_control" else "unique",
                    "output_path": str(dest),
                    "prompt": request.prompt,
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
    log_payload = {
        "model_id": health.get("model_id"),
        "model_revision": health.get("model_revision"),
        "device": health.get("device"),
        "entries": log_entries,
    }
    log_path = out_dir / "generation_log.json"
    log_path.write_text(json.dumps(log_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Log: {}", log_path)

    sheet_path = _make_contact_sheet(out_dir, concepts)
    logger.info("Contact sheet: {}", sheet_path)

    ok_count = sum(1 for entry in log_entries if entry["ok"])
    total = len(log_entries)
    logger.info("Done: {}/{} images generated successfully", ok_count, total)
    if ok_count < total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
