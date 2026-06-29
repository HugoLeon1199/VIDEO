"""Generate 10 style concept pairs for Klein 9B reference image selection.

Each concept produces 2 images:
  A: hero scene 16:9 (1024×576) — action shot
  B: character lineup 1:1 (576×576) — frontal, full body, costume visible

Output: reference_images/concepts/<concept_id>/hero.png + lineup.png
        reference_images/concepts/contact_sheet.png  (grid for selection)

Usage:
  python scripts/gen_style_concepts.py --dry-run
  python scripts/gen_style_concepts.py --vast-host 1.2.3.4
  python scripts/gen_style_concepts.py --vast-host 1.2.3.4 --vast-port 8080 --vast-port 8080
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

import config
from image_generation.schemas import SceneRequest, SceneResult

# ---------------------------------------------------------------------------
# Style lock — prepended to all prompts
# ---------------------------------------------------------------------------

_STYLE_LOCK = (
    "simple hand-drawn educational explainer illustration, "
    "warm off-white paper background, rough black ink outlines, "
    "flat muted earth colors, simple rounded characters, "
    "minimal background, one clear focal action, clean composition, "
    "no text, no logo, no watermark"
)

_SEED = 11001


# ---------------------------------------------------------------------------
# 10 Concept definitions
# ---------------------------------------------------------------------------

CONCEPTS = [
    {
        "id": "C01",
        "name": "ink_earth",
        "hero": "prehistoric man crouches at river edge sharpening a stone blade, medium shot",
        "lineup": "prehistoric man standing front view, full body, simple tunic and animal hide, arms at sides, plain white background",
    },
    {
        "id": "C02",
        "name": "cave_warm",
        "hero": "prehistoric woman tends small fire in cave, warm amber glow, medium shot",
        "lineup": "prehistoric woman standing front view, full body, woven cloth wrap, arms at sides, plain white background",
    },
    {
        "id": "C03",
        "name": "savanna_wide",
        "hero": "prehistoric family walks across open savanna at dawn, wide shot, silhouettes against horizon",
        "lineup": "prehistoric couple standing side by side front view, full body, simple earth-tone clothing, plain white background",
    },
    {
        "id": "C04",
        "name": "hunt_action",
        "hero": "prehistoric man throws spear toward horizon, dynamic pose, side view, motion lines",
        "lineup": "muscular prehistoric man standing front view, full body, carrying spear, animal hide vest, plain white background",
    },
    {
        "id": "C05",
        "name": "gather_fruit",
        "hero": "prehistoric woman reaches up to pick berries from a bush at forest edge, dappled light",
        "lineup": "prehistoric woman standing front view, full body, basket on arm, gathered cloth wrap, plain white background",
    },
    {
        "id": "C06",
        "name": "child_play",
        "hero": "prehistoric child runs along riverbank with a stick, joyful, wide shot, open sky",
        "lineup": "prehistoric child standing front view, full body, simple tunic, bare feet, plain white background",
    },
    {
        "id": "C07",
        "name": "elder_teach",
        "hero": "old man sits and draws animals on cave wall with charcoal, warm torch light, medium shot",
        "lineup": "elderly man standing front view, full body, weathered face, animal hide robe, plain white background",
    },
    {
        "id": "C08",
        "name": "night_fire",
        "hero": "group of three prehistoric people sit around campfire at night, warm glow, wide shot",
        "lineup": "three prehistoric people standing in a row front view, varied earth-tone clothing, plain white background",
    },
    {
        "id": "C09",
        "name": "stone_tool",
        "hero": "close-up of hands shaping a flint blade on a stone, detailed craft, macro shot",
        "lineup": "pair of hands holding stone tool front view, close crop, no face visible, plain white background",
    },
    {
        "id": "C10",
        "name": "coast_explore",
        "hero": "prehistoric woman stands on coastal cliff looking out to sea, wide shot, wind-blown hair",
        "lineup": "coastal woman standing front view, full body, wind-blown hair, wrapped cloth, plain white background",
    },
]


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Image save helper
# ---------------------------------------------------------------------------

def _save_result(result: SceneResult, dest: Path) -> bool:
    if not result.candidates:
        return False
    best = result.candidates[0]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if best.local_path and Path(best.local_path).exists():
        import shutil
        shutil.copy2(best.local_path, dest)
        return True

    if hasattr(best, "base64_data") and best.base64_data:
        from PIL import Image as PILImage
        raw = base64.b64decode(best.base64_data)
        with PILImage.open(io.BytesIO(raw)) as img:
            img.convert("RGB").save(dest, format="PNG")
        return True

    return False


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------

def _make_contact_sheet(concepts_dir: Path, concepts: list[dict]) -> Path:
    from PIL import Image as PILImage, ImageDraw, ImageFont

    cell_w, cell_h = 512, 288  # hero scaled
    lineup_cell = 288           # lineup square, same height
    col_w = cell_w + lineup_cell + 20
    row_h = cell_h + 40
    n_cols = 2
    n_rows = (len(concepts) + 1) // 2

    sheet_w = n_cols * col_w + (n_cols + 1) * 10
    sheet_h = n_rows * row_h + (n_rows + 1) * 10
    sheet = PILImage.new("RGB", (sheet_w, sheet_h), (250, 248, 240))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for i, concept in enumerate(concepts):
        col = i % n_cols
        row = i // n_cols
        x0 = 10 + col * (col_w + 10)
        y0 = 10 + row * (row_h + 10)

        hero_path = concepts_dir / concept["id"] / "hero.png"
        lineup_path = concepts_dir / concept["id"] / "lineup.png"

        if hero_path.exists():
            with PILImage.open(hero_path) as img:
                img = img.convert("RGB").resize((cell_w, cell_h), PILImage.LANCZOS)
                sheet.paste(img, (x0, y0))
        else:
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=(200, 200, 200))
            draw.text((x0 + 10, y0 + 10), "MISSING", fill=(100, 100, 100), font=font)

        lx0 = x0 + cell_w + 10
        if lineup_path.exists():
            with PILImage.open(lineup_path) as img:
                img = img.convert("RGB").resize((lineup_cell, lineup_cell), PILImage.LANCZOS)
                sheet.paste(img, (lx0, y0))
        else:
            draw.rectangle([lx0, y0, lx0 + lineup_cell, y0 + lineup_cell], fill=(200, 200, 200))

        label = f"{concept['id']} {concept['name']}"
        draw.text((x0, y0 + cell_h + 5), label, fill=(40, 40, 40), font=font)

    out = concepts_dir / "contact_sheet.png"
    sheet.save(out, format="PNG")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 10 style concept pairs for Klein 9B reference selection")
    parser.add_argument("--backend", choices=["runpod", "vast"], default="vast")
    parser.add_argument("--vast-host", default="", metavar="HOST")
    parser.add_argument("--vast-port", type=int, default=8080)
    parser.add_argument("--out-dir", default="reference_images/concepts", metavar="DIR")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts only, no generation")
    parser.add_argument("--concepts", default="", metavar="C01,C02,...", help="Subset of concept IDs to generate")
    args = parser.parse_args()

    concepts = CONCEPTS
    if args.concepts:
        ids = {c.strip().upper() for c in args.concepts.split(",")}
        concepts = [c for c in CONCEPTS if c["id"] in ids]
        if not concepts:
            logger.error("No matching concepts for: {}", args.concepts)
            sys.exit(1)

    out_dir = Path(args.out_dir)

    if args.dry_run:
        logger.info("DRY RUN — {} concepts, 2 images each", len(concepts))
        for c in concepts:
            logger.info("")
            logger.info("  {} {}", c["id"], c["name"])
            logger.info("    hero:   {}", c["hero"])
            logger.info("    lineup: {}", c["lineup"])
        logger.info("")
        logger.info("Output dir: {}", out_dir)
        return

    backend = _build_backend(args)

    results_log = []
    for c in concepts:
        for variant, prompt_suffix, w, h in [
            ("hero",   c["hero"],   1024, 576),
            ("lineup", c["lineup"],  576, 576),
        ]:
            dest = out_dir / c["id"] / f"{variant}.png"
            if dest.exists():
                logger.info("{}/{}: already exists, skip", c["id"], variant)
                results_log.append({"id": c["id"], "variant": variant, "ok": True, "skipped": True})
                continue

            full_prompt = f"{_STYLE_LOCK}. {prompt_suffix}"
            req = SceneRequest(
                video_id="style_concepts",
                scene_id=f"{c['id']}_{variant}",
                prompt=full_prompt,
                clip_prompt=prompt_suffix[:200],
                width=w,
                height=h,
                steps=config.IMAGE_STEPS,
                guidance_scale=config.IMAGE_GUIDANCE_SCALE,
                candidate_seeds=[_SEED],
                output_format="PNG",
                quality=100,
            )
            t0 = time.time()
            try:
                result = backend.generate(req)
                ok = _save_result(result, dest)
                elapsed = round(time.time() - t0, 1)
                status = "ok" if ok else "FAIL (no image)"
                logger.info("{}/{}: {} in {:.1f}s", c["id"], variant, status, elapsed)
                results_log.append({"id": c["id"], "variant": variant, "ok": ok, "elapsed": elapsed})
            except Exception as exc:
                elapsed = round(time.time() - t0, 1)
                logger.error("{}/{}: ERROR in {:.1f}s — {}", c["id"], variant, elapsed, exc)
                results_log.append({"id": c["id"], "variant": variant, "ok": False, "error": str(exc)})

    # Write results log
    log_path = out_dir / "generation_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(results_log, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Log: {}", log_path)

    # Build contact sheet
    try:
        sheet_path = _make_contact_sheet(out_dir, concepts)
        logger.info("Contact sheet: {}", sheet_path)
    except Exception as exc:
        logger.warning("Contact sheet failed (PIL issue?): {}", exc)

    ok_count = sum(1 for r in results_log if r.get("ok"))
    total = len(results_log)
    logger.info("Done: {}/{} images generated successfully", ok_count, total)
    if ok_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
