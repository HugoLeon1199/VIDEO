"""Generate character sheets for selected style concepts (Step 2 of reference image workflow).

After Leon selects top 3 concepts from the contact sheet, this script generates:
  - character_male.png  (1:1, frontal, full body, plain background)
  - character_female.png (1:1, frontal, full body, plain background)
for each selected concept.

Output: reference_images/candidates/<concept_id>/character_male.png
                                                  character_female.png

Usage:
  python scripts/gen_character_sheets.py --concepts C01,C04,C07 --backend runpod
  python scripts/gen_character_sheets.py --concepts C01 --backend vast --vast-host 1.2.3.4

After Leon picks the final winner, manually copy files to reference_images/:
  reference_images/character_male.png    <- from winning concept
  reference_images/character_female.png  <- from winning concept
  reference_images/style_sheet.png       <- hero.png from winning concept
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
from scripts.gen_style_concepts import CONCEPTS, _STYLE_LOCK, _build_backend, _save_result

_SEED = 11001

# Character sheet prompts keyed by concept ID
# Male: KARO archetype — strong, capable, carries tools
# Female: LUMA archetype — resourceful, fully clothed, carries basket or tool

_MALE_PROMPTS: dict[str, str] = {
    "C01": "prehistoric man named Karo standing front view, full body, simple tunic and animal hide vest, short dark hair, carrying stone blade tool, arms at sides, plain white background, no weapons drawn",
    "C02": "prehistoric man named Karo standing front view, full body, cave dweller, simple woven tunic, arms at sides, plain white background",
    "C03": "prehistoric man named Karo standing front view, full body, savanna traveler, earth-tone robe, carrying walking stick, plain white background",
    "C04": "prehistoric man named Karo standing front view, full body, hunter, animal hide vest and loincloth, spear held upright, plain white background",
    "C05": "prehistoric man named Karo standing front view, full body, forager, simple cloth tunic, belt with pouch, plain white background",
    "C06": "prehistoric man named Karo standing front view, full body, young adult, simple earth-tone tunic, bare feet, plain white background",
    "C07": "prehistoric man named Karo standing front view, full body, craftsman, leather apron over tunic, arms at sides, plain white background",
    "C08": "prehistoric man named Karo standing front view, full body, tribal member, earth-tone wrapped cloth, plain white background",
    "C09": "prehistoric man named Karo standing front view, full body, tool-maker, simple tunic, hands visible, plain white background",
    "C10": "prehistoric man named Karo standing front view, full body, coastal explorer, woven wrap cloth, plain white background",
}

_FEMALE_PROMPTS: dict[str, str] = {
    "C01": "prehistoric woman named Luma standing front view, full body, simple woven dress to knees, dark hair tied back, carrying small clay pot, arms at sides, plain white background, fully clothed",
    "C02": "prehistoric woman named Luma standing front view, full body, cave dweller, layered cloth wrap dress, plain white background, fully clothed",
    "C03": "prehistoric woman named Luma standing front view, full body, savanna traveler, flowing earth-tone robe, carrying woven basket, plain white background, fully clothed",
    "C04": "prehistoric woman named Luma standing front view, full body, gatherer, knee-length tunic dress, belt with pouch, plain white background, fully clothed",
    "C05": "prehistoric woman named Luma standing front view, full body, forager, woven cloth dress, basket on arm, plain white background, fully clothed",
    "C06": "prehistoric woman named Luma standing front view, full body, simple tunic dress, bare feet, plain white background, fully clothed",
    "C07": "prehistoric woman named Luma standing front view, full body, artisan, long wrapped cloth dress, arms at sides, plain white background, fully clothed",
    "C08": "prehistoric woman named Luma standing front view, full body, tribal member, layered cloth wrap, plain white background, fully clothed",
    "C09": "prehistoric woman named Luma standing front view, full body, craftswoman, simple long tunic, plain white background, fully clothed",
    "C10": "prehistoric woman named Luma standing front view, full body, coastal gatherer, wind-blown hair, wrapped cloth dress, plain white background, fully clothed",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate character sheets for selected Klein 9B concepts")
    parser.add_argument("--concepts", required=True, metavar="C01,C04,C07",
                        help="Comma-separated concept IDs to generate character sheets for")
    parser.add_argument("--backend", choices=["runpod", "vast"], default="runpod")
    parser.add_argument("--vast-host", default="", metavar="HOST")
    parser.add_argument("--vast-port", type=int, default=8080)
    parser.add_argument("--out-dir", default="reference_images/candidates", metavar="DIR")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    concept_ids = [c.strip().upper() for c in args.concepts.split(",")]
    selected = {c["id"]: c for c in CONCEPTS if c["id"] in concept_ids}
    missing_ids = [cid for cid in concept_ids if cid not in selected]
    if missing_ids:
        logger.error("Unknown concept IDs: {}", missing_ids)
        sys.exit(1)

    out_dir = Path(args.out_dir)

    if args.dry_run:
        logger.info("DRY RUN — {} concepts × 2 characters = {} images", len(selected), len(selected) * 2)
        for cid, concept in selected.items():
            logger.info("")
            logger.info("  {} {}", cid, concept["name"])
            logger.info("    male:   {}", _MALE_PROMPTS.get(cid, "N/A"))
            logger.info("    female: {}", _FEMALE_PROMPTS.get(cid, "N/A"))
        return

    backend = _build_backend(args)
    results_log = []

    for cid, concept in selected.items():
        for character, prompt_suffix in [
            ("character_male",   _MALE_PROMPTS.get(cid, "")),
            ("character_female", _FEMALE_PROMPTS.get(cid, "")),
        ]:
            if not prompt_suffix:
                logger.warning("{}/{}: no prompt defined, skip", cid, character)
                continue

            dest = out_dir / cid / f"{character}.png"
            if dest.exists():
                logger.info("{}/{}: already exists, skip", cid, character)
                results_log.append({"id": cid, "character": character, "ok": True, "skipped": True})
                continue

            full_prompt = f"{_STYLE_LOCK}. {prompt_suffix}"
            req = SceneRequest(
                video_id="char_sheets",
                scene_id=f"{cid}_{character}",
                prompt=full_prompt,
                clip_prompt=prompt_suffix[:200],
                width=576,
                height=576,
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
                logger.info("{}/{}: {} in {:.1f}s", cid, character, status, elapsed)
                results_log.append({"id": cid, "character": character, "ok": ok, "elapsed": elapsed})
            except Exception as exc:
                elapsed = round(time.time() - t0, 1)
                logger.error("{}/{}: ERROR in {:.1f}s — {}", cid, character, elapsed, exc)
                results_log.append({"id": cid, "character": character, "ok": False, "error": str(exc)})

    log_path = out_dir / "generation_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(results_log, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Log: {}", log_path)

    ok_count = sum(1 for r in results_log if r.get("ok"))
    total = len(results_log)
    logger.info("Done: {}/{} character sheets generated", ok_count, total)

    if ok_count == total:
        logger.info("")
        logger.info("Next step: pick the winning concept, then copy to reference_images/:")
        for cid in concept_ids:
            logger.info("  cp {} reference_images/character_male.png", out_dir / cid / "character_male.png")
            logger.info("  cp {} reference_images/character_female.png", out_dir / cid / "character_female.png")
            logger.info("  cp {} reference_images/style_sheet.png", Path("reference_images/concepts") / cid / "hero.png")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
