"""Generate the 10 master style seed images for the prehistoric flat-vector series.

Each seed teaches FLUX how to draw lines, fill colors, and handle 2D flat lighting
for all future scenes. Run once, save forever.

Usage:
    $python scripts/generate_master_seeds.py [--output-dir master_style_seeds]
"""
import argparse
import logging
import os
import sys
import time
import concurrent.futures
from pathlib import Path

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Style prefix shared by all seeds — cinematic 2D painted documentary
_PAINTED_PREFIX = (
    "cinematic 2D painted documentary illustration, semi-realistic prehistoric humans, "
    "painted in the style of a serious historical documentary, "
    "warm natural earthy lighting, serious mature tone, cinematic 16:9 composition, "
    "no text, no watermark, "
)

# ── 10 master seed definitions ────────────────────────────────────────────────
MASTER_SEEDS = [
    {
        "id": "seed_01",
        "filename": "seed_01.png",
        "label": "character_male — Close-up Male Portrait",
        "width": 720, "height": 720,
        "prompt": (
            _PAINTED_PREFIX +
            "close-up portrait of a prehistoric Homo sapiens man, rugged weathered face, "
            "dark brown skin, deep-set eyes, messy dark hair, wearing thick rough animal fur "
            "draped over both shoulders fully covering chest, side warm firelight, "
            "dark rocky cave background, anatomically coherent, natural human proportions"
        ),
    },
    {
        "id": "seed_02",
        "filename": "seed_02.png",
        "label": "character_female — Close-up Female Portrait",
        "width": 720, "height": 720,
        "prompt": (
            _PAINTED_PREFIX +
            "close-up portrait of a prehistoric Homo sapiens woman, wearing a sewn animal-hide "
            "top fully covering her chest and shoulders, dark brown skin, braided hair with "
            "bone ornament, calm expression, soft warm firelight from the side, "
            "dark cave background, anatomically coherent, natural human proportions"
        ),
    },
    {
        "id": "seed_03",
        "filename": "seed_03.png",
        "label": "character_full_body — Full Body Walking Figure",
        "width": 720, "height": 720,
        "prompt": (
            _PAINTED_PREFIX +
            "full body view of a lone prehistoric Homo sapiens man walking on open savanna, "
            "simple stable walking pose, arms and legs clearly separated, "
            "wearing rough hide wrap around waist and shoulders, wooden spear in one hand, "
            "clean readable silhouette against warm golden sky, anatomically coherent human figure, "
            "exactly two arms and two legs, natural human proportions"
        ),
    },
    {
        "id": "seed_04",
        "filename": "seed_04.png",
        "label": "night_fire — Cave Interior Campfire",
        "width": 1280, "height": 720,
        "prompt": (
            _PAINTED_PREFIX +
            "interior of a dark primitive cave at night, glowing campfire in the center "
            "casting warm flickering orange light on rough rock walls, deep surrounding darkness, "
            "two prehistoric human silhouettes seated around the fire, "
            "smoke rising toward a crack in the cave ceiling, moody atmospheric contrast"
        ),
    },
    {
        "id": "seed_05",
        "filename": "seed_05.png",
        "label": "day_wilderness — Daytime African Savanna",
        "width": 1280, "height": 720,
        "prompt": (
            _PAINTED_PREFIX +
            "wide establishing shot of an expansive prehistoric African savanna at golden hour, "
            "flat grassy plains stretching to distant mountains, scattered acacia trees, "
            "warm amber sunlight with long shadows, blue-gold horizon, "
            "small human silhouettes visible in the middle distance"
        ),
    },
    {
        "id": "seed_06",
        "filename": "seed_06.png",
        "label": "cosmic_sky — Night Sky Stars Moon",
        "width": 1280, "height": 720,
        "prompt": (
            _PAINTED_PREFIX +
            "wide shot of a vast prehistoric night sky filled with stars and a large full moon, "
            "blue-silver moonlight over silhouettes of ancient mountain ridgeline, "
            "milky way visible across the sky, cool blue-purple tones, "
            "one lone human silhouette standing and looking upward"
        ),
    },
    {
        "id": "seed_07",
        "filename": "seed_07.png",
        "label": "scientific_diagram — Brain Evolution Diagram",
        "width": 1280, "height": 720,
        "prompt": (
            "cinematic 2D painted educational documentary illustration, "
            "scientific diagram showing two prehistoric human skulls side by side on a stone surface, "
            "left skull smaller right skull visibly larger, labeled with clean minimal text, "
            "warm firelight illumination, dark earthy background, "
            "serious historical documentary aesthetic, cinematic 16:9, no watermark"
        ),
    },
    {
        "id": "seed_08",
        "filename": "seed_08.png",
        "label": "timeline_cycle — Historical Timeline Diagram",
        "width": 1280, "height": 720,
        "prompt": (
            "cinematic 2D painted educational documentary illustration, "
            "a historical timeline diagram showing prehistoric human eras as horizontal flow, "
            "simple painted arrow shapes connecting four distinct period markers, "
            "earthy warm color palette, dark background, clean readable layout, "
            "serious documentary educational style, cinematic 16:9, no watermark"
        ),
    },
    {
        "id": "seed_09",
        "filename": "seed_09.png",
        "label": "object_macro — Stone Tool Close-Up",
        "width": 1280, "height": 720,
        "prompt": (
            _PAINTED_PREFIX +
            "extreme close-up macro shot of a prehistoric hand-knapped flint stone tool "
            "lying on rough rock ground, sharp edges clearly visible, "
            "leather-wrapped handle, warm dramatic side lighting casting deep shadows, "
            "earthy muted brown and gray tones, historical documentary aesthetic"
        ),
    },
    {
        "id": "seed_10",
        "filename": "seed_10.png",
        "label": "grain_overlay — Background Texture Template",
        "width": 1280, "height": 720,
        "prompt": (
            "cinematic 2D painted documentary illustration, "
            "solid dark charcoal background with subtle painted canvas texture, "
            "warm brown-black earthy tones, faint horizontal brush marks, "
            "cinematic vignette around edges, atmospheric moody empty background, "
            "historical documentary aesthetic, 16:9 composition, no text, no figures"
        ),
    },
]

NEGATIVE = (
    "photorealistic, realistic photography, hyperrealistic, 3D render, CGI, "
    "anime, cartoon, Pixar, Disney, extra limbs, extra arms, extra legs, "
    "fused limbs, malformed hands, deformed anatomy, nudity, bare chest, "
    "exposed torso, text, watermark, logo, modern objects, technology"
)


def _gen_one(seed_def, out_dir, backend):
    from image_generation.schemas import SceneRequest
    from PIL import Image

    filename = out_dir / seed_def["filename"]
    if filename.exists():
        logger.info("SKIP (exists): %s", filename.name)
        return seed_def["id"], True

    req = SceneRequest(
        video_id="master-seeds",
        scene_id=seed_def["id"],
        prompt=seed_def["prompt"],
        negative_prompt=NEGATIVE,
        width=seed_def["width"],
        height=seed_def["height"],
        steps=22,
        guidance_scale=3.5,
        candidate_seeds=[11001],
        output_format="WEBP",
        quality=95,
        output_mode="base64",
    )

    logger.info("Generating %-10s %s", seed_def["id"], seed_def["label"])
    t0 = time.time()
    result = backend.generate(req)

    if result.errors:
        logger.error("FAILED %s: %s", seed_def["id"], result.errors)
        return seed_def["id"], False

    if not result.candidates or not result.candidates[0].local_path:
        logger.error("No candidate for %s", seed_def["id"])
        return seed_def["id"], False

    src = Path(result.candidates[0].local_path)
    img = Image.open(src).convert("RGB")
    img.save(filename, format="PNG", optimize=True)
    logger.info("Saved %-20s  %.1fs", filename.name, time.time() - t0)
    return seed_def["id"], True


def generate_seeds(output_dir: str, workers: int = 2):
    # Load .env so RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID are available
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    from image_generation.runpod_serverless_backend import RunPodServerlessBackend
    backend = RunPodServerlessBackend()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ok = 0
    failed = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_gen_one, s, out, backend): s for s in MASTER_SEEDS}
        for fut in concurrent.futures.as_completed(futures):
            sid, success = fut.result()
            if success:
                ok += 1
            else:
                failed.append(sid)

    print("\n" + "=" * 52)
    print(f"Master seeds: {ok}/{len(MASTER_SEEDS)} done")
    if failed:
        print(f"FAILED: {failed}")
    print(f"Output: {output_dir}/")
    print("=" * 52)
    return len(failed) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 10 master style seed images")
    parser.add_argument("--output-dir", default="master_style_seeds",
                        help="Directory to save seeds (default: master_style_seeds)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers (default 2)")
    args = parser.parse_args()

    success = generate_seeds(args.output_dir, args.workers)
    sys.exit(0 if success else 1)
