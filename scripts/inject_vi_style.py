"""One-shot script: inject VI 2D documentary style prefix into image_prompts.json."""
import json, shutil
from pathlib import Path

STYLE_PREFIX = (
    "cinematic 2D painted documentary illustration, semi-realistic prehistoric humans, "
    "mature historical animation, hand-painted texture, simplified anatomically correct bodies, "
    "all characters fully clothed in prehistoric hide garments covering chest and torso, "
    "long animal-hide tunics and wraparound skirts reaching mid-thigh or below, "
    "no bare skin except face forearms and lower legs, "
    "clean separated silhouettes, warm golden-amber natural lighting, "
    "detailed prehistoric environment, serious educational documentary tone, "
    "cinematic composition, 16:9, no text, no watermark, "
)
ANATOMY_CLAUSE = (
    ", anatomically coherent human figures, exactly two arms and two legs per visible person, "
    "natural human proportions, clean separated silhouettes, clearly readable limbs, natural hands"
)
NEGATIVE_PROMPT = (
    "nudity, bare skin, topless, shirtless female, exposed chest, exposed breasts, exposed belly, "
    "bikini, swimwear, revealing outfit, skimpy clothing, loincloth only, "
    "sexual content, suggestive pose, "
    "extra arms, extra legs, extra limbs, duplicated body parts, fused limbs, overlapping bodies, "
    "malformed hands, extra fingers, missing fingers, twisted joints, disconnected limbs, "
    "duplicate people, merged faces, deformed anatomy, cropped head, cropped feet, "
    "photorealistic photography, 3D render, anime, chibi, Pixar, Disney, text, letters, subtitles, logo, watermark"
)

HUMAN_KEYWORDS = [
    "man", "woman", "hunter", "gatherer", "farmer", "elder", "figure", "human",
    "people", "person", "child", "hands", "face", "portrait", "silhouette",
    "generations", "community", "dancing", "sitting", "resting", "walking", "crouching",
    "storyteller", "figures", "workers", "ancient humans",
]

# Photorealistic suffixes to strip (longest first)
STRIP_SUFFIXES = [
    ", earthy cinematic photography, photorealistic",
    ", cinematic photorealistic wide shot",
    ", cinematic documentary photorealistic",
    ", dramatic cinematic wide shot, photorealistic",
    ", cinematic photorealistic portrait",
    ", intimate cinematic portrait, photorealistic",
    ", cinematic earthy wide shot, photorealistic",
    ", slow cinematic wide shot, photorealistic",
    ", stark cinematic photorealistic portrait",
    ", cinematic documentary, photorealistic",
    ", cinematic macro photorealistic",
    ", cinematic documentary style, photorealistic",
    ", cinematic wide shot, photorealistic",
    ", earthy cinematic wide shot, photorealistic",
    ", earthy cinematic tones, photorealistic",
    ", cinematic earthy tones, photorealistic",
    ", cinematic photorealistic",
    ", photorealistic wide shot",
    ", photorealistic, cinematic wide shot",
    ", photorealistic, ancient wilderness",
    ", photorealistic, no modern objects",
    ", photorealistic, survival atmosphere",
    ", photorealistic, earthy tones",
    ", photorealistic, peaceful",
    ", photorealistic, warm tones",
    ", photorealistic.",
    ", photorealistic",
    "photorealistic, ",
    "photorealistic",
]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--video-id", required=True)
parser.add_argument("--output-root", default="output")
args = parser.parse_args()

p = Path(args.output_root) / args.video_id / "image_prompts.json"
data = json.loads(p.read_text(encoding="utf-8"))

backup = p.with_name("image_prompts_photorealistic_backup.json")
shutil.copy2(p, backup)
print(f"Backup: {backup}")

changed = 0
for entry in data:
    orig = entry["prompt"]
    cleaned = orig
    for tag in STRIP_SUFFIXES:
        cleaned = cleaned.replace(tag, "")
    cleaned = cleaned.strip().rstrip(",").strip()

    has_humans = any(kw in cleaned.lower() for kw in HUMAN_KEYWORDS)
    anatomy = ANATOMY_CLAUSE if has_humans else ""

    entry["prompt"] = STYLE_PREFIX + cleaned + anatomy
    entry["negative_prompt"] = NEGATIVE_PROMPT
    entry.setdefault("global_style", "")
    changed += 1

tmp = p.with_suffix(".tmp")
tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
tmp.replace(p)
print(f"Updated {changed}/{len(data)} prompts")
print(f"\nScene 1:\n{data[0]['prompt'][:220]}")
print(f"\nScene 7 (human):\n{data[6]['prompt'][:220]}")
