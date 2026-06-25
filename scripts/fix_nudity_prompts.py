"""Fix nudity issue: update negative_prompt in existing image_prompts.json."""
import json, argparse
from pathlib import Path

NEW_NEGATIVE = (
    "nudity, bare female chest, topless female, exposed breasts, revealing clothing, "
    "extra arms, extra legs, extra limbs, duplicated body parts, fused limbs, overlapping bodies, "
    "malformed hands, extra fingers, missing fingers, twisted joints, disconnected limbs, "
    "duplicate people, merged faces, deformed anatomy, cropped head, cropped feet, "
    "photorealistic photography, 3D cartoon, anime, chibi, text, letters, subtitles, logo, watermark"
)

# Also patch style prefix to require clothing
OLD_CLOTHING = "simplified anatomically correct bodies, clean separated silhouettes"
NEW_CLOTHING = "simplified anatomically correct bodies, all characters wearing hide wraps or minimal prehistoric clothing covering torso, clean separated silhouettes"

parser = argparse.ArgumentParser()
parser.add_argument("--video-id", required=True)
parser.add_argument("--output-root", default="output")
args = parser.parse_args()

p = Path(args.output_root) / args.video_id / "image_prompts.json"
data = json.loads(p.read_text(encoding="utf-8"))

for entry in data:
    entry["negative_prompt"] = NEW_NEGATIVE
    if OLD_CLOTHING in entry.get("prompt", ""):
        entry["prompt"] = entry["prompt"].replace(OLD_CLOTHING, NEW_CLOTHING)

tmp = p.with_suffix(".tmp")
tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
tmp.replace(p)
print(f"Updated {len(data)} prompts with nudity fix")
print(f"Sample negative: {data[0]['negative_prompt'][:80]}...")
print(f"Sample prompt clothing clause: {'hide wraps' in data[0]['prompt']}")
