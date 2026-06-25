"""
Build script_units.json from image_prompts.json scene_text.

Extracts exactly 64 canonical text units that match the sentence indices
used by image_prompts.json. This file becomes the single source of truth
for stable-ts alignment and timestamp generation.

Usage:
    python scripts/build_script_units.py --video-id <vid>
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def split_into_sentences(text: str) -> list[str]:
    """Split a multi-sentence text at sentence boundaries (.!?)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()

    video_dir = Path(config.OUTPUT_DIR) / args.video_id
    prompts_path = video_dir / "image_prompts.json"
    out_path = video_dir / "script_units.json"

    if not prompts_path.exists():
        print(f"ERROR: image_prompts.json not found: {prompts_path}", file=sys.stderr)
        sys.exit(1)

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))

    units: dict[int, str] = {}

    for scene in prompts:
        indices = scene.get("sentences", [])
        scene_text = scene.get("scene_text", "").strip()

        if not indices or not scene_text:
            print(f"WARNING: Scene {scene['index']} missing sentences or scene_text", file=sys.stderr)
            continue

        if len(indices) == 1:
            units[indices[0]] = scene_text
        else:
            # Multiple sentence slots — split scene_text at sentence boundaries
            parts = split_into_sentences(scene_text)
            if len(parts) == len(indices):
                for idx, part in zip(indices, parts):
                    units[idx] = part
            else:
                # Fallback: split evenly or assign full text to first, rest empty
                print(
                    f"  WARNING: Scene {scene['index']} has {len(indices)} indices "
                    f"but {len(parts)} sentence parts — assigning by position",
                    file=sys.stderr,
                )
                for i, idx in enumerate(indices):
                    units[idx] = parts[i] if i < len(parts) else scene_text

    # Sort by index and validate
    sorted_indices = sorted(units.keys())
    expected = list(range(1, len(sorted_indices) + 1))
    if sorted_indices != expected:
        print(f"WARNING: Indices are not consecutive 1..N: {sorted_indices[:10]}...", file=sys.stderr)

    result = [{"index": idx, "text": units[idx]} for idx in sorted_indices]

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(result)} units to {out_path}")
    for u in result:
        text_preview = u['text'][:80].encode('ascii', errors='replace').decode('ascii')
        print(f"  [{u['index']:02d}] {text_preview}")


if __name__ == "__main__":
    main()
