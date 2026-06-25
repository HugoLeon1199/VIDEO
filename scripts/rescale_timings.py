"""Rescale image_prompts.json scene timings from timestamps.json.

Reads the existing image_prompts.json (with manually written prompts),
maps each scene's sentence indices to real timestamps from timestamps.json,
and writes back updated start/end without touching prompts or overlays.

Usage:
    python scripts/rescale_timings.py --video-id <vid>
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()

    video_dir = Path(config.OUTPUT_DIR) / args.video_id
    prompts_path = video_dir / "image_prompts.json"
    timestamps_path = video_dir / "timestamps.json"

    if not prompts_path.exists():
        print(f"ERROR: image_prompts.json not found: {prompts_path}", file=sys.stderr)
        sys.exit(1)
    if not timestamps_path.exists():
        print(f"ERROR: timestamps.json not found: {timestamps_path}", file=sys.stderr)
        sys.exit(1)

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    timestamps = json.loads(timestamps_path.read_text(encoding="utf-8"))

    # Build index: sentence_number (1-based) → {start, end}
    ts_map: dict[int, dict] = {t["index"]: t for t in timestamps}
    total_ts = len(timestamps)

    print(f"Loaded {len(prompts)} scenes, {total_ts} sentence timestamps")

    updated = 0
    warned = 0
    for scene in prompts:
        sent_indices = scene.get("sentences", [])
        if not sent_indices:
            print(f"  WARNING: Scene {scene['index']} has no sentences — skipping", file=sys.stderr)
            warned += 1
            continue

        first_idx = min(sent_indices)
        last_idx = max(sent_indices)

        first_ts = ts_map.get(first_idx)
        last_ts = ts_map.get(last_idx)

        if first_ts is None:
            print(
                f"  WARNING: Scene {scene['index']} references sentence {first_idx} "
                f"which doesn't exist in timestamps.json (max={total_ts})",
                file=sys.stderr,
            )
            warned += 1
            continue
        if last_ts is None:
            print(
                f"  WARNING: Scene {scene['index']} references sentence {last_idx} "
                f"which doesn't exist in timestamps.json (max={total_ts})",
                file=sys.stderr,
            )
            warned += 1
            continue

        old_start = scene.get("start", 0)
        old_end = scene.get("end", 0)
        scene["start"] = round(first_ts["start"], 3)
        scene["end"] = round(last_ts["end"], 3)

        if scene["start"] != old_start or scene["end"] != old_end:
            updated += 1

    # Cap last scene end to last timestamp end (avoid floating point overshoot)
    if prompts and timestamps:
        last_ts_end = timestamps[-1]["end"]
        prompts[-1]["end"] = round(last_ts_end, 3)

    prompts_path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Done: {updated}/{len(prompts)} scenes updated"
        + (f", {warned} warnings" if warned else "")
        + f" → {prompts_path}"
    )


if __name__ == "__main__":
    main()
