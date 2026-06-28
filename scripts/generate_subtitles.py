from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steps import subtitles as subtitle_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate subtitle artifacts and preview videos")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--style", default=subtitle_step.STYLE_CINEMATIC_CLEAN, choices=sorted(subtitle_step.VALID_STYLES))
    parser.add_argument("--preview-seconds", type=float, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    result = subtitle_step.generate(args.video_id, style=args.style)
    print(f"Generated subtitles for {args.video_id}: {len(result['cues'])} cues")
    print(f"Diagnostics: output/{args.video_id}/subtitle_diagnostics.json")

    if args.validate_only:
        print("Validation only complete.")
        return

    if args.preview_seconds is not None:
        preview_path = subtitle_step.render_preview(
            args.video_id,
            style=args.style,
            preview_seconds=args.preview_seconds,
        )
        print(f"Preview rendered: {preview_path}")


if __name__ == "__main__":
    main()
