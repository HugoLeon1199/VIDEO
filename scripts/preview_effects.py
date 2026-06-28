from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steps import render_video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a local effects preview using the production renderer")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--seconds", type=float, default=45.0)
    args = parser.parse_args()

    render_video.render_preview(args.video_id, seconds=args.seconds)


if __name__ == "__main__":
    main()
