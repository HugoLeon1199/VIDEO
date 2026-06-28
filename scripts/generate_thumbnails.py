from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

import config
from steps.thumbnails import generate_thumbnail_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thumbnail backgrounds and final JPGs")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--regenerate", type=int, action="append", default=[])
    parser.add_argument("--allow-stale-package", action="store_true")
    args = parser.parse_args()

    diagnostics = generate_thumbnail_assets(
        args.video_id,
        regenerate=args.regenerate,
        allow_stale_package=args.allow_stale_package,
    )
    logger.info(
        "Thumbnails done: {} generated, failed ids={}",
        diagnostics["thumbnail_generated_count"],
        diagnostics["thumbnail_failed_ids"],
    )


if __name__ == "__main__":
    main()
