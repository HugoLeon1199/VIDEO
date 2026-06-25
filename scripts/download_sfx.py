"""Prepare SFX assets for the soundscape pipeline.

WORKFLOW:
1. Run: python scripts/download_sfx.py --list
   → See which files are missing + Pixabay search links

2. For each missing file:
   - Open the Pixabay link in your browser
   - Click any sound → click "Free Download" → save as MP3
   - Move the downloaded file to assets/sfx/raw/<tag>.mp3

3. Run: python scripts/download_sfx.py --normalize
   → Converts all raw/*.mp3 to normalized WAV in assets/sfx/

Usage:
    python scripts/download_sfx.py --list
    python scripts/download_sfx.py --normalize
    python scripts/download_sfx.py --normalize --tag birds
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SFX_DIR = Path("assets/sfx")
RAW_DIR = SFX_DIR / "raw"

# Pixabay search URLs — open in browser, download any result as MP3 → rename to <tag>.mp3
PIXABAY_SEARCH: dict[str, str] = {
    "birds":         "https://pixabay.com/sound-effects/search/birds-forest-morning/",
    "wind":          "https://pixabay.com/sound-effects/search/wind-outdoor-nature/",
    "fire":          "https://pixabay.com/sound-effects/search/campfire-crackling/",
    "water":         "https://pixabay.com/sound-effects/search/river-stream-water/",
    "crowd_murmur":  "https://pixabay.com/sound-effects/search/crowd-murmur-ambient/",
    "night_insects": "https://pixabay.com/sound-effects/search/crickets-night-insects/",
    "footsteps":     "https://pixabay.com/sound-effects/search/footsteps-dirt-grass/",
    "impact_soft":   "https://pixabay.com/sound-effects/search/thud-impact-soft/",
    "impact_hard":   "https://pixabay.com/sound-effects/search/impact-hit-hard/",
    "whoosh":        "https://pixabay.com/sound-effects/search/whoosh-transition-swipe/",
    "ui_alert":      "https://pixabay.com/sound-effects/search/notification-ding-alert/",
    "drone_tension": "https://pixabay.com/sound-effects/search/tension-drone-dark-ambient/",
}


def _ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        winget_path = (
            r"C:\Users\LEON_RM\AppData\Local\Microsoft\WinGet\Packages"
            r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
        )
        if Path(winget_path).exists():
            return winget_path
        print("ERROR: ffmpeg not found. Install: winget install Gyan.FFmpeg", file=sys.stderr)
        sys.exit(1)
    return ffmpeg


def _normalize_to_wav(src: Path, dst: Path, ffmpeg: str) -> bool:
    """Convert any audio file to 44100Hz stereo WAV."""
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-ar", "44100", "-ac", "2",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:].decode(errors='ignore')}")
        return False
    kb = dst.stat().st_size // 1024
    print(f"  OK: {dst.name} ({kb}KB)")
    return True


def cmd_list() -> None:
    print(f"SFX assets status ({len(PIXABAY_SEARCH)} tags):\n")
    any_missing = False
    for tag, url in PIXABAY_SEARCH.items():
        wav = SFX_DIR / f"{tag}.wav"
        raw_mp3 = RAW_DIR / f"{tag}.mp3"
        raw_wav = RAW_DIR / f"{tag}.wav"
        if wav.exists():
            kb = wav.stat().st_size // 1024
            print(f"  [OK     ] {tag}.wav ({kb}KB)")
        elif raw_mp3.exists() or raw_wav.exists():
            print(f"  [RAW OK ] {tag} — run --normalize to convert")
        else:
            print(f"  [MISSING] {tag}")
            print(f"            Download: {url}")
            print(f"            Save as:  assets/sfx/raw/{tag}.mp3")
            any_missing = True

    if any_missing:
        print(f"\nRaw folder: {RAW_DIR.absolute()}")
        print("After downloading all files, run: python scripts/download_sfx.py --normalize")
    else:
        print("\nAll tags ready!")


def cmd_normalize(tag_filter: str | None = None) -> None:
    ffmpeg = _ensure_ffmpeg()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SFX_DIR.mkdir(parents=True, exist_ok=True)

    tags = [tag_filter] if tag_filter else list(PIXABAY_SEARCH.keys())
    ok = 0
    fail = 0
    skip = 0

    for tag in tags:
        wav_out = SFX_DIR / f"{tag}.wav"
        if wav_out.exists():
            print(f"  [SKIP] {tag}.wav already exists")
            skip += 1
            continue

        # Look for raw file in various formats
        src = None
        for ext in ("mp3", "wav", "ogg", "flac", "m4a"):
            candidate = RAW_DIR / f"{tag}.{ext}"
            if candidate.exists():
                src = candidate
                break

        if src is None:
            print(f"  [MISS] {tag} — no raw file found in {RAW_DIR}/")
            print(f"         Download from: {PIXABAY_SEARCH.get(tag, '')}")
            fail += 1
            continue

        print(f"  Normalizing {src.name}...", end=" ")
        if _normalize_to_wav(src, wav_out, ffmpeg):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} normalized, {skip} skipped, {fail} missing/failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SFX WAV files for soundscape pipeline"
    )
    parser.add_argument("--list", action="store_true", help="Show status of all tags")
    parser.add_argument("--normalize", action="store_true", help="Convert raw files to WAV")
    parser.add_argument("--tag", help="Process only this tag (use with --normalize)")
    args = parser.parse_args()

    if args.list or (not args.normalize and not args.list):
        cmd_list()

    if args.normalize:
        print(f"\nNormalizing raw files from {RAW_DIR}/ to {SFX_DIR}/\n")
        cmd_normalize(tag_filter=args.tag)


if __name__ == "__main__":
    main()
