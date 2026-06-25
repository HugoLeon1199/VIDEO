"""
Regenerate timestamps.json for VI video using stable-ts forced alignment.

Aligns audio.mp3 against script_units.json (single source of truth),
producing exactly N timestamp entries — one per script unit.

Usage:
    python scripts/regen_timestamps_vi.py --video-id <vid> [--model medium] [--device cpu]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from steps.transcribe import _assign_words_to_sentences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    video_dir = Path(config.OUTPUT_DIR) / args.video_id
    units_path = video_dir / "script_units.json"
    audio_path = video_dir / "audio.mp3"
    out_path = video_dir / "timestamps.json"

    if not units_path.exists():
        print(f"ERROR: script_units.json not found — run build_script_units.py first", file=sys.stderr)
        sys.exit(1)
    if not audio_path.exists():
        print(f"ERROR: audio.mp3 not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    units = json.loads(units_path.read_text(encoding="utf-8"))
    texts = [u["text"] for u in units]
    print(f"Loaded {len(texts)} units from script_units.json")

    canonical_text = " ".join(texts)
    print(f"Canonical text: {len(canonical_text)} chars")

    import stable_whisper
    import mutagen.mp3

    print(f"Loading stable-ts model ({args.model}, {args.device})...")
    model = stable_whisper.load_model(args.model, device=args.device)

    print(f"Aligning {audio_path.name}...")
    result = model.align(str(audio_path), canonical_text, language="vi")

    words = []
    for seg in result.segments:
        for w in seg.words:
            if w.start is not None and w.end is not None:
                words.append({"word": w.word, "start": w.start, "end": w.end})

    print(f"stable-ts produced {len(words)} word timestamps")

    if not words:
        print("ERROR: No word timestamps produced — check audio/script match", file=sys.stderr)
        sys.exit(1)

    audio_duration = mutagen.mp3.MP3(str(audio_path)).info.length
    print(f"Audio duration: {audio_duration:.1f}s")

    timestamps = _assign_words_to_sentences(texts, words, audio_duration)

    out_path.write_text(json.dumps(timestamps, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(timestamps)} entries → {out_path}")
    print(f"Span: {timestamps[0]['start']:.2f}s — {timestamps[-1]['end']:.2f}s")

    # Quick quality check
    durations = [t["end"] - t["start"] for t in timestamps]
    gaps = [timestamps[i]["start"] - timestamps[i-1]["end"] for i in range(1, len(timestamps))]
    print(f"\nQuality check:")
    print(f"  Duration range: {min(durations):.2f}s — {max(durations):.2f}s (avg {sum(durations)/len(durations):.2f}s)")
    if gaps:
        print(f"  Gap range: {min(gaps):.2f}s — {max(gaps):.2f}s")
    print(f"  Last entry end: {timestamps[-1]['end']:.2f}s vs audio: {audio_duration:.2f}s")


if __name__ == "__main__":
    main()
