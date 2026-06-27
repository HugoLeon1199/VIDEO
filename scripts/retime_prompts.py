"""Re-time image_prompts.json start/end against the real transcript (timestamps.json).

Why: a video's image_prompts.json can carry timings computed by character-ratio
estimation on an OLD/different audio, so images drift out of sync with the voice
(and the final image freezes for tens of seconds at the end). The scene_text values
themselves are an in-order subset of the transcript sentences, so we can re-anchor
each scene to the FIRST transcript sentence that belongs to it and rewrite start/end.

Usage:
    python scripts/retime_prompts.py --video-id <slug>            # apply
    python scripts/retime_prompts.py --video-id <slug> --dry-run  # show only
"""

import argparse
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def _overlap(a: str, b: str) -> float:
    """Word-overlap ratio relative to the shorter token set."""
    A, B = set(a.split()), set(b.split())
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, min(len(A), len(B)))


def _belongs(ts_norm: str, scene_norm: str, threshold: float = 0.6) -> bool:
    return bool(ts_norm) and (ts_norm in scene_norm or _overlap(ts_norm, scene_norm) >= threshold)


def retime(prompts: list[dict], timestamps: list[dict]) -> list[dict]:
    """Return prompts with start/end re-anchored to the real transcript.

    Each scene's start = the FIRST transcript sentence belonging to it (range start,
    not best-overlap). A constraint window [max(cursor,k), N-M+k] guarantees a
    strictly-increasing, gap-free, monotonic timeline without backward clamping.
    """
    N, M = len(timestamps), len(prompts)
    if M == 0:
        return prompts
    if M > N:
        logger_warn = f"More prompts ({M}) than transcript sentences ({N}) — cannot re-time safely."
        raise ValueError(logger_warn)

    ts_norm = [_norm(t["text"]) for t in timestamps]
    audio_end = timestamps[-1]["end"]

    anchors: list[int] = []
    cursor = 0
    for k, p in enumerate(prompts):
        scene = _norm(p.get("scene_text", ""))
        lo = max(cursor, k)
        hi = N - M + k  # leave at least one sentence for each remaining scene
        start_i = None
        for i in range(lo, hi + 1):
            if _belongs(ts_norm[i], scene):
                start_i = i
                break
        if start_i is None:
            start_i = lo
        anchors.append(start_i)
        # consume the trailing sentences that clearly belong to this same scene
        j = start_i
        while j + 1 <= hi and _belongs(ts_norm[j + 1], scene):
            j += 1
        cursor = j + 1

    out = []
    for i, p in enumerate(prompts):
        start = timestamps[anchors[i]]["start"]
        end = timestamps[anchors[i + 1]]["start"] if i < M - 1 else audio_end
        np = dict(p)
        np["start"] = round(start, 3)
        np["end"] = round(end, 3)
        out.append(np)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    video_dir = Path(config.OUTPUT_DIR) / args.video_id
    prompts_path = video_dir / "image_prompts.json"
    timestamps_path = video_dir / "timestamps.json"
    for p in (prompts_path, timestamps_path):
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            sys.exit(1)

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    timestamps = json.loads(timestamps_path.read_text(encoding="utf-8"))

    new_prompts = retime(prompts, timestamps)

    # report
    durs = []
    print(f"{'P#':>4} {'old_start':>10} {'new_start':>10} {'hold(s)':>8}  scene_text")
    for i, (old, new) in enumerate(zip(prompts, new_prompts)):
        nxt = new_prompts[i + 1]["start"] if i < len(new_prompts) - 1 else timestamps[-1]["end"]
        hold = nxt - new["start"]
        durs.append(hold)
        st = (new.get("scene_text") or "")[:42]
        print(f"{new['index']:>4} {old['start']:>10.2f} {new['start']:>10.2f} {hold:>8.1f}  {st}")

    bad = sum(1 for d in durs if d <= 0)
    mono = all(new_prompts[i]["start"] < new_prompts[i + 1]["start"] for i in range(len(new_prompts) - 1))
    print()
    print(f"strictly-increasing: {mono} | non-positive durations: {bad}")
    print(f"first start: {new_prompts[0]['start']} | last end: {new_prompts[-1]['end']} | audio end: {timestamps[-1]['end']}")
    print(f"max hold: {max(durs):.1f}s | min hold: {min(durs):.2f}s")

    if bad or not mono:
        print("ABORT: timeline not clean — not writing.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\n(dry-run) no files written.")
        return

    backup = prompts_path.with_suffix(".json.bak")
    shutil.copy2(prompts_path, backup)
    prompts_path.write_text(json.dumps(new_prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {prompts_path} (backup: {backup.name})")


if __name__ == "__main__":
    main()
