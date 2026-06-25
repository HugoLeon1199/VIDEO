"""Generate image_prompts.json directly — no external API needed.
Run this script via Claude Code which provides the generation logic inline.
"""
import re, json, subprocess
from pathlib import Path

STOP = ("COMMENT SEED:", "RESEARCH NOTES:", "Your script is ready", "Save as:")

def load_script(path):
    text = Path(path).read_text(encoding="utf-8").lstrip("﻿")
    for m in STOP:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
    return text.strip()

def split_sentences(text):
    paragraphs = re.split(r"\n{2,}", text)
    sentences = []
    first = True
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if first:
            first = False
            if not re.search(r"[.!?]$", para) and len(para.split()) <= 12:
                continue
        for part in re.split(r"(?<=[.!?])\s+", para):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences

def get_audio_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())

def compute_times(sentences, total_dur):
    total_chars = sum(len(s) for s in sentences)
    cursor = 0.0
    times = []
    for i, s in enumerate(sentences, 1):
        dur = total_dur * len(s) / total_chars
        times.append({"index": i, "start": round(cursor, 3), "end": round(cursor + dur, 3), "text": s})
        cursor += dur
    return times

vid = "to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi"
script_path = f"output/{vid}/script.txt"
audio_path = f"output/{vid}/audio.mp3"

text = load_script(script_path)
sentences = split_sentences(text)
total_dur = get_audio_duration(audio_path)
times = compute_times(sentences, total_dur)

print(f"Sentences: {len(sentences)}, Audio: {total_dur:.1f}s")
print(json.dumps(times, ensure_ascii=False, indent=2))
