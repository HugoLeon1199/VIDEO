"""Generate audio.mp3 for a given VieNeu voice then exit."""
import os, sys, json, re, subprocess, tempfile
import numpy as np
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, ".")

from pathlib import Path
from vieneu import Vieneu

voice = sys.argv[1]          # e.g. "Bình An"
vid   = sys.argv[2]          # source video id
out_mp3 = sys.argv[3]        # output path for audio.mp3

script_path = Path("output") / vid / "script.txt"
text = script_path.read_text(encoding="utf-8").strip()

tts = Vieneu()

# Use voice= name → reserved_id speaker token (most stable path)
# Full script in one call; VieNeu chunks internally + crossfade
print(f"Voice: {voice} | full script ({len(text)} chars) in one call")
combined = tts.infer(
    text,
    voice=voice,
    temperature=0.5,
    top_k=20,
    top_p=0.90,
    repetition_penalty=1.2,
    max_chars=256,
    crossfade_p=0.1,
    silence_p=0.12,
    apply_watermark=False,
)
if combined.ndim > 1:
    combined = combined.mean(axis=1)
combined = combined.astype(np.float32)
print(f"Done: {len(combined)/tts.sample_rate:.1f}s")

import soundfile as sf
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tmp_path = tmp.name
sf.write(tmp_path, combined, tts.sample_rate)

subprocess.run(
    ["ffmpeg", "-y", "-i", tmp_path, "-codec:a", "libmp3lame", "-qscale:a", "0", out_mp3],
    check=True, capture_output=True,
)
Path(tmp_path).unlink(missing_ok=True)
print(f"Saved: {out_mp3}")
