import sys, os
sys.path.insert(0, ".")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import re, subprocess, tempfile
import numpy as np
from pathlib import Path
from vieneu import Vieneu

voice = "Duc Tri"  # will use the full name below
voice_full = "Đức Trí"
vid = "to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi"
out_mp3 = f"output/{vid}_duc-tri/audio.mp3"

script_path = Path("output") / vid / "script.txt"
text = script_path.read_text(encoding="utf-8").strip()

tts = Vieneu()
preset = tts.get_preset_voice(voice_full)
ref_codes = preset["codes"]

chunks = []
for para in re.split(r"\n+", text):
    para = para.strip()
    if not para:
        continue
    if len(para) <= 200:
        chunks.append(para)
    else:
        for sent in re.split(r"(?<=[.!?…])\s+", para):
            s = sent.strip()
            if s:
                chunks.append(s[:200])

print(f"Voice: Duc Tri | {len(chunks)} chunks", flush=True)
sample_rate = 48000
silence = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
parts = []

for i, chunk in enumerate(chunks, 1):
    wav = tts.infer(chunk, ref_codes=ref_codes, temperature=0.5, top_k=20, top_p=0.90)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    parts.append(wav.astype(np.float32))
    parts.append(silence)
    if i % 10 == 0 or i == len(chunks):
        print(f"  {i}/{len(chunks)}", flush=True)

combined = np.concatenate(parts)

import soundfile as sf
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tmp_path = tmp.name
sf.write(tmp_path, combined, sample_rate)

subprocess.run(
    ["ffmpeg", "-y", "-i", tmp_path, "-codec:a", "libmp3lame", "-qscale:a", "0", out_mp3],
    check=True, capture_output=True,
)
from pathlib import Path as P
P(tmp_path).unlink(missing_ok=True)
print(f"Saved: {out_mp3}", flush=True)
