"""Quick local test: F5-TTS clone giong tu file mau, doc doan dau script."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from f5_tts.api import F5TTS

REF_AUDIO_MP3 = r"d:\CODE\VIDEO\YOUTUBE\output\ancient-child-surgery-31000-years-vi\voice_ref_nhat.mp3"
REF_AUDIO = r"d:\CODE\VIDEO\YOUTUBE\output\ancient-child-surgery-31000-years-vi\voice_ref_nhat.wav"
OUTPUT = r"d:\CODE\VIDEO\YOUTUBE\output\ancient-child-surgery-31000-years-vi\test_f5tts.mp3"

# Convert MP3 -> WAV for torchaudio compatibility on Windows
import subprocess as _sp
_sp.run(["ffmpeg", "-y", "-i", REF_AUDIO_MP3, "-ar", "24000", "-ac", "1", REF_AUDIO],
        check=True, capture_output=True)

TEXT = (
    "Chân Trái Của Một Đứa Trẻ Bị Cắt Cụt 31.000 Năm Trước. Và Nó Đã Sống. "
    "Đặt tay lên ống chân của bạn. Cảm nhận khúc xương dài ngay dưới da, và xương mỏng hơn bên cạnh nó. "
    "Rồi thử tưởng tượng nửa dưới cái chân đó biến mất hoàn toàn — không phải trong bệnh viện với mặt nạ trên mặt "
    "và máy móc đếm nhịp tim, mà giữa cái nóng ẩm ướt của rừng nhiệt đới, 31.000 năm trước, "
    "với một hòn đá mài nhọn và những người yêu thương bạn đứng xung quanh."
)

print("Loading F5-TTS model (first time downloads ~1GB)...")
tts = F5TTS()

print(f"Cloning voice from: {REF_AUDIO}")
print(f"Text length: {len(TEXT)} chars")

wav, sr, _ = tts.infer(
    ref_file=REF_AUDIO,
    ref_text="Xin kiến chào quý bạn và các vị. Món rằng dọng nói này sẽ phù hợp với bất kỳ dự án nào của bạn. Cảm ơn đã lựa chọn. Hãy thử ngay nhé.",
    gen_text=TEXT,
    speed=0.95,
)

import numpy as np, soundfile as sf, io, subprocess
buf = io.BytesIO()
wav_np = wav.squeeze().cpu().numpy() if hasattr(wav, "cpu") else np.array(wav).squeeze()
sf.write(buf, wav_np, sr, format="WAV")
buf.seek(0)

# Convert WAV -> MP3
proc = subprocess.run(
    ["ffmpeg", "-y", "-i", "pipe:0", "-codec:a", "libmp3lame", "-qscale:a", "2", OUTPUT],
    input=buf.read(), capture_output=True,
)
if proc.returncode == 0:
    size = Path(OUTPUT).stat().st_size // 1024
    print(f"\nDone! Output: {OUTPUT} ({size} KB)")
    print("Mo file nay de nghe thu.")
else:
    print("ffmpeg error:", proc.stderr.decode())
    sys.exit(1)
