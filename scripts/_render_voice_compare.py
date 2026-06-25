"""Render a voice-compare video using images + image_prompts from source video,
but audio from a different folder. Re-scales timings to match new audio duration."""
import json, subprocess, sys, shutil, tempfile
from pathlib import Path

src_vid  = sys.argv[1]   # source video id (images + prompts)
cmp_dir  = sys.argv[2]   # folder with new audio.mp3

base_src = Path("output") / src_vid
cmp_path = Path(cmp_dir)

# Load source image_prompts (with scaled timings from Thai Son run)
ip = json.loads((base_src / "image_prompts.json").read_text(encoding="utf-8"))

# Get new audio duration
r = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "default=noprint_wrappers=1:nokey=1", str(cmp_path / "audio.mp3")],
    capture_output=True, text=True,
)
new_dur = float(r.stdout.strip())
old_dur = ip[-1]["end"]
scale   = new_dur / old_dur
print(f"Audio: {new_dur:.1f}s | Source: {old_dur:.1f}s | Scale: {scale:.4f}")

# Scale timings
scaled_ip = []
for e in ip:
    ne = dict(e)
    ne["start"] = round(e["start"] * scale, 3)
    ne["end"]   = round(e["end"]   * scale, 3)
    scaled_ip.append(ne)
scaled_ip[-1]["end"] = round(new_dur, 3)

# Write scaled prompts to cmp folder
(cmp_path / "image_prompts.json").write_text(
    json.dumps(scaled_ip, ensure_ascii=False, indent=2), encoding="utf-8"
)

# Find images dir from source
images_dir = None
for candidate in ["images_flat2d", "images_en", "images_vi", "images"]:
    d = base_src / candidate
    if (d / "img_001.png").exists():
        images_dir = d
        break

if not images_dir:
    print("ERROR: no images found in source video")
    sys.exit(1)
print(f"Images: {images_dir}")

# Build clips
n = len(scaled_ip)
clip_durations = []
for i in range(n):
    start = scaled_ip[i]["start"]
    end   = scaled_ip[i + 1]["start"] if i + 1 < n else new_dur
    clip_durations.append(round(end - start, 3))

with tempfile.TemporaryDirectory(prefix="render_cmp_") as tmpdir:
    tmpdir = Path(tmpdir)
    clip_list = []

    ffmpeg = shutil.which("ffmpeg") or r"C:\Users\LEON_RM\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1-full_build\bin\ffmpeg.exe"

    print(f"Converting {n} images to clips...")
    for i, (item, dur) in enumerate(zip(scaled_ip, clip_durations)):
        img = images_dir / f"img_{item['index']:03d}.png"
        clip = tmpdir / f"clip_{i:04d}.mp4"
        subprocess.run([
            ffmpeg, "-y", "-loop", "1", "-i", str(img),
            "-t", str(dur), "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-r", "30", clip,
        ], check=True, capture_output=True)
        clip_list.append(clip)
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"  {i+1}/{n} clips done")

    # Concat list
    concat_file = tmpdir / "clips.txt"
    concat_file.write_text("\n".join(f"file '{c}'" for c in clip_list))

    out_mp4 = cmp_path / "final.mp4"
    print("Concatenating + muxing audio...")
    subprocess.run([
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-i", str(cmp_path / "audio.mp3"),
        "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-b:v", "8M",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2",
        "-shortest", "-r", "30",
        str(out_mp4),
    ], check=True, capture_output=True)

size_mb = out_mp4.stat().st_size / 1024 / 1024
print(f"Done: {out_mp4} ({size_mb:.1f} MB)")
