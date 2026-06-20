"""Quick test: submit 1 scene to RunPod and save the result locally."""
import base64
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Load .env
env_file = pathlib.Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from image_generation.runpod_client import RunPodClient

client = RunPodClient()

job_input = {
    "video_id": "test",
    "scene_id": "001",
    "mode": "text_to_image",
    "prompt": "ancient human sitting by fire, cave painting style, ochre on stone",
    "global_style": "prehistoric cave painting, ochre and charcoal, no text",
    "width": 512,
    "height": 288,
    "steps": 4,
    "guidance_scale": 1.0,
    "candidate_seeds": [42],
    "output_format": "WEBP",
    "quality": 80,
    "output_mode": "base64",
}

print("Submitting test job...")
job_id = client.submit(job_input)
print(f"Job submitted: {job_id}")
print("Polling for result (may take 1-3 min on cold start)...")

result = client.poll_until_done(job_id)
output = result.get("output", {})
imgs = output.get("images", [])
errors = output.get("errors", [])

print(f"Images: {len(imgs)}, Errors: {errors}")

if imgs:
    data = base64.b64decode(imgs[0]["base64"])
    out_path = pathlib.Path("test_scene001.webp")
    out_path.write_bytes(data)
    sha = imgs[0]["sha256"]
    print(f"Saved {out_path} ({len(data)} bytes) sha256={sha[:16]}...")
    print("SUCCESS")
else:
    print("No images returned. Full output:")
    print(json.dumps(output, indent=2)[:800])
    sys.exit(1)
