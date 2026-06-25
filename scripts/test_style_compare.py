"""
Generate 4 style test images (same scene, different styles) for style selection.
Output: output/style_test/style_A.png, style_B.png, style_C.png, style_D.png
"""
import time
import base64
from pathlib import Path
import httpx

# Load .env manually
_env_file = Path(__file__).parent.parent / ".env"
_env = {}
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _env[_k.strip()] = _v.strip()

RUNPOD_API_KEY = _env.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = _env.get("RUNPOD_ENDPOINT_ID", "")
BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

SCENE_DESC = (
    "A Paleolithic man slowly sitting up at dawn inside a simple branch shelter, "
    "other sleeping figures near glowing embers in background, medium shot"
)

NEGATIVE_COMMON = (
    "nudity, bare chest, exposed torso, topless, shirtless, "
    "extra limbs, fused limbs, deformed anatomy, "
    "text, watermark, logo, blurry"
)

STYLES = {
    "style_A_watercolor": (
        f"watercolor illustration, prehistoric humans fully clothed in animal-hide robes, "
        f"warm earthy tones, loose painterly brushstrokes, National Geographic documentary style, "
        f"soft golden light, African savanna, highly detailed, 16:9, "
        f"{SCENE_DESC}, "
        f"correct anatomy, clean poses"
    ),
    "style_B_flat2D": (
        f"2D animated illustration, flat design, prehistoric people wearing full robes and cloaks, "
        f"bold clean outlines, warm color palette, simple shapes, "
        f"Disney nature documentary style, earthy browns and oranges, 16:9, "
        f"{SCENE_DESC}, "
        f"correct anatomy, clean poses"
    ),
    "style_C_inkwash": (
        f"ink wash illustration, prehistoric humans wearing draped robes, bold ink outlines, "
        f"limited earthy color palette, expressive brushwork, "
        f"historical documentary art style, warm sepia tones, 16:9, "
        f"{SCENE_DESC}, "
        f"correct anatomy, clean poses"
    ),
    "style_D_storybook": (
        f"illustrated storybook style, prehistoric people wearing full body robes and cloaks, "
        f"detailed hand-drawn art, warm amber lighting, earthy color palette, "
        f"historical illustration, clean linework, classic book illustration style, 16:9, "
        f"{SCENE_DESC}, "
        f"correct anatomy, clean poses"
    ),
}

out_dir = Path("output/style_test")
out_dir.mkdir(parents=True, exist_ok=True)

def submit_job(prompt):
    payload = {
        "input": {
            "prompt": prompt,
            "negative_prompt": NEGATIVE_COMMON,
            "width": 1024,
            "height": 576,
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "seed": 42,
            "output_format": "WEBP",
        }
    }
    r = httpx.post(f"{BASE_URL}/run", headers=HEADERS, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["id"]

def poll_job(job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = httpx.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status == "COMPLETED":
            return data["output"]
        elif status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Job {job_id} {status}: {data}")
        time.sleep(3)
    raise TimeoutError(f"Job {job_id} timed out")

# Submit all 4 in parallel
print("Submitting 4 style test jobs...")
jobs = {}
for name, prompt in STYLES.items():
    job_id = submit_job(prompt)
    jobs[name] = job_id
    print(f"  {name}: {job_id}")

# Poll and save
print("Waiting for results...")
for name, job_id in jobs.items():
    try:
        output = poll_job(job_id)
        # Output format: {"images": [{"base64": "data:image/webp;base64,..."}], ...}
        images_list = output.get("images", [])
        if not images_list:
            raise ValueError(f"No images in output: {list(output.keys())}")
        img_entry = images_list[0]
        img_b64 = img_entry["base64"] if isinstance(img_entry, dict) else img_entry
        if img_b64.startswith("data:"):
            img_b64 = img_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(img_b64)
        out_path = out_dir / f"{name}.webp"
        out_path.write_bytes(img_bytes)
        print(f"  Saved {out_path}")
    except Exception as e:
        print(f"  ERROR {name}: {e}")

print(f"\nDone. Check output/style_test/")
