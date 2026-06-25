"""Create 5 style test prompts for comparison."""
import json
from pathlib import Path

SCENE = (
    "prehistoric man wearing full animal hide clothing sitting peacefully "
    "under a large acacia tree in African savanna at golden hour, "
    "elephants in background, warm afternoon, at rest, work done"
)
NEGATIVE = (
    "nudity, bare chest, bare skin, topless, shirtless, exposed torso, "
    "text, watermark, logo, extra limbs, deformed"
)

styles = [
    {
        "index": 1, "start": 0.0, "end": 5.0, "text": "style test 1",
        "prompt": (
            "flat 2D vector illustration, Kurzgesagt style, "
            "warm amber golden palette, simple bold shapes, clean silhouettes, " + SCENE
        ),
        "negative_prompt": NEGATIVE, "global_style": "",
    },
    {
        "index": 2, "start": 0.0, "end": 5.0, "text": "style test 2",
        "prompt": (
            "BBC nature documentary digital painting, semi-realistic illustration, "
            "painterly brushstrokes, warm cinematic lighting, rich earthy colors, " + SCENE
        ),
        "negative_prompt": NEGATIVE, "global_style": "",
    },
    {
        "index": 3, "start": 0.0, "end": 5.0, "text": "style test 3",
        "prompt": (
            "watercolor illustration, soft washes of warm amber and ochre, "
            "loose painterly style, National Geographic book illustration, " + SCENE
        ),
        "negative_prompt": NEGATIVE, "global_style": "",
    },
    {
        "index": 4, "start": 0.0, "end": 5.0, "text": "style test 4",
        "prompt": (
            "epic cinematic matte painting, dramatic golden hour lighting, "
            "illustrated figures in vast landscape, sweeping wide angle, "
            "high detail environment, documentary epic style, " + SCENE
        ),
        "negative_prompt": NEGATIVE, "global_style": "",
    },
    {
        "index": 5, "start": 0.0, "end": 5.0, "text": "style test 5",
        "prompt": (
            "graphic novel illustration, bold ink outlines, warm flat colors, "
            "expressive faces, serious mature tone, clean composition, " + SCENE
        ),
        "negative_prompt": NEGATIVE, "global_style": "",
    },
]

out = Path("output/style_test")
out.mkdir(parents=True, exist_ok=True)
(out / "image_prompts.json").write_text(
    json.dumps(styles, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("Saved 5 style test prompts")
for s in styles:
    print(f"  {s['index']}: {s['prompt'][:80]}...")
