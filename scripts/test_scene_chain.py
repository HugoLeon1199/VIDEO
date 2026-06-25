"""
Test the scene chain pipeline with 3 scenes:
  Scene 1 (new)   -> t2i: campfire at night
  Scene 2 (angle) -> img2img 0.72: zoom in to elder face at same campfire
  Scene 3 (pose)  -> img2img 0.55: elder raises hand, same composition

Run:
  $python scripts/test_scene_chain.py
"""
import logging
import os
import sys

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from image_generation.runpod_serverless_backend import RunPodServerlessBackend
from image_generation.scene_chain import SceneChain, ChainedScene

backend = RunPodServerlessBackend()

chain = SceneChain(
    backend=backend,
    video_id="chain-test",
    width=1024,
    height=576,
    steps=22,
    guidance_scale=3.5,
    candidate_seeds=[11001],
)

scenes = [
    ChainedScene(
        scene_id="001",
        transition="new",
        prompt=(
            "prehistoric tribe sitting around a large campfire at night, "
            "African savanna, starry sky, wide establishing shot, "
            "warm orange firelight, silhouettes of 4-5 people"
        ),
    ),
    ChainedScene(
        scene_id="002",
        transition="angle",   # strength=0.72: keep palette, change composition
        prompt=(
            "close-up portrait of prehistoric elder man sitting at campfire, "
            "same warm orange firelight, wise expression, looking into fire"
        ),
    ),
    ChainedScene(
        scene_id="003",
        transition="pose",    # strength=0.55: same elder, slight pose change
        prompt=(
            "close-up portrait of prehistoric elder man at campfire, "
            "raising one hand to gesture while speaking, same warm firelight"
        ),
    ),
]

print("Running 3-scene chain test...\n")
results = chain.run(scenes)

for r in results:
    status = "OK" if not r.errors else "FAIL"
    print(f"  Scene {r.scene_id} [{status}] mode={r.mode} strength={r.strength:.2f} "
          f"time={r.duration_seconds:.1f}s")
    if r.local_path:
        print(f"    -> {r.local_path}")
    if r.errors:
        print(f"    ERRORS: {r.errors}")

print("\nDone. Open output/chain-test/images/ to inspect.")
