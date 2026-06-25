"""Apply watercolor National Geographic style to all prompts in a video.

Strategy for clothing compliance with FLUX.1-dev:
- FLUX weights early tokens most heavily (~75 token window)
- OLD: [long style prefix 50 tokens] [clothing 20 tokens] [scene] -- clothing buried too deep
- NEW: [clothing 15 tokens] [scene] [style suffix 20 tokens] -- clothing in first 15 tokens

Strategy for anatomy (3 arms/legs):
- Append simple impressionistic style cue that naturally blurs anatomy
- Add explicit anatomy constraint as suffix (short, not buried)
"""
import json
import re
import argparse
from pathlib import Path

# SHORT style suffix — goes AFTER scene description, keeps token count low
STYLE_SUFFIX = (
    ", watercolor illustration, soft amber ochre washes, National Geographic style, "
    "warm golden light, loose painterly brushwork, cinematic 16:9, no text, no watermark"
)

# Clothing prefix — goes FIRST so FLUX sees it in the strongest token window
CLOTHING_PREFIX_WITH_HUMANS = (
    "all characters fully clothed, thick animal hide vest covering chest, "
    "long hide skirt or loincloth covering waist to knees, no bare torso, no bare chest, "
)
CLOTHING_PREFIX_NO_HUMANS = ""  # landscape/object scenes need no clothing prefix

# Anatomy suffix — appended last, short
ANATOMY_SUFFIX = (
    ", simple clean figures, two arms two legs per person, "
    "no extra limbs, loose impressionistic anatomy"
)

NEW_NEGATIVE = (
    "nudity, bare chest, bare skin, bare torso, topless, shirtless, exposed breasts, "
    "cleavage, bikini top, revealing clothes, loincloth only, skimpy outfit, "
    "large breasts, oversized chest, sexualized body, sexual content, suggestive pose, "
    "3D render, anime, chibi, Pixar, Disney, flat vector, "
    "extra limbs, extra arms, extra legs, three arms, three legs, "
    "deformed anatomy, fused limbs, text, watermark, logo"
)

# All known old prefixes to strip
KNOWN_PREFIXES = [
    "watercolor illustration, soft washes of warm amber and ochre, loose painterly style, National Geographic book illustration, characters wearing full animal hide and grass fiber clothing covering torso, warm golden light, African savanna setting, clean cinematic 16:9 composition, no text, no watermark, ",
    "flat 2D animated illustration, Kurzgesagt-inspired style, warm amber and golden color palette, simple rounded prehistoric human figures fully clothed in brown hide garments, peaceful and contemplative mood, soft warm lighting, clean composition, cinematic 16:9, no text, no watermark, ",
    "flat 2D vector illustration, Kurzgesagt style, warm amber golden palette, simple bold shapes, clean silhouettes, ",
    "cinematic 2D painted illustration, warm and inviting style, semi-realistic prehistoric humans with friendly natural expressions, relaxed peaceful atmosphere, characters at ease and content, ALL characters male and female wearing thick animal fur vests and hide wraps covering chest shoulders and torso completely, fur pelts covering upper body, hide skirts or pants below waist, no bare chest no bare torso on anyone, warm golden sunlight, lush earthy prehistoric landscape, approachable lighthearted documentary tone, cinematic 16:9 composition, no text, no watermark, ",
    "cinematic 2D painted documentary illustration, semi-realistic prehistoric humans, painted in the style of a serious historical documentary, characters wearing thick full-coverage animal fur and hide clothing, fur pelts draped over both shoulders covering entire chest and torso, rough animal skin garments tied at waist, no exposed chest no exposed breasts no bare torso on any character, warm natural earthy lighting, detailed rocky prehistoric landscape, serious mature tone, cinematic 16:9 composition, no text, no watermark, ",
    "cinematic 2D painted documentary illustration, semi-realistic prehistoric humans, mature historical animation, hand-painted texture, simplified anatomically correct bodies, all characters fully clothed in prehistoric hide garments covering chest and torso, long animal-hide tunics and wraparound skirts reaching mid-thigh or below, no bare skin except face forearms and lower legs, clean separated silhouettes, warm golden-amber natural lighting, detailed prehistoric environment, serious educational documentary tone, cinematic composition, 16:9, no text, no watermark, ",
]

OLD_ANATOMY_SUFFIX = (
    ", anatomically coherent human figures, exactly two arms and two legs per visible person, "
    "natural human proportions, clean separated silhouettes, clearly readable limbs, natural hands"
)
OLD_ANATOMY_SUFFIX2 = (
    ", simple clean figures, two arms two legs per person, "
    "no extra limbs, loose impressionistic anatomy"
)

# Old clothing injections to strip before rebuilding
OLD_CLOTHING_INJECTIONS = [
    " wearing thick woven grass top and long wraparound hide skirt reaching ankles, fully covering chest and torso,",
    " wearing thick animal hide vest covering chest and long hide loincloth to knees, fully covering torso,",
    " all dressed in thick animal hide vests and long hide skirts covering chest and torso,",
]

HUMAN_KEYWORDS = re.compile(
    r'\b(man|woman|women|men|hunter|gatherer|elder|figure|figures|human|humans|'
    r'person|people|ancestor|ancestors|family|tribe|community|group|child|children|'
    r'warrior|farmer|storyteller|silhouette|silhouettes|prehistoric|ancient|'
    r'homo sapiens|neanderthal|hadza|ju.hoansi)\b',
    re.I
)

# Scenes where torso is explicitly shown — force waist-up rewrite
TORSO_RISK_PATTERNS = re.compile(
    r'\b(full.body|full body|standing|walking|dancing|running|crouching|kneeling|'
    r'seated|sitting|lying|resting|close.up portrait|portrait)\b',
    re.I
)


def strip_scene(prompt):
    """Remove all known prefixes, old clothing injections, and anatomy suffixes."""
    # Strip prefix
    for old in sorted(KNOWN_PREFIXES, key=len, reverse=True):
        if prompt.startswith(old):
            prompt = prompt[len(old):]
            break
    # Strip old anatomy suffixes
    prompt = prompt.replace(OLD_ANATOMY_SUFFIX, "")
    prompt = prompt.replace(OLD_ANATOMY_SUFFIX2, "")
    # Strip old inline clothing injections
    for inj in OLD_CLOTHING_INJECTIONS:
        prompt = prompt.replace(inj, "")
    return prompt.strip().rstrip(",").strip()


def build_prompt(scene_text):
    """Build the final prompt with clothing prefix first, then scene, then style suffix."""
    has_humans = bool(HUMAN_KEYWORDS.search(scene_text))

    if has_humans:
        prefix = CLOTHING_PREFIX_WITH_HUMANS
        suffix = STYLE_SUFFIX + ANATOMY_SUFFIX
    else:
        prefix = CLOTHING_PREFIX_NO_HUMANS
        suffix = STYLE_SUFFIX

    return prefix + scene_text + suffix


parser = argparse.ArgumentParser()
parser.add_argument("--video-id", required=True)
parser.add_argument("--output-root", default="output")
args = parser.parse_args()

p = Path(args.output_root) / args.video_id / "image_prompts.json"
data = json.loads(p.read_text(encoding="utf-8"))

updated = 0
human_scenes = 0
for e in data:
    scene = strip_scene(e["prompt"])
    has_humans = bool(HUMAN_KEYWORDS.search(scene))
    if has_humans:
        human_scenes += 1
    e["prompt"] = build_prompt(scene)
    e["negative_prompt"] = NEW_NEGATIVE
    e.setdefault("global_style", "")
    updated += 1

tmp = p.with_suffix(".tmp")
tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
tmp.replace(p)
print(f"Updated {updated}/{len(data)} prompts")
print(f"Human scenes (clothing prefix applied): {human_scenes}/{len(data)}")
print(f"\nSample with humans (scene 1):")
print(data[0]["prompt"][:300])
print(f"\nSample without humans (scene 25 - fire only):")
print(data[24]["prompt"][:300])
