import re as _re

STYLE_VERSION = "prehistoric-flat-vector-v1"

STYLE_PREFIX = (
    "v3ct0r style, simple flat vector art, 2D digital illustration, "
    "minimalist flat vector design, clean outlines, simplified geometric shapes, "
    "solid matte colors, completely flat composition, "
    "hard-edged two-tone shadow shapes only, no gradients, "
    "no realistic rendering, no three-dimensional shading, "
)

PREHISTORIC_CORE = (
    "prehistoric humans wearing simple animal-hide clothing, "
    "primitive stone and wooden tools, historically inspired early human life, "
)

STYLE_SUFFIX = (
    ", mature educational historical documentary style, "
    "dark muted earth-tone palette, corporate explainer-video graphic asset, "
    "clean separated silhouettes, consistent character proportions, "
    "uncluttered composition, 16:9 widescreen, no text, no logo, no watermark"
)

ANATOMY_BLOCK = (
    ", anatomically coherent simplified human figures, "
    "exactly two arms and two legs per visible person, "
    "clean separated silhouettes, no overlapping bodies, natural readable poses"
)

GROUP_BLOCK = (
    ", show no more than two detailed foreground characters, "
    "place all additional people far in the background as small simplified silhouettes"
)

PALETTE_BLOCK = (
    ", use a restricted flat palette based on: charcoal (#2D2926), "
    "dark brown (#4A3427), fur ochre (#A77A3D), warm brown skin (#B9784D), "
    "muted forest green (#4F5D3A), deep night blue (#273449), "
    "restrained fire orange (#D77732), stone gray (#77736A), sand beige (#B49A73)"
)

PREHISTORIC_PALETTE = {
    "charcoal": "#2D2926",
    "dark_brown": "#4A3427",
    "fur_ochre": "#A77A3D",
    "skin_warm": "#B9784D",
    "forest_green": "#4F5D3A",
    "night_blue": "#273449",
    "fire_orange": "#D77732",
    "stone_gray": "#77736A",
    "sand_beige": "#B49A73",
}

NEGATIVE_PROMPT = (
    "photorealistic, realistic photography, hyperrealistic, DSLR, cinematic photo, "
    "3D render, CGI, glossy surfaces, realistic skin pores, volumetric lighting, "
    "ray tracing, bokeh, shallow depth of field, lens flare, gradients, "
    "complex realistic textures, extra arms, extra legs, extra limbs, "
    "duplicated body parts, fused bodies, overlapping limbs, malformed hands, "
    "extra fingers, missing fingers, twisted joints, merged faces, duplicate people, "
    "cropped head, text, captions, logo, watermark, "
    "exposed female breasts, nipples, bare chest on woman, topless woman"
)

# Keywords that indicate a female character is present in the scene
_FEMALE_KEYWORDS = _re.compile(
    r"\b(woman|women|female|mother|girl|grandmother|elder woman|she\b|her\b"
    r"|grandma|matriarch|huntress|gatherer woman)\b",
    _re.IGNORECASE,
)

# Inserted right after female subject noun to force clothing coverage
_FEMALE_CLOTHING = (
    "wearing a sewn animal-hide top fully covering her chest and shoulders, "
    "long wraparound hide skirt reaching her knees, fully clothed"
)

# Words that push FLUX toward realism — map to safe replacements
_REALISM_REPLACEMENTS = {
    "photorealistic": "flat vector",
    "realistic photograph": "flat illustration",
    "dslr": "clean digital art",
    "cinematic photography": "flat cinematic composition",
    "skin pores": "simplified skin",
    "volumetric lighting": "flat color contrast",
    "ray tracing": "flat shading",
    "3d render": "2d flat illustration",
    "highly detailed skin": "simplified flat skin",
    "shallow depth of field": "flat composition",
    "bokeh": "clean background",
    "lens flare": "flat light shapes",
    "glossy surface": "matte surface",
    "dramatic realistic shadows": "hard-edged two-tone shadow shapes",
    "textured brush strokes": "simplified texture marks",
    "hyperrealistic": "flat vector style",
    "glowing fire": "flat orange light shapes",
    "detailed texture": "simplified texture marks",
    "realistic lighting": "controlled flat color contrast",
}


def sanitize_scene_text(scene_text: str) -> str:
    """Remove or replace terms that push FLUX toward realism. Logs removed terms."""
    import re
    import logging
    logger = logging.getLogger(__name__)
    result = scene_text
    removed = []
    for bad, replacement in _REALISM_REPLACEMENTS.items():
        pattern = re.compile(re.escape(bad), re.IGNORECASE)
        if pattern.search(result):
            removed.append(bad)
            result = pattern.sub(replacement, result)
    if removed:
        logger.warning("sanitize_scene_text: replaced realism terms: %s", removed)
    return result


def has_female_character(scene_text: str) -> bool:
    """Return True if scene_text mentions any female character."""
    return bool(_FEMALE_KEYWORDS.search(scene_text))


def build_scene_prompt(
    scene_text: str,
    character_blocks: list,  # list of str — exact character description blocks
    environment_block: str = "",
    shot_block: str = "",
    palette_block: str = "",
    include_anatomy: bool = True,
    include_group_rule: bool = False,
) -> str:
    """Build a complete flat-vector scene prompt.

    Order: STYLE_PREFIX + PREHISTORIC_CORE + characters + scene + environment + shot + palette + anatomy + STYLE_SUFFIX
    Female clothing is automatically injected when a female character is detected.
    """
    scene_clean = sanitize_scene_text(scene_text)

    # Female clothing guard injected RIGHT AFTER style prefix — within first 30 tokens
    if has_female_character(scene_clean):
        parts = [STYLE_PREFIX, _FEMALE_CLOTHING + ", ", PREHISTORIC_CORE]
    else:
        parts = [STYLE_PREFIX, PREHISTORIC_CORE]

    for char_block in character_blocks:
        parts.append(char_block.rstrip(", ") + ", ")

    parts.append(scene_clean)

    if environment_block:
        parts.append(", " + environment_block.strip(", "))
    if shot_block:
        parts.append(", " + shot_block.strip(", "))

    parts.append(palette_block or PALETTE_BLOCK)

    if include_anatomy:
        parts.append(ANATOMY_BLOCK)
    if include_group_rule:
        parts.append(GROUP_BLOCK)

    parts.append(STYLE_SUFFIX)

    return "".join(parts)
