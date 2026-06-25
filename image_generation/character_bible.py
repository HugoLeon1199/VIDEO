"""Fixed character descriptions. Never paraphrase between scenes."""

CHARACTERS = {
    "karo": {
        "id": "karo",
        "description": (
            "KARO, adult prehistoric man, broad rectangular face, warm brown skin, "
            "shoulder-length dark wavy hair, short square beard, thick eyebrows, "
            "ochre one-shoulder animal-hide tunic with three dark brown patches"
        ),
        "gender": "male",
        "reference_key": "character_male",
    },
    "luma": {
        "id": "luma",
        "description": (
            "LUMA, adult prehistoric woman, oval face, warm brown skin, "
            "long dark braided hair, calm dark eyes, simple brown animal-hide dress "
            "with an ochre shoulder strap"
        ),
        "gender": "female",
        "reference_key": "character_female",
    },
}

# Keywords that trigger character lookup from a scene description
CHARACTER_TRIGGER_PATTERNS = {
    "karo": ["karo", "the man", "adult man"],
    "luma": ["luma", "the woman", "adult woman"],
}


def get_character_block(char_id: str) -> str:
    """Return the exact injected description for a character. Raises KeyError if unknown."""
    char = CHARACTERS[char_id.lower()]
    return char["description"] + ", "


def detect_characters_in_text(scene_text: str) -> list:
    """Return list of character IDs mentioned in scene_text."""
    import re
    found = []
    lower = scene_text.lower()
    for char_id, triggers in CHARACTER_TRIGGER_PATTERNS.items():
        for trigger in triggers:
            if re.search(r'\b' + re.escape(trigger) + r'\b', lower):
                if char_id not in found:
                    found.append(char_id)
                break
    return found
