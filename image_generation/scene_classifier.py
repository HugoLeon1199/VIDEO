"""Classify each scene to determine: reference image, mode, strength."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SceneClassification:
    scene_type: str          # character_closeup_male | character_closeup_female | character_full_body
                             # night_fire | day_wilderness | cosmic_sky | scientific_diagram
                             # timeline_cycle | object_macro | new_context
                             # same_scene_minor_pose | same_scene_camera_shift
    use_previous_image: bool
    reference_key: Optional[str]   # key into master_seed manifest, or None
    change_type: str               # "new" | "pose" | "angle" | "continuity"
    expected_people: int
    shot_type: str                 # "wide" | "medium" | "closeup" | "macro" | "overhead"
    confidence: float              # 0.0 to 1.0


# Keyword patterns for classification
_NIGHT_FIRE = re.compile(r'\b(night|campfire|fire|cave|dark|torch|bonfire|flame)\b', re.I)
_DAY_WILD   = re.compile(r'\b(savanna|valley|sunrise|sunset|landscape|plains|horizon|forest|jungle|desert)\b', re.I)
_COSMIC     = re.compile(r'\b(sky|stars|moon|cosmos|galaxy|space|night sky|milky way)\b', re.I)
_SCI_DIAG   = re.compile(r'\b(diagram|brain|skull|dna|chart|evolution|comparison|anatomy|cross.section)\b', re.I)
_TIMELINE   = re.compile(r'\b(timeline|cycle|process|infographic|era|period|stages|sequence)\b', re.I)
_OBJECT     = re.compile(r'\b(close.up of|macro|stone axe|tool|spear|flint|bone|artifact|seed|berry|root)\b', re.I)
_CLOSEUP    = re.compile(r'\b(close.up|portrait|face|expression|eyes|extreme close|detail)\b', re.I)
_WIDE       = re.compile(r'\b(wide|establishing|panorama|overhead|aerial|distant|far away|horizon)\b', re.I)
_POSE_CHANGE = re.compile(r'\b(raises?|lifts?|points?|turns?|looks?|nods?|reaches?|moves? (?:arm|hand|leg))\b', re.I)
_ANGLE_CHANGE = re.compile(r'\b(close.up of|zoom|pan to|pull back|cut to|now showing|portrait of)\b', re.I)
_PEOPLE_COUNT = re.compile(r'\b(one|two|three|four|five|six|a lone|a single|group of|tribe|crowd|family)\b', re.I)
_FEMALE_KW  = re.compile(r'\b(woman|female|luma|girl|mother)\b', re.I)
_MALE_KW    = re.compile(r'\b(man|male|karo|hunter|warrior|father)\b', re.I)


def _count_people(text: str) -> int:
    word_map = {"one": 1, "a lone": 1, "a single": 1, "two": 2, "three": 3,
                "four": 4, "five": 5, "six": 6, "group of": 5, "tribe": 8, "crowd": 10, "family": 4}
    lower = text.lower()
    for phrase, n in sorted(word_map.items(), key=lambda x: -len(x[0])):
        if phrase in lower:
            return n
    return 2  # default


def classify_scene(
    scene_text: str,
    previous_scene_text: Optional[str] = None,
    characters: Optional[list] = None,
    continuity_group: Optional[str] = None,
    previous_continuity_group: Optional[str] = None,
) -> SceneClassification:
    """Classify a scene using deterministic keyword rules. Falls back to new_context."""

    text = scene_text
    chars = characters or []
    same_group = (
        continuity_group is not None
        and continuity_group == previous_continuity_group
    )

    # --- Continuity within same group ---
    if same_group and previous_scene_text:
        if _ANGLE_CHANGE.search(text):
            return SceneClassification(
                scene_type="same_scene_camera_shift",
                use_previous_image=True,
                reference_key=None,
                change_type="angle",
                expected_people=_count_people(text),
                shot_type=_shot_type(text),
                confidence=0.85,
            )
        if _POSE_CHANGE.search(text):
            return SceneClassification(
                scene_type="same_scene_minor_pose",
                use_previous_image=True,
                reference_key=None,
                change_type="pose",
                expected_people=_count_people(text),
                shot_type=_shot_type(text),
                confidence=0.80,
            )
        # Same group, no pose/angle signal: treat as camera shift (safer)
        return SceneClassification(
            scene_type="same_scene_camera_shift",
            use_previous_image=True,
            reference_key=None,
            change_type="angle",
            expected_people=_count_people(text),
            shot_type=_shot_type(text),
            confidence=0.65,
        )

    # --- Object / macro ---
    if _OBJECT.search(text) and not _FEMALE_KW.search(text) and not _MALE_KW.search(text):
        return SceneClassification(
            scene_type="object_macro",
            use_previous_image=False,
            reference_key="object_macro",
            change_type="new",
            expected_people=0,
            shot_type="macro",
            confidence=0.80,
        )

    # --- Scientific diagram ---
    if _SCI_DIAG.search(text):
        return SceneClassification(
            scene_type="scientific_diagram",
            use_previous_image=False,
            reference_key="scientific_diagram",
            change_type="new",
            expected_people=0,
            shot_type="medium",
            confidence=0.85,
        )

    # --- Timeline / cycle ---
    if _TIMELINE.search(text):
        return SceneClassification(
            scene_type="timeline_cycle",
            use_previous_image=False,
            reference_key="timeline_cycle",
            change_type="new",
            expected_people=0,
            shot_type="wide",
            confidence=0.80,
        )

    # --- Cosmic sky ---
    if _COSMIC.search(text) and not _NIGHT_FIRE.search(text):
        return SceneClassification(
            scene_type="cosmic_sky",
            use_previous_image=False,
            reference_key="cosmic_sky",
            change_type="new",
            expected_people=0,
            shot_type="wide",
            confidence=0.75,
        )

    # --- Character close-up ---
    if _CLOSEUP.search(text):
        if _FEMALE_KW.search(text) or ("luma" in chars):
            return SceneClassification(
                scene_type="character_closeup_female",
                use_previous_image=False,
                reference_key="character_female",
                change_type="new",
                expected_people=1,
                shot_type="closeup",
                confidence=0.82,
            )
        if _MALE_KW.search(text) or ("karo" in chars):
            return SceneClassification(
                scene_type="character_closeup_male",
                use_previous_image=False,
                reference_key="character_male",
                change_type="new",
                expected_people=1,
                shot_type="closeup",
                confidence=0.82,
            )

    # --- Night / fire scene ---
    if _NIGHT_FIRE.search(text):
        return SceneClassification(
            scene_type="night_fire",
            use_previous_image=False,
            reference_key="night_fire",
            change_type="new",
            expected_people=_count_people(text),
            shot_type=_shot_type(text),
            confidence=0.78,
        )

    # --- Daytime wilderness ---
    if _DAY_WILD.search(text):
        return SceneClassification(
            scene_type="day_wilderness",
            use_previous_image=False,
            reference_key="day_wilderness",
            change_type="new",
            expected_people=_count_people(text),
            shot_type=_shot_type(text),
            confidence=0.75,
        )

    # Default: new context, text-to-image
    return SceneClassification(
        scene_type="new_context",
        use_previous_image=False,
        reference_key=None,
        change_type="new",
        expected_people=_count_people(text),
        shot_type=_shot_type(text),
        confidence=0.50,
    )


def _shot_type(text: str) -> str:
    if _WIDE.search(text):
        return "wide"
    if _CLOSEUP.search(text):
        return "closeup"
    if _OBJECT.search(text):
        return "macro"
    return "medium"
