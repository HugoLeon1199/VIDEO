from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_STOP_MARKERS = (
    "COMMENT SEED:",
    "RESEARCH NOTES:",
    "Your script is ready",
    "Save as:",
    "Then: python",
)


@dataclass(frozen=True)
class SentenceUnit:
    sentence_index: int
    text: str
    paragraph_index: int

    def to_dict(self) -> dict:
        return asdict(self)


def strip_script_metadata(text: str) -> str:
    cleaned = text.lstrip("\ufeff")
    for marker in SCRIPT_STOP_MARKERS:
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx]
    return cleaned.strip()


def load_script_text(script_path: Path) -> str:
    return strip_script_metadata(script_path.read_text(encoding="utf-8"))


def split_paragraph_texts(script_text: str) -> list[str]:
    paragraphs = re.split(r"\n{2,}", strip_script_metadata(script_text))
    cleaned: list[str] = []
    first = True
    for para in paragraphs:
        para = re.sub(r"\s+", " ", para.strip())
        if not para:
            continue
        if first:
            first = False
            if not re.search(r"[.!?]$", para) and len(para.split()) <= 12:
                continue
        cleaned.append(para)
    return cleaned


def split_sentence_units(script_text: str) -> list[SentenceUnit]:
    units: list[SentenceUnit] = []
    sentence_index = 1
    for paragraph_index, para in enumerate(split_paragraph_texts(script_text), start=1):
        for part in re.split(r"(?<=[.!?])\s+", para):
            part = re.sub(r"\s+", " ", part.strip())
            if not part:
                continue
            units.append(
                SentenceUnit(
                    sentence_index=sentence_index,
                    text=part,
                    paragraph_index=paragraph_index,
                )
            )
            sentence_index += 1
    return units


def load_sentence_units(script_path: Path) -> list[SentenceUnit]:
    return split_sentence_units(load_script_text(script_path))
