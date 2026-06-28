from __future__ import annotations

from functools import lru_cache
from typing import Any
import unicodedata


def normalize_prompt_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text))
    if "\ufffd" in normalized:
        raise ValueError("Prompt contains U+FFFD replacement character")
    return normalized


def token_count_from_tokenizer(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=True)
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids", [])
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


@lru_cache(maxsize=8)
def load_tokenizer(model_id: str, subfolder: str, revision: str | None = None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        subfolder=subfolder,
        revision=revision or None,
        use_fast=True,
    )


def token_count_for_model(model_id: str, text: str, subfolder: str, revision: str | None = None) -> int:
    tokenizer = load_tokenizer(model_id, subfolder, revision)
    return token_count_from_tokenizer(tokenizer, text)
