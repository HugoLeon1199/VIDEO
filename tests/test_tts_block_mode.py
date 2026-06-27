from pathlib import Path

from steps import tts
from steps.text_units import load_sentence_units, split_sentence_units, strip_script_metadata


class StubVieNeuRuntime:
    def __init__(self):
        self.block_config = {
            "block_target_max_seconds": 16.0,
            "block_hard_max_seconds": 20.0,
            "block_soft_max_normalized_chars": 240,
            "block_hard_max_normalized_chars": 280,
            "initial_chars_per_second": 14.0,
        }

    def normalized_chars(self, text: str) -> int:
        return len(text)

    def estimate_seconds(self, text: str) -> float:
        return len(text) / self.block_config["initial_chars_per_second"]


class StubKokoroRuntime:
    def phoneme_chars(self, text: str) -> int:
        return len(text)

    def estimate_seconds(self, text: str) -> float:
        return len(text.split()) / 2.6


def test_strip_script_metadata():
    raw = "Line one.\n\nCOMMENT SEED: ignore me\nRESEARCH NOTES:\n- hidden"
    assert strip_script_metadata(raw) == "Line one."


def test_split_sentence_units_preserves_paragraph_index():
    script = "Title\n\nFirst sentence. Second sentence.\n\nThird sentence?"
    units = split_sentence_units(script)
    assert [unit.sentence_index for unit in units] == [1, 2, 3]
    assert [unit.paragraph_index for unit in units] == [1, 1, 2]
    assert units[0].text == "First sentence."


def test_vieneu_block_builder_respects_soft_limit():
    sentence_units = split_sentence_units(
        "A" * 120 + ". " + "B" * 120 + ". " + "C" * 120 + "."
    )
    blocks = tts._build_vieneu_blocks(sentence_units, StubVieNeuRuntime())
    assert len(blocks) == 3
    assert blocks[0].sentence_indices == [1]
    assert blocks[1].sentence_indices == [2]


def test_kokoro_block_builder_uses_phoneme_soft_limit():
    sentence_units = split_sentence_units(
        ("alpha " * 20).strip() + ". " + ("beta " * 20).strip() + ". " + ("gamma " * 20).strip() + "."
    )
    runtime = StubKokoroRuntime()
    cfg = {
        "block_soft_max_phoneme_chars": 120,
        "block_hard_max_phoneme_chars": 500,
    }
    blocks = tts._build_kokoro_blocks(sentence_units, runtime, cfg)
    assert len(blocks) >= 2
    assert [idx for block in blocks for idx in block.sentence_indices] == [1, 2, 3]


def test_load_sentence_units_from_fixture():
    script_path = Path("output") / "to-tien-ban-lam-gi-ca-ngay-vi" / "script.txt"
    units = load_sentence_units(script_path)
    assert units
    assert units[0].sentence_index == 1
