"""Round-gated Kokoro voice lab with blind review and lineage tracking."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import list_repo_files
from kokoro import KPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steps.text_units import load_sentence_units  # noqa: E402

LAB_VERSION = 2
DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SPEED = 0.95
DEFAULT_SEED = 20260627
DEFAULT_OUTPUT_DIR = Path("output") / "voice_lab"
ROUND_SEQUENCE = ["base", "topic", "blend", "speed", "final"]
ROUND_PREVIOUS = {
    "topic": "base",
    "blend": "topic",
    "speed": "blend",
    "final": "speed",
}
DECISION_PRIORITY = {
    "Keep": 0,
    "Maybe": 1,
    "Reject": 2,
    "": 3,
}
DECISIONS_FIELDNAMES = [
    "blind_id",
    "round",
    "decision",
    "revealed",
    "notes",
    "revealed_voice",
    "revealed_family",
    "revealed_source",
    "revealed_lineage",
]
ROUND_LIMITS = {
    "topic": 6,
    "blend": 3,
    "speed": 2,
    "final": 2,
}

BASE_SAMPLE = (
    "What would you trust first, the scar, the tool, or the year 31000 BCE carved into the argument itself? "
    "In 2023, researchers compared DNA evidence, AI reconstruction, and a NASA-style imaging workflow before naming Liang Tebo, "
    "Tim Maloney, Maxime Aubert, and Melandri Vlok in the same breath, and that long chain matters because a voice has to stay clear "
    "through numbers, proper names, and one sentence that deliberately keeps stretching while the meaning still lands cleanly at the end."
)

BLEND_SAMPLE = (
    "You hear the setup, then the turn. A bone cut, a careful hand, and a question that still refuses to settle. "
    "By 2026, AI models can sort patterns fast, but a human narrator still has to carry tension, clarity, and names like Liang Tebo "
    "without losing warmth when the story pivots from evidence to interpretation."
)

TOPIC_SEGMENTS: list[dict[str, str]] = [
    {
        "slug": "history",
        "text": (
            "Long before the first city wall, people memorized rivers, tracks, and seasons well enough to survive without a map."
        ),
    },
    {
        "slug": "mystery",
        "text": (
            "Then one detail refuses to fit, and that is usually where the real story begins to breathe."
        ),
    },
    {
        "slug": "science",
        "text": (
            "Science moves by testing pressure, timing, and repeatable evidence, even when the conclusion sounds dramatic."
        ),
    },
    {
        "slug": "finance",
        "text": (
            "Finance is the same in another costume, because every budget is really a story about tradeoffs under pressure."
        ),
    },
    {
        "slug": "reflective",
        "text": (
            "A quiet voice matters here, because reflection often makes the final line feel more honest than a shout ever could."
        ),
    },
    {
        "slug": "pronunciation_stress",
        "text": (
            "Now hold Liang Tebo, Borneo, Maxime Aubert, DNA, AI, and NASA together without flattening the voice."
        ),
    },
]
TOPIC_REEL_TEXT = " ".join(segment["text"] for segment in TOPIC_SEGMENTS)

FINAL_SCRIPT = (
    "What if the oldest medical story in the room is not dramatic because it is ancient, but because it sounds uncomfortably modern? "
    "A child loses part of a leg, survives, heals, and keeps living long enough for bone to record patience instead of panic. "
    "That is the headline version. The slower version is more interesting. It asks who carried water, who cleaned wounds, who watched for fever, "
    "and who knew that one wrong cut could turn help into harm. In the cave record, certainty never arrives alone. Evidence comes in fragments, "
    "and fragments have to be read against weather, time, and damage that happened long after any witness was gone. Still, some clues stay stubborn. "
    "The cut edge looks deliberate. The healing looks prolonged. The survival looks social. That matters because no one heals like this in isolation. "
    "Someone had knowledge. Someone had restraint. Someone kept the body alive through pain, risk, and the slow mathematics of recovery. "
    "Modern listeners hear surgery and imagine clean rooms, steel tools, and bright procedural light, but the deeper story may be simpler and harder at once. "
    "It may be about planning. It may be about memory. It may be about the kind of practical intelligence that never wrote a textbook but still moved carefully "
    "through blood, infection, and consequence. That is why the argument still feels alive today. When AI, DNA, imaging, and field archaeology meet in the same conversation, "
    "they are not replacing the ancient people in the story. They are circling back toward them, trying to understand how much skill, care, and discipline had to exist before history ever learned how to name it."
)


@dataclasses.dataclass
class LabArtifact:
    blind_id: str
    kind: str
    round: str
    round_order: int
    label: str
    lang_code: str
    source_ref: str
    audio_wav: str
    audio_mp3: str
    sample_id: str
    speed: float
    duration_seconds: float
    lineage: dict[str, Any]
    metadata: dict[str, Any]


class PipelineCache:
    def __init__(self, repo_id: str, device: str | None = None):
        self.repo_id = repo_id
        self.device = device
        self._pipelines: dict[str, KPipeline] = {}

    def get(self, lang_code: str) -> KPipeline:
        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id=self.repo_id, device=self.device)
        return self._pipelines[lang_code]


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _ffmpeg_path() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        r"C:\Users\LEON_RM\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1-full_build\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("ffmpeg not found")


def _voice_family(voice: str) -> str:
    return voice.split("_", 1)[0]


def _voice_lang_code(voice: str) -> str:
    return voice[0].lower()


def _list_official_voice_names(repo_id: str) -> set[str]:
    names: set[str] = set()
    try:
        for file_name in list_repo_files(repo_id):
            if file_name.startswith("voices/") and file_name.endswith(".pt"):
                names.add(Path(file_name).stem)
    except Exception:
        pass
    return names


def _list_cached_voice_names(repo_id: str) -> set[str]:
    names: set[str] = set()
    cache_roots = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_roots.append(Path(hf_home) / "hub")
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    repo_slug = f"models--{repo_id.replace('/', '--')}"
    for root in cache_roots:
        if not root.exists():
            continue
        for path in root.rglob("voices/*.pt"):
            if repo_slug in path.as_posix():
                names.add(path.stem)
    return names


def discover_english_voices(repo_id: str) -> list[str]:
    voices = _list_official_voice_names(repo_id) | _list_cached_voice_names(repo_id)
    english = [voice for voice in voices if voice and voice[0] in {"a", "b"}]
    return sorted(english)


def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_wav(audio: np.ndarray, sample_rate: int, wav_path: Path) -> None:
    import wave

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _wav_duration_seconds(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    ffmpeg = _ffmpeg_path()
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-q:a", "3", str(mp3_path)],
        check=True,
        capture_output=True,
    )


def _render_text(pipeline: KPipeline, text: str, voice: Any, speed: float) -> np.ndarray:
    outputs = []
    for result in pipeline(text, voice=voice, speed=speed, split_pattern=None):
        if result.audio is None:
            continue
        chunk = result.audio.detach().cpu().numpy()
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        outputs.append(chunk.astype(np.float32))
    if not outputs:
        raise RuntimeError("Kokoro returned no audio")
    return np.concatenate(outputs)


def _phoneme_chars(pipeline: KPipeline, text: str) -> int:
    _, tokens = pipeline.g2p(text)
    return len(KPipeline.tokens_to_ps(tokens))


def _build_blocks(
    pipeline: KPipeline,
    sentences: list[str],
    soft_cap: int,
    hard_cap: int,
) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = current + [sentence]
        candidate_text = " ".join(candidate)
        phoneme_chars = _phoneme_chars(pipeline, candidate_text)
        if phoneme_chars > hard_cap and not current:
            raise ValueError(f"Sentence exceeds hard cap: {phoneme_chars} > {hard_cap} :: {sentence[:80]}")
        if current and phoneme_chars > soft_cap:
            blocks.append(current)
            current = [sentence]
            continue
        if phoneme_chars > hard_cap:
            blocks.append(current)
            current = [sentence]
            continue
        current = candidate
    if current:
        blocks.append(current)
    return blocks


def _generate_blind_ids(count: int, seed: int) -> list[str]:
    ids = [f"A{i:03d}" for i in range(1, max(count, 1) + 1)]
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[:count]


def _allocate_blind_id(existing_ids: set[str], seed: int) -> str:
    for candidate in _generate_blind_ids(999, seed):
        if candidate not in existing_ids:
            return candidate
    raise RuntimeError("Ran out of blind ids")


def _resolve_decisions_path(out_dir: Path, decisions: Path | None) -> Path:
    if decisions is not None:
        return decisions
    return out_dir / "decisions.csv"


def _normalize_round_counts(artifacts: list[LabArtifact]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for round_name in ROUND_SEQUENCE:
        counts[round_name] = sum(1 for artifact in artifacts if artifact.round == round_name)
    return counts


def _resolve_manifest(out_dir: Path) -> dict[str, Any]:
    return _load_json(
        out_dir / "manifest.json",
        default={
            "version": LAB_VERSION,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "repo_id": DEFAULT_REPO_ID,
            "active_round": None,
            "artifacts": [],
            "round_counts": {},
            "round_configs": {},
        },
    )


def _load_all_artifacts(manifest: dict[str, Any]) -> list[LabArtifact]:
    return [LabArtifact(**item) for item in manifest.get("artifacts", [])]


def _artifacts_for_round(manifest: dict[str, Any], round_name: str) -> list[LabArtifact]:
    items = [artifact for artifact in _load_all_artifacts(manifest) if artifact.round == round_name]
    return sorted(items, key=lambda item: (item.round_order, item.blind_id))


def _next_round_order(existing_artifacts: list[LabArtifact], round_name: str) -> int:
    current = [artifact.round_order for artifact in existing_artifacts if artifact.round == round_name]
    return max(current, default=0) + 1


def _mapping_entry_from_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    metadata = artifact.get("metadata", {})
    lineage = artifact.get("lineage", {})
    entry = {
        "kind": artifact["kind"],
        "round": artifact["round"],
        "family": metadata.get("family"),
        "lineage": lineage,
        "material_type": metadata.get("material_type"),
        "voice": metadata.get("voice"),
        "tensor_path": metadata.get("tensor_path"),
        "tensor_sha256": metadata.get("tensor_sha256"),
        "source_blind_id": metadata.get("source_blind_id"),
        "speed": artifact.get("speed"),
        "components": metadata.get("components"),
        "source_description": metadata.get("source_description"),
    }
    return entry


def _resolve_mapping(out_dir: Path) -> dict[str, Any]:
    mapping = _load_json(out_dir / "blind_mapping.json", default={})
    manifest = _resolve_manifest(out_dir)
    for artifact in manifest.get("artifacts", []):
        blind_id = artifact.get("blind_id")
        if blind_id and blind_id not in mapping:
            mapping[blind_id] = _mapping_entry_from_artifact(artifact)
    return mapping


def _save_manifest_bundle(out_dir: Path, manifest: dict[str, Any], mapping: dict[str, Any]) -> None:
    _save_json(out_dir / "manifest.json", manifest)
    _save_json(out_dir / "blind_mapping.json", mapping)


def _blank_decision_row() -> dict[str, str]:
    return {field: "" for field in DECISIONS_FIELDNAMES}


def _load_decisions(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            blind_id = (row.get("blind_id") or "").strip()
            if not blind_id:
                continue
            normalized = _blank_decision_row()
            for field in DECISIONS_FIELDNAMES:
                normalized[field] = (row.get(field) or "").strip()
            rows[blind_id] = normalized
    return rows


def _write_decisions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DECISIONS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in DECISIONS_FIELDNAMES})


def _describe_lineage(lineage: dict[str, Any]) -> str:
    parts = []
    base_ids = lineage.get("base_ids") or []
    topic_ids = lineage.get("topic_ids") or []
    if base_ids:
        parts.append(f"base={','.join(base_ids)}")
    if topic_ids:
        parts.append(f"topic={','.join(topic_ids)}")
    if lineage.get("blend_id"):
        parts.append(f"blend={lineage['blend_id']}")
    if lineage.get("speed_id"):
        parts.append(f"speed={lineage['speed_id']}")
    if lineage.get("final_id"):
        parts.append(f"final={lineage['final_id']}")
    return " | ".join(parts)


def _merge_decisions(
    path: Path,
    artifacts: list[LabArtifact],
    mapping: dict[str, Any],
) -> None:
    existing = _load_decisions(path)
    for artifact in artifacts:
        row = existing.get(artifact.blind_id, _blank_decision_row())
        entry = mapping.get(artifact.blind_id, {})
        row["blind_id"] = artifact.blind_id
        row["round"] = artifact.round
        row["decision"] = row.get("decision", "")
        row["revealed"] = row.get("revealed", "")
        row["notes"] = row.get("notes", "")
        row["revealed_voice"] = row.get("revealed_voice", entry.get("voice", "") or "")
        row["revealed_family"] = row.get("revealed_family", entry.get("family", "") or "")
        row["revealed_source"] = row.get("revealed_source", entry.get("source_description", "") or "")
        row["revealed_lineage"] = row.get("revealed_lineage", _describe_lineage(artifact.lineage))
        existing[artifact.blind_id] = row
    rows = list(existing.values())
    rows.sort(key=lambda item: (ROUND_SEQUENCE.index(item["round"]) if item["round"] in ROUND_SEQUENCE else 999, item["blind_id"]))
    _write_decisions(path, rows)


def _decision_sort_key(artifact: LabArtifact, decisions: dict[str, dict[str, str]]) -> tuple[int, int, str]:
    row = decisions.get(artifact.blind_id, {})
    decision = row.get("decision", "")
    return (DECISION_PRIORITY.get(decision, 3), artifact.round_order, artifact.blind_id)


def _select_finalists(
    manifest: dict[str, Any],
    decisions: dict[str, dict[str, str]],
    previous_round: str,
    limit: int,
    allowed_kinds: set[str] | None = None,
) -> list[LabArtifact]:
    candidates = _artifacts_for_round(manifest, previous_round)
    if allowed_kinds is not None:
        candidates = [artifact for artifact in candidates if artifact.kind in allowed_kinds]
    ranked = sorted(candidates, key=lambda artifact: _decision_sort_key(artifact, decisions))
    ranked = [artifact for artifact in ranked if decisions.get(artifact.blind_id, {}).get("decision", "")]
    if not ranked:
        raise RuntimeError(f"No reviewed finalists found for round '{previous_round}'")
    return ranked[:limit]


def _validate_duration(
    label: str,
    duration: float,
    target_min: float,
    target_max: float,
    acceptable_min: float,
    acceptable_max: float,
) -> None:
    if duration < acceptable_min or duration > acceptable_max:
        raise RuntimeError(
            f"{label} duration {duration:.3f}s is outside acceptable range {acceptable_min:.1f}-{acceptable_max:.1f}s"
        )
    if duration < target_min or duration > target_max:
        warnings.warn(
            f"{label} duration {duration:.3f}s is outside target range {target_min:.1f}-{target_max:.1f}s",
            stacklevel=2,
        )


def _validate_final_duration(label: str, duration: float) -> None:
    if duration < 90.0 or duration > 120.0:
        raise RuntimeError(f"{label} duration {duration:.3f}s must be within 90.0-120.0s")


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _update_manifest(
    out_dir: Path,
    manifest: dict[str, Any],
    mapping: dict[str, Any],
    round_name: str,
    artifacts: list[LabArtifact],
    round_config: dict[str, Any],
) -> dict[str, Any]:
    existing = [artifact for artifact in _load_all_artifacts(manifest) if artifact.round != round_name]
    combined = existing + artifacts
    combined.sort(key=lambda item: (ROUND_SEQUENCE.index(item.round), item.round_order, item.blind_id))
    manifest.update(
        {
            "version": LAB_VERSION,
            "updated_at": _now_iso(),
            "active_round": round_name,
            "artifacts": [dataclasses.asdict(item) for item in combined],
            "round_counts": _normalize_round_counts(combined),
        }
    )
    round_configs = dict(manifest.get("round_configs", {}))
    round_configs[round_name] = round_config
    manifest["round_configs"] = round_configs
    _save_manifest_bundle(out_dir, manifest, mapping)
    return manifest


def _resolve_voice_material(blind_id: str, mapping: dict[str, Any], out_dir: Path) -> tuple[Any, str, str | None, float]:
    entry = mapping[blind_id]
    material_type = entry.get("material_type")
    if material_type == "voice":
        voice = entry["voice"]
        return voice, _voice_lang_code(voice), entry.get("family"), float(entry.get("speed", DEFAULT_SPEED))
    if material_type == "tensor":
        tensor_path = out_dir / entry["tensor_path"]
        tensor = torch.load(tensor_path, weights_only=True).float()
        return tensor, "a", entry.get("family"), float(entry.get("speed", DEFAULT_SPEED))
    if material_type == "derived":
        source_id = entry["source_blind_id"]
        voice_ref, lang_code, family, inherited_speed = _resolve_voice_material(source_id, mapping, out_dir)
        speed = float(entry.get("speed", inherited_speed))
        return voice_ref, lang_code, family, speed
    raise ValueError(f"Unsupported material type for {blind_id}: {material_type}")


def _resolve_source_description(blind_id: str, mapping: dict[str, Any]) -> str:
    entry = mapping[blind_id]
    if entry.get("kind") == "blend":
        components = entry.get("components") or []
        parts = [f"{item['ref']}@{int(round(float(item['weight']) * 100))}" for item in components]
        return "blend " + " + ".join(parts)
    if entry.get("voice"):
        return entry["voice"]
    source_id = entry.get("source_blind_id")
    if source_id:
        return f"{blind_id} -> {source_id}"
    return blind_id


def _synthesize_sample(
    pipelines: PipelineCache,
    text: str,
    voice_ref: Any,
    lang_code: str,
    speed: float,
    wav_path: Path,
    mp3_path: Path,
) -> float:
    pipeline = pipelines.get(lang_code)
    audio = _render_text(pipeline, text, voice_ref, speed)
    _write_wav(audio, DEFAULT_SAMPLE_RATE, wav_path)
    _wav_to_mp3(wav_path, mp3_path)
    return _wav_duration_seconds(wav_path)


def _save_round_script(out_dir: Path, round_name: str, file_name: str, text: str) -> str:
    script_path = out_dir / "scripts" / round_name / file_name
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(text, encoding="utf-8")
    return _relative(script_path, out_dir)


def _base_lineage(blind_id: str) -> dict[str, Any]:
    return {
        "base_ids": [blind_id],
        "topic_ids": [],
        "blend_id": None,
        "speed_id": None,
        "final_id": None,
    }


def _inherit_lineage(artifact: LabArtifact) -> dict[str, Any]:
    return json.loads(json.dumps(artifact.lineage))


def _base_round(
    out_dir: Path,
    repo_id: str,
    seed: int,
    device: str | None,
    manifest: dict[str, Any],
    mapping: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[LabArtifact]]:
    voice_names = discover_english_voices(repo_id)
    if not voice_names:
        raise RuntimeError("No English Kokoro voices discovered")
    existing_artifacts = _load_all_artifacts(manifest)
    next_order = _next_round_order(existing_artifacts, "base")
    existing_ids = set(mapping.keys())
    pipelines = PipelineCache(repo_id, device=device)
    script_rel = _save_round_script(out_dir, "base", "base_sample.txt", BASE_SAMPLE)
    artifacts: list[LabArtifact] = []
    new_mapping = dict(mapping)
    for index, voice in enumerate(voice_names, start=0):
        blind_id = _allocate_blind_id(existing_ids, seed + index)
        existing_ids.add(blind_id)
        family = _voice_family(voice)
        wav_path = out_dir / "audio" / "base" / f"{blind_id}.wav"
        mp3_path = out_dir / "audio" / "base" / f"{blind_id}.mp3"
        duration = _synthesize_sample(pipelines, BASE_SAMPLE, voice, _voice_lang_code(voice), DEFAULT_SPEED, wav_path, mp3_path)
        _validate_duration(f"base:{blind_id}", duration, 20.0, 25.0, 18.0, 30.0)
        lineage = _base_lineage(blind_id)
        artifact = LabArtifact(
            blind_id=blind_id,
            kind="base",
            round="base",
            round_order=next_order + index,
            label="base_calibration",
            lang_code=_voice_lang_code(voice),
            source_ref=voice,
            audio_wav=_relative(wav_path, out_dir),
            audio_mp3=_relative(mp3_path, out_dir),
            sample_id="base_sample",
            speed=DEFAULT_SPEED,
            duration_seconds=duration,
            lineage=lineage,
            metadata={
                "material_type": "voice",
                "voice": voice,
                "family": family,
                "speed": DEFAULT_SPEED,
                "script_path": script_rel,
                "source_description": voice,
            },
        )
        artifacts.append(artifact)
        new_mapping[blind_id] = {
            "kind": "base",
            "round": "base",
            "material_type": "voice",
            "voice": voice,
            "family": family,
            "speed": DEFAULT_SPEED,
            "lineage": lineage,
            "source_description": voice,
        }
    round_config = {
        "script_path": script_rel,
        "target_seconds": [20, 25],
        "acceptable_seconds": [18, 30],
    }
    manifest = _update_manifest(out_dir, manifest, new_mapping, "base", artifacts, round_config)
    return manifest, new_mapping, artifacts


def _topic_round(
    out_dir: Path,
    repo_id: str,
    seed: int,
    device: str | None,
    manifest: dict[str, Any],
    mapping: dict[str, Any],
    decisions_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[LabArtifact]]:
    decisions = _load_decisions(decisions_path)
    finalists = _select_finalists(manifest, decisions, "base", ROUND_LIMITS["topic"], {"base"})
    existing_artifacts = _load_all_artifacts(manifest)
    next_order = _next_round_order(existing_artifacts, "topic")
    existing_ids = set(mapping.keys())
    pipelines = PipelineCache(repo_id, device=device)
    script_rel = _save_round_script(out_dir, "topic", "topic_reel.txt", TOPIC_REEL_TEXT)
    artifacts: list[LabArtifact] = []
    new_mapping = dict(mapping)
    for index, source in enumerate(finalists, start=0):
        blind_id = _allocate_blind_id(existing_ids, seed + 100 + index)
        existing_ids.add(blind_id)
        voice_ref, lang_code, family, speed = _resolve_voice_material(source.blind_id, mapping, out_dir)
        wav_path = out_dir / "audio" / "topic" / f"{blind_id}.wav"
        mp3_path = out_dir / "audio" / "topic" / f"{blind_id}.mp3"
        duration = _synthesize_sample(pipelines, TOPIC_REEL_TEXT, voice_ref, lang_code, speed, wav_path, mp3_path)
        if duration < 45.0 or duration > 60.0:
            raise RuntimeError(f"topic:{blind_id} duration {duration:.3f}s must be within 45.0-60.0s")
        lineage = _inherit_lineage(source)
        lineage["topic_ids"] = [blind_id]
        artifact = LabArtifact(
            blind_id=blind_id,
            kind="topic",
            round="topic",
            round_order=next_order + index,
            label="topic_reel",
            lang_code=lang_code,
            source_ref=source.blind_id,
            audio_wav=_relative(wav_path, out_dir),
            audio_mp3=_relative(mp3_path, out_dir),
            sample_id="topic_reel",
            speed=speed,
            duration_seconds=duration,
            lineage=lineage,
            metadata={
                "material_type": "derived",
                "source_blind_id": source.blind_id,
                "family": family,
                "speed": speed,
                "script_path": script_rel,
                "source_description": f"topic from {source.blind_id}",
            },
        )
        artifacts.append(artifact)
        new_mapping[blind_id] = {
            "kind": "topic",
            "round": "topic",
            "material_type": "derived",
            "source_blind_id": source.blind_id,
            "family": family,
            "speed": speed,
            "lineage": lineage,
            "source_description": f"topic from {source.blind_id}",
        }
    round_config = {
        "script_path": script_rel,
        "finalists_from": "base",
        "finalist_limit": ROUND_LIMITS["topic"],
        "target_seconds": [45, 60],
    }
    manifest = _update_manifest(out_dir, manifest, new_mapping, "topic", artifacts, round_config)
    return manifest, new_mapping, artifacts


def _default_blend_specs(finalists: list[LabArtifact]) -> list[dict[str, Any]]:
    a, b, c = finalists
    return [
        {"name": "blend_ab_70_30", "components": [{"ref": a.blind_id, "weight": 0.7}, {"ref": b.blind_id, "weight": 0.3}]},
        {"name": "blend_ac_70_30", "components": [{"ref": a.blind_id, "weight": 0.7}, {"ref": c.blind_id, "weight": 0.3}]},
        {"name": "blend_ba_70_30", "components": [{"ref": b.blind_id, "weight": 0.7}, {"ref": a.blind_id, "weight": 0.3}]},
        {"name": "blend_ca_70_30", "components": [{"ref": c.blind_id, "weight": 0.7}, {"ref": a.blind_id, "weight": 0.3}]},
        {"name": "blend_ab_80_20", "components": [{"ref": a.blind_id, "weight": 0.8}, {"ref": b.blind_id, "weight": 0.2}]},
        {"name": "blend_ab_50_50", "components": [{"ref": a.blind_id, "weight": 0.5}, {"ref": b.blind_id, "weight": 0.5}]},
    ]


def _round_lineage_compare(source: LabArtifact) -> dict[str, Any]:
    lineage = _inherit_lineage(source)
    lineage["topic_ids"] = list(source.lineage.get("topic_ids") or [source.blind_id])
    return lineage


def _blend_round(
    out_dir: Path,
    repo_id: str,
    seed: int,
    device: str | None,
    manifest: dict[str, Any],
    mapping: dict[str, Any],
    decisions_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[LabArtifact]]:
    decisions = _load_decisions(decisions_path)
    finalists = _select_finalists(manifest, decisions, "topic", ROUND_LIMITS["blend"], {"topic"})
    if len(finalists) < 3:
        raise RuntimeError("Blend round requires 3 topic finalists")
    existing_artifacts = _load_all_artifacts(manifest)
    next_order = _next_round_order(existing_artifacts, "blend")
    existing_ids = set(mapping.keys())
    pipelines = PipelineCache(repo_id, device=device)
    script_rel = _save_round_script(out_dir, "blend", "blend_sample.txt", BLEND_SAMPLE)
    artifacts: list[LabArtifact] = []
    new_mapping = dict(mapping)

    for index, source in enumerate(finalists, start=0):
        blind_id = _allocate_blind_id(existing_ids, seed + 200 + index)
        existing_ids.add(blind_id)
        voice_ref, lang_code, family, speed = _resolve_voice_material(source.blind_id, mapping, out_dir)
        wav_path = out_dir / "audio" / "blend" / "base_refs" / f"{blind_id}.wav"
        mp3_path = out_dir / "audio" / "blend" / "base_refs" / f"{blind_id}.mp3"
        duration = _synthesize_sample(pipelines, BLEND_SAMPLE, voice_ref, lang_code, speed, wav_path, mp3_path)
        _validate_duration(f"blend-base:{blind_id}", duration, 25.0, 30.0, 22.0, 35.0)
        lineage = _round_lineage_compare(source)
        artifact = LabArtifact(
            blind_id=blind_id,
            kind="base",
            round="blend",
            round_order=next_order + index,
            label=f"base_compare_{source.blind_id}",
            lang_code=lang_code,
            source_ref=source.blind_id,
            audio_wav=_relative(wav_path, out_dir),
            audio_mp3=_relative(mp3_path, out_dir),
            sample_id="blend_compare_base",
            speed=speed,
            duration_seconds=duration,
            lineage=lineage,
            metadata={
                "material_type": "derived",
                "source_blind_id": source.blind_id,
                "family": family,
                "speed": speed,
                "script_path": script_rel,
                "source_description": f"base compare from {source.blind_id}",
            },
        )
        artifacts.append(artifact)
        new_mapping[blind_id] = {
            "kind": "base",
            "round": "blend",
            "material_type": "derived",
            "source_blind_id": source.blind_id,
            "family": family,
            "speed": speed,
            "lineage": lineage,
            "source_description": f"base compare from {source.blind_id}",
        }

    specs = _default_blend_specs(finalists)
    for index, spec in enumerate(specs, start=0):
        components = spec["components"]
        refs = [item["ref"] for item in components]
        resolved = [mapping[ref] for ref in refs]
        families = {entry.get("family") for entry in resolved}
        if len(families) != 1:
            raise RuntimeError(f"Blend family mismatch for {spec['name']}: {families}")
        pipeline = pipelines.get("a")
        packs = [pipeline.load_voice(_resolve_voice_material(ref, mapping, out_dir)[0]).float() for ref in refs]
        blend_tensor = torch.zeros_like(packs[0])
        for pack, item in zip(packs, components):
            blend_tensor = blend_tensor + pack * float(item["weight"])
        blind_id = _allocate_blind_id(existing_ids, seed + 300 + index)
        existing_ids.add(blind_id)
        tensor_path = out_dir / "tensors" / "blend" / f"{blind_id}.pt"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blend_tensor.cpu(), tensor_path)
        wav_path = out_dir / "audio" / "blend" / "blends" / f"{blind_id}.wav"
        mp3_path = out_dir / "audio" / "blend" / "blends" / f"{blind_id}.mp3"
        duration = _synthesize_sample(pipelines, BLEND_SAMPLE, blend_tensor.float(), "a", DEFAULT_SPEED, wav_path, mp3_path)
        _validate_duration(f"blend:{blind_id}", duration, 25.0, 30.0, 22.0, 35.0)
        base_ids = sorted({base_id for ref in refs for base_id in mapping[ref]["lineage"].get("base_ids", [])})
        topic_ids = sorted({topic_id for ref in refs for topic_id in mapping[ref]["lineage"].get("topic_ids", [])})
        lineage = {
            "base_ids": base_ids,
            "topic_ids": topic_ids,
            "blend_id": blind_id,
            "speed_id": None,
            "final_id": None,
        }
        artifact = LabArtifact(
            blind_id=blind_id,
            kind="blend",
            round="blend",
            round_order=next_order + len(finalists) + index,
            label=spec["name"],
            lang_code="a",
            source_ref=spec["name"],
            audio_wav=_relative(wav_path, out_dir),
            audio_mp3=_relative(mp3_path, out_dir),
            sample_id="blend_sample",
            speed=DEFAULT_SPEED,
            duration_seconds=duration,
            lineage=lineage,
            metadata={
                "material_type": "tensor",
                "tensor_path": _relative(tensor_path, out_dir),
                "tensor_sha256": _sha256_file(tensor_path),
                "family": next(iter(families)),
                "speed": DEFAULT_SPEED,
                "components": [{"ref": item["ref"], "weight": float(item["weight"])} for item in components],
                "script_path": script_rel,
                "source_description": spec["name"],
            },
        )
        artifacts.append(artifact)
        new_mapping[blind_id] = {
            "kind": "blend",
            "round": "blend",
            "material_type": "tensor",
            "tensor_path": _relative(tensor_path, out_dir),
            "tensor_sha256": _sha256_file(tensor_path),
            "family": next(iter(families)),
            "speed": DEFAULT_SPEED,
            "components": [{"ref": item["ref"], "weight": float(item["weight"])} for item in components],
            "lineage": lineage,
            "source_description": spec["name"],
        }

    round_config = {
        "script_path": script_rel,
        "finalists_from": "topic",
        "finalist_limit": ROUND_LIMITS["blend"],
        "base_compare_count": len(finalists),
        "blend_count": len(specs),
        "target_seconds": [25, 30],
        "acceptable_seconds": [22, 35],
    }
    manifest = _update_manifest(out_dir, manifest, new_mapping, "blend", artifacts, round_config)
    return manifest, new_mapping, artifacts


def _speed_round(
    out_dir: Path,
    repo_id: str,
    seed: int,
    device: str | None,
    manifest: dict[str, Any],
    mapping: dict[str, Any],
    decisions_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[LabArtifact]]:
    decisions = _load_decisions(decisions_path)
    finalists = _select_finalists(manifest, decisions, "blend", ROUND_LIMITS["speed"], {"base", "blend"})
    existing_artifacts = _load_all_artifacts(manifest)
    next_order = _next_round_order(existing_artifacts, "speed")
    existing_ids = set(mapping.keys())
    pipelines = PipelineCache(repo_id, device=device)
    script_rel = _save_round_script(out_dir, "speed", "speed_sample.txt", BLEND_SAMPLE)
    speed_values = [0.95, 0.98]
    artifacts: list[LabArtifact] = []
    new_mapping = dict(mapping)
    for finalist_index, source in enumerate(finalists, start=0):
        voice_ref, lang_code, family, _ = _resolve_voice_material(source.blind_id, mapping, out_dir)
        for speed_index, speed_value in enumerate(speed_values, start=0):
            blind_id = _allocate_blind_id(existing_ids, seed + 400 + finalist_index * 10 + speed_index)
            existing_ids.add(blind_id)
            wav_path = out_dir / "audio" / "speed" / f"{blind_id}.wav"
            mp3_path = out_dir / "audio" / "speed" / f"{blind_id}.mp3"
            duration = _synthesize_sample(pipelines, BLEND_SAMPLE, voice_ref, lang_code, speed_value, wav_path, mp3_path)
            _validate_duration(f"speed:{blind_id}", duration, 25.0, 30.0, 22.0, 35.0)
            lineage = _inherit_lineage(source)
            lineage["speed_id"] = blind_id
            artifact = LabArtifact(
                blind_id=blind_id,
                kind="speed",
                round="speed",
                round_order=next_order + finalist_index * len(speed_values) + speed_index,
                label=f"{source.blind_id}@{speed_value:.2f}",
                lang_code=lang_code,
                source_ref=source.blind_id,
                audio_wav=_relative(wav_path, out_dir),
                audio_mp3=_relative(mp3_path, out_dir),
                sample_id="speed_sample",
                speed=speed_value,
                duration_seconds=duration,
                lineage=lineage,
                metadata={
                    "material_type": "derived",
                    "source_blind_id": source.blind_id,
                    "family": family,
                    "speed": speed_value,
                    "script_path": script_rel,
                    "source_description": f"speed from {source.blind_id}",
                },
            )
            artifacts.append(artifact)
            new_mapping[blind_id] = {
                "kind": "speed",
                "round": "speed",
                "material_type": "derived",
                "source_blind_id": source.blind_id,
                "family": family,
                "speed": speed_value,
                "lineage": lineage,
                "source_description": f"speed from {source.blind_id}",
            }
    round_config = {
        "script_path": script_rel,
        "finalists_from": "blend",
        "finalist_limit": ROUND_LIMITS["speed"],
        "speed_values": speed_values,
        "target_seconds": [25, 30],
        "acceptable_seconds": [22, 35],
    }
    manifest = _update_manifest(out_dir, manifest, new_mapping, "speed", artifacts, round_config)
    return manifest, new_mapping, artifacts


def _export_boundary_clip(source_wav: Path, clip_path: Path, start: float, duration: float) -> None:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _ffmpeg_path(),
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source_wav),
            "-t",
            f"{duration:.3f}",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(clip_path),
        ],
        check=True,
        capture_output=True,
    )


def _select_suspicious_boundaries(block_records: list[dict[str, Any]], max_count: int = 5) -> list[dict[str, Any]]:
    if len(block_records) < 2:
        return []
    ranked = []
    for i in range(len(block_records) - 1):
        left = block_records[i]
        right = block_records[i + 1]
        gap = max(0.0, float(right["start"]) - float(left["end"]))
        score = abs(gap - 0.3) * 40 + abs(float(left["duration_seconds"]) - float(right["duration_seconds"])) * 2
        ranked.append(
            {
                "boundary_index": i + 1,
                "left_block": left["block_index"],
                "right_block": right["block_index"],
                "start": max(0.0, float(left["end"]) - 2.2),
                "dur": 4.8,
                "gap": round(gap, 3),
                "score": round(score, 3),
                "clip": f"boundary_{i+1:03d}_b{left['block_index']:03d}_to_b{right['block_index']:03d}.mp3",
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:max_count]


def _render_final_artifact(
    out_dir: Path,
    artifact_root: Path,
    pipelines: PipelineCache,
    voice_ref: Any,
    lang_code: str,
    speed: float,
    script_text: str,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]], str]:
    script_path = artifact_root / "final_script.txt"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_text, encoding="utf-8")
    sentence_units = load_sentence_units(script_path)
    pipeline = pipelines.get(lang_code)
    blocks = _build_blocks(pipeline, [unit.text for unit in sentence_units], 420, 500)
    block_dir = artifact_root / "blocks"
    block_records = []
    concat_parts: list[Path] = []
    audio_cursor = 0.0
    for block_index, block_sentences in enumerate(blocks, start=1):
        block_text = " ".join(block_sentences)
        wav_path = block_dir / f"block_{block_index:03d}.wav"
        mp3_path = block_dir / f"block_{block_index:03d}.mp3"
        audio = _render_text(pipeline, block_text, voice_ref, speed)
        _write_wav(audio, DEFAULT_SAMPLE_RATE, wav_path)
        _wav_to_mp3(wav_path, mp3_path)
        duration = _wav_duration_seconds(wav_path)
        block_records.append(
            {
                "block_index": block_index,
                "text": block_text,
                "wav_path": _relative(wav_path, out_dir),
                "mp3_path": _relative(mp3_path, out_dir),
                "start": round(audio_cursor, 3),
                "end": round(audio_cursor + duration, 3),
                "duration_seconds": round(duration, 3),
                "sentence_count": len(block_sentences),
                "phoneme_chars": _phoneme_chars(pipeline, block_text),
            }
        )
        concat_parts.append(wav_path)
        audio_cursor += duration + 0.3

    final_wav = artifact_root / "audio_master.wav"
    final_mp3 = artifact_root / "audio.mp3"
    if concat_parts:
        combined = []
        for wav_path in concat_parts:
            import wave

            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                combined.append(arr)
                combined.append(np.zeros(int(DEFAULT_SAMPLE_RATE * 0.3), dtype=np.float32))
        merged = np.concatenate(combined[:-1]) if len(combined) > 1 else combined[0]
        _write_wav(merged, DEFAULT_SAMPLE_RATE, final_wav)
        _wav_to_mp3(final_wav, final_mp3)

    suspicious = _select_suspicious_boundaries(block_records, max_count=5)
    for item in suspicious:
        clip_path = artifact_root / "boundary_clips" / item["clip"]
        _export_boundary_clip(final_wav, clip_path, float(item["start"]), float(item["dur"]))
    _save_json(artifact_root / "blocks.json", {"blocks": block_records, "lang_code": lang_code})
    _save_json(artifact_root / "suspicious_boundaries.json", suspicious)
    return _wav_duration_seconds(final_wav), block_records, suspicious, _relative(script_path, out_dir)


def _final_round(
    out_dir: Path,
    repo_id: str,
    seed: int,
    device: str | None,
    manifest: dict[str, Any],
    mapping: dict[str, Any],
    decisions_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[LabArtifact]]:
    decisions = _load_decisions(decisions_path)
    finalists = _select_finalists(manifest, decisions, "speed", ROUND_LIMITS["final"], {"speed"})
    existing_artifacts = _load_all_artifacts(manifest)
    next_order = _next_round_order(existing_artifacts, "final")
    existing_ids = set(mapping.keys())
    pipelines = PipelineCache(repo_id, device=device)
    artifacts: list[LabArtifact] = []
    new_mapping = dict(mapping)
    for index, source in enumerate(finalists, start=0):
        blind_id = _allocate_blind_id(existing_ids, seed + 500 + index)
        existing_ids.add(blind_id)
        voice_ref, lang_code, family, speed = _resolve_voice_material(source.blind_id, mapping, out_dir)
        artifact_root = out_dir / "final" / blind_id
        duration, block_records, suspicious, script_rel = _render_final_artifact(
            out_dir,
            artifact_root,
            pipelines,
            voice_ref,
            lang_code,
            speed,
            FINAL_SCRIPT,
        )
        _validate_final_duration(f"final:{blind_id}", duration)
        lineage = _inherit_lineage(source)
        lineage["final_id"] = blind_id
        artifact = LabArtifact(
            blind_id=blind_id,
            kind="final",
            round="final",
            round_order=next_order + index,
            label=f"final_{source.blind_id}",
            lang_code=lang_code,
            source_ref=source.blind_id,
            audio_wav=_relative(artifact_root / "audio_master.wav", out_dir),
            audio_mp3=_relative(artifact_root / "audio.mp3", out_dir),
            sample_id="final_longform",
            speed=speed,
            duration_seconds=duration,
            lineage=lineage,
            metadata={
                "material_type": "derived",
                "source_blind_id": source.blind_id,
                "family": family,
                "speed": speed,
                "script_path": script_rel,
                "block_count": len(block_records),
                "boundary_clip_count": len(suspicious),
                "source_description": f"final from {source.blind_id}",
            },
        )
        artifacts.append(artifact)
        new_mapping[blind_id] = {
            "kind": "final",
            "round": "final",
            "material_type": "derived",
            "source_blind_id": source.blind_id,
            "family": family,
            "speed": speed,
            "lineage": lineage,
            "source_description": f"final from {source.blind_id}",
        }
    round_config = {
        "script_text": "FINAL_SCRIPT",
        "finalists_from": "speed",
        "finalist_limit": ROUND_LIMITS["final"],
        "required_seconds": [90, 120],
    }
    manifest = _update_manifest(out_dir, manifest, new_mapping, "final", artifacts, round_config)
    return manifest, new_mapping, artifacts


def _round_rows_for_report(
    manifest: dict[str, Any],
    decisions: dict[str, dict[str, str]],
    active_round: str,
) -> list[dict[str, Any]]:
    artifacts = _artifacts_for_round(manifest, active_round)
    rows = []
    for artifact in artifacts:
        decision = decisions.get(artifact.blind_id, _blank_decision_row())
        rows.append(
            {
                "blind_id": artifact.blind_id,
                "round": artifact.round,
                "kind": artifact.kind,
                "round_order": artifact.round_order,
                "audio_mp3": artifact.audio_mp3,
                "audio_wav": artifact.audio_wav,
                "decision": decision.get("decision", ""),
                "revealed": decision.get("revealed", ""),
                "notes": decision.get("notes", ""),
                "revealed_voice": decision.get("revealed_voice", ""),
                "revealed_family": decision.get("revealed_family", ""),
                "revealed_source": decision.get("revealed_source", ""),
                "revealed_lineage": decision.get("revealed_lineage", ""),
            }
        )
    rows.sort(key=lambda item: (DECISION_PRIORITY.get(item["decision"], 3), item["round_order"], item["blind_id"]))
    return rows


def _build_review_html(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    reveal_map: dict[str, dict[str, str]],
    decisions_path: Path,
) -> str:
    manifest_json = json.dumps(manifest, ensure_ascii=False)
    rows_json = json.dumps(rows, ensure_ascii=False)
    reveal_json = json.dumps(reveal_map, ensure_ascii=False)
    decisions_name = "decisions.csv"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Kokoro Voice Lab</title>
  <style>
    :root {{
      --bg: #0f1418;
      --panel: #151d23;
      --panel-2: #1b252d;
      --text: #edf3f7;
      --muted: #93a6b6;
      --line: #2a3945;
      --accent: #cdb380;
      --ok: #8fd694;
      --warn: #f0c674;
      --bad: #f28b82;
    }}
    body {{ margin: 0; font-family: Segoe UI, system-ui, sans-serif; background: linear-gradient(180deg, #0d1115, #121920); color: var(--text); }}
    header {{ padding: 24px 28px 16px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .sub {{ color: var(--muted); font-size: 14px; }}
    .toolbar {{ display: flex; gap: 12px; flex-wrap: wrap; padding: 16px 28px; position: sticky; top: 0; background: rgba(13,17,21,0.92); backdrop-filter: blur(8px); border-bottom: 1px solid var(--line); z-index: 5; }}
    .toolbar input, .toolbar select, .toolbar button {{ background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; }}
    .toolbar button {{ cursor: pointer; }}
    main {{ padding: 20px 28px 36px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 12px 8px; vertical-align: top; }}
    th {{ text-align: left; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    tr.card td {{ background: var(--panel); }}
    tr.card:nth-child(even) td {{ background: var(--panel-2); }}
    .blind {{ font-weight: 700; font-size: 16px; }}
    .muted {{ color: var(--muted); }}
    .hiddenMeta {{ color: var(--muted); font-size: 12px; min-height: 38px; }}
    .decisionRow {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .decisionRow label {{ display: inline-flex; align-items: center; gap: 5px; }}
    .notes {{ width: 240px; min-height: 48px; background: #10161b; color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 8px; }}
    audio {{ width: 240px; }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 999px; background: #202c36; color: var(--accent); font-size: 12px; }}
    .keep {{ color: var(--ok); }}
    .maybe {{ color: var(--warn); }}
    .reject {{ color: var(--bad); }}
    .revealBtn[disabled] {{ opacity: .5; cursor: not-allowed; }}
  </style>
</head>
<body>
  <header>
    <h1>Kokoro Voice Lab</h1>
    <div class="sub">Blind review for the active round. Voice, family, and source stay hidden until you choose Keep, Maybe, or Reject and press Reveal.</div>
  </header>
  <div class="toolbar">
    <input id="filterInput" placeholder="Filter by blind ID or round" />
    <select id="sortSelect">
      <option value="round_order">Sort by round order</option>
      <option value="decision">Sort by decision</option>
      <option value="blind_id">Sort by blind ID</option>
    </select>
    <button id="exportBtn">Export decisions.csv</button>
    <button id="clearBtn">Clear local review</button>
  </div>
  <main>
    <table id="labTable">
      <thead>
        <tr>
          <th>Blind</th>
          <th>Round</th>
          <th>Audio</th>
          <th>Decision</th>
          <th>Reveal</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </main>
  <script>
    window.VOICE_LAB_MANIFEST = {manifest_json};
    window.VOICE_LAB_ROWS = {rows_json};
    window.VOICE_LAB_REVEAL = {reveal_json};
    window.VOICE_LAB_DECISIONS_NAME = {json.dumps(decisions_name)};
    const LOCAL_KEY = 'kokoro_voice_lab_review';
    const rows = window.VOICE_LAB_ROWS || [];
    const revealMap = window.VOICE_LAB_REVEAL || {{}};
    const saved = JSON.parse(localStorage.getItem(LOCAL_KEY) || '[]');
    const savedMap = new Map(saved.map((row) => [row.blind_id, row]));
    const tbody = document.querySelector('#labTable tbody');
    const filterInput = document.querySelector('#filterInput');
    const sortSelect = document.querySelector('#sortSelect');
    const exportBtn = document.querySelector('#exportBtn');
    const clearBtn = document.querySelector('#clearBtn');

    function restoreRow(blindId) {{
      return savedMap.get(blindId) || {{}};
    }}

    function collectRows() {{
      const out = [];
      tbody.querySelectorAll('tr').forEach((tr) => {{
        const decision = tr.querySelector('input[type="radio"]:checked')?.value || '';
        const revealed = tr.querySelector('.revealBtn').dataset.revealed || '';
        const notes = tr.querySelector('.notes').value.trim();
        const hidden = revealMap[tr.dataset.blind] || {{}};
        out.push({{
          blind_id: tr.dataset.blind,
          round: tr.dataset.round,
          decision,
          revealed,
          notes,
          revealed_voice: revealed ? (hidden.revealed_voice || '') : '',
          revealed_family: revealed ? (hidden.revealed_family || '') : '',
          revealed_source: revealed ? (hidden.revealed_source || '') : '',
          revealed_lineage: revealed ? (hidden.revealed_lineage || '') : '',
        }});
      }});
      return out;
    }}

    function saveLocal() {{
      localStorage.setItem(LOCAL_KEY, JSON.stringify(collectRows()));
    }}

    function applyReveal(tr) {{
      const hidden = revealMap[tr.dataset.blind] || {{}};
      const meta = tr.querySelector('.hiddenMeta');
      meta.textContent = [hidden.revealed_voice, hidden.revealed_family, hidden.revealed_source, hidden.revealed_lineage]
        .filter(Boolean)
        .join(' | ') || 'No reveal metadata';
      const button = tr.querySelector('.revealBtn');
      button.dataset.revealed = 'yes';
      saveLocal();
    }}

    function updateRevealState(tr) {{
      const selected = tr.querySelector('input[type="radio"]:checked');
      const button = tr.querySelector('.revealBtn');
      button.disabled = !selected;
    }}

    function decisionClass(decision) {{
      if (decision === 'Keep') return 'keep';
      if (decision === 'Maybe') return 'maybe';
      if (decision === 'Reject') return 'reject';
      return '';
    }}

    function makeRow(item) {{
      const tr = document.createElement('tr');
      tr.className = 'card';
      tr.dataset.blind = item.blind_id;
      tr.dataset.round = item.round;
      const restored = restoreRow(item.blind_id);
      tr.innerHTML = `
        <td><div class="blind">${{item.blind_id}}</div><div class="muted">${{item.kind}}</div></td>
        <td><span class="pill">${{item.round}}</span></td>
        <td><audio controls preload="none" src="${{item.audio_mp3 || item.audio_wav}}"></audio></td>
        <td>
          <div class="decisionRow">
            <label class="keep"><input type="radio" name="decision_${{item.blind_id}}" value="Keep">Keep</label>
            <label class="maybe"><input type="radio" name="decision_${{item.blind_id}}" value="Maybe">Maybe</label>
            <label class="reject"><input type="radio" name="decision_${{item.blind_id}}" value="Reject">Reject</label>
          </div>
        </td>
        <td>
          <button class="revealBtn" disabled data-revealed="">Reveal</button>
          <div class="hiddenMeta">Hidden until reveal</div>
        </td>
        <td><textarea class="notes" placeholder="Notes"></textarea></td>
      `;
      if (restored.decision) {{
        const input = tr.querySelector(`input[value="${{restored.decision}}"]`);
        if (input) input.checked = true;
      }}
      tr.querySelector('.notes').value = restored.notes || '';
      tr.querySelectorAll('input[type="radio"]').forEach((input) => {{
        input.addEventListener('change', () => {{
          updateRevealState(tr);
          tr.dataset.decision = input.value;
          saveLocal();
        }});
      }});
      tr.querySelector('.notes').addEventListener('input', saveLocal);
      tr.querySelector('.revealBtn').addEventListener('click', () => applyReveal(tr));
      updateRevealState(tr);
      if (restored.revealed === 'yes') {{
        applyReveal(tr);
      }}
      tr.dataset.decision = restored.decision || '';
      tr.classList.add(decisionClass(restored.decision || ''));
      return tr;
    }}

    function render() {{
      const filter = filterInput.value.trim().toLowerCase();
      const sortBy = sortSelect.value;
      let items = rows.slice();
      if (filter) {{
        items = items.filter((item) => `${{item.blind_id}} ${{item.round}} ${{item.kind}}`.toLowerCase().includes(filter));
      }}
      items.sort((a, b) => {{
        const ra = restoreRow(a.blind_id);
        const rb = restoreRow(b.blind_id);
        if (sortBy === 'decision') {{
          const da = {{Keep: 0, Maybe: 1, Reject: 2}}[ra.decision] ?? 3;
          const db = {{Keep: 0, Maybe: 1, Reject: 2}}[rb.decision] ?? 3;
          return da - db || a.round_order - b.round_order || a.blind_id.localeCompare(b.blind_id);
        }}
        if (sortBy === 'blind_id') {{
          return a.blind_id.localeCompare(b.blind_id);
        }}
        return a.round_order - b.round_order || a.blind_id.localeCompare(b.blind_id);
      }});
      tbody.innerHTML = '';
      items.forEach((item) => tbody.appendChild(makeRow(item)));
    }}

    function exportCsv() {{
      const currentRows = collectRows();
      const header = {json.dumps(DECISIONS_FIELDNAMES)};
      const lines = [header.join(',')];
      for (const row of currentRows) {{
        lines.push(header.map((key) => JSON.stringify(row[key] || '')).join(','));
      }}
      const blob = new Blob([lines.join('\\n')], {{type: 'text/csv;charset=utf-8;'}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = window.VOICE_LAB_DECISIONS_NAME;
      a.click();
      URL.revokeObjectURL(url);
      saveLocal();
    }}

    function clearLocal() {{
      localStorage.removeItem(LOCAL_KEY);
      location.reload();
    }}

    filterInput.addEventListener('input', render);
    sortSelect.addEventListener('change', render);
    exportBtn.addEventListener('click', exportCsv);
    clearBtn.addEventListener('click', clearLocal);
    render();
  </script>
</body>
</html>
"""


def _report(
    out_dir: Path,
    repo_id: str,
    manifest: dict[str, Any],
    mapping: dict[str, Any],
    decisions_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    active_round = manifest.get("active_round")
    if not active_round:
        raise RuntimeError("Manifest has no active_round")
    decisions = _load_decisions(decisions_path)
    rows = _round_rows_for_report(manifest, decisions, active_round)
    reveal_map = {
        row["blind_id"]: {
            "revealed_voice": row["revealed_voice"],
            "revealed_family": row["revealed_family"],
            "revealed_source": row["revealed_source"] or _resolve_source_description(row["blind_id"], mapping),
            "revealed_lineage": row["revealed_lineage"],
        }
        for row in rows
    }
    leaderboard_path = out_dir / "leaderboard.md"
    lines = [
        "# Kokoro Voice Lab Leaderboard",
        "",
        f"- Repo: `{repo_id}`",
        f"- Active round: `{active_round}`",
        f"- Updated: `{_now_iso()}`",
        "",
        "| Blind | Decision | Kind | Source | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        source = row["revealed_source"] or ""
        lines.append(f"| {row['blind_id']} | {row['decision'] or 'pending'} | {row['kind']} | {source} | {row['notes'] or ''} |")
    leaderboard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    html_path = out_dir / "review.html"
    html_path.write_text(_build_review_html(manifest, rows, reveal_map, decisions_path), encoding="utf-8")
    summary = {
        "repo_id": repo_id,
        "active_round": active_round,
        "artifact_count": len(rows),
        "reviewed_count": sum(1 for row in rows if row["decision"]),
        "leaderboard_path": _relative(leaderboard_path, out_dir),
        "html_path": _relative(html_path, out_dir),
        "decisions_path": str(decisions_path),
    }
    _save_json(out_dir / "report.json", summary)
    return summary, rows


def cmd_base(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _resolve_manifest(out_dir)
    mapping = _resolve_mapping(out_dir)
    decisions_path = _resolve_decisions_path(out_dir, args.decisions)
    manifest, mapping, artifacts = _base_round(out_dir, args.repo_id, args.seed, args.device, manifest, mapping)
    _merge_decisions(decisions_path, artifacts, mapping)
    print(f"Generated {len(artifacts)} base voices in {out_dir}")
    return 0


def cmd_topic(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _resolve_manifest(out_dir)
    mapping = _resolve_mapping(out_dir)
    decisions_path = _resolve_decisions_path(out_dir, args.decisions)
    manifest, mapping, artifacts = _topic_round(out_dir, args.repo_id, args.seed, args.device, manifest, mapping, decisions_path)
    _merge_decisions(decisions_path, artifacts, mapping)
    print(f"Generated {len(artifacts)} topic reels in {out_dir}")
    return 0


def cmd_blend(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _resolve_manifest(out_dir)
    mapping = _resolve_mapping(out_dir)
    decisions_path = _resolve_decisions_path(out_dir, args.decisions)
    manifest, mapping, artifacts = _blend_round(out_dir, args.repo_id, args.seed, args.device, manifest, mapping, decisions_path)
    _merge_decisions(decisions_path, artifacts, mapping)
    print(f"Generated {len(artifacts)} blend-round artifacts in {out_dir}")
    return 0


def cmd_speed(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _resolve_manifest(out_dir)
    mapping = _resolve_mapping(out_dir)
    decisions_path = _resolve_decisions_path(out_dir, args.decisions)
    manifest, mapping, artifacts = _speed_round(out_dir, args.repo_id, args.seed, args.device, manifest, mapping, decisions_path)
    _merge_decisions(decisions_path, artifacts, mapping)
    print(f"Generated {len(artifacts)} speed artifacts in {out_dir}")
    return 0


def cmd_final(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _resolve_manifest(out_dir)
    mapping = _resolve_mapping(out_dir)
    decisions_path = _resolve_decisions_path(out_dir, args.decisions)
    manifest, mapping, artifacts = _final_round(out_dir, args.repo_id, args.seed, args.device, manifest, mapping, decisions_path)
    _merge_decisions(decisions_path, artifacts, mapping)
    print(f"Generated {len(artifacts)} final artifacts in {out_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _resolve_manifest(out_dir)
    mapping = _resolve_mapping(out_dir)
    decisions_path = _resolve_decisions_path(out_dir, args.decisions)
    summary, _rows = _report(out_dir, args.repo_id, manifest, mapping, decisions_path)
    print(json.dumps(summary, indent=2))
    print(f"Leaderboard written to {out_dir / 'leaderboard.md'}")
    print(f"Review HTML written to {out_dir / 'review.html'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Round-gated Kokoro voice lab")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--decisions", type=Path, default=None, help="Path to decisions.csv (default: <output-dir>/decisions.csv)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_base = sub.add_parser("base", help="Generate all base English voice samples")
    p_base.set_defaults(func=cmd_base)

    p_topic = sub.add_parser("topic", help="Generate topic reels from reviewed base finalists")
    p_topic.set_defaults(func=cmd_topic)
    p_topics = sub.add_parser("topics", help="Alias for topic")
    p_topics.set_defaults(func=cmd_topic)

    p_blend = sub.add_parser("blend", help="Generate base comparisons and blends from reviewed topic finalists")
    p_blend.set_defaults(func=cmd_blend)
    p_blends = sub.add_parser("blends", help="Alias for blend")
    p_blends.set_defaults(func=cmd_blend)

    p_speed = sub.add_parser("speed", help="Generate speed variants from reviewed blend finalists")
    p_speed.set_defaults(func=cmd_speed)

    p_final = sub.add_parser("final", help="Generate longform finals from reviewed speed finalists")
    p_final.set_defaults(func=cmd_final)
    p_longform = sub.add_parser("longform", help="Alias for final")
    p_longform.set_defaults(func=cmd_final)

    p_report = sub.add_parser("report", help="Build review HTML and leaderboard from decisions.csv")
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
