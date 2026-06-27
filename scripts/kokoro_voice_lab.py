"""Independent Kokoro voice lab for blind testing base voices, blends, speed, and longform."""

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
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from huggingface_hub import list_repo_files
from kokoro import KPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steps.text_units import load_sentence_units  # noqa: E402

LAB_VERSION = 1
DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SPEED = 0.95
DEFAULT_SEED = 20260627
DEFAULT_OUTPUT_DIR = Path("output") / "voice_lab"
CALIBRATION_SAMPLE = (
    "A signal crosses the room, quiet at first, then unmistakable.\n\n"
    "You can hear the old machinery of the story: pressure, timing, and a voice that stays calm "
    "even when the room itself is not."
)

TOPIC_PACKS: list[dict[str, str]] = [
    {
        "slug": "history_documentary",
        "title": "History Documentary",
        "text": (
            "Long before the first city wall, people already knew how to wait for weather, "
            "how to read tracks, and how to remember the shape of a river. "
            "That kind of knowledge does not survive in books first. "
            "It survives in habits, in warnings, and in the names of places."
        ),
    },
    {
        "slug": "dark_mystery",
        "title": "Dark Mystery",
        "text": (
            "At the edge of the firelight, the evidence looks simple. "
            "But the simple version is usually the one that hides the real answer. "
            "When the body says one thing and the timeline says another, the silence becomes part of the story."
        ),
    },
    {
        "slug": "science_technology",
        "title": "Science and Technology",
        "text": (
            "A good tool is not just harder than the material it cuts. "
            "It is also a way of thinking about force, angle, and timing. "
            "The same logic lives in a stone blade, a microscope, and a machine that can hear a word before we do."
        ),
    },
    {
        "slug": "finance_business",
        "title": "Finance and Business",
        "text": (
            "Every budget hides a decision, and every decision hides a tradeoff. "
            "That is true for a household, a factory, and a company trying to survive a bad quarter. "
            "Money is only the visible layer; the real story is always allocation."
        ),
    },
    {
        "slug": "calm_reflective",
        "title": "Calm Reflective",
        "text": (
            "Some stories do not need to shout to be remembered. "
            "They ask for a slower breath, a steadier pace, and a willingness to sit with what is left unsaid. "
            "That is where a voice feels human instead of merely accurate."
        ),
    },
    {
        "slug": "energetic_hook",
        "title": "Energetic Hook",
        "text": (
            "What if the oldest lesson in the room is not the one you expected? "
            "What if the clue you skipped over is the one holding everything together? "
            "Stay with it, because the answer changes the way the whole story lands."
        ),
    },
    {
        "slug": "dialogue_quotation",
        "title": "Dialogue and Quotation",
        "text": (
            "He said, 'Start with the evidence and ignore the drama.' "
            "Then she replied, 'That only works if the evidence is allowed to speak.' "
            "The line between those two sentences is where a lot of truth lives."
        ),
    },
    {
        "slug": "pronunciation_stress_test",
        "title": "Pronunciation Stress Test",
        "text": (
            "Liang Tebo, Borneo, Tim Maloney, Maxime Aubert, and Melandri Vlok all sound different, "
            "and that is exactly the point. "
            "The lab should keep these names clear, steady, and understandable without flattening the voice."
        ),
    },
]

WEIGHTS = {
    "naturalness": 0.30,
    "clarity": 0.20,
    "comfort": 0.20,
    "emotional_range": 0.10,
    "versatility": 0.10,
    "distinctiveness": 0.10,
}


@dataclasses.dataclass
class LabArtifact:
    blind_id: str
    kind: str
    label: str
    lang_code: str
    family: str
    source_ref: str
    audio_wav: str
    audio_mp3: str
    sample_id: str
    speed: float
    duration_seconds: float
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


def _slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower() or "item"


def _voice_family(voice: str) -> str:
    return voice.split("_", 1)[0]


def _voice_lang_code(voice: str) -> str:
    return voice[0].lower()


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


def _blank_scores_row(blind_id: str, kind: str, family: str) -> dict[str, str]:
    row = {
        "blind_id": blind_id,
        "kind": kind,
        "family": family,
        "naturalness": "",
        "clarity": "",
        "comfort": "",
        "emotional_range": "",
        "versatility": "",
        "distinctiveness": "",
        "weighted_total": "",
        "notes": "",
    }
    return row


def _weighted_total(row: dict[str, str]) -> str:
    total = 0.0
    filled = False
    for key, weight in WEIGHTS.items():
        value = row.get(key, "").strip()
        if not value:
            return ""
        filled = True
        total += float(value) * weight
    return f"{total:.2f}" if filled else ""


def _load_scores(path: Path) -> dict[str, dict[str, str]]:
    scores: dict[str, dict[str, str]] = {}
    if not path.exists():
        return scores
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            blind_id = (row.get("blind_id") or "").strip()
            if blind_id:
                scores[blind_id] = row
    return scores


def _write_scores(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "blind_id",
        "kind",
        "family",
        "naturalness",
        "clarity",
        "comfort",
        "emotional_range",
        "versatility",
        "distinctiveness",
        "weighted_total",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_review_html(manifest: dict[str, Any], scores: list[dict[str, str]]) -> str:
    manifest_json = json.dumps(manifest, ensure_ascii=False)
    scores_json = json.dumps(scores, ensure_ascii=False)
    weights_json = json.dumps(WEIGHTS, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Kokoro Voice Lab</title>
  <style>
    :root {{
      --bg: #0c1116;
      --panel: #131a21;
      --panel-2: #18222c;
      --text: #e8eef4;
      --muted: #8fa1b5;
      --accent: #89c2d9;
      --line: #253241;
      --good: #73d09b;
      --warn: #f3c969;
    }}
    body {{ margin: 0; font-family: Segoe UI, system-ui, sans-serif; background: linear-gradient(180deg, #0c1116, #111821); color: var(--text); }}
    header {{ padding: 24px 28px 16px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .sub {{ color: var(--muted); font-size: 14px; }}
    .toolbar {{ display: flex; gap: 12px; flex-wrap: wrap; padding: 16px 28px; position: sticky; top: 0; background: rgba(12,17,22,0.9); backdrop-filter: blur(8px); border-bottom: 1px solid var(--line); z-index: 5; }}
    .toolbar input, .toolbar select, .toolbar button {{ background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; }}
    main {{ padding: 20px 28px 36px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; vertical-align: top; }}
    th {{ text-align: left; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    tr.card td {{ background: var(--panel); }}
    tr.card:nth-child(even) td {{ background: var(--panel-2); }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 999px; background: #1b2633; color: var(--accent); font-size: 12px; }}
    .voice {{ font-weight: 700; font-size: 16px; }}
    audio {{ width: 240px; }}
    input.score {{ width: 64px; padding: 8px; background: #0f151c; color: var(--text); border: 1px solid var(--line); border-radius: 8px; }}
    .score-total {{ font-weight: 700; color: var(--good); }}
    .notes {{ width: 220px; min-height: 44px; background: #0f151c; color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 8px; }}
    .muted {{ color: var(--muted); }}
    .footer {{ color: var(--muted); padding: 18px 28px 34px; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>Kokoro Voice Lab</h1>
    <div class="sub">Blind test base voices, blends, speed variants, and longform finalists. Scores auto-calc in browser and persist in localStorage.</div>
  </header>
  <div class="toolbar">
    <input id="filterInput" placeholder="Filter by blind ID or kind" />
    <select id="sortSelect">
      <option value="blind_id">Sort by blind id</option>
      <option value="score">Sort by weighted score</option>
      <option value="kind">Sort by kind</option>
    </select>
    <button id="saveBtn">Save local scores</button>
    <button id="exportBtn">Export CSV</button>
    <button id="clearBtn">Clear local scores</button>
  </div>
  <main>
    <table id="labTable">
      <thead>
        <tr>
          <th>Blind</th>
          <th>Kind</th>
          <th>Audio</th>
          <th>Naturalness</th>
          <th>Clarity</th>
          <th>Comfort</th>
          <th>Emotion</th>
          <th>Versatility</th>
          <th>Distinct.</th>
          <th>Total</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </main>
  <div class="footer">
    Weights: {json.dumps(WEIGHTS)}. Use the report command to regenerate leaderboard.md from the saved CSV.
  </div>
  <script>
    window.VOICE_LAB_MANIFEST = {manifest_json};
    window.VOICE_LAB_SCORES = {scores_json};
    window.VOICE_LAB_WEIGHTS = {weights_json};
    const rows = window.VOICE_LAB_MANIFEST.artifacts || [];
    const defaultScores = new Map((window.VOICE_LAB_SCORES || []).map((row) => [row.blind_id, row]));
    const tbody = document.querySelector('#labTable tbody');
    const filterInput = document.querySelector('#filterInput');
    const sortSelect = document.querySelector('#sortSelect');
    const saveBtn = document.querySelector('#saveBtn');
    const exportBtn = document.querySelector('#exportBtn');
    const clearBtn = document.querySelector('#clearBtn');

    function weightedTotalFromRow(row) {{
      const weights = window.VOICE_LAB_WEIGHTS;
      let total = 0;
      for (const [key, weight] of Object.entries(weights)) {{
        const value = row.querySelector(`[data-score="${{key}}"]`).value.trim();
        if (!value) return '';
        total += Number(value) * weight;
      }}
      return total.toFixed(2);
    }}

    function makeRow(item) {{
      const tr = document.createElement('tr');
      tr.className = 'card';
      tr.dataset.blind = item.blind_id;
      tr.dataset.kind = item.kind;
      tr.dataset.score = defaultScores.has(item.blind_id) ? (defaultScores.get(item.blind_id).weighted_total || '') : '';
      tr.innerHTML = `
        <td><div class="voice">${{item.blind_id}}</div><div class="muted">${{item.family || ''}}</div></td>
        <td><span class="pill">${{item.kind}}</span></td>
        <td>
          <audio controls preload="none" src="${{item.audio_mp3 || item.audio_wav}}"></audio>
          <div class="muted">${{item.sample_id || ''}}</div>
        </td>
        <td><input class="score" data-score="naturalness" type="number" min="1" max="10" step="1"></td>
        <td><input class="score" data-score="clarity" type="number" min="1" max="10" step="1"></td>
        <td><input class="score" data-score="comfort" type="number" min="1" max="10" step="1"></td>
        <td><input class="score" data-score="emotional_range" type="number" min="1" max="10" step="1"></td>
        <td><input class="score" data-score="versatility" type="number" min="1" max="10" step="1"></td>
        <td><input class="score" data-score="distinctiveness" type="number" min="1" max="10" step="1"></td>
        <td class="score-total" data-total>—</td>
        <td><textarea class="notes" placeholder="Notes"></textarea></td>
      `;
      const base = defaultScores.get(item.blind_id);
      if (base) {{
        for (const key of Object.keys(window.VOICE_LAB_WEIGHTS)) {{
          tr.querySelector(`[data-score="${{key}}"]`).value = base[key] || '';
        }}
        tr.querySelector('[data-total]').textContent = base.weighted_total || '—';
        tr.querySelector('.notes').value = base.notes || '';
      }} else {{
        tr.querySelector('[data-total]').textContent = weightedTotalFromRow(tr) || '—';
      }}
      tr.querySelectorAll('input.score').forEach((input) => {{
        input.addEventListener('input', () => {{
          const total = weightedTotalFromRow(tr);
          tr.querySelector('[data-total]').textContent = total || '—';
          tr.dataset.score = total || '';
        }});
      }});
      return tr;
    }}

    function render() {{
      const filter = filterInput.value.trim().toLowerCase();
      const sortBy = sortSelect.value;
      let items = rows.slice();
      if (filter) {{
        items = items.filter((item) => `${{item.blind_id}} ${{item.kind}} ${{item.family || ''}}`.toLowerCase().includes(filter));
      }}
      items.sort((a, b) => {{
        if (sortBy === 'kind') return a.kind.localeCompare(b.kind) || a.blind_id.localeCompare(b.blind_id);
        if (sortBy === 'score') {{
          const sa = Number(defaultScores.get(a.blind_id)?.weighted_total || 0);
          const sb = Number(defaultScores.get(b.blind_id)?.weighted_total || 0);
          return sb - sa || a.blind_id.localeCompare(b.blind_id);
        }}
        return a.blind_id.localeCompare(b.blind_id);
      }});
      tbody.innerHTML = '';
      items.forEach((item) => tbody.appendChild(makeRow(item)));
    }}

    function collectRows() {{
      const out = [];
      tbody.querySelectorAll('tr').forEach((tr) => {{
        const row = {{
          blind_id: tr.dataset.blind,
          kind: tr.dataset.kind,
          naturalness: tr.querySelector('[data-score="naturalness"]').value.trim(),
          clarity: tr.querySelector('[data-score="clarity"]').value.trim(),
          comfort: tr.querySelector('[data-score="comfort"]').value.trim(),
          emotional_range: tr.querySelector('[data-score="emotional_range"]').value.trim(),
          versatility: tr.querySelector('[data-score="versatility"]').value.trim(),
          distinctiveness: tr.querySelector('[data-score="distinctiveness"]').value.trim(),
          weighted_total: tr.querySelector('[data-total]').textContent.trim().replace(/^—$/, ''),
          notes: tr.querySelector('.notes').value.trim(),
        }};
        out.push(row);
      }});
      return out;
    }}

    function saveLocal() {{
      localStorage.setItem('kokoro_voice_lab_scores', JSON.stringify(collectRows()));
      alert('Saved to localStorage');
    }}

    function exportCsv() {{
      const rows = collectRows();
      const header = ['blind_id','kind','naturalness','clarity','comfort','emotional_range','versatility','distinctiveness','weighted_total','notes'];
      const lines = [header.join(',')];
      for (const row of rows) {{
        lines.push(header.map((key) => JSON.stringify(row[key] || '')).join(','));
      }}
      const blob = new Blob([lines.join('\\n')], {{type: 'text/csv;charset=utf-8;'}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'scores.csv';
      a.click();
      URL.revokeObjectURL(url);
    }}

    function clearLocal() {{
      localStorage.removeItem('kokoro_voice_lab_scores');
      alert('Cleared localStorage. Reload the page to reset the fields.');
    }}

    filterInput.addEventListener('input', render);
    sortSelect.addEventListener('change', render);
    saveBtn.addEventListener('click', saveLocal);
    exportBtn.addEventListener('click', exportCsv);
    clearBtn.addEventListener('click', clearLocal);
    render();
  </script>
</body>
</html>
"""


def _make_manifest(out_dir: Path, repo_id: str, kind: str, artifacts: list[LabArtifact]) -> dict[str, Any]:
    manifest = _load_json(out_dir / "manifest.json", default={})
    manifest.update(
        {
            "version": LAB_VERSION,
            "created_at": manifest.get("created_at") or _now_iso(),
            "updated_at": _now_iso(),
            "repo_id": repo_id,
            "kind": kind,
            "artifacts": [dataclasses.asdict(item) for item in artifacts],
            "artifact_count": len(artifacts),
        }
    )
    return manifest


def _save_manifest_bundle(out_dir: Path, manifest: dict[str, Any], mapping: dict[str, Any]) -> None:
    _save_json(out_dir / "manifest.json", manifest)
    _save_json(out_dir / "blind_mapping.json", mapping)


def _voice_records_to_scores(artifacts: list[LabArtifact]) -> list[dict[str, str]]:
    rows = []
    for artifact in artifacts:
        row = _blank_scores_row(artifact.blind_id, artifact.kind, artifact.family)
        rows.append(row)
    return rows


def _merge_artifacts(existing: list[LabArtifact], new: list[LabArtifact], kind: str) -> list[LabArtifact]:
    kept = [artifact for artifact in existing if artifact.kind != kind]
    return kept + new


def _merge_score_rows(existing: dict[str, dict[str, str]], new_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    merged = dict(existing)
    for row in new_rows:
        blind_id = row["blind_id"]
        if blind_id not in merged:
            merged[blind_id] = row
            continue
        current = merged[blind_id]
        for key, value in row.items():
            if not current.get(key):
                current[key] = value
    return list(merged.values())


def _resolve_voice_material(blind_id: str, mapping: dict[str, Any], out_dir: Path) -> tuple[Any, str, str, float]:
    entry = mapping[blind_id]
    kind = entry["kind"]
    if kind == "base":
        return entry["voice"], entry["lang_code"], entry["family"], DEFAULT_SPEED
    if kind == "blend":
        tensor_path = out_dir / entry["tensor_path"]
        tensor = torch.load(tensor_path, weights_only=True).float()
        return tensor, entry["lang_code"], entry["family"], DEFAULT_SPEED
    if kind in {"speed", "longform"}:
        source_id = entry["source_blind_id"]
        voice_ref, lang_code, family, inherited_speed = _resolve_voice_material(source_id, mapping, out_dir)
        speed = float(entry.get("speed", inherited_speed))
        return voice_ref, lang_code, family, speed
    raise ValueError(f"Unsupported artifact kind: {kind}")


def _resolve_blind_ref(ref: str, mapping: dict[str, Any]) -> dict[str, Any]:
    if ref in mapping:
        return mapping[ref]
    return {"voice": ref, "lang_code": _voice_lang_code(ref), "family": _voice_family(ref)}


def _synthesize_sample(
    pipelines: PipelineCache,
    text: str,
    voice_ref: Any,
    lang_code: str,
    speed: float,
    wav_path: Path,
    mp3_path: Path,
) -> tuple[int, float]:
    pipeline = pipelines.get(lang_code)
    audio = _render_text(pipeline, text, voice_ref, speed)
    _write_wav(audio, DEFAULT_SAMPLE_RATE, wav_path)
    _wav_to_mp3(wav_path, mp3_path)
    return DEFAULT_SAMPLE_RATE, _wav_duration_seconds(wav_path)


def _base_artifacts(
    out_dir: Path,
    repo_id: str,
    limit: int | None,
    seed: int,
    device: str | None,
    voice_filter: set[str] | None,
    family_filter: set[str] | None,
) -> tuple[dict[str, Any], dict[str, Any], list[LabArtifact]]:
    voice_names = discover_english_voices(repo_id)
    if voice_filter:
        voice_names = [voice for voice in voice_names if voice in voice_filter]
    if family_filter:
        voice_names = [voice for voice in voice_names if _voice_family(voice) in family_filter]
    if not voice_names:
        raise RuntimeError("No English Kokoro voices discovered")
    if limit is not None:
        voice_names = voice_names[:limit]
    existing_ids = set()
    blind_ids = []
    for _ in voice_names:
        blind_id = _allocate_blind_id(existing_ids, seed)
        existing_ids.add(blind_id)
        blind_ids.append(blind_id)
    mapping: dict[str, Any] = {}
    artifacts: list[LabArtifact] = []
    pipelines = PipelineCache(repo_id, device=device)
    for blind_id, voice in zip(blind_ids, voice_names):
        lang_code = _voice_lang_code(voice)
        family = _voice_family(voice)
        audio_dir = out_dir / "audio" / "base"
        wav_path = audio_dir / f"{blind_id}.wav"
        mp3_path = audio_dir / f"{blind_id}.mp3"
        _synthesize_sample(pipelines, CALIBRATION_SAMPLE, voice, lang_code, DEFAULT_SPEED, wav_path, mp3_path)
        mapping[blind_id] = {
            "kind": "base",
            "voice": voice,
            "lang_code": lang_code,
            "family": family,
        }
        artifacts.append(
            LabArtifact(
                blind_id=blind_id,
                kind="base",
                label="calibration",
                lang_code=lang_code,
                family=family,
                source_ref=voice,
                audio_wav=str(wav_path.relative_to(out_dir)).replace("\\", "/"),
                audio_mp3=str(mp3_path.relative_to(out_dir)).replace("\\", "/"),
                sample_id="calibration",
                speed=DEFAULT_SPEED,
                duration_seconds=_wav_duration_seconds(wav_path),
                metadata={"voice": voice, "source": "official"},
            )
        )
    manifest = _make_manifest(out_dir, repo_id, "base", artifacts)
    return manifest, mapping, artifacts


def _topic_manifest(out_dir: Path, repo_id: str) -> dict[str, Any]:
    topic_dir = out_dir / "topics"
    topic_dir.mkdir(parents=True, exist_ok=True)
    topic_items = []
    for topic in TOPIC_PACKS:
        text_path = topic_dir / f"{topic['slug']}.txt"
        text_path.write_text(topic["text"], encoding="utf-8")
        topic_items.append(
            {
                "slug": topic["slug"],
                "title": topic["title"],
                "text_path": str(text_path.relative_to(out_dir)).replace("\\", "/"),
                "sentence_count": len(load_sentence_units(text_path)),
            }
        )
    manifest = _load_json(out_dir / "manifest.json", default={})
    manifest.update(
        {
            "version": LAB_VERSION,
            "updated_at": _now_iso(),
            "repo_id": repo_id,
            "topic_packs": topic_items,
            "topic_pack_count": len(topic_items),
        }
    )
    return manifest


def _default_blend_specs(base_artifacts: list[LabArtifact]) -> list[dict[str, Any]]:
    grouped: dict[str, list[LabArtifact]] = defaultdict(list)
    for artifact in base_artifacts:
        grouped[artifact.family].append(artifact)
    specs: list[dict[str, Any]] = []
    for family, items in grouped.items():
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda item: item.blind_id)
        for i in range(0, min(len(items) - 1, 4), 2):
            left = items[i]
            right = items[i + 1]
            spec = {
                "name": f"blend_{family}_{i // 2 + 1:02d}",
                "lang_code": left.lang_code,
                "components": [
                    {"ref": left.blind_id, "weight": 0.6},
                    {"ref": right.blind_id, "weight": 0.4},
                ],
            }
            specs.append(spec)
            if len(specs) >= 12:
                return specs
    return specs


def _load_base_artifacts(manifest: dict[str, Any]) -> list[LabArtifact]:
    artifacts = []
    for item in manifest.get("artifacts", []):
        artifacts.append(LabArtifact(**item))
    return artifacts


def _load_all_artifacts(manifest: dict[str, Any]) -> list[LabArtifact]:
    return _load_base_artifacts(manifest)


def _mapping_entry_from_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    kind = artifact["kind"]
    if kind == "base":
        return {
            "kind": "base",
            "voice": artifact["source_ref"],
            "lang_code": artifact["lang_code"],
            "family": artifact["family"],
        }
    if kind == "blend":
        return {
            "kind": "blend",
            "lang_code": artifact["lang_code"],
            "family": artifact["family"],
            "components": artifact.get("metadata", {}).get("components", []),
            "tensor_path": artifact.get("metadata", {}).get("tensor_path"),
            "tensor_sha256": artifact.get("metadata", {}).get("tensor_sha256"),
            "kokoro_version": artifact.get("metadata", {}).get("kokoro_version"),
        }
    if kind == "speed":
        return {
            "kind": "speed",
            "source_blind_id": artifact.get("metadata", {}).get("source_blind_id"),
            "lang_code": artifact["lang_code"],
            "family": artifact["family"],
            "speed": artifact.get("speed"),
        }
    if kind == "longform":
        return {
            "kind": "longform",
            "source_blind_id": artifact.get("metadata", {}).get("source_blind_id"),
            "lang_code": artifact["lang_code"],
            "family": artifact["family"],
            "speed": artifact.get("speed"),
            "block_count": artifact.get("metadata", {}).get("block_count"),
        }
    return {
        "kind": kind,
        "lang_code": artifact.get("lang_code"),
        "family": artifact.get("family"),
    }


def _render_blend_artifact(
    out_dir: Path,
    repo_id: str,
    seed: int,
    device: str | None,
    base_manifest: dict[str, Any],
    mapping: dict[str, Any],
    specs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[LabArtifact]]:
    base_artifacts = _load_base_artifacts(base_manifest)
    existing_artifacts = _load_all_artifacts(base_manifest)
    base_lookup = {item.blind_id: item for item in base_artifacts}
    pipelines = PipelineCache(repo_id, device=device)
    blend_dir = out_dir / "audio" / "blends"
    tensor_dir = out_dir / "tensors" / "blends"
    artifacts: list[LabArtifact] = []
    new_mapping = dict(mapping)
    for index, spec in enumerate(specs, start=1):
        components = spec["components"]
        weights = [float(c["weight"]) for c in components]
        total = sum(weights)
        if not np.isclose(total, 1.0):
            weights = [w / total for w in weights]
        refs = [c["ref"] for c in components]
        resolved = [_resolve_blind_ref(ref, new_mapping) for ref in refs]
        voices = [entry["voice"] for entry in resolved]
        lang_code = spec.get("lang_code") or resolved[0]["lang_code"]
        if any(entry["lang_code"] != lang_code for entry in resolved):
            raise ValueError(f"Blend {spec['name']} mixes lang codes")
        families = {entry["family"] for entry in resolved}
        if len(families) != 1:
            raise ValueError(f"Blend {spec['name']} mixes accent families: {families}")
        pipeline = pipelines.get(lang_code)
        packs = [pipeline.load_voice(voice).float() for voice in voices]
        blend_tensor = torch.zeros_like(packs[0])
        for weight, pack in zip(weights, packs):
            blend_tensor = blend_tensor + pack * float(weight)
        blend_id = _allocate_blind_id(set(new_mapping.keys()), seed + 1000 + index)
        tensor_path = tensor_dir / f"{blend_id}.pt"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blend_tensor.cpu(), tensor_path)
        wav_path = blend_dir / f"{blend_id}.wav"
        mp3_path = blend_dir / f"{blend_id}.mp3"
        audio = _render_text(pipeline, CALIBRATION_SAMPLE, blend_tensor.float(), DEFAULT_SPEED)
        _write_wav(audio, DEFAULT_SAMPLE_RATE, wav_path)
        _wav_to_mp3(wav_path, mp3_path)
        sha256 = _sha256_file(tensor_path)
        new_mapping[blend_id] = {
            "kind": "blend",
            "lang_code": lang_code,
            "family": next(iter(families)),
            "components": [{"ref": ref, "weight": weight} for ref, weight in zip(refs, weights)],
            "tensor_path": str(tensor_path.relative_to(out_dir)).replace("\\", "/"),
            "tensor_sha256": sha256,
            "kokoro_version": "0.9.4",
        }
        artifacts.append(
            LabArtifact(
                blind_id=blend_id,
                kind="blend",
                label=spec["name"],
                lang_code=lang_code,
                family=next(iter(families)),
                source_ref=spec["name"],
                audio_wav=str(wav_path.relative_to(out_dir)).replace("\\", "/"),
                audio_mp3=str(mp3_path.relative_to(out_dir)).replace("\\", "/"),
                sample_id="calibration",
                speed=DEFAULT_SPEED,
                duration_seconds=_wav_duration_seconds(wav_path),
                metadata={
                    "components": [{"ref": ref, "weight": weight} for ref, weight in zip(refs, weights)],
                    "tensor_path": str(tensor_path.relative_to(out_dir)).replace("\\", "/"),
                    "tensor_sha256": sha256,
                    "kokoro_version": "0.9.4",
                },
            )
        )
    manifest = _load_json(out_dir / "manifest.json", default={})
    manifest.update(
        {
            "version": LAB_VERSION,
            "updated_at": _now_iso(),
            "repo_id": repo_id,
            "blend_specs": specs,
            "blend_count": len(artifacts),
            "artifacts": [dataclasses.asdict(item) for item in _merge_artifacts(existing_artifacts, artifacts, "blend")],
        }
    )
    return manifest, new_mapping, artifacts


def _load_selected_ids(
    args_ids: list[str] | None,
    leaderboard_path: Path,
    default_ids: list[str],
    limit: int | None,
    allowed_ids: set[str] | None = None,
) -> list[str]:
    if args_ids:
        selected = list(dict.fromkeys(args_ids))
    elif leaderboard_path.exists():
        selected = []
        with leaderboard_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("| ") and not line.startswith("| Blind"):
                    parts = [p.strip() for p in line.strip().strip("|").split("|")]
                    if parts and parts[0] and parts[0] != "---":
                        blind_id = parts[0]
                        if allowed_ids is None or blind_id in allowed_ids:
                            selected.append(blind_id)
    else:
        selected = list(default_ids)
    if limit is not None:
        selected = selected[:limit]
    return selected


def _base_for_id(artifact_lookup: dict[str, LabArtifact], blind_id: str, mapping: dict[str, Any]) -> tuple[str, Any, float, str]:
    entry = mapping.get(blind_id)
    if not entry:
        raise KeyError(f"Unknown blind id: {blind_id}")
    if entry["kind"] == "base":
        return entry["voice"], entry["voice"], DEFAULT_SPEED, entry["lang_code"]
    if entry["kind"] == "blend":
        return blind_id, blind_id, DEFAULT_SPEED, entry["lang_code"]
    if entry["kind"] == "speed":
        source = entry["source_blind_id"]
        source_entry = mapping[source]
        voice_ref = source_entry["voice"] if source_entry["kind"] == "base" else source
        return voice_ref, voice_ref, float(entry["speed"]), entry["lang_code"]
    if entry["kind"] == "longform":
        source = entry["source_blind_id"]
        source_entry = mapping[source]
        voice_ref = source_entry["voice"] if source_entry["kind"] == "base" else source
        return voice_ref, voice_ref, float(entry["speed"]), entry["lang_code"]
    raise ValueError(f"Unsupported artifact kind: {entry['kind']}")


def _render_speed_artifacts(
    out_dir: Path,
    repo_id: str,
    device: str | None,
    mapping: dict[str, Any],
    finalists: list[str],
    limit: int | None,
) -> tuple[dict[str, Any], dict[str, Any], list[LabArtifact]]:
    speed_values = [0.92, 0.95, 0.98, 1.00]
    candidates = finalists[:]
    if not candidates:
        raise RuntimeError("No finalists supplied for speed testing")
    if limit is not None:
        candidates = candidates[:limit]
    artifacts: list[LabArtifact] = []
    new_mapping = dict(mapping)
    pipelines = PipelineCache(repo_id, device=device)
    speed_dir = out_dir / "audio" / "speed"
    for source_id in candidates:
        voice_ref, lang_code, family, _ = _resolve_voice_material(source_id, mapping, out_dir)
        for speed in speed_values:
            blind_id = _allocate_blind_id(set(new_mapping.keys()), DEFAULT_SEED + 2000 + len(artifacts) + 1)
            wav_path = speed_dir / f"{blind_id}.wav"
            mp3_path = speed_dir / f"{blind_id}.mp3"
            pipeline = pipelines.get(lang_code)
            audio = _render_text(pipeline, CALIBRATION_SAMPLE, voice_ref, speed)
            _write_wav(audio, DEFAULT_SAMPLE_RATE, wav_path)
            _wav_to_mp3(wav_path, mp3_path)
            new_mapping[blind_id] = {
                "kind": "speed",
                "source_blind_id": source_id,
                "lang_code": lang_code,
                "family": family,
                "speed": speed,
            }
            artifacts.append(
                LabArtifact(
                    blind_id=blind_id,
                    kind="speed",
                    label=f"{source_id}@{speed:.2f}",
                    lang_code=lang_code,
                    family=family,
                    source_ref=source_id,
                    audio_wav=str(wav_path.relative_to(out_dir)).replace("\\", "/"),
                    audio_mp3=str(mp3_path.relative_to(out_dir)).replace("\\", "/"),
                    sample_id="calibration",
                    speed=speed,
                    duration_seconds=_wav_duration_seconds(wav_path),
                    metadata={"source_blind_id": source_id, "speed": speed},
                )
            )
    manifest = _load_json(out_dir / "manifest.json", default={})
    existing_artifacts = _load_all_artifacts(manifest)
    manifest.update(
        {
            "version": LAB_VERSION,
            "updated_at": _now_iso(),
            "repo_id": repo_id,
            "speed_values": speed_values,
            "artifacts": [dataclasses.asdict(item) for item in _merge_artifacts(existing_artifacts, artifacts, "speed")],
            "speed_count": len(artifacts),
        }
    )
    return manifest, new_mapping, artifacts


def _render_longform_artifacts(
    out_dir: Path,
    repo_id: str,
    device: str | None,
    mapping: dict[str, Any],
    finalists: list[str],
    limit: int | None,
    minutes: float,
) -> tuple[dict[str, Any], dict[str, Any], list[LabArtifact]]:
    if not finalists:
        raise RuntimeError("No finalists supplied for longform testing")
    if limit is not None:
        finalists = finalists[:limit]
    longform_dir = out_dir / "longform"
    longform_dir.mkdir(parents=True, exist_ok=True)
    longform_script_path = longform_dir / "longform_script.txt"
    longform_text = "\n\n".join(topic["text"] for topic in TOPIC_PACKS)
    longform_script_path.write_text(longform_text, encoding="utf-8")
    sentence_units = load_sentence_units(longform_script_path)
    artifacts: list[LabArtifact] = []
    new_mapping = dict(mapping)
    pipelines = PipelineCache(repo_id, device=device)
    for source_id in finalists:
        voice_ref, lang_code, family, speed_override = _resolve_voice_material(source_id, mapping, out_dir)
        voice_label = source_id
        pipeline = pipelines.get(lang_code)
        blocks = _build_blocks(pipeline, [unit.text for unit in sentence_units], 420, 500)
        block_dir = longform_dir / source_id
        block_audio_dir = block_dir / "blocks"
        block_audio_dir.mkdir(parents=True, exist_ok=True)
        block_records = []
        concat_parts = []
        audio_cursor = 0.0
        for block_index, block_sentences in enumerate(blocks, start=1):
            block_text = " ".join(block_sentences)
            wav_path = block_audio_dir / f"block_{block_index:03d}.wav"
            mp3_path = block_audio_dir / f"block_{block_index:03d}.mp3"
            audio = _render_text(pipeline, block_text, voice_ref, speed_override)
            _write_wav(audio, DEFAULT_SAMPLE_RATE, wav_path)
            _wav_to_mp3(wav_path, mp3_path)
            duration = _wav_duration_seconds(wav_path)
            block_records.append(
                {
                    "block_index": block_index,
                    "text": block_text,
                    "wav_path": str(wav_path.relative_to(out_dir)).replace("\\", "/"),
                    "mp3_path": str(mp3_path.relative_to(out_dir)).replace("\\", "/"),
                    "start": round(audio_cursor, 3),
                    "end": round(audio_cursor + duration, 3),
                    "duration_seconds": round(duration, 3),
                    "sentence_count": len(block_sentences),
                    "phoneme_chars": _phoneme_chars(pipeline, block_text),
                }
            )
            concat_parts.append(wav_path)
            audio_cursor += duration + 0.3
        final_wav = block_dir / "audio_master.wav"
        final_mp3 = block_dir / "audio.mp3"
        if concat_parts:
            combined = []
            for wav_path in concat_parts:
                import wave

                with wave.open(str(wav_path), "rb") as wf:
                    frames = wf.readframes(wf.getnframes())
                    arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                    combined.append(arr)
                    combined.append(np.zeros(int(DEFAULT_SAMPLE_RATE * 0.3), dtype=np.float32))
            if combined:
                merged = np.concatenate(combined[:-1]) if len(combined) > 1 else combined[0]
                _write_wav(merged, DEFAULT_SAMPLE_RATE, final_wav)
                _wav_to_mp3(final_wav, final_mp3)
        suspicious = _select_suspicious_boundaries(block_records, max_count=10)
        _save_json(block_dir / "blocks.json", {"blocks": block_records, "voice_ref": voice_label, "lang_code": lang_code})
        _save_json(block_dir / "suspicious_boundaries.json", suspicious)
        for item in suspicious:
            clip_path = block_dir / "boundary_clips" / item["clip"]
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    _ffmpeg_path(),
                    "-y",
                    "-ss",
                    f"{item['start']:.3f}",
                    "-i",
                    str(final_wav),
                    "-t",
                    f"{item['dur']:.3f}",
                    "-codec:a",
                    "libmp3lame",
                    "-q:a",
                    "4",
                    str(clip_path),
                ],
                check=True,
                capture_output=True,
            )
        blind_id = _allocate_blind_id(set(new_mapping.keys()), DEFAULT_SEED + 3000 + len(artifacts) + 1)
        new_mapping[blind_id] = {
            "kind": "longform",
            "source_blind_id": source_id,
            "lang_code": lang_code,
            "family": family,
            "speed": speed_override,
            "minutes": minutes,
            "block_count": len(block_records),
        }
        artifacts.append(
            LabArtifact(
                blind_id=blind_id,
                kind="longform",
                label=f"longform_{source_id}",
                lang_code=lang_code,
                family=family,
                source_ref=source_id,
                audio_wav=str(final_wav.relative_to(out_dir)).replace("\\", "/"),
                audio_mp3=str(final_mp3.relative_to(out_dir)).replace("\\", "/"),
                sample_id="longform",
                speed=DEFAULT_SPEED,
                duration_seconds=_wav_duration_seconds(final_wav),
                metadata={
                    "source_blind_id": source_id,
                    "block_count": len(block_records),
                    "suspicious_boundary_count": len(suspicious),
                    "longform_script": str(longform_script_path.relative_to(out_dir)).replace("\\", "/"),
                },
            )
        )
    manifest = _load_json(out_dir / "manifest.json", default={})
    existing_artifacts = _load_all_artifacts(manifest)
    manifest.update(
        {
            "version": LAB_VERSION,
            "updated_at": _now_iso(),
            "repo_id": repo_id,
            "longform_minutes": minutes,
            "artifacts": [dataclasses.asdict(item) for item in _merge_artifacts(existing_artifacts, artifacts, "longform")],
            "longform_count": len(artifacts),
        }
    )
    return manifest, new_mapping, artifacts


def _select_suspicious_boundaries(block_records: list[dict[str, Any]], max_count: int = 10) -> list[dict[str, Any]]:
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


def _report(
    out_dir: Path,
    repo_id: str,
    manifest: dict[str, Any],
    mapping: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    artifacts = [LabArtifact(**item) for item in manifest.get("artifacts", [])]
    scores_path = out_dir / "scores.csv"
    current_scores = _load_scores(scores_path)
    rows = []
    for artifact in artifacts:
        row = _blank_scores_row(artifact.blind_id, artifact.kind, artifact.family)
        if artifact.blind_id in current_scores:
            row.update(current_scores[artifact.blind_id])
        row["weighted_total"] = _weighted_total(row)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row["weighted_total"]) if row["weighted_total"] else -1.0,
            row["kind"],
            row["blind_id"],
        ),
        reverse=True,
    )
    _write_scores(scores_path, rows)
    leaderboard_path = out_dir / "leaderboard.md"
    lines = [
        "# Kokoro Voice Lab Leaderboard",
        "",
        f"- Repo: `{repo_id}`",
        f"- Updated: `{_now_iso()}`",
        "",
        "| Blind | Kind | Family | Weighted Score | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['blind_id']} | {row['kind']} | {row['family']} | {row['weighted_total'] or 'pending'} | {row['notes'] or ''} |"
        )
    leaderboard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    html_path = out_dir / "review.html"
    html_path.write_text(_build_review_html(manifest, rows), encoding="utf-8")
    summary = {
        "repo_id": repo_id,
        "artifact_count": len(artifacts),
        "scored_count": sum(1 for row in rows if row["weighted_total"]),
        "leaderboard_path": str(leaderboard_path.relative_to(out_dir)).replace("\\", "/"),
        "html_path": str(html_path.relative_to(out_dir)).replace("\\", "/"),
    }
    _save_json(out_dir / "report.json", summary)
    return summary, rows


def _resolve_manifest(out_dir: Path) -> dict[str, Any]:
    return _load_json(out_dir / "manifest.json", default={"version": LAB_VERSION, "artifacts": []})


def _resolve_mapping(out_dir: Path) -> dict[str, Any]:
    mapping = _load_json(out_dir / "blind_mapping.json", default={})
    manifest = _load_json(out_dir / "manifest.json", default={"artifacts": []})
    for artifact in manifest.get("artifacts", []):
        blind_id = artifact.get("blind_id")
        if blind_id and blind_id not in mapping:
            mapping[blind_id] = _mapping_entry_from_artifact(artifact)
    return mapping


def _write_outputs(out_dir: Path, manifest: dict[str, Any], mapping: dict[str, Any], scores: list[dict[str, str]] | None = None) -> None:
    _save_json(out_dir / "manifest.json", manifest)
    _save_json(out_dir / "blind_mapping.json", mapping)
    if scores is not None:
        existing_scores = _load_scores(out_dir / "scores.csv")
        merged_scores = _merge_score_rows(existing_scores, scores)
        _write_scores(out_dir / "scores.csv", merged_scores)


def cmd_base(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = _resolve_manifest(out_dir)
    existing_mapping = _resolve_mapping(out_dir)
    existing_artifacts = _load_all_artifacts(existing_manifest)
    voice_filter = set(args.voices) if args.voices else None
    family_filter = set(args.families) if args.families else None
    manifest, mapping, artifacts = _base_artifacts(out_dir, args.repo_id, args.limit, args.seed, args.device, voice_filter, family_filter)
    scores = _voice_records_to_scores(artifacts)
    combined_artifacts = _merge_artifacts(existing_artifacts, artifacts, "base")
    manifest["artifacts"] = [dataclasses.asdict(item) for item in combined_artifacts]
    manifest["artifact_count"] = len(combined_artifacts)
    _save_manifest_bundle(out_dir, manifest, {**existing_mapping, **mapping})
    merged_scores = _merge_score_rows(_load_scores(out_dir / "scores.csv"), scores)
    _write_scores(out_dir / "scores.csv", merged_scores)
    print(f"Generated {len(artifacts)} base voices in {out_dir}")
    return 0


def cmd_topics(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _topic_manifest(out_dir, args.repo_id)
    _save_json(out_dir / "manifest.json", manifest)
    print(f"Wrote {len(manifest.get('topic_packs', []))} topic packs to {out_dir / 'topics'}")
    return 0


def _default_blend_config(out_dir: Path, repo_id: str, device: str | None) -> list[dict[str, Any]]:
    manifest = _resolve_manifest(out_dir)
    artifacts = [artifact for artifact in _load_all_artifacts(manifest) if artifact.kind == "base"]
    if not artifacts:
        raise RuntimeError("Run base first or provide a blend config")
    return _default_blend_specs(artifacts)


def cmd_blends(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _resolve_manifest(out_dir)
    mapping = _resolve_mapping(out_dir)
    base_artifacts = _load_base_artifacts(manifest)
    if not base_artifacts:
        raise RuntimeError("Run base first")
    if args.config:
        specs = json.loads(Path(args.config).read_text(encoding="utf-8"))
    else:
        specs = _default_blend_config(out_dir, args.repo_id, args.device)
    if args.limit is not None:
        specs = specs[: args.limit]
    manifest, mapping, artifacts = _render_blend_artifact(out_dir, args.repo_id, args.seed, args.device, manifest, mapping, specs)
    scores = _voice_records_to_scores(artifacts)
    _save_manifest_bundle(out_dir, manifest, mapping)
    if scores:
        _write_scores(out_dir / "scores.csv", _merge_score_rows(_load_scores(out_dir / "scores.csv"), scores))
    print(f"Generated {len(artifacts)} blends in {out_dir}")
    return 0


def cmd_speed(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _resolve_manifest(out_dir)
    mapping = _resolve_mapping(out_dir)
    base_ids = [a.blind_id for a in _load_all_artifacts(manifest) if a.kind == "base"]
    finalists = _load_selected_ids(args.finalists, out_dir / "leaderboard.md", base_ids, args.limit, set(base_ids))
    manifest, mapping, artifacts = _render_speed_artifacts(out_dir, args.repo_id, args.device, mapping, finalists, args.limit)
    scores = _voice_records_to_scores(artifacts)
    _save_manifest_bundle(out_dir, manifest, mapping)
    _write_scores(out_dir / "scores.csv", _merge_score_rows(_load_scores(out_dir / "scores.csv"), scores))
    print(f"Generated {len(artifacts)} speed variants in {out_dir}")
    return 0


def cmd_longform(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _resolve_manifest(out_dir)
    mapping = _resolve_mapping(out_dir)
    base_ids = [a.blind_id for a in _load_all_artifacts(manifest) if a.kind == "base"]
    allowed_ids = {artifact.blind_id for artifact in _load_all_artifacts(manifest) if artifact.kind in {"base", "blend"}}
    finalists = _load_selected_ids(args.finalists, out_dir / "leaderboard.md", base_ids, args.limit, allowed_ids)
    manifest, mapping, artifacts = _render_longform_artifacts(
        out_dir,
        args.repo_id,
        args.device,
        mapping,
        finalists,
        args.limit,
        args.minutes,
    )
    scores = _voice_records_to_scores(artifacts)
    _save_manifest_bundle(out_dir, manifest, mapping)
    _write_scores(out_dir / "scores.csv", _merge_score_rows(_load_scores(out_dir / "scores.csv"), scores))
    print(f"Generated {len(artifacts)} longform finalists in {out_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _resolve_manifest(out_dir)
    mapping = _resolve_mapping(out_dir)
    summary, rows = _report(out_dir, args.repo_id, manifest, mapping)
    print(json.dumps(summary, indent=2))
    print(f"Leaderboard written to {out_dir / 'leaderboard.md'}")
    print(f"Review HTML written to {out_dir / 'review.html'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent Kokoro Voice Lab")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sub = parser.add_subparsers(dest="command", required=True)

    p_base = sub.add_parser("base", help="Generate blind calibration samples for all English voices")
    p_base.add_argument("--limit", type=int, default=None)
    p_base.add_argument("--voices", nargs="+", default=None)
    p_base.add_argument("--families", nargs="+", default=None)
    p_base.set_defaults(func=cmd_base)

    p_topics = sub.add_parser("topics", help="Write topic pack scripts")
    p_topics.set_defaults(func=cmd_topics)

    p_blends = sub.add_parser("blends", help="Generate weighted blend samples")
    p_blends.add_argument("--config", type=str, default=None)
    p_blends.add_argument("--limit", type=int, default=12)
    p_blends.set_defaults(func=cmd_blends)

    p_speed = sub.add_parser("speed", help="Generate speed variants for finalists")
    p_speed.add_argument("--finalists", nargs="+", default=None)
    p_speed.add_argument("--limit", type=int, default=None)
    p_speed.set_defaults(func=cmd_speed)

    p_long = sub.add_parser("longform", help="Stress-test finalists with block-mode longform audio")
    p_long.add_argument("--finalists", nargs="+", default=None)
    p_long.add_argument("--limit", type=int, default=3)
    p_long.add_argument("--minutes", type=float, default=4.5)
    p_long.set_defaults(func=cmd_longform)

    p_report = sub.add_parser("report", help="Build scores CSV, leaderboard, and review HTML")
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
