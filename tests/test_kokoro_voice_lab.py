from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from scripts import kokoro_voice_lab as lab


class FakePipeline:
    def __init__(self, lang_code: str, repo_id: str, device: str | None = None):
        self.lang_code = lang_code
        self.repo_id = repo_id
        self.device = device

    def load_voice(self, voice):
        if isinstance(voice, torch.Tensor):
            return voice
        base = sum(ord(ch) for ch in str(voice)) % 10 + 1
        return torch.tensor([float(base), float(base + 1)], dtype=torch.float32)

    def g2p(self, text: str):
        return None, list(text)


def _decision_row(blind_id: str, round_name: str, decision: str) -> dict[str, str]:
    row = {field: "" for field in lab.DECISIONS_FIELDNAMES}
    row["blind_id"] = blind_id
    row["round"] = round_name
    row["decision"] = decision
    return row


def _write_decisions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=lab.DECISIONS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def patched_lab(monkeypatch, tmp_path):
    durations = {
        lab.BASE_SAMPLE: 21.0,
        lab.TOPIC_REEL_TEXT: 50.0,
        lab.BLEND_SAMPLE: 27.0,
    }

    def fake_synthesize(_pipelines, text, _voice_ref, _lang_code, _speed, wav_path, mp3_path):
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        mp3_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"wav")
        mp3_path.write_bytes(b"mp3")
        return durations[text]

    def fake_render_final_artifact(out_dir, artifact_root, _pipelines, _voice_ref, _lang_code, _speed, _script_text):
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "audio_master.wav").write_bytes(b"wav")
        (artifact_root / "audio.mp3").write_bytes(b"mp3")
        (artifact_root / "blocks.json").write_text("{}", encoding="utf-8")
        suspicious = [
            {"clip": f"clip_{index}.mp3", "start": 0.0, "dur": 4.8, "score": float(index)}
            for index in range(5)
        ]
        for item in suspicious:
            clip_path = artifact_root / "boundary_clips" / item["clip"]
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            clip_path.write_bytes(b"clip")
        blocks = [
            {
                "block_index": 1,
                "text": "one",
                "wav_path": "final/block_001.wav",
                "mp3_path": "final/block_001.mp3",
                "start": 0.0,
                "end": 10.0,
                "duration_seconds": 10.0,
                "sentence_count": 1,
                "phoneme_chars": 100,
            }
        ]
        script_path = artifact_root / "final_script.txt"
        script_path.write_text("final", encoding="utf-8")
        return 100.0, blocks, suspicious, str(script_path.relative_to(out_dir)).replace("\\", "/")

    monkeypatch.setattr(lab, "discover_english_voices", lambda _repo_id: [f"a_voice_{i}" for i in range(1, 8)])
    monkeypatch.setattr(lab, "PipelineCache", lambda repo_id, device=None: type("FakeCache", (), {"get": lambda self, _lang: FakePipeline("a", repo_id, device)})())
    monkeypatch.setattr(lab, "_synthesize_sample", fake_synthesize)
    monkeypatch.setattr(lab, "_render_final_artifact", fake_render_final_artifact)
    return tmp_path


def _run_base(out_dir: Path):
    manifest = lab._resolve_manifest(out_dir)
    mapping = lab._resolve_mapping(out_dir)
    return lab._base_round(out_dir, lab.DEFAULT_REPO_ID, lab.DEFAULT_SEED, None, manifest, mapping)


def test_base_round_assigns_round_order_and_warnings(monkeypatch, tmp_path):
    monkeypatch.setattr(lab, "discover_english_voices", lambda _repo_id: ["a_voice_1"])
    monkeypatch.setattr(lab, "PipelineCache", lambda repo_id, device=None: type("FakeCache", (), {"get": lambda self, _lang: FakePipeline("a", repo_id, device)})())

    def synth_warn(_pipelines, _text, _voice_ref, _lang_code, _speed, wav_path, mp3_path):
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        mp3_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"wav")
        mp3_path.write_bytes(b"mp3")
        return 19.0

    monkeypatch.setattr(lab, "_synthesize_sample", synth_warn)
    with pytest.warns(UserWarning):
        manifest, mapping, artifacts = lab._base_round(tmp_path, lab.DEFAULT_REPO_ID, lab.DEFAULT_SEED, None, lab._resolve_manifest(tmp_path), {},)
    assert manifest["active_round"] == "base"
    assert artifacts[0].round == "base"
    assert artifacts[0].round_order == 1
    assert artifacts[0].lineage["base_ids"] == [artifacts[0].blind_id]
    assert mapping[artifacts[0].blind_id]["kind"] == "base"

    def synth_fail(_pipelines, _text, _voice_ref, _lang_code, _speed, wav_path, mp3_path):
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        mp3_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"wav")
        mp3_path.write_bytes(b"mp3")
        return 31.0

    monkeypatch.setattr(lab, "_synthesize_sample", synth_fail)
    with pytest.raises(RuntimeError, match="outside acceptable range"):
        lab._base_round(tmp_path / "fail", lab.DEFAULT_REPO_ID, lab.DEFAULT_SEED, None, lab._resolve_manifest(tmp_path / "fail"), {})


def test_round_flow_and_report(patched_lab):
    out_dir = patched_lab
    decisions_path = out_dir / "decisions.csv"

    manifest, mapping, base_artifacts = _run_base(out_dir)
    lab._merge_decisions(decisions_path, base_artifacts, mapping)
    assert len(base_artifacts) == 7
    assert all(artifact.round == "base" for artifact in base_artifacts)
    assert [artifact.round_order for artifact in base_artifacts] == list(range(1, 8))

    _write_decisions(
        decisions_path,
        [_decision_row(artifact.blind_id, "base", "Keep") for artifact in base_artifacts],
    )
    manifest, mapping, topic_artifacts = lab._topic_round(out_dir, lab.DEFAULT_REPO_ID, lab.DEFAULT_SEED, None, manifest, mapping, decisions_path)
    lab._merge_decisions(decisions_path, topic_artifacts, mapping)
    assert manifest["active_round"] == "topic"
    assert len(topic_artifacts) == 6
    assert all(artifact.kind == "topic" and artifact.round == "topic" for artifact in topic_artifacts)
    assert all(artifact.sample_id == "topic_reel" for artifact in topic_artifacts)

    _write_decisions(
        decisions_path,
        [_decision_row(artifact.blind_id, "topic", "Keep") for artifact in topic_artifacts[:3]],
    )
    manifest, mapping, blend_artifacts = lab._blend_round(out_dir, lab.DEFAULT_REPO_ID, lab.DEFAULT_SEED, None, manifest, mapping, decisions_path)
    lab._merge_decisions(decisions_path, blend_artifacts, mapping)
    assert manifest["active_round"] == "blend"
    assert len(blend_artifacts) == 9
    assert len([artifact for artifact in blend_artifacts if artifact.kind == "base"]) == 3
    blends = [artifact for artifact in blend_artifacts if artifact.kind == "blend"]
    assert len(blends) == 6
    allowed_ratios = {tuple(round(item["weight"], 2) for item in artifact.metadata["components"]) for artifact in blends}
    assert allowed_ratios <= {(0.7, 0.3), (0.8, 0.2), (0.5, 0.5)}

    selected_speed_sources = [
        _decision_row(next(artifact.blind_id for artifact in blend_artifacts if artifact.kind == "base"), "blend", "Keep"),
        _decision_row(next(artifact.blind_id for artifact in blend_artifacts if artifact.kind == "blend"), "blend", "Maybe"),
    ]
    _write_decisions(decisions_path, selected_speed_sources)
    manifest, mapping, speed_artifacts = lab._speed_round(out_dir, lab.DEFAULT_REPO_ID, lab.DEFAULT_SEED, None, manifest, mapping, decisions_path)
    lab._merge_decisions(decisions_path, speed_artifacts, mapping)
    assert manifest["active_round"] == "speed"
    assert len(speed_artifacts) == 4
    assert {artifact.speed for artifact in speed_artifacts} == {0.95, 0.98}
    blend_derived = [artifact for artifact in speed_artifacts if artifact.lineage["blend_id"]]
    assert blend_derived, "Expected at least one speed artifact sourced from a blend artifact"

    _write_decisions(
        decisions_path,
        [_decision_row(artifact.blind_id, "speed", "Keep") for artifact in speed_artifacts[:2]],
    )
    manifest, mapping, final_artifacts = lab._final_round(out_dir, lab.DEFAULT_REPO_ID, lab.DEFAULT_SEED, None, manifest, mapping, decisions_path)
    lab._merge_decisions(decisions_path, final_artifacts, mapping)
    assert manifest["active_round"] == "final"
    assert len(final_artifacts) == 2
    assert all(artifact.lineage["final_id"] == artifact.blind_id for artifact in final_artifacts)
    assert all(artifact.metadata["boundary_clip_count"] <= 5 for artifact in final_artifacts)

    summary, rows = lab._report(out_dir, lab.DEFAULT_REPO_ID, manifest, mapping, decisions_path)
    assert summary["active_round"] == "final"
    assert len(rows) == 2
    html = (out_dir / "review.html").read_text(encoding="utf-8")
    assert "localStorage.getItem(LOCAL_KEY)" in html
    assert "Hidden until reveal" in html
    assert "window.VOICE_LAB_DECISIONS_NAME" in html
    assert "Reveal</button>" in html
    assert "${item.kind}" in html
    assert "revealed_voice" in html


def test_report_only_ranks_active_round_and_honors_decisions_override(patched_lab):
    out_dir = patched_lab
    decisions_path = out_dir / "custom-decisions.csv"
    manifest, mapping, base_artifacts = _run_base(out_dir)
    _write_decisions(
        decisions_path,
        [
            _decision_row(base_artifacts[0].blind_id, "base", "Maybe"),
            _decision_row(base_artifacts[1].blind_id, "base", "Keep"),
        ],
    )
    summary, rows = lab._report(out_dir, lab.DEFAULT_REPO_ID, lab._resolve_manifest(out_dir), lab._resolve_mapping(out_dir), decisions_path)
    assert summary["decisions_path"].endswith("custom-decisions.csv")
    assert all(row["round"] == "base" for row in rows)
    assert rows[0]["blind_id"] == base_artifacts[1].blind_id
