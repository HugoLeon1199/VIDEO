from __future__ import annotations

import warnings
from pathlib import Path

from scripts import vieneu_voice_lab as lab


def _artifact(
    blind_id: str,
    round_name: str,
    round_order: int,
    source_voice: str = "Voice A",
    preset: str = "production_default",
    duration_seconds: float = 24.0,
    source_ref: str = "",
):
    return lab.LabArtifact(
        blind_id=blind_id,
        kind=round_name,
        round=round_name,
        round_order=round_order,
        source_voice=source_voice,
        preset=preset,
        effective_infer_params={"voice": source_voice, "preset": preset},
        duration_seconds=duration_seconds,
        audio_wav=f"audio/{round_name}/{blind_id}.wav",
        audio_mp3=f"audio/{round_name}/{blind_id}.mp3",
        source_ref=source_ref,
        sample_id=f"{round_name}_sample",
        metadata={},
    )


def _seed_manifest(tmp_path: Path, round_name: str, artifacts: list[lab.LabArtifact]) -> dict:
    manifest = lab._load_manifest(tmp_path)
    return lab._update_manifest(tmp_path, manifest, round_name, artifacts, {"seeded": True})


def test_base_round_creates_one_sample_per_discovered_voice(tmp_path, monkeypatch):
    voices = [
        {"display_name": "Ngọc Lan — nữ", "voice_name": "Ngọc Lan"},
        {"display_name": "Bình An — nam", "voice_name": "Bình An"},
        {"display_name": "Trọng Hữu — nam", "voice_name": "Trọng Hữu"},
    ]
    monkeypatch.setattr(lab, "discover_vieneu_voices", lambda: voices)
    monkeypatch.setattr(
        lab,
        "_render_direct_sample",
        lambda *args, **kwargs: ("audio/base/x.wav", "audio/base/x.mp3", 22.4, {"voice": args[4]}),
    )

    manifest = lab._load_manifest(tmp_path)
    manifest, artifacts = lab._base_round(tmp_path, 123, manifest)

    assert manifest["active_round"] == "base"
    assert len(artifacts) == len(voices)
    assert [artifact.source_voice for artifact in artifacts] == [voice["voice_name"] for voice in voices]
    assert all(artifact.round == "base" for artifact in artifacts)
    assert [artifact.round_order for artifact in artifacts] == [1, 2, 3]


def test_duration_validation_warns_inside_acceptable_and_fails_outside():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lab._validate_duration("base:V001", 19.0, 20.0, 25.0, 18.0, 30.0)
    assert len(caught) == 1
    assert "outside target range" in str(caught[0].message)

    try:
        lab._validate_duration("base:V002", 31.0, 20.0, 25.0, 18.0, 30.0)
    except RuntimeError as exc:
        assert "outside acceptable range" in str(exc)
    else:
        raise AssertionError("Expected duration validation to fail outside acceptable range")


def test_topic_round_uses_only_base_finalists_and_creates_one_reel_each(tmp_path, monkeypatch):
    base_artifacts = [_artifact(f"B{i}", "base", i, source_voice=f"Voice {i}") for i in range(1, 7)]
    manifest = _seed_manifest(tmp_path, "base", base_artifacts)
    decisions_path = tmp_path / "decisions.csv"
    lab._write_decisions(
        decisions_path,
        [
            {"blind_id": "B1", "round": "base", "decision": "Keep"},
            {"blind_id": "B2", "round": "base", "decision": "Keep"},
            {"blind_id": "B3", "round": "base", "decision": "Keep"},
            {"blind_id": "B4", "round": "base", "decision": "Maybe"},
            {"blind_id": "B5", "round": "base", "decision": "Maybe"},
            {"blind_id": "B6", "round": "base", "decision": "Reject"},
        ],
    )
    monkeypatch.setattr(
        lab,
        "_render_direct_sample",
        lambda *args, **kwargs: ("audio/topic/x.wav", "audio/topic/x.mp3", 50.0, {"voice": args[4]}),
    )

    manifest, artifacts = lab._topic_round(tmp_path, 99, manifest, decisions_path)

    assert manifest["active_round"] == "topic"
    assert len(artifacts) == 5
    assert all(artifact.sample_id == "topic_reel" for artifact in artifacts)
    assert [artifact.source_ref for artifact in artifacts] == ["B1", "B2", "B3", "B4", "B5"]


def test_style_round_limits_to_top_three_and_only_two_presets(tmp_path, monkeypatch):
    topic_artifacts = [_artifact(f"T{i}", "topic", i, source_voice=f"Voice {i}", source_ref=f"B{i}") for i in range(1, 5)]
    manifest = _seed_manifest(tmp_path, "topic", topic_artifacts)
    decisions_path = tmp_path / "decisions.csv"
    lab._write_decisions(
        decisions_path,
        [
            {"blind_id": "T1", "round": "topic", "decision": "Keep"},
            {"blind_id": "T2", "round": "topic", "decision": "Keep"},
            {"blind_id": "T3", "round": "topic", "decision": "Maybe"},
            {"blind_id": "T4", "round": "topic", "decision": "Reject"},
        ],
    )
    monkeypatch.setattr(
        lab,
        "_render_direct_sample",
        lambda *args, **kwargs: ("audio/style/x.wav", "audio/style/x.mp3", 27.0, {"voice": args[4], "preset": args[5]}),
    )

    manifest, artifacts = lab._style_round(tmp_path, 77, manifest, decisions_path)

    assert manifest["active_round"] == "style"
    assert len(artifacts) == 6
    assert {artifact.preset for artifact in artifacts} == {"production_default", "natural_calm"}
    assert {artifact.source_ref for artifact in artifacts} == {"T1", "T2", "T3"}


def test_final_round_limits_to_top_two_and_preserves_boundary_limit(tmp_path, monkeypatch):
    style_artifacts = [
        _artifact("S1", "style", 1, source_voice="Voice 1", preset="production_default", source_ref="T1"),
        _artifact("S2", "style", 2, source_voice="Voice 2", preset="natural_calm", source_ref="T2"),
        _artifact("S3", "style", 3, source_voice="Voice 3", preset="production_default", source_ref="T3"),
    ]
    manifest = _seed_manifest(tmp_path, "style", style_artifacts)
    decisions_path = tmp_path / "decisions.csv"
    lab._write_decisions(
        decisions_path,
        [
            {"blind_id": "S1", "round": "style", "decision": "Keep"},
            {"blind_id": "S2", "round": "style", "decision": "Maybe"},
            {"blind_id": "S3", "round": "style", "decision": "Reject"},
        ],
    )
    monkeypatch.setattr(
        lab,
        "_render_final_artifact",
        lambda *args, **kwargs: (
            "final/x/audio_master.wav",
            "final/x/audio.mp3",
            101.5,
            {"voice": args[2], "preset": args[3]},
            [{"clip": f"c{i}.mp3"} for i in range(5)],
            {"suspicious_boundaries": [{"clip": f"c{i}.mp3"} for i in range(5)]},
        ),
    )

    manifest, artifacts = lab._final_round(tmp_path, 55, manifest, decisions_path)

    assert manifest["active_round"] == "final"
    assert len(artifacts) == 2
    assert [artifact.source_ref for artifact in artifacts] == ["S1", "S2"]
    assert all(artifact.metadata["boundary_clip_count"] <= 5 for artifact in artifacts)


def test_rank_suspicious_boundaries_caps_at_five(tmp_path, monkeypatch):
    video_dir = tmp_path / "final" / "V001"
    video_dir.mkdir(parents=True)
    manifest = {
        "block_config": {"trim_trailing_threshold": 0.01, "gap_after_ms": 300},
        "blocks": [
            {"block_index": index, "wav_path": f"tts_blocks/block_{index:03d}.wav", "audio_end": float(index), "audio_start": float(index) - 0.5}
            for index in range(1, 8)
        ],
    }
    monkeypatch.setattr(
        lab,
        "_analyze_block_metrics",
        lambda wav_path, threshold: {
            "rms": float(wav_path.stem[-3:]) / 1000.0,
            "peak": 0.5,
            "trailing_silence_seconds": 0.1,
            "duration_seconds": 2.0,
        },
    )

    ranked = lab._rank_suspicious_boundaries(video_dir, manifest, max_count=5)

    assert len(ranked) == 5
    assert all("clip" in item for item in ranked)


def test_decisions_csv_round_trip(tmp_path):
    path = tmp_path / "decisions.csv"
    lab._write_decisions(
        path,
        [
            {
                "blind_id": "V001",
                "round": "base",
                "decision": "Keep",
                "revealed": "yes",
                "notes": "best",
                "revealed_voice": "Ngọc Lan",
                "revealed_preset": "production_default",
                "revealed_source": "base",
                "revealed_params": '{"voice":"Ngọc Lan"}',
            }
        ],
    )

    loaded = lab._load_decisions(path)

    assert loaded["V001"]["decision"] == "Keep"
    assert loaded["V001"]["revealed_voice"] == "Ngọc Lan"


def test_report_uses_active_round_only_and_ranks_by_decision_then_round_order(tmp_path):
    manifest = lab._load_manifest(tmp_path)
    manifest = lab._update_manifest(tmp_path, manifest, "base", [_artifact("B1", "base", 1)], {"seeded": True})
    style_artifacts = [
        _artifact("S1", "style", 1, source_voice="Voice 1", preset="production_default"),
        _artifact("S2", "style", 2, source_voice="Voice 2", preset="natural_calm"),
        _artifact("S3", "style", 3, source_voice="Voice 3", preset="production_default"),
    ]
    manifest = lab._update_manifest(tmp_path, manifest, "style", style_artifacts, {"seeded": True})
    decisions_path = tmp_path / "decisions.csv"
    lab._write_decisions(
        decisions_path,
        [
            {"blind_id": "B1", "round": "base", "decision": "Keep"},
            {"blind_id": "S1", "round": "style", "decision": "Maybe"},
            {"blind_id": "S2", "round": "style", "decision": "Keep"},
            {"blind_id": "S3", "round": "style", "decision": "Reject"},
        ],
    )

    summary, rows = lab._report(tmp_path, manifest, decisions_path)

    assert summary["active_round"] == "style"
    assert [row["blind_id"] for row in rows] == ["S2", "S1", "S3"]
    assert all(row["round"] == "style" for row in rows)


def test_review_html_hides_voice_until_reveal_and_restores_local_state():
    html = lab._build_review_html(
        {"active_round": "base", "round_counts": {"base": 1}, "updated_at": "now"},
        [
            {
                "blind_id": "V001",
                "round": "base",
                "kind": "base",
                "round_order": 1,
                "audio_mp3": "audio/base/V001.mp3",
                "audio_wav": "audio/base/V001.wav",
                "source_voice": "Ngọc Lan",
                "preset": "production_default",
                "effective_infer_params": {"voice": "Ngọc Lan"},
                "duration_seconds": 22.0,
                "decision": "",
                "revealed": "",
                "notes": "",
            }
        ],
    )

    assert "Ngọc Lan" not in html
    assert "localStorage.getItem(LOCAL_KEY)" in html
    assert "class=\"revealBtn\" disabled" in html
    assert "Hidden until reveal" in html
    assert "decisions.csv" in html
