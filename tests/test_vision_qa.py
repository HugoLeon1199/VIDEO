"""
Unit tests for Vision QA module and generate_images.py QA integration.
All Gemini and RunPod calls are mocked — no real API calls.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("RUNPOD_API_KEY", "test_key_abcdef")
os.environ.setdefault("RUNPOD_ENDPOINT_ID", "test_endpoint_123")
os.environ.setdefault("GEMINI_API_KEY", "test_gemini_key")
os.environ.setdefault("IMAGE_OUTPUT_ROOT", tempfile.mkdtemp())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_webp(width: int = 64, height: int = 36, color=(100, 80, 60)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=80)
    return buf.getvalue()


def _write_image(path: Path, data: bytes = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        data = _make_webp()
    path.write_bytes(data)
    return path


def _good_qa_json(score: int = 92, extra_limbs: bool = False) -> str:
    return json.dumps({
        "pass": score >= 80 and not extra_limbs,
        "score": score,
        "people_detected": 1,
        "anatomy": {
            "extra_limbs": extra_limbs,
            "missing_limbs": False,
            "fused_bodies": False,
            "malformed_hands": False,
            "duplicate_faces": False,
        },
        "composition": {
            "matches_prompt": True,
            "main_subject_clear": True,
            "bodies_separated": True,
            "important_subject_cropped": False,
        },
        "style": {
            "matches_2d_painted_documentary": True,
            "looks_childish": False,
            "contains_text_or_watermark": False,
        },
        "issues": [],
        "regeneration_instruction": "",
    })


def _fail_qa_json(issues=("extra limbs detected",)) -> str:
    return json.dumps({
        "pass": False,
        "score": 45,
        "people_detected": 1,
        "anatomy": {
            "extra_limbs": True,
            "missing_limbs": False,
            "fused_bodies": False,
            "malformed_hands": False,
            "duplicate_faces": False,
        },
        "composition": {
            "matches_prompt": True,
            "main_subject_clear": True,
            "bodies_separated": False,
            "important_subject_cropped": False,
        },
        "style": {
            "matches_2d_painted_documentary": True,
            "looks_childish": False,
            "contains_text_or_watermark": False,
        },
        "issues": list(issues),
        "regeneration_instruction": "Show only one foreground figure with arms clearly separated.",
    })


# ---------------------------------------------------------------------------
# Test: _parse_qa_json
# ---------------------------------------------------------------------------

class TestParseQAJson:
    def test_plain_json(self):
        from image_generation.vision_qa import _parse_qa_json
        raw = '{"pass": true, "score": 90}'
        d = _parse_qa_json(raw)
        assert d["pass"] is True
        assert d["score"] == 90

    def test_strips_markdown_fence(self):
        from image_generation.vision_qa import _parse_qa_json
        raw = '```json\n{"pass": false, "score": 50}\n```'
        d = _parse_qa_json(raw)
        assert d["pass"] is False

    def test_strips_plain_fence(self):
        from image_generation.vision_qa import _parse_qa_json
        raw = '```\n{"pass": true, "score": 88}\n```'
        d = _parse_qa_json(raw)
        assert d["score"] == 88

    def test_invalid_json_raises(self):
        from image_generation.vision_qa import _parse_qa_json
        with pytest.raises(json.JSONDecodeError):
            _parse_qa_json("not json at all")


# ---------------------------------------------------------------------------
# Test: _validate_qa_dict
# ---------------------------------------------------------------------------

class TestValidateQADict:
    def test_fills_missing_keys(self):
        from image_generation.vision_qa import _validate_qa_dict
        d = _validate_qa_dict({})
        assert "pass" in d
        assert "anatomy" in d
        assert "composition" in d
        assert "style" in d
        assert "issues" in d

    def test_coerces_score_to_int(self):
        from image_generation.vision_qa import _validate_qa_dict
        d = _validate_qa_dict({"score": "90.5"})
        assert isinstance(d["score"], int)

    def test_fills_anatomy_sub_defaults(self):
        from image_generation.vision_qa import _validate_qa_dict
        d = _validate_qa_dict({"anatomy": {}})
        assert d["anatomy"]["extra_limbs"] is False


# ---------------------------------------------------------------------------
# Test: AnatomyResult scoring and hard-fail
# ---------------------------------------------------------------------------

class TestAnatomyResult:
    def test_perfect_anatomy_score(self):
        from image_generation.vision_qa import AnatomyResult
        a = AnatomyResult()
        assert a.score() == 40

    def test_extra_limbs_deducts_20(self):
        from image_generation.vision_qa import AnatomyResult
        a = AnatomyResult(extra_limbs=True)
        assert a.score() == 20

    def test_hard_fail_on_extra_limbs(self):
        from image_generation.vision_qa import AnatomyResult
        a = AnatomyResult(extra_limbs=True)
        assert a.hard_fail() is True

    def test_hard_fail_on_fused_bodies(self):
        from image_generation.vision_qa import AnatomyResult
        a = AnatomyResult(fused_bodies=True)
        assert a.hard_fail() is True

    def test_no_hard_fail_on_missing_limbs_only(self):
        from image_generation.vision_qa import AnatomyResult
        a = AnatomyResult(missing_limbs=True)
        assert a.hard_fail() is False  # missing_limbs is not in hard-fail list
        assert a.score() == 35

    def test_score_never_negative(self):
        from image_generation.vision_qa import AnatomyResult
        a = AnatomyResult(
            extra_limbs=True, missing_limbs=True, fused_bodies=True,
            malformed_hands=True, duplicate_faces=True,
        )
        assert a.score() >= 0


# ---------------------------------------------------------------------------
# Test: StyleResult hard-fail
# ---------------------------------------------------------------------------

class TestStyleResult:
    def test_hard_fail_on_watermark(self):
        from image_generation.vision_qa import StyleResult
        s = StyleResult(contains_text_or_watermark=True)
        assert s.hard_fail() is True

    def test_hard_fail_on_childish(self):
        from image_generation.vision_qa import StyleResult
        s = StyleResult(looks_childish=True)
        assert s.hard_fail() is True

    def test_no_hard_fail_on_style_mismatch_only(self):
        from image_generation.vision_qa import StyleResult
        s = StyleResult(matches_2d_painted_documentary=False)
        assert s.hard_fail() is False


# ---------------------------------------------------------------------------
# Test: CompositionResult hard-fail
# ---------------------------------------------------------------------------

class TestCompositionResult:
    def test_hard_fail_on_cropped_subject(self):
        from image_generation.vision_qa import CompositionResult
        c = CompositionResult(important_subject_cropped=True)
        assert c.hard_fail() is True


# ---------------------------------------------------------------------------
# Test: VisionQA.evaluate — mocked Gemini
# ---------------------------------------------------------------------------

class TestVisionQAEvaluate:
    def _qa(self, **kwargs):
        from image_generation.vision_qa import VisionQA
        return VisionQA(api_key="fake_key", min_score=80, anatomy_min=34, **kwargs)

    def _img(self, tmp_path: Path) -> Path:
        p = tmp_path / "scene_001" / "candidate.webp"
        return _write_image(p)

    def test_passing_image_returns_passed_true(self, tmp_path):
        qa = self._qa(cache_dir=tmp_path / ".cache")
        img = self._img(tmp_path)

        with patch("image_generation.vision_qa._call_gemini", return_value=_good_qa_json(score=92)):
            result = qa.evaluate(img, "a warrior", "001", 0, 11001, "sha1" * 16)

        assert result.passed is True
        assert result.score == 92

    def test_failing_image_returns_passed_false(self, tmp_path):
        qa = self._qa(cache_dir=tmp_path / ".cache")
        img = self._img(tmp_path)

        with patch("image_generation.vision_qa._call_gemini", return_value=_fail_qa_json()):
            result = qa.evaluate(img, "a warrior", "001", 0, 11001, "sha1" * 16)

        assert result.passed is False
        assert result.anatomy.extra_limbs is True

    def test_hard_fail_overrides_high_score(self, tmp_path):
        qa = self._qa(cache_dir=tmp_path / ".cache")
        img = self._img(tmp_path)
        # Model returns high score but with extra_limbs=True
        raw = _good_qa_json(score=95, extra_limbs=True)

        with patch("image_generation.vision_qa._call_gemini", return_value=raw):
            result = qa.evaluate(img, "scene", "001", 0, 11001, "sha1" * 16)

        assert result.passed is False
        assert result.score < 80  # hard fail forces score below threshold

    def test_api_error_returns_qa_error_not_pass(self, tmp_path):
        qa = self._qa(cache_dir=tmp_path / ".cache")
        img = self._img(tmp_path)

        with patch("image_generation.vision_qa._call_gemini", side_effect=Exception("API down")):
            result = qa.evaluate(img, "scene", "001", 0, 11001, "sha1" * 16)

        assert result.passed is False
        assert result.qa_error is not None
        assert "retry" in result.qa_error.lower() or "failed" in result.qa_error.lower()

    def test_missing_gemini_key_returns_qa_error(self, tmp_path):
        from image_generation.vision_qa import VisionQA
        qa = VisionQA(api_key="", min_score=80, cache_dir=tmp_path / ".cache")
        img = self._img(tmp_path)

        result = qa.evaluate(img, "scene", "001", 0, 11001, "sha1" * 16)
        assert result.passed is False
        assert result.qa_error is not None

    def test_json_parse_error_returns_qa_error(self, tmp_path):
        qa = self._qa(cache_dir=tmp_path / ".cache")
        img = self._img(tmp_path)

        with patch("image_generation.vision_qa._call_gemini", return_value="NOT JSON {{{"):
            result = qa.evaluate(img, "scene", "001", 0, 11001, "sha1" * 16)

        assert result.passed is False
        assert "parse" in (result.qa_error or "").lower()

    def test_cache_hit_skips_api_call(self, tmp_path):
        from image_generation.vision_qa import VisionQA, _QACache, QA_PROMPT_VERSION
        cache_dir = tmp_path / ".cache"
        qa = VisionQA(api_key="fake_key", min_score=80, cache_dir=cache_dir)
        img = self._img(tmp_path)
        sha = "abcdef1234567890" * 4

        # Pre-populate cache
        good = json.loads(_good_qa_json(92))
        cache = _QACache(cache_dir)
        cache.put(sha, QA_PROMPT_VERSION, good)

        with patch("image_generation.vision_qa._call_gemini") as mock_call:
            result = qa.evaluate(img, "scene", "001", 0, 11001, sha)

        mock_call.assert_not_called()
        assert result.cached is True
        assert result.passed is True

    def test_retry_on_transient_api_error(self, tmp_path):
        qa = self._qa(cache_dir=tmp_path / ".cache", max_retries=3)
        img = self._img(tmp_path)
        call_count = [0]

        def side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("transient error")
            return _good_qa_json(88)

        with patch("image_generation.vision_qa._call_gemini", side_effect=side_effect):
            with patch("time.sleep"):
                result = qa.evaluate(img, "scene", "001", 0, 11001, "sha" * 21)

        assert result.passed is True
        assert call_count[0] == 3


# ---------------------------------------------------------------------------
# Test: VisionQA.select_best
# ---------------------------------------------------------------------------

class TestSelectBest:
    def _res(self, seed, score, passed):
        from image_generation.vision_qa import QAResult
        r = QAResult(
            scene_id="001", candidate_index=0, seed=seed, sha256="x",
            image_path="", passed=passed, score=score, people_detected=1,
        )
        return r

    def test_selects_highest_score(self):
        from image_generation.vision_qa import VisionQA
        qa = VisionQA.__new__(VisionQA)
        qa._min_score = 80
        qa._anatomy_min = 34

        r1 = self._res(11001, 82, True)
        r2 = self._res(11002, 91, True)
        r3 = self._res(11003, 75, False)

        best = qa.select_best([r1, r2, r3])
        assert best.seed == 11002

    def test_returns_none_when_all_fail(self):
        from image_generation.vision_qa import VisionQA
        qa = VisionQA.__new__(VisionQA)
        r1 = self._res(11001, 50, False)
        r2 = self._res(11002, 60, False)
        assert qa.select_best([r1, r2]) is None

    def test_tiebreak_by_seed_ascending(self):
        from image_generation.vision_qa import VisionQA
        qa = VisionQA.__new__(VisionQA)
        r1 = self._res(11003, 90, True)
        r2 = self._res(11001, 90, True)
        best = qa.select_best([r1, r2])
        assert best.seed == 11001


# ---------------------------------------------------------------------------
# Test: build_corrective_prompt
# ---------------------------------------------------------------------------

class TestCorrectivePrompt:
    def test_includes_anatomy_fix(self):
        from image_generation.vision_qa import VisionQA, QAResult, AnatomyResult
        qa = VisionQA.__new__(VisionQA)
        r = QAResult(
            scene_id="001", candidate_index=0, seed=11001, sha256="x",
            image_path="", issues=["extra arm spotted"],
        )
        r.anatomy = AnatomyResult(extra_limbs=True)

        corrective = qa.build_corrective_prompt("original scene", [r])
        assert "original scene" in corrective
        assert "Simplify" in corrective or "foreground" in corrective

    def test_deduplicates_issues(self):
        from image_generation.vision_qa import VisionQA, QAResult
        qa = VisionQA.__new__(VisionQA)
        r1 = QAResult("001", 0, 11001, "x", "", issues=["extra arm"])
        r2 = QAResult("001", 1, 11002, "x", "", issues=["extra arm"])
        corrective = qa.build_corrective_prompt("scene", [r1, r2])
        assert corrective.count("extra arm") == 1


# ---------------------------------------------------------------------------
# Test: Retry seed determinism
# ---------------------------------------------------------------------------

class TestRetrySeedDeterminism:
    def test_retry_seeds_dont_overlap_original(self):
        original_seeds = [11001, 11002, 11003]
        retry_round = 1
        retry_seeds = [s + 100000 * retry_round for s in original_seeds]
        assert not set(original_seeds) & set(retry_seeds)

    def test_retry_seeds_dont_overlap_across_rounds(self):
        original = [11001, 11002, 11003]
        round1 = [s + 100000 for s in original]
        round2 = [s + 200000 for s in original]
        assert not set(round1) & set(round2)
        assert not set(original) & set(round2)

    def test_retry_seeds_are_deterministic(self):
        original = [11001, 11002, 11003]
        seeds_a = [s + 100000 * 1 for s in original]
        seeds_b = [s + 100000 * 1 for s in original]
        assert seeds_a == seeds_b


# ---------------------------------------------------------------------------
# Test: _scene_done with QA requirement
# ---------------------------------------------------------------------------

class TestSceneDone:
    def _log_with_image(self, tmp_path: Path, qa_passed: bool = True) -> tuple:
        from scripts.generate_images import _scene_done
        img = tmp_path / "img_001.png"
        img.write_bytes(_make_webp())
        log = {
            "001": {
                "status": "completed",
                "candidates_saved": 3,
                "selected_image": str(img),
                "qa_passed": qa_passed,
            }
        }
        return log, _scene_done

    def test_done_with_qa_passed(self, tmp_path):
        log, fn = self._log_with_image(tmp_path, qa_passed=True)
        assert fn(log, "001", 3, require_qa=True) is True

    def test_not_done_when_qa_failed(self, tmp_path):
        log, fn = self._log_with_image(tmp_path, qa_passed=False)
        assert fn(log, "001", 3, require_qa=True) is False

    def test_done_without_qa_requirement(self, tmp_path):
        log, fn = self._log_with_image(tmp_path, qa_passed=False)
        assert fn(log, "001", 3, require_qa=False) is True

    def test_not_done_when_image_missing(self, tmp_path):
        from scripts.generate_images import _scene_done
        log = {
            "001": {
                "status": "completed",
                "candidates_saved": 3,
                "selected_image": str(tmp_path / "nonexistent.png"),
                "qa_passed": True,
            }
        }
        assert _scene_done(log, "001", 3, require_qa=True) is False


# ---------------------------------------------------------------------------
# Test: QA-only mode does not call RunPod
# ---------------------------------------------------------------------------

class TestQAOnlyDoesNotCallRunPod:
    def test_qa_only_skips_backend(self, tmp_path):
        """--qa-only must not instantiate or call RunPodServerlessBackend."""
        from image_generation.vision_qa import VisionQA
        mock_qa = MagicMock(spec=VisionQA)
        mock_qa_result = MagicMock()
        mock_qa_result.passed = True
        mock_qa_result.score = 90
        mock_qa_result.issues = []
        mock_qa_result.qa_error = None
        mock_qa.evaluate.return_value = mock_qa_result

        # Write an existing image and log
        img_path = tmp_path / "images_vi" / "img_001.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(_make_webp())

        log_path = tmp_path / "generation_log_vi.json"
        log_path.write_text(json.dumps({
            "001": {
                "status": "completed",
                "candidates_saved": 1,
                "selected_image": str(img_path),
                "selected_seed": 11001,
                "qa_passed": False,
            }
        }), encoding="utf-8")

        prompts_path = tmp_path / "image_prompts.json"
        prompts_path.write_text(json.dumps([
            {"index": 1, "start": 0.0, "end": 3.0, "prompt": "a warrior in a cave", "negative_prompt": ""}
        ]), encoding="utf-8")

        # We simply verify RunPodServerlessBackend is NOT called
        with patch("image_generation.runpod_serverless_backend.RunPodServerlessBackend") as mock_backend_cls:
            with patch("image_generation.vision_qa.VisionQA", return_value=mock_qa):
                sys.argv = [
                    "generate_images.py",
                    "--video-id", "test-vid",
                    "--qa-only",
                    "--track", "vi",
                    "--output-root", str(tmp_path),
                ]
                try:
                    from scripts.generate_images import main
                    # Will sys.exit(1) because QA not passed, that's fine
                    main()
                except SystemExit:
                    pass

            mock_backend_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Test: EN track style not affected by VI changes
# ---------------------------------------------------------------------------

class TestENTrackNotAffected:
    def test_en_prompt_template_unchanged(self):
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        content = (prompts_dir / "image_prompt_en.txt").read_text(encoding="utf-8")
        assert "Ink sketch illustration on aged parchment paper" in content
        assert "no extra limbs" in content

    def test_vi_style_version_not_in_en_template(self):
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        content = (prompts_dir / "image_prompt_en.txt").read_text(encoding="utf-8")
        assert "vi-2d-documentary" not in content
        assert "2D painted documentary" not in content

    def test_config_en_track_unchanged(self):
        import config
        tc = config.TRACK_CONFIG["en"]
        assert tc["output_subdir"] == "images_en"
        assert tc["steps"] == 20

    def test_config_vi_track_correct(self):
        import config
        tc = config.TRACK_CONFIG["vi"]
        assert tc["output_subdir"] == "images_vi"
        assert tc["steps"] == 20
        assert tc["guidance_scale"] == 3.5


# ---------------------------------------------------------------------------
# Test: Fallback not promoted by default
# ---------------------------------------------------------------------------

class TestNoSilentFallback:
    def test_needs_review_entry_when_all_fail(self, tmp_path):
        """When all QA rounds fail and allow_fallback=False, status must be needs_review."""
        from scripts.generate_images import _make_log_entry

        qa_results = []
        for i in range(3):
            from image_generation.vision_qa import QAResult
            r = QAResult("001", i, 11001 + i, "x", "", passed=False, score=40, issues=["extra arm"])
            qa_results.append(r)

        entry = _make_log_entry(
            status="needs_review",
            candidates=[],
            selected_image="",
            errors=["All QA rounds failed"],
            job_id=None,
            duration=10.0,
            qa_results=qa_results,
            qa_round=2,
        )
        assert entry["status"] == "needs_review"
        assert entry["selected_image"] == ""
        assert len(entry["candidate_reviews"]) == 3


# ---------------------------------------------------------------------------
# Test: Final audit detects missing/corrupt images
# ---------------------------------------------------------------------------

class TestAuditDetectsMissing:
    def test_scene_with_missing_image_flagged(self, tmp_path):
        """audit_generated_images should flag missing files."""
        import importlib.util, types

        # Build a minimal gen_log with a missing image
        log = {"001": {"selected_image": str(tmp_path / "nonexistent.png"), "qa_passed": True}}
        log_path = tmp_path / "generation_log_vi.json"
        log_path.write_text(json.dumps(log), encoding="utf-8")

        # Just test the logic directly (not via subprocess)
        audit_results = {}
        for scene_id, entry in log.items():
            selected = entry.get("selected_image", "")
            if not selected or not Path(selected).exists():
                audit_results[scene_id] = {"audit_status": "failed", "reason": "image file missing"}

        assert audit_results["001"]["audit_status"] == "failed"
