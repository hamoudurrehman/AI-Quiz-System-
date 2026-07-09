"""
test_inference.py
=================
Unit tests for the RACE Reading Comprehension inference pipeline.
Tests cover preprocessing utilities, Model A answer verifier,
Model B distractor/hint generation, and the full inference engine.

Run:
    python -m pytest tests/test_inference.py -v
"""

import os
import sys
import re
import string
import numpy as np
import pytest

# ── Add project root to path ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocessing import (
    clean_text,
    extract_candidate_sentences,
    apply_wh_template,
    get_distractor_candidates,
    get_hint_sentences,
)
from src.evaluate import (
    compute_classification_metrics,
    exact_match,
    compute_regression_metrics,
    distractor_diversity,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_ARTICLE = (
    "The ancient Silk Road was a network of trade routes that connected China "
    "to Europe and the Middle East. Merchants travelled thousands of miles "
    "carrying silk, spices, and precious metals. The route passed through "
    "deserts, mountains, and oasis cities. It was not a single road but a "
    "web of paths used by different cultures. The Silk Road also carried "
    "ideas, religions, and diseases across continents. Buddhism, Islam, and "
    "Christianity all spread along these routes. The road flourished for "
    "centuries until sea trade routes became dominant."
)

SAMPLE_QUESTION = "What did merchants carry along the Silk Road?"

SAMPLE_OPTIONS = {
    "A": "silk, spices, and precious metals",
    "B": "modern electronics and furniture",
    "C": "oil and natural gas pipelines",
    "D": "books and printing equipment",
}

CORRECT_ANSWER = "A"


# ═══════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCleanText:
    def test_lowercase(self):
        assert clean_text("Hello World") == "hello world"

    def test_punctuation_removed(self):
        result = clean_text("Hello, World!")
        for ch in string.punctuation:
            assert ch not in result

    def test_extra_whitespace(self):
        result = clean_text("  too   many   spaces  ")
        assert "  " not in result
        assert result == result.strip()

    def test_non_string_input(self):
        # Should return empty string for non-string input
        assert clean_text(None) == ""
        assert clean_text(123) == ""

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_normal_sentence(self):
        result = clean_text("The cat sat on the mat.")
        assert result == "the cat sat on the mat"


class TestExtractCandidateSentences:
    def test_returns_list(self):
        result = extract_candidate_sentences(SAMPLE_ARTICLE, SAMPLE_OPTIONS["A"])
        assert isinstance(result, list)

    def test_returns_top_k(self):
        top_k = 3
        result = extract_candidate_sentences(SAMPLE_ARTICLE, SAMPLE_OPTIONS["A"], top_k=top_k)
        assert len(result) <= top_k

    def test_non_empty_sentences(self):
        result = extract_candidate_sentences(SAMPLE_ARTICLE, SAMPLE_OPTIONS["A"])
        for sent in result:
            assert len(sent.strip()) > 0

    def test_correct_answer_boosts_relevant_sentence(self):
        """Sentence mentioning 'silk, spices, metals' should rank highly."""
        result = extract_candidate_sentences(SAMPLE_ARTICLE, "silk spices metals", top_k=1)
        assert len(result) == 1
        # Top sentence should contain at least one answer keyword
        top = result[0].lower()
        assert any(kw in top for kw in ["silk", "spice", "metal"])


class TestApplyWhTemplate:
    def test_returns_string(self):
        result = apply_wh_template("What is the capital of France?")
        assert isinstance(result, str)

    def test_capitalised(self):
        result = apply_wh_template("the students studied hard")
        assert result[0].isupper()

    def test_fallback_for_no_wh_word(self):
        result = apply_wh_template("students studied hard all night")
        assert "inferred" in result.lower() or len(result) > 5

    def test_who_template(self):
        result = apply_wh_template("who discovered penicillin in 1928")
        assert result.startswith("Who")


# ═══════════════════════════════════════════════════════════════════════════
# 2. DISTRACTOR GENERATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestGetDistractorCandidates:
    def test_returns_list(self):
        result = get_distractor_candidates(SAMPLE_ARTICLE, SAMPLE_OPTIONS["A"])
        assert isinstance(result, list)

    def test_returns_candidates(self):
        result = get_distractor_candidates(SAMPLE_ARTICLE, SAMPLE_OPTIONS["A"], n_candidates=5)
        assert len(result) >= 1

    def test_candidates_are_strings(self):
        result = get_distractor_candidates(SAMPLE_ARTICLE, SAMPLE_OPTIONS["A"])
        for c in result:
            assert isinstance(c, str)
            assert len(c.strip()) > 0

    def test_candidates_not_correct_answer(self):
        """No candidate should exactly equal the correct answer."""
        correct = clean_text(SAMPLE_OPTIONS["A"])
        result = get_distractor_candidates(SAMPLE_ARTICLE, SAMPLE_OPTIONS["A"])
        for c in result:
            assert clean_text(c) != correct

    def test_length_cap(self):
        """Candidates should be capped at 120 characters."""
        result = get_distractor_candidates(SAMPLE_ARTICLE, SAMPLE_OPTIONS["A"])
        for c in result:
            assert len(c) <= 120


class TestGetHintSentences:
    def test_returns_list(self):
        result = get_hint_sentences(SAMPLE_ARTICLE, SAMPLE_QUESTION)
        assert isinstance(result, list)

    def test_top_k_respected(self):
        result = get_hint_sentences(SAMPLE_ARTICLE, SAMPLE_QUESTION, top_k=3)
        assert len(result) <= 3

    def test_hints_are_strings(self):
        result = get_hint_sentences(SAMPLE_ARTICLE, SAMPLE_QUESTION)
        for hint in result:
            assert isinstance(hint, str)
            assert len(hint.strip()) > 0

    def test_hint_ordering(self):
        """Hint 3 (index 2) should be more specific (higher keyword overlap) than Hint 1."""
        hints = get_hint_sentences(SAMPLE_ARTICLE, SAMPLE_QUESTION, top_k=3)
        if len(hints) == 3:
            q_words = set(clean_text(SAMPLE_QUESTION).split())
            def overlap(h):
                hw = set(clean_text(h).split())
                return len(hw & q_words)
            # Hint 3 (most specific) should have overlap >= Hint 1 (most general)
            assert overlap(hints[2]) >= overlap(hints[0])


# ═══════════════════════════════════════════════════════════════════════════
# 3. EVALUATION METRIC TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestClassificationMetrics:
    def test_perfect_accuracy(self):
        y = [0, 1, 0, 1, 1]
        metrics = compute_classification_metrics(y, y, "test_model")
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_zero_accuracy(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_output_keys(self):
        metrics = compute_classification_metrics([0, 1], [0, 1])
        for key in ["accuracy", "macro_f1", "precision", "recall", "confusion_matrix"]:
            assert key in metrics

    def test_all_metrics_between_0_and_1(self):
        y_true = [0, 1, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        for key in ["accuracy", "macro_f1", "precision", "recall"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"

    def test_confusion_matrix_shape(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 0, 1, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        cm = metrics["confusion_matrix"]
        # 2x2 for binary classification
        assert len(cm) == 2
        assert len(cm[0]) == 2


class TestExactMatch:
    def test_perfect_match(self):
        assert exact_match(["A", "B", "C"], ["A", "B", "C"]) == pytest.approx(1.0, abs=1e-4)

    def test_zero_match(self):
        assert exact_match(["A", "B"], ["C", "D"]) == pytest.approx(0.0, abs=1e-4)

    def test_partial_match(self):
        score = exact_match(["A", "B", "C", "D"], ["A", "B", "X", "X"])
        assert 0.0 < score < 1.0

    def test_case_sensitive(self):
        # "a" != "A" — exact match is case-sensitive after strip
        score = exact_match(["A"], ["a"])
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_empty_lists(self):
        score = exact_match([], [])
        assert score == pytest.approx(0.0, abs=1e-4)


class TestRegressionMetrics:
    def test_perfect_r2(self):
        y = [1.0, 2.0, 3.0, 4.0]
        metrics = compute_regression_metrics(y, y)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["mse"] == pytest.approx(0.0, abs=1e-4)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-4)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-4)

    def test_output_keys(self):
        metrics = compute_regression_metrics([0, 1], [0.5, 0.5])
        for key in ["r2", "mse", "mae", "rmse"]:
            assert key in metrics

    def test_rmse_equals_sqrt_mse(self):
        y_t = [1, 2, 3]
        y_p = [1.5, 2.5, 2.5]
        metrics = compute_regression_metrics(y_t, y_p)
        assert metrics["rmse"] == pytest.approx(metrics["mse"] ** 0.5, abs=1e-4)


class TestDistractorDiversity:
    def test_identical_distractors_low_diversity(self):
        d = ["the cat sat on the mat"] * 3
        score = distractor_diversity(d)
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_completely_different_distractors_high_diversity(self):
        d = ["alpha beta gamma", "delta epsilon zeta", "theta iota kappa"]
        score = distractor_diversity(d)
        assert score > 0.9

    def test_single_distractor_returns_zero(self):
        score = distractor_diversity(["only one"])
        assert score == 0.0

    def test_returns_float(self):
        score = distractor_diversity(["aaa bbb", "ccc ddd"])
        assert isinstance(score, float)

    def test_score_between_0_and_1(self):
        d = ["the quick brown fox", "jumped over the lazy dog", "a very fast animal"]
        score = distractor_diversity(d)
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. INFERENCE ENGINE INTEGRATION TESTS (optional — skip if models missing)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "models", "model_a", "traditional", "ensemble.pkl")),
    reason="Trained models not found — run model_a_train.py and model_b_train.py first"
)
class TestRACEInferenceEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        from src.inference import RACEInferenceEngine
        eng = RACEInferenceEngine()
        success = eng.load()
        assert success, f"Engine failed to load: {eng.load_error}"
        return eng

    def test_engine_is_ready(self, engine):
        assert engine.is_ready

    def test_predict_answer_returns_dict(self, engine):
        result = engine.predict_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS)
        assert isinstance(result, dict)

    def test_predict_answer_keys(self, engine):
        result = engine.predict_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS)
        assert "predicted_letter" in result
        assert "confidence_scores" in result
        assert "latency_ms" in result

    def test_predicted_letter_valid(self, engine):
        result = engine.predict_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS)
        assert result["predicted_letter"] in ["A", "B", "C", "D"]

    def test_confidence_scores_sum_to_one(self, engine):
        result = engine.predict_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS)
        total = sum(result["confidence_scores"].values())
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_latency_under_10_seconds(self, engine):
        """Per spec: single inference must complete in < 10 seconds."""
        result = engine.predict_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS)
        assert result["latency_ms"] < 10_000, (
            f"Inference too slow: {result['latency_ms']:.0f} ms"
        )

    def test_generate_distractors_returns_3(self, engine):
        distractors = engine.generate_distractors(SAMPLE_ARTICLE, SAMPLE_OPTIONS["A"])
        assert len(distractors) == 3

    def test_generate_hints_returns_3(self, engine):
        hints = engine.generate_hints(SAMPLE_ARTICLE, SAMPLE_QUESTION)
        assert len(hints) == 3

    def test_distractors_are_strings(self, engine):
        distractors = engine.generate_distractors(SAMPLE_ARTICLE, SAMPLE_OPTIONS["A"])
        for d in distractors:
            assert isinstance(d, str) and len(d.strip()) > 0

    def test_hints_are_strings(self, engine):
        hints = engine.generate_hints(SAMPLE_ARTICLE, SAMPLE_QUESTION)
        for h in hints:
            assert isinstance(h, str) and len(h.strip()) > 0

    def test_run_with_options(self, engine):
        result = engine.run(SAMPLE_ARTICLE, SAMPLE_QUESTION, SAMPLE_OPTIONS)
        assert result["mode"] == "verification"
        assert "predicted_letter" in result
        assert "hints" in result


# ═══════════════════════════════════════════════════════════════════════════
# RUN DIRECTLY
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
