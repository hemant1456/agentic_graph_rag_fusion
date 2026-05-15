"""
Step 03 — Evaluation Framework: unit tests.

Unit tests mock the LLM judge so no API calls are made.
The integration test (marked) runs end-to-end with real LLM calls.

Run unit tests:
    uv run pytest step_03_evaluation/tests/ -v -k "not integration"

Run integration test:
    uv run pytest step_03_evaluation/tests/ -v -m integration
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from step_03_evaluation.implementation.judge import _extract_json
from step_03_evaluation.implementation.metrics import (
    MetricResult,
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
    multihop_success,
)


# ── _extract_json ─────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_plain_json(self):
        raw = '{"score": 0.8, "reasoning": "good"}'
        assert _extract_json(raw) == {"score": 0.8, "reasoning": "good"}

    def test_json_in_code_block(self):
        raw = '```json\n{"score": 0.5, "reasoning": "ok"}\n```'
        assert _extract_json(raw) == {"score": 0.5, "reasoning": "ok"}

    def test_json_in_bare_code_block(self):
        raw = '```\n{"score": 0.3}\n```'
        assert _extract_json(raw) == {"score": 0.3}

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            _extract_json("not json at all")


# ── MetricResult ──────────────────────────────────────────────────────────────

class TestMetricResult:
    def test_fields(self):
        mr = MetricResult(score=0.7, reasoning="test reason")
        assert mr.score == 0.7
        assert mr.reasoning == "test reason"

    def test_score_range_valid(self):
        for s in [0.0, 0.5, 1.0]:
            mr = MetricResult(score=s, reasoning="")
            assert 0.0 <= mr.score <= 1.0

    def test_not_applicable_sentinel(self):
        # multihop_success uses -1.0 for N/A
        mr = MetricResult(score=-1.0, reasoning="N/A")
        assert mr.score == -1.0


# ── faithfulness ─────────────────────────────────────────────────────────────

class TestFaithfulness:
    def test_high_score(self):
        mock_response = {"score": 0.95, "reasoning": "all claims supported", "unsupported_claims": []}
        with patch("step_03_evaluation.implementation.metrics.judge", return_value=mock_response):
            result = faithfulness("Q?", "Answer.", "Context with relevant info.")
        assert result.score == 0.95
        assert "supported" in result.reasoning

    def test_low_score_hallucination(self):
        mock_response = {"score": 0.1, "reasoning": "claims not in context", "unsupported_claims": ["fact X"]}
        with patch("step_03_evaluation.implementation.metrics.judge", return_value=mock_response):
            result = faithfulness("Q?", "Answer with invented fact.", "Unrelated context.")
        assert result.score == 0.1

    def test_score_clamped_above_1(self):
        mock_response = {"score": 1.5, "reasoning": ""}
        with patch("step_03_evaluation.implementation.metrics.judge", return_value=mock_response):
            result = faithfulness("Q?", "A.", "C.")
        assert result.score == 1.0

    def test_score_clamped_below_0(self):
        mock_response = {"score": -0.3, "reasoning": ""}
        with patch("step_03_evaluation.implementation.metrics.judge", return_value=mock_response):
            result = faithfulness("Q?", "A.", "C.")
        assert result.score == 0.0

    def test_missing_score_defaults_to_half(self):
        mock_response = {"reasoning": "no score field"}
        with patch("step_03_evaluation.implementation.metrics.judge", return_value=mock_response):
            result = faithfulness("Q?", "A.", "C.")
        assert result.score == 0.5


# ── answer_relevance ──────────────────────────────────────────────────────────

class TestAnswerRelevance:
    def test_relevant_answer(self):
        mock_response = {"score": 1.0, "reasoning": "directly addresses the question"}
        with patch("step_03_evaluation.implementation.metrics.judge", return_value=mock_response):
            result = answer_relevance("What is X?", "X is Y.")
        assert result.score == 1.0

    def test_irrelevant_answer(self):
        mock_response = {"score": 0.0, "reasoning": "does not address the question"}
        with patch("step_03_evaluation.implementation.metrics.judge", return_value=mock_response):
            result = answer_relevance("What is X?", "I don't know.")
        assert result.score == 0.0


# ── context_precision ─────────────────────────────────────────────────────────

class TestContextPrecision:
    def test_empty_chunks(self):
        result = context_precision("Q?", [])
        assert result.score == 0.0
        assert "No chunks" in result.reasoning

    def test_all_relevant(self):
        mock_response = {"score": 1.0, "reasoning": "all chunks relevant", "chunk_scores": [1, 1, 1]}
        with patch("step_03_evaluation.implementation.metrics.judge", return_value=mock_response):
            result = context_precision("Q?", ["chunk1", "chunk2", "chunk3"])
        assert result.score == 1.0

    def test_mixed_relevance(self):
        mock_response = {"score": 0.6, "reasoning": "3 of 5 relevant", "chunk_scores": [1, 1, 0, 1, 0]}
        with patch("step_03_evaluation.implementation.metrics.judge", return_value=mock_response):
            result = context_precision("Q?", ["c1", "c2", "c3", "c4", "c5"])
        assert result.score == 0.6


# ── context_recall ────────────────────────────────────────────────────────────

class TestContextRecall:
    def test_no_required_facts(self):
        result = context_recall("Q?", [], "some context")
        assert result.score == 1.0
        assert "No required facts" in result.reasoning

    def test_all_facts_present(self):
        mock_response = {
            "score": 1.0,
            "reasoning": "all facts in context",
            "facts_present": ["fact A", "fact B"],
            "facts_missing": [],
        }
        with patch("step_03_evaluation.implementation.metrics.judge", return_value=mock_response):
            result = context_recall("Q?", ["fact A", "fact B"], "context with fact A and fact B")
        assert result.score == 1.0

    def test_facts_missing(self):
        mock_response = {
            "score": 0.5,
            "reasoning": "only half the facts present",
            "facts_present": ["fact A"],
            "facts_missing": ["fact B"],
        }
        with patch("step_03_evaluation.implementation.metrics.judge", return_value=mock_response):
            result = context_recall("Q?", ["fact A", "fact B"], "context with only fact A")
        assert result.score == 0.5


# ── multihop_success ──────────────────────────────────────────────────────────

class TestMultihopSuccess:
    def test_non_multihop_returns_na(self):
        result = multihop_success("PASS", "simple_lookup")
        assert result.score == -1.0
        assert "N/A" in result.reasoning

    def test_pass_grade(self):
        result = multihop_success("PASS", "multi_hop")
        assert result.score == 1.0

    def test_partial_grade(self):
        result = multihop_success("PARTIAL", "multi_hop")
        assert result.score == 0.5

    def test_fail_grade(self):
        result = multihop_success("FAIL", "multi_hop")
        assert result.score == 0.0

    def test_all_question_types_except_multihop(self):
        for qtype in ["simple_lookup", "simple_aggregation", "comparative", "temporal", "aggregation"]:
            result = multihop_success("PASS", qtype)
            assert result.score == -1.0


# ── Integration test ──────────────────────────────────────────────────────────

@pytest.mark.integration
class TestEvaluatorIntegration:
    """Requires live API keys and a built ChromaDB index."""

    def test_single_question_full_pipeline(self):
        from step_01_baseline_rag.evaluation.golden_questions import GOLDEN_QUESTIONS
        from step_01_baseline_rag.implementation.pipeline import BaselineRAG
        from step_03_evaluation.implementation.evaluator import Evaluator

        rag = BaselineRAG(k=5).build()
        evaluator = Evaluator()
        q = GOLDEN_QUESTIONS[0]  # Q01 — simple lookup anchor

        result = rag.query(q.question)
        record = evaluator.evaluate(q, result)

        assert 0.0 <= record.faithfulness.score <= 1.0
        assert 0.0 <= record.answer_relevance.score <= 1.0
        assert 0.0 <= record.context_precision.score <= 1.0
        assert 0.0 <= record.context_recall.score <= 1.0
        assert record.grade in ("PASS", "PARTIAL", "FAIL")

        row = evaluator.to_dict(record)
        assert row["id"] == "Q01"
        assert "faithfulness" in row
        assert "context_recall_reasoning" in row
