import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from step_01_baseline_rag.evaluation.golden_questions import GoldenQuestion
from step_01_baseline_rag.evaluation.run_eval import score as pass_fail_score
from step_01_baseline_rag.implementation.pipeline import RAGResult

from .metrics import (
    MetricResult,
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
    multihop_success,
)

_INTER_METRIC_DELAY_S = 0.4   # stay under rate limits between judge LLM calls


@dataclass
class EvalRecord:
    question_id: str
    question_type: str
    question: str
    answer: str
    grade: str                   # PASS / PARTIAL / FAIL from existing scorer
    faithfulness: MetricResult
    answer_relevance: MetricResult
    context_precision: MetricResult
    context_recall: MetricResult
    multihop_success: MetricResult


class Evaluator:
    """Runs all five metrics on each (golden question, RAGResult) pair."""

    def evaluate(self, golden: GoldenQuestion, result: RAGResult) -> EvalRecord:
        scoring = pass_fail_score(result, golden)
        grade = scoring["grade"]
        context = result.context_sent
        chunk_texts = [c.text for c in result.retrieved_chunks]

        faith = faithfulness(golden.question, result.answer, context)
        time.sleep(_INTER_METRIC_DELAY_S)

        relevance = answer_relevance(golden.question, result.answer)
        time.sleep(_INTER_METRIC_DELAY_S)

        precision = context_precision(golden.question, chunk_texts)
        time.sleep(_INTER_METRIC_DELAY_S)

        recall = context_recall(golden.question, golden.required_facts, context)

        mh = multihop_success(grade, golden.type)

        return EvalRecord(
            question_id=golden.id,
            question_type=golden.type,
            question=golden.question,
            answer=result.answer,
            grade=grade,
            faithfulness=faith,
            answer_relevance=relevance,
            context_precision=precision,
            context_recall=recall,
            multihop_success=mh,
        )

    @staticmethod
    def to_dict(record: EvalRecord) -> dict:
        return {
            "id": record.question_id,
            "type": record.question_type,
            "grade": record.grade,
            "faithfulness": round(record.faithfulness.score, 3),
            "faithfulness_reasoning": record.faithfulness.reasoning,
            "answer_relevance": round(record.answer_relevance.score, 3),
            "answer_relevance_reasoning": record.answer_relevance.reasoning,
            "context_precision": round(record.context_precision.score, 3),
            "context_precision_reasoning": record.context_precision.reasoning,
            "context_recall": round(record.context_recall.score, 3),
            "context_recall_reasoning": record.context_recall.reasoning,
            "multihop_score": record.multihop_success.score,
            "question": record.question,
            "answer": record.answer,
        }
