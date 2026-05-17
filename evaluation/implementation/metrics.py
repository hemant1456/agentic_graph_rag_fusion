from dataclasses import dataclass

from .judge import judge

# Context fed to judge is capped to avoid blowing token budgets.
_CONTEXT_LIMIT = 3000
_CHUNK_PREVIEW = 400


@dataclass
class MetricResult:
    score: float      # 0.0 (worst) → 1.0 (best); -1.0 = not applicable
    reasoning: str    # one-sentence explanation from the judge


def faithfulness(question: str, answer: str, context: str) -> MetricResult:
    """
    Are all factual claims in the answer supported by the retrieved context?
    High score → no hallucination.  Low score → answer contains facts not in context.
    """
    prompt = f"""You are a RAG evaluation judge assessing FAITHFULNESS.

Faithfulness = every factual claim in the answer can be traced to the context.
Score 1.0 if every claim is supported. Score 0.0 if claims contradict or are absent from context.

Question: {question}

Answer: {answer}

Retrieved Context (truncated to {_CONTEXT_LIMIT} chars):
{context[:_CONTEXT_LIMIT]}

Instructions:
1. List each factual claim in the answer.
2. Mark each claim as SUPPORTED or UNSUPPORTED based on the context.
3. Score = supported_claims / total_claims.

Respond with valid JSON only — no prose outside the JSON:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>", "unsupported_claims": [<strings>]}}"""

    result = judge(prompt)
    return MetricResult(
        score=min(1.0, max(0.0, float(result.get("score", 0.5)))),
        reasoning=result.get("reasoning", ""),
    )


def answer_relevance(question: str, answer: str) -> MetricResult:
    """
    Does the answer directly and completely address the question?
    High score → focused, on-point.  Low score → evasive, off-topic, or a non-answer.
    """
    prompt = f"""You are a RAG evaluation judge assessing ANSWER RELEVANCE.

Answer relevance = the answer directly addresses what was asked.
Score 1.0: fully addresses the question.
Score 0.5: partially addresses or is incomplete.
Score 0.0: irrelevant, refuses to answer, or addresses a different question.

Question: {question}

Answer: {answer}

Respond with valid JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""

    result = judge(prompt)
    return MetricResult(
        score=min(1.0, max(0.0, float(result.get("score", 0.5)))),
        reasoning=result.get("reasoning", ""),
    )


def context_precision(question: str, chunk_texts: list[str]) -> MetricResult:
    """
    Of the retrieved chunks, what fraction were actually useful for answering?
    High score → precise retrieval (every chunk helps).
    Low score → noisy retrieval (many irrelevant chunks padding the context).
    """
    if not chunk_texts:
        return MetricResult(score=0.0, reasoning="No chunks retrieved")

    chunks_fmt = "\n\n".join(
        f"[Chunk {i + 1}]: {text[:_CHUNK_PREVIEW]}" for i, text in enumerate(chunk_texts)
    )
    n = len(chunk_texts)

    prompt = f"""You are a RAG evaluation judge assessing CONTEXT PRECISION.

Context precision = fraction of retrieved chunks that were actually useful for answering the question.
Score 1 for a chunk if it contains information relevant to answering the question, 0 otherwise.
Overall score = sum(chunk_scores) / {n}.

Question: {question}

Retrieved Chunks ({n} total):
{chunks_fmt}

Respond with valid JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>", "chunk_scores": [<{n} values, each 0 or 1>]}}"""

    result = judge(prompt)
    return MetricResult(
        score=min(1.0, max(0.0, float(result.get("score", 0.5)))),
        reasoning=result.get("reasoning", ""),
    )


def context_recall(
    question: str,
    required_facts: list[str],
    context: str,
) -> MetricResult:
    """
    Were all the facts needed to answer the question present in the retrieved context?
    High score → context contained everything needed (retrieval did its job).
    Low score → key facts missing from context (retrieval failure, not generation failure).
    """
    if not required_facts:
        return MetricResult(score=1.0, reasoning="No required facts defined")

    facts_fmt = "\n".join(f"  - {f}" for f in required_facts)

    prompt = f"""You are a RAG evaluation judge assessing CONTEXT RECALL.

Context recall = fraction of required facts that are present (explicitly or implicitly) in the context.
Score = facts_present / total_required_facts.

Question: {question}

Required facts (ground truth — all must be findable to answer correctly):
{facts_fmt}

Retrieved Context (truncated to {_CONTEXT_LIMIT} chars):
{context[:_CONTEXT_LIMIT]}

For each required fact, determine if the relevant information appears in the context.
Respond with valid JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>", "facts_present": [<strings>], "facts_missing": [<strings>]}}"""

    result = judge(prompt)
    return MetricResult(
        score=min(1.0, max(0.0, float(result.get("score", 0.5)))),
        reasoning=result.get("reasoning", ""),
    )


def multihop_success(grade: str, question_type: str) -> MetricResult:
    """
    For multi_hop questions: did the system successfully chain through all hops?
    Derived from the existing PASS/PARTIAL/FAIL grade — no additional LLM call.
    Returns score=-1.0 for non-multi-hop questions (not applicable).
    """
    if question_type != "multi_hop":
        return MetricResult(score=-1.0, reasoning="N/A — not a multi-hop question")

    score_map = {"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0}
    score = score_map.get(grade, 0.0)
    return MetricResult(score=score, reasoning=f"Existing grade: {grade}")
