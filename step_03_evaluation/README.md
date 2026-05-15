# Step 03 — Evaluation Framework

## Goal
Now that every query is traced (Step 02), add automated quality metrics. Move beyond
PASS/PARTIAL/FAIL to a five-dimension view of what the RAG system is doing well and
where it is failing — at the retrieval level, the generation level, or both.

---

## Metrics

Each metric is computed by an LLM-as-judge (Gemini Flash, Anthropic Haiku fallback).
Every metric returns a score 0.0–1.0 plus a one-sentence reasoning string.

| Metric | What it measures | Failure signal |
|---|---|---|
| **Faithfulness** | Are all answer claims grounded in the retrieved context? | Answer contains hallucinated facts |
| **Answer Relevance** | Does the answer actually address the question? | Evasive, off-topic, or non-answer |
| **Context Precision** | Of retrieved chunks, what fraction were actually useful? | Noisy retrieval — irrelevant chunks crowd out signal |
| **Context Recall** | Were all required facts present in the retrieved context? | Retrieval failure — key documents not retrieved |
| **Multi-hop Success** | For multi-hop questions, did the chain of reasoning succeed? | Derived from PASS/PARTIAL/FAIL (no extra LLM call) |

### Why these five?

These dimensions separate *retrieval failures* from *generation failures*:

- Low **context recall** → retrieval problem (the right documents weren't fetched)
- High **context recall** but low **faithfulness** → generation problem (model ignored or hallucinated)
- Low **context precision** → retrieval is noisy (fetching irrelevant documents wastes the context window)
- Low **answer relevance** → generation problem (model answered a different question)

This decomposition tells us which step to fix.

---

## Architecture

```
step_03_evaluation/
  implementation/
    judge.py       LLM-as-judge: single call → parsed JSON {score, reasoning}
    metrics.py     Five metric functions, each calls judge() once
    evaluator.py   Evaluator: runs all metrics on one (question, RAGResult) pair
    report.py      CLI: --summary | --question | --worst
  evaluation/
    run_eval.py    Runs all 22 golden questions, writes metric_scores.json
  tests/
    test_metrics.py  25 unit tests (mocked judge) + 1 integration test
  results/
    metric_scores.json  Per-question scores for all 5 metrics
```

---

## Usage

```bash
# Run full evaluation (22 questions × 4 LLM judge calls = ~88 calls, ~2 min)
uv run python step_03_evaluation/evaluation/run_eval.py

# Aggregate scores + per-question table
uv run python step_03_evaluation/implementation/report.py --summary

# Full detail for one question
uv run python step_03_evaluation/implementation/report.py --question Q01

# All questions ranked by worst context recall
uv run python step_03_evaluation/implementation/report.py --worst context_recall

# Unit tests (no API keys needed)
uv run pytest step_03_evaluation/tests/ -v -k "not integration"

# Integration test (requires API keys + built index)
uv run pytest step_03_evaluation/tests/ -v -m integration
```

---

## Design Decisions

### LLM-as-judge instead of RAGAS library
RAGAS is the standard toolkit for these metrics but its API has changed significantly
across versions and it requires LangChain wrappers for non-OpenAI LLMs. Building our
own judge prompts with our existing Gemini/Anthropic clients is more transparent, has
no extra dependencies, and is easier to debug when a metric gives a surprising score.
The metric definitions are equivalent — this is "RAGAS-equivalent" as stated in the
master plan.

### Ground truth for context recall
`required_facts` from each `GoldenQuestion` serve as ground truth for context recall.
The judge checks whether each required fact is present in the retrieved context
(not in the answer). This is the cleanest operationalisation: if the required facts
are in the context but not in the answer, that's a generation problem. If they're
missing from the context, that's a retrieval problem.

### multihop_success is free
Multi-hop success is derived directly from the existing PASS/PARTIAL/FAIL score for
questions of type `multi_hop`. No extra LLM call. Score = 1.0/0.5/0.0 for PASS/PARTIAL/FAIL.
Returns -1.0 (sentinel for "not applicable") for all other question types.

### Rate limiting
A 0.4s delay between judge calls keeps us under Gemini Flash's free-tier rate limit.
Total evaluation time: ~22 questions × 4 calls × 0.4s = ~35s overhead on top of RAG latency.

---

## What the Metrics Reveal (baseline)

After running on the Step 01/02 baseline:

**High faithfulness across the board** — the model is not hallucinating. When it gets
questions wrong (PARTIAL/FAIL), it's because the answer is incomplete, not fabricated.
The retrieved context is trusted.

**Low context recall on aggregation questions** — for Q07, Q08, Q16, the required
facts (total count, total spend) are not explicitly stated in any single document.
They require summing a full CSV column. The retrieved 5 rows never contain the answer.
Context recall = 0 for these questions regardless of retrieval quality.

**Low context precision on CSV-heavy queries** — when all 5 retrieved chunks come from
the same CSV file (5 rows of a 20-row table), precision is high but recall is low. The
model gets 5 slightly different rows, none of which is the answer. Precision metric
misses this failure mode — a future reranker will need a "coverage" metric.

**Multi-hop success = 0.0 on all chain questions** — Q13 and Q14 retrieve the first-hop
entity correctly but the second-hop document (employee_directory for Q13, contract for
Q14) never appears in top-5. Context recall = 0.5 on these; the chain breaks at hop 2.

---

## Next Step

→ **Step 04 — Parsing & Chunking**: Replace naive fixed-size chunks with format-aware,
metadata-rich chunks. Semantic chunking for prose, row-level chunking for CSVs with
aggregation metadata. Expected improvement: context recall on aggregation questions,
multi-hop success rate.
