# Evaluation Framework

A single-folder framework for scoring each of the 10 RAG steps against a 15-question golden set, using the RAGAS library with a free-tier LLM judge.

This folder is a utility — it is not a numbered step. It is invoked manually after building a step's pipeline to measure quality.

## How it works

```
question
  ├──> step's pipeline ──> answer + retrieved_contexts
  │
  ├──> reference answer (from golden_questions.py)
  │
  └──> RAGAS (5 metrics) ──> JSON results
```

Each step exposes its pipeline class (e.g. `Step03HybridRAG`). The runner picks the step via a small per-step adapter function, queries it with each golden question, and feeds (question, answer, contexts, reference) into RAGAS.

## The 5 metrics

RAGAS scores every (question, answer, retrieved_contexts, reference) tuple on five dimensions, each in [0, 1]:

| Metric | What it measures | Reveals |
|---|---|---|
| **answer_correctness** | Factual + semantic match against the reference answer | Whether the answer is *right* (format-tolerant) |
| **faithfulness** | Are answer claims grounded in the retrieved contexts? | Hallucination |
| **answer_relevancy** | Does the answer address the question that was asked? | Evasion / off-topic responses |
| **context_precision** | Are retrieved chunks relevant to the answer? | Retrieval noise |
| **context_recall** | Did retrieval surface every fact in the reference? | Retrieval miss |

`answer_correctness` is what drives the overall PASS/PARTIAL/FAIL grade. The other four exist to *diagnose* failures:

- low `context_recall` + low `answer_correctness` → retrieval missed the document → fix retrieval (next step in the progression)
- high `context_recall` + low `answer_correctness` → retrieval found it, generation failed → fix prompt / model / context engineering
- low `context_precision` → too much noise in the retrieved chunks → add reranking / dedup
- low `faithfulness` → the model is making things up despite having context → tighten the system prompt

This decomposition is exactly what the numbered steps in this project unlock, one at a time.

## Grade thresholds

`answer_correctness` is bucketed for a readable grade:

| Range | Grade |
|---|---|
| ≥ 0.7 | PASS |
| 0.4 – 0.7 | PARTIAL |
| < 0.4 | FAIL |

The four diagnostic metrics are kept as continuous scores in the JSON output.

## LLM judge — llm_gatewayV2

RAGAS needs an LLM. We route it through the project's local gateway on port 8100 (`llm_gatewayV2/`), which itself routes among free-tier providers.

Provider preference (`evaluation/judge_llm.py`):
1. **groq** (llama-3.3-70b) — ~200 ms/call, 30 RPM, fastest free option
2. **gemini** (3.1-flash-lite) — ~900 ms/call, 15 RPM, fallback

The two slower / broken free providers in the gateway are skipped intentionally:
- **nvidia** (deepseek-v4-flash) — ~30 s/call, too slow for batch eval
- **cerebras** (qwen-3-235b) — returns empty responses, broken on the free tier

Override the order with `JUDGE_PROVIDERS=groq,gemini` in `.env`. Fall back to direct paid OpenAI with `JUDGE_PROVIDER=openai` (uses `OPENAI_API_KEY`).

## Golden questions

15 questions live at `step_01_baseline_rag/evaluation/golden_questions.py`. Each one has:

- `question` — the prompt
- `required_facts` — short strings the answer must contain
- `disqualifiers` — strings whose presence forces FAIL
- `reference_answer` — a natural-language gold answer (used by RAGAS)
- `fixed_by_step` — which step the question is designed to test

Questions are organized into 6 tiers, one tier per step capability:

| Tier | IDs | Type | Fixed at |
|---|---|---|---|
| 1 | Q01–Q02 | simple retrieval | step_01 |
| 2 | Q03–Q04 | format-aware chunking | step_02 |
| 3 | Q05–Q07 | CSV aggregate | step_03 |
| 4 | Q08–Q10 | BM25 keyword-exact | step_04 |
| 5 | Q11–Q13 | knowledge-graph multi-hop | step_05 |
| 6 | Q14–Q15 | cross-document reasoning | step_07 |

## Files in this folder

```
evaluation/
├── __init__.py
├── README.md          this file
├── judge_llm.py       GatewayChat + build_judge_llm()
├── run_eval.py        CLI + one answer_step_NN(question) adapter per step
└── results/           empty by design — results land in <step>/results/
```

Each step's results go into `<step_name>/results/eval_results.json` (not into this folder), so each step owns its own latest scores.

## Running

```bash
# List the 10 available steps
uv run python evaluation/run_eval.py --list

# Score one step
uv run python evaluation/run_eval.py --step step_03_hybrid_retrieval

# Score every step (writes a JSON per step)
uv run python evaluation/run_eval.py --all
```

The gateway must be running first:

```bash
cd llm_gatewayV2 && ./run.sh
```

## Per-step JSON output

Each `<step>/results/eval_results.json` has this shape:

```jsonc
{
  "step": "step_03_hybrid_retrieval",
  "timestamp": "2026-05-17T21:35:12",
  "total_questions": 15,
  "grade_counts": {"PASS": 8, "PARTIAL": 4, "FAIL": 3},
  "pass_rate": 0.53,
  "ragas_averages": {
    "answer_correctness": 0.64,
    "faithfulness":       0.91,
    "answer_relevancy":   0.78,
    "context_precision":  0.55,
    "context_recall":     0.62
  },
  "thresholds": {"PASS": "answer_correctness>=0.7", "PARTIAL": ">=0.4", "FAIL": "<0.4"},
  "results": [ /* 15 per-question rows */ ]
}
```

## Cost & timing

15 questions × 5 metrics × ~3 LLM calls per metric = ~225 judge calls per step. At groq's 30 RPM with `max_workers=4`, that's ~10 minutes per step end-to-end. Running `--all` takes roughly 100 minutes.

## Why we don't use the built-in OpenAI judge

RAGAS defaults to OpenAI. For a learning project meant to be cheap to run, we route through `llm_gatewayV2` to use free-tier providers with automatic failover. The same `LangchainLLMWrapper(GatewayChat(...))` plugs into RAGAS in 10 lines (see `evaluation/run_eval.py:_build_ragas_objects`).
