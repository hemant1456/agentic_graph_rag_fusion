# Step 03 — CSV Tools

## What it adds
Introduces a structured-data tool: a pandas-backed CSV query function that runs alongside dense retrieval. When the question matches an aggregate intent (total ARR, Q3 revenue, headcount by city), the tool computes the exact answer from the source CSV and prepends the result to the LLM context. Newly handles Tier 3 questions (Q05-Q07) where vector retrieval can find the relevant rows but cannot sum them.

## Design
- **Class:** `Step02ToolsRAG` in `step_02_tools/implementation/pipeline.py`
- **Inherits from:** composes `BaselineRAG`'s ChromaDB collection (`vertexia_smart`), retrieval, and generation
- **Key components:**
  - `step_02_tools/implementation/csv_tool.py` — `detect_intent()` regex/keyword router and `run_query()` pandas executor
  - Step 02's `vertexia_smart` ChromaDB collection (no re-ingest)

## How it works
At query time, the pipeline runs the Step 02 dense retrieval to produce a vector context. In parallel, `detect_intent(question)` pattern-matches the question against a small registry of aggregate intents (total ARR, quarterly revenue sum, employee count by location, etc.). If an intent matches, `run_query(intent)` loads the relevant CSV from `dataset/company_data/` with pandas, computes the aggregate, and renders it as a labelled text block. The CSV result is prepended to the vector context, then both are sent to the LLM. If no intent matches, the pipeline degrades to pure dense retrieval.

## Run
```bash
uv run python evaluation/run_eval.py --step step_02_tools
```

## Results
See `step_02_tools/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Dense retrieval can locate the customer-list CSV chunks for Q05 but cannot sum 20 ARR values. The LLM can sometimes add small numbers it sees in context, but it routinely truncates the CSV, hallucinates rows, or drops precision. A deterministic pandas call is both correct and traceable. This step makes the structured side of the corpus first-class instead of treating CSVs as if they were prose.

<!-- RESULTS_DETAIL_START -->

## Eval results

**Run summary** — 6 PASS · 2 PARTIAL · 6 FAIL out of 14 questions (43% pass rate).

RAGAS averages:

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.514 | 0.986 | 0.700 | 0.243 | 0.529 |

### Per-question detail

| ID | Grade | correctness | Fixed-by step | Notes |
|---|---|---:|---|---|
| **Q01** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q02** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q03** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q04** | PARTIAL | 0.50 | `step_02_tools` | Should PASS at this tier but only PARTIAL. The actual answer correctly reports both the $11M CSV total and the $16.5M finance report figure, acknowledging the discrepancy without inventing explanations, but the reference a… |
| **Q05** | PASS | 1.00 | `step_02_tools` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q06** | PASS | 1.00 | `step_02_tools` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q07** | PASS | 1.00 | `step_03_hybrid_retrieval` | Unexpected PASS — question targets step 3's capability, but retrieved context happened to contain enough signal. |
| **Q08** | FAIL | 0.00 | `step_03_hybrid_retrieval` | Expected FAIL — required capability arrives at step 3. |
| **Q09** | FAIL | 0.00 | `step_03_hybrid_retrieval` | Expected FAIL — required capability arrives at step 3. |
| **Q10** | FAIL | 0.30 | `step_04_knowledge_graph` | Expected FAIL — required capability arrives at step 4. |
| **Q11** | PARTIAL | 0.40 | `step_04_knowledge_graph` | Expected — capability arrives at step 4. PARTIAL means retrieval brought some related context. The answer failed to identify PulseConnect and DataCraft as services, incorrectly attributed DataCraft as a consumer of… |
| **Q12** | FAIL | 0.00 | `step_04_knowledge_graph` | Expected FAIL — required capability arrives at step 4. |
| **Q13** | FAIL | 0.00 | `step_05_multi_agent` | Expected FAIL — required capability arrives at step 5. |
| **Q14** | FAIL | 0.00 | `step_05_multi_agent` | Expected FAIL — required capability arrives at step 5. |

> Each question's text + reference answer lives in `step_01_baseline_rag/evaluation/golden_questions.py`. The full per-question JSON (including the judge's reasoning) is in `results/eval_results.json`.

<!-- RESULTS_DETAIL_END -->
