# Step 01 — Baseline RAG

## What it adds
The minimal RAG loop: corpus is split into paragraph chunks, embedded with HuggingFace MiniLM, and stored in ChromaDB. A question is embedded, top-5 cosine matches are retrieved, and a single LLM call generates the answer. This step is the floor that every later step builds on, and it already handles the simplest factoid lookups (Tier 1: Q01-Q02).

## Design
- **Class:** `BaselineRAG` in `step_01_baseline_rag/implementation/pipeline.py`
- **Inherits from:** none (root of the chain)
- **Key components:**
  - `step_01_baseline_rag/implementation/ingest.py` — paragraph chunker (~1000 chars, 200 overlap) and ChromaDB writer
  - `step_01_baseline_rag/implementation/retrieve.py` — top-k cosine retrieval and context formatter
  - `step_01_baseline_rag/implementation/generate.py` — prompt assembly and LLM call via `llm_gatewayV2`
  - Shared `chroma_db/` collection `vertexia_baseline`

## How it works
Ingestion walks `dataset/company_data/`, reads each file as plain text, splits on paragraphs with a sliding window (1000 chars, 200 overlap), embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2`, and writes to ChromaDB. At query time, the question is embedded with the same model, ChromaDB returns the top-5 nearest chunks by cosine distance, they are concatenated into a context block, and a single LLM call (Groq llama-3.3-70b with Gemini fallback) produces the answer. There is no reranking, no query rewriting, no metadata filtering.

## Run
```bash
uv run python evaluation/run_eval.py --step step_01_baseline_rag
```

## Results
See `step_01_baseline_rag/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Every later step is justified by a failure mode of this one. Naive paragraph chunking shreds structured markdown (vendor matrices, runbook alerts), so Tier 2 questions like Q03 and Q04 fail. Dense embeddings alone miss keyword-exact tokens (Tier 4), cannot aggregate CSVs (Tier 3), and cannot traverse relations across files (Tier 5). The baseline gives us a concrete starting score against which each upgrade is measured.

<!-- RESULTS_DETAIL_START -->

## Eval results

**Run summary** — 4 PASS · 1 PARTIAL · 9 FAIL out of 14 questions (29% pass rate).

RAGAS averages:

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.350 | 0.986 | 0.621 | 0.264 | 0.343 |

### Per-question detail

| ID | Grade | correctness | Fixed-by step | Notes |
|---|---|---:|---|---|
| **Q01** | PASS | 1.00 | `step_01_baseline_rag` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q02** | PASS | 1.00 | `step_01_baseline_rag` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q03** | PASS | 1.00 | `step_01_baseline_rag` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q04** | FAIL | 0.00 | `step_02_tools` | Expected FAIL — required capability arrives at step 2. |
| **Q05** | FAIL | 0.00 | `step_02_tools` | Expected FAIL — required capability arrives at step 2. |
| **Q06** | FAIL | 0.20 | `step_02_tools` | Expected FAIL — required capability arrives at step 2. |
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
