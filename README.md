# Agentic Graph RAG Fusion

A 7-step learning project that builds a production-grade RAG system from scratch, diagnosing and fixing one failure mode at a time over synthetic company data (Vertexia Inc.).

## Pipeline progression

7 numbered steps, each cumulatively builds on the previous. Three utility folders (`dataset/`, `observability/`, `evaluation/`) sit alongside but are not part of the numbered progression.

| Step | Name | What it adds | Tier fixed |
|------|------|--------------|-----------|
| [01](step_01_baseline_rag/README.md) | Baseline RAG + Format-aware Chunking | ChromaDB + HuggingFace MiniLM embeddings, top-k cosine retrieval, markdown section splits, per-row CSV chunks | Tier 1 |
| [02](step_02_tools/README.md) | CSV Tool Calling | Pandas tools for exact aggregates (total ARR, Q3 revenue, headcount) | Tier 2 |
| [03](step_03_hybrid_retrieval/README.md) | BM25 + Dense Hybrid | BM25 fused with dense via Reciprocal Rank Fusion for keyword-exact lookups | Tier 3 |
| [04](step_04_knowledge_graph/README.md) | Knowledge Graph + Graph RAG | Entity nodes + edges from CSVs (reports_to, depends_on, uses), alias resolution + BFS dependency-chain traversal | Tier 4 |
| [05](step_05_multi_agent/README.md) | Multi-Agent | QueryAnalyst → Retrieval/Graph/CSV agents → Synthesis → Critic | Tier 5 |
| [06](step_06_context_engineering/README.md) | Context Engineering + VSA | CrossEncoder rerank → Jaccard dedup → extractive compress → XML budget, dispatched by a Finance/HR/Engineering/General domain router |  |
| [07](step_07_production/README.md) | Production Hardening | Semantic cache + retry/backoff + confidence scoring + health monitor |  |

Latest eval results are written to each step's `results/eval_results.json` and rolled up below.

## Question set & tiers

15 golden questions, each tier requires the next capability to PASS:

| Tier | Range | Type | Fixed by |
|------|-------|------|----------|
| 1 | Q01–Q04 | Simple retrieval + format-aware chunking | Step 01 baseline |
| 2 | Q05–Q07 | CSV aggregate computation | Step 02 (tools) |
| 3 | Q08–Q10 | BM25 keyword-exact | Step 03 (hybrid) |
| 4 | Q11–Q13 | Knowledge-graph multi-hop | Step 04 (knowledge graph) |
| 5 | Q14–Q15 | Cross-document / multi-step | Step 05 (multi-agent) |

Reduced from 31 to 15 for faster iteration on the free-tier LLM judge.

## Latest eval results

Scored with RAGAS via `llm_gatewayV2` (cerebras → gemini → groq fallback). `answer_correctness ≥ 0.7` = PASS, `≥ 0.4` = PARTIAL, else FAIL. The four diagnostic metrics localize *where* a failure happens (low recall = retrieval missed; low precision = retrieval noisy; low faithfulness = hallucination; low relevancy = off-topic).

<!-- RESULTS_TABLE_START -->

| Step | PASS | PART | FAIL | answer_correctness | faithfulness | context_recall | context_precision | answer_relevancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step_01_baseline_rag | 8 | 2 | 5 | 0.607 | 1.000 | 0.640 | 0.300 | 0.827 |
| step_02_tools | 9 | 3 | 3 | 0.687 | 1.000 | 0.773 | 0.360 | 0.847 |
| step_03_hybrid_retrieval | 12 | 1 | 2 | 0.813 | 1.000 | 0.840 | 0.213 | 0.893 |
| step_04_knowledge_graph | 13 | 1 | 1 | 0.853 | 0.967 | 0.867 | 0.200 | 0.940 |
| step_05_multi_agent | _pending_ | | | | | | | |
| step_06_context_engineering | _pending_ | | | | | | | |
| step_07_production | _pending_ | | | | | | | |

<!-- RESULTS_TABLE_END -->

Eval rows are pending after the 2026-05-18 restructure (10 steps → 8). All previous numbers were captured against the old structure and will be re-run.



## Evaluation

All evaluation lives in one folder: `evaluation/`.

- `evaluation/run_eval.py` — CLI runner with one `answer_step_NN(question)` adapter per step
- `evaluation/judge_llm.py` — LangChain wrapper around `llm_gatewayV2` (prefers groq → gemini)
- Scoring metric: **RAGAS `answer_correctness`** (semantic + factual, format-tolerant)
- Threshold: `≥0.7` PASS, `≥0.4` PARTIAL, `<0.4` FAIL

```bash
uv run python evaluation/run_eval.py --list                       # show all steps
uv run python evaluation/run_eval.py --step step_03_hybrid_retrieval
uv run python evaluation/run_eval.py --all                        # run everything
```

Results land in `<step_name>/results/eval_results.json` with per-question scores and grades.

## LLM Gateway V2

All LLM calls (both pipeline answer generation and RAGAS judge) route through a local gateway:

```bash
cd llm_gatewayV2
uv run uvicorn main:app --port 8100
```

Providers: Groq (llama-3.3-70b, fastest), Gemini (3.1 Flash-Lite, fallback), NVIDIA NIM, Cerebras — all free tier.

## Storage

One shared ChromaDB at the project root: `chroma_db/`.
- `vertexia_smart` collection — format-aware section chunks (used by all steps)

## Running the dashboard

```bash
uv run streamlit run dashboard.py
```

## Repository layout

```
agentic_graph_rag_fusion/
├── dataset/                 # synthetic Vertexia corpus (50 files, 7 departments)
├── observability/           # Arize Phoenix + JSONL trace store (utility)
├── evaluation/              # RAGAS scorer + per-step adapters (utility)
├── chroma_db/               # shared vector index, 2 collections
├── step_01_baseline_rag/   ┐
├── step_02_tools/          │
├── step_03_hybrid_retrieval│  ← 7 numbered steps, cumulative
├── step_04_knowledge_graph/│    each step inherits from the previous
├── step_05_multi_agent/    │
├── step_06_context_engineering/
└── step_07_production/     ┘
```
