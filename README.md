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

14 golden questions, each tier requires the next capability to PASS:

| Tier | Range | Type | Fixed by |
|------|-------|------|----------|
| 1 | Q01–Q03 | Simple retrieval + format-aware chunking | Step 01 baseline |
| 2 | Q04–Q06 | CSV aggregate computation | Step 02 (tools) |
| 3 | Q07–Q09 | BM25 keyword-exact | Step 03 (hybrid) |
| 4 | Q10–Q12 | Knowledge-graph multi-hop | Step 04 (knowledge graph) |
| 5 | Q13–Q14 | Cross-document / multi-CSV composition | Step 05 (multi-agent) |

Reduced from 31 to 14 — each tier is corpus-audited to be unreachable by prior steps (answer facts appear in no prose doc that earlier-step retrieval would surface).

## Latest eval results

Scored with RAGAS via `llm_gatewayV2` (groq → cerebras → gemini → nvidia fallback chain). `answer_correctness ≥ 0.7` = PASS, `≥ 0.4` = PARTIAL, else FAIL. The four diagnostic metrics localize *where* a failure happens (low recall = retrieval missed; low precision = retrieval noisy; low faithfulness = hallucination; low relevancy = off-topic).

<!-- RESULTS_TABLE_START -->

| Step | PASS | PART | FAIL | answer_correctness | faithfulness | context_recall | context_precision | answer_relevancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step_01_baseline_rag | 4 | 1 | 9 | 0.336 | 0.986 | 0.314 | 0.186 | 0.686 |
| step_02_tools | 7 | 1 | 6 | 0.586 | 0.986 | 0.593 | 0.271 | 0.750 |
| step_03_hybrid_retrieval | 10 | 0 | 4 | 0.774 | 0.950 | 0.743 | 0.206 | 0.874 |
| step_04_knowledge_graph | 12 | 2 | 0 | 0.929 | 0.871 | 0.914 | 0.386 | 0.979 |
| step_05_multi_agent | 10 | 2 | 2 | 0.800 | 0.786 | 0.843 | 0.164 | 1.000 |
| step_06_context_engineering | 10 | 2 | 2 | 0.771 | 0.771 | 0.679 | 0.429 | 1.000 |
| step_07_production | 11 | 1 | 2 | 0.843 | 0.657 | 0.564 | 0.279 | 0.986 |

<!-- RESULTS_TABLE_END -->

All rows are from the latest run against the 14-question golden set. Step 07 (production hardening) adds the reliability layer (cache, retry, confidence, health monitor) on top of step 06 — it's scored against the same questions so accuracy is directly comparable to the prior steps.



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

## Side experiments

- [extras/hybrid_rerank/](extras/hybrid_rerank/) — a Step 03 variant that adds a CrossEncoder rerank stage on top of BM25+dense+RRF (no dedup, no compression). Self-contained, evaluated against the same 14-question golden set, useful as an ablation for "does the rerank alone help, or do you need the full Step 06 context-engineering stack?"

## Repository layout

```
agentic_graph_rag_fusion/
├── dataset/                 # synthetic Vertexia corpus (50 files, 7 departments)
├── observability/           # Arize Phoenix + JSONL trace store (utility)
├── evaluation/              # RAGAS scorer + per-step adapters (utility)
├── chroma_db/               # shared vector index, 2 collections
├── extras/                  # side experiments (rerank-only ablation)
├── step_01_baseline_rag/   ┐
├── step_02_tools/          │
├── step_03_hybrid_retrieval│  ← 7 numbered steps, cumulative
├── step_04_knowledge_graph/│    each step inherits from the previous
├── step_05_multi_agent/    │
├── step_06_context_engineering/
└── step_07_production/     ┘
```
