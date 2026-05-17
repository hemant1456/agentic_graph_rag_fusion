# Agentic Graph RAG Fusion

A 10-step learning project that builds a production-grade RAG system from scratch, diagnosing and fixing one failure mode at a time over synthetic company data (Vertexia Inc.).

## Pipeline progression

10 numbered steps, each cumulatively builds on the previous. Three utility folders (`dataset/`, `observability/`, `evaluation/`) sit alongside but are not part of the numbered progression.

| Step | Name | What it adds | Tier fixed |
|------|------|--------------|-----------|
| [01](step_01_baseline_rag/README.md) | Baseline RAG | ChromaDB + HuggingFace MiniLM embeddings, top-k cosine retrieval | Tier 1 |
| [02](step_02_chunking/README.md) | Format-aware Chunking | Markdown section splits + contextual headers (`[FILE | DOC | SECTION]`), per-row CSV | Tier 2 |
| [03](step_03_tools/README.md) | CSV Tool Calling | Pandas tools for exact aggregates (total ARR, Q3 revenue, headcount) | Tier 3 |
| [04](step_04_hybrid_retrieval/README.md) | BM25 + Dense Hybrid | BM25 fused with dense via Reciprocal Rank Fusion for keyword-exact lookups | Tier 4 |
| [05](step_05_knowledge_graph/README.md) | Knowledge Graph | Entity nodes + edges from CSVs (reports_to, depends_on, uses) | Tier 5 |
| [06](step_06_graph_rag/README.md) | Graph RAG | Alias resolution + BFS dependency-chain traversal | Tier 5 |
| [07](step_07_multi_agent/README.md) | Multi-Agent | QueryAnalyst → Retrieval/Graph/CSV agents → Synthesis → Critic | Tier 6 |
| [08](step_08_context_engineering/README.md) | Context Engineering | CrossEncoder rerank → Jaccard dedup → extractive compress → XML budget |  |
| [09](step_09_vsa/README.md) | Vertical Slice Architecture | Keyword router dispatches to Finance/HR/Engineering domain slices |  |
| [10](step_10_production/README.md) | Production Hardening | Semantic cache + retry/backoff + confidence scoring + health monitor |  |

Latest eval results are written to each step's `results/eval_results.json` and rolled up below.

## Question set & tiers

15 golden questions, each tier requires the next capability to PASS:

| Tier | Range | Type | Fixed by |
|------|-------|------|----------|
| 1 | Q01–Q02 | Simple retrieval | Step 01 baseline |
| 2 | Q03–Q04 | Format-aware chunking | Step 02 |
| 3 | Q05–Q07 | CSV aggregate computation | Step 03 |
| 4 | Q08–Q10 | BM25 keyword-exact | Step 04 |
| 5 | Q11–Q13 | Knowledge-graph multi-hop | Step 05 |
| 6 | Q14–Q15 | Cross-document / multi-step | Step 07 |

Reduced from 31 to 15 for faster iteration on the free-tier LLM judge.

## Latest eval results

Scored with RAGAS via `llm_gatewayV2` (groq → gemini fallback). `answer_correctness ≥ 0.7` = PASS, `≥ 0.4` = PARTIAL, else FAIL. The four diagnostic metrics localize *where* a failure happens (low recall = retrieval missed; low precision = retrieval noisy; low faithfulness = hallucination; low relevancy = off-topic).

| Step | PASS | PART | FAIL | answer_correctness | faithfulness | context_recall | context_precision | answer_relevancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step_01_baseline_rag | 2 | 3 | 10 | 0.299 | 0.907 | 0.300 | 0.100 | 0.440 |
| step_02_chunking | _running_ | | | | | | | |
| step_03_tools | _pending_ | | | | | | | |
| step_04_hybrid_retrieval | _pending_ | | | | | | | |
| step_05_knowledge_graph | _pending_ | | | | | | | |
| step_06_graph_rag | _pending_ | | | | | | | |
| step_07_multi_agent | _pending_ | | | | | | | |
| step_08_context_engineering | _pending_ | | | | | | | |
| step_09_vsa | _pending_ | | | | | | | |
| step_10_production | _pending_ | | | | | | | |

**Step 01 baseline observations** — faithfulness is high (0.91, model doesn't hallucinate), but retrieval is the bottleneck: `context_recall=0.30` means most required facts never land in the top-k chunks. The 10 failing questions are exactly the tiers that subsequent steps unlock: CSV aggregates need Step 03's Pandas tool, keyword-exact queries need Step 04's BM25, multi-hop chains need Step 05's graph.



## Evaluation

All evaluation lives in one folder: `evaluation/`.

- `evaluation/run_eval.py` — CLI runner with one `answer_step_NN(question)` adapter per step
- `evaluation/judge_llm.py` — LangChain wrapper around `llm_gatewayV2` (prefers groq → gemini)
- Scoring metric: **RAGAS `answer_correctness`** (semantic + factual, format-tolerant)
- Threshold: `≥0.7` PASS, `≥0.4` PARTIAL, `<0.4` FAIL

```bash
uv run python evaluation/run_eval.py --list                       # show all steps
uv run python evaluation/run_eval.py --step step_04_hybrid_retrieval
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
- `vertexia_baseline` collection — naive paragraph chunks (used by step_01)
- `vertexia_smart` collection — format-aware section chunks (used by step_02+)

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
├── step_02_chunking/       │
├── step_03_tools/          │
├── step_04_hybrid_retrieval│  ← 10 numbered steps, cumulative
├── step_05_knowledge_graph/│    each step inherits from the previous
├── step_06_graph_rag/      │
├── step_07_multi_agent/    │
├── step_08_context_engineering/
├── step_09_vsa/            │
└── step_10_production/     ┘
```
