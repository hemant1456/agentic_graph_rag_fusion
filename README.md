# Agentic Graph RAG Fusion

A 12-step learning project that builds a production-grade RAG system from scratch, diagnosing and fixing each failure mode one step at a time over synthetic company data (Vertexia Inc.).

## Progression

| Step | Name | Pass Rate | What it adds |
|------|------|-----------|-------------|
| [00](dataset/README.md) | Dataset | — | 48 synthetic files across 7 departments (CSV, Markdown, TXT, JSON) |
| [01](step_01_baseline_rag/README.md) | Baseline Vector RAG | 26% (7/27) | ChromaDB + Gemini embeddings, top-5 cosine retrieval |
| [02](observability/README.md) | Observability | — | JSONL trace store + Arize Phoenix integration |
| [03](evaluation/README.md) | Evaluation Framework | — | 5 RAGAS-style LLM-as-judge metrics |
| [04](step_02_chunking/README.md) | Format-aware Chunking | — | Markdown section splits, text structure detection (row-by-row CSV) |
| [05](step_03_tools/README.md) | CSV Tool Calling | — | Pandas query tools for exact aggregate computation (total ARR, Q3 revenue, headcount) |
| [06](step_04_hybrid_retrieval/README.md) | Hybrid BM25 + Dense | — | BM25 + dense RRF merge for keyword-exact retrieval (version strings, vendor names) |
| [07](step_05_knowledge_graph/README.md) | Knowledge Graph | — | Entity nodes + relationship edges from CSVs (reports_to, depends_on, uses) |
| [08](step_06_graph_rag/README.md) | Graph RAG | — | Alias resolution + full dependency chain traversal |
| [09](step_07_multi_agent/README.md) | Multi-Agent System | 93% (25/27) | 6 specialised agents + orchestrator + Critic + synthesis precision rules |
| [10](step_08_context_engineering/README.md) | Context Engineering | 85% (23/27) † | CrossEncoder rerank → Jaccard dedup → extractive compress → XML budget |
| [11](step_09_vsa/README.md) | Vertical Slice Architecture | 89% (24/27) | Keyword router dispatches to Finance/HR/Engineering/General domain slices |
| [12](step_10_production/README.md) | Production Hardening | 89% (24/27) + reliability | Semantic cache + retry/backoff + confidence scoring + health monitor |

> † Step 10's extractive compression introduces a tradeoff: Q18 (cross-reference disambiguation) and Q22 (blast-radius completeness) regress as aggressive sentence filtering removes context that multi-agent reasoning preserved. The Finance/HR slices in Step 11 recover these losses through domain-specific prompts and routing.
>
> Q23–Q27 target the agent-tier steps (09–11). Run eval scripts to populate results.

## Architecture at Step 12

```
Query
  │
  ▼
Semantic Cache ──hit──► cached answer
  │ miss
  ▼
VSA Router  ──► Finance / HR / Engineering / General slice
  │              (keyword scoring, zero LLM calls)
  ▼
QueryAnalyst → RetrievalSpecialist → GraphNavigator → StructuredData
  │            (BM25 + dense, k=20)  (entity + BFS)   (Pandas CSV)
  ▼
Context Engineering
  CrossEncoder rerank → Jaccard dedup → extractive compress → XML
  │
  ▼
Synthesis (slice-specific system prompt)
  │
  ▼
Critic → Confidence Score → Health Monitor → answer
```

## Running the Dashboard

```bash
uv run streamlit run dashboard.py
```

## LLM Gateway V2

All LLM calls from Step 09 onward route through a local gateway:

```bash
cd llm_gatewayV2
uv run uvicorn main:app --port 8100
```

Providers: Gemini 3.1 Flash-Lite Preview, NVIDIA NIM, Groq, Cerebras — all free tier.

## Running Evaluations

```bash
uv run python step_01_baseline_rag/evaluation/run_eval.py
uv run python step_02_chunking/evaluation/run_eval.py
uv run python step_03_tools/evaluation/run_eval.py
uv run python step_04_hybrid_retrieval/evaluation/run_eval.py
uv run python step_05_knowledge_graph/evaluation/run_eval.py
uv run python step_06_graph_rag/evaluation/run_eval.py
uv run python step_07_multi_agent/evaluation/run_eval.py
uv run python step_08_context_engineering/evaluation/run_eval.py
uv run python step_09_vsa/evaluation/run_eval.py
uv run python step_10_production/evaluation/run_eval.py
```
