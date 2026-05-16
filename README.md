# Agentic Graph RAG Fusion

A 12-step learning project that builds a production-grade RAG system from scratch, diagnosing and fixing each failure mode one step at a time over synthetic company data (Vertexia Inc.).

## Progression

| Step | Name | Pass Rate | What it adds |
|------|------|-----------|-------------|
| [00](step_00_dataset/README.md) | Dataset | — | 48 synthetic files across 7 departments (CSV, Markdown, TXT, JSON) |
| [01](step_01_baseline_rag/README.md) | Baseline Vector RAG | 27% | ChromaDB + Gemini embeddings, top-5 cosine retrieval |
| [02](step_02_observability/README.md) | Observability | — | JSONL trace store + Arize Phoenix integration |
| [03](step_03_evaluation/README.md) | Evaluation Framework | — | 5 RAGAS-style LLM-as-judge metrics |
| [04](step_04_chunking/README.md) | Format-aware Chunking | 59% | CSV aggregate chunks, Markdown section splits, text structure detection |
| [05](step_05_knowledge_graph/README.md) | Knowledge Graph | 82% | Entity nodes + relationship edges from CSVs (reports_to, depends_on, uses) |
| [06](step_06_graph_rag/README.md) | Graph RAG | 91% | Alias resolution + full dependency chain traversal |
| [07](step_07_rag_fusion/README.md) | RAG Fusion + BM25 | 95% | BM25 + dense RRF merge + Pandas structured CSV query tool |
| [08](step_08_agentic_rag/README.md) | Agentic RAG | 100% | Tool-calling loop via LLM Gateway V2 (Gemini/NVIDIA/Groq/Cerebras) |
| [09](step_09_multi_agent/README.md) | Multi-Agent System | — | 6 specialised agents + orchestrator + Critic with typed contracts |
| [10](step_10_context_engineering/README.md) | Context Engineering | — | CrossEncoder rerank → Jaccard dedup → extractive compress → XML budget |
| [11](step_11_vsa/README.md) | Vertical Slice Architecture | — | Keyword router dispatches to Finance/HR/Engineering/General domain slices |
| [12](step_12_production/README.md) | Production Hardening | — | Semantic cache + retry/backoff + confidence scoring + health monitor |

> Steps 09–12 have placeholder eval results — run the evaluation scripts to populate them.

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

All LLM calls from Step 08 onward route through a local gateway:

```bash
cd llm_gatewayV2
uv run uvicorn main:app --port 8100
```

Providers: Gemini 3.1 Flash-Lite Preview, NVIDIA NIM, Groq, Cerebras — all free tier.

## Running Evaluations

```bash
uv run python step_01_baseline_rag/evaluation/run_eval.py
uv run python step_04_chunking/evaluation/run_eval.py
uv run python step_05_knowledge_graph/evaluation/run_eval.py
uv run python step_06_graph_rag/evaluation/run_eval.py
uv run python step_07_rag_fusion/evaluation/run_eval.py
uv run python -m step_08_agentic_rag.evaluation.run_eval
uv run python step_09_multi_agent/evaluation/run_eval.py
uv run python step_10_context_engineering/evaluation/run_eval.py
uv run python step_11_vsa/evaluation/run_eval.py
uv run python step_12_production/evaluation/run_eval.py
```
