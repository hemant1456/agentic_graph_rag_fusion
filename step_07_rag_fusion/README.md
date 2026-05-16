# Step 07 — RAG Fusion (BM25 + Dense) + Structured CSV Tool

> **Problem**: Dense vector retrieval favours semantic similarity — exact keywords like product codes and employee IDs can rank poorly. Q20 (Q3 2023 revenue sum) also requires arithmetic over the full table, not retrieved prose.  
> **Fix**: Fuse BM25 keyword retrieval with dense retrieval via Reciprocal Rank Fusion. Add a Pandas-backed structured query tool for CSV arithmetic.

## How It Works

```
Query: "What was the Q3 2023 total revenue?"
        │
        ├──► BM25Index.search()      keyword match on all chunk texts  ──┐
        │    (BM25Okapi, rank list)                                       │
        ├──► ChromaDB.query()        dense cosine similarity             ─┤ RRF merge
        │    (gemini-embedding-2)                                         │ score = Σ 1/(k+rank)
        └──────────────────────────────────────────────────────────────── ►  top-K fused list
                                                                          │
        ┌─────────────────────────────────────────────────────────────────┘
        │
        ▼
  detect_intent(query)  ──► "revenue_sum" / "headcount_filter" / "budget_ratio" / None
        │
        ├── if intent found ──► csv_tool.run_query()  ──► Pandas operation on raw CSVs
        │                       returns exact numeric result
        │
        └── graph_context from Step 06 (alias + dependency chain)
                │
                ▼
        [graph_context + csv_result + fused_chunks] → LLM → answer
```

## RRF Explained

Each retriever produces a ranked list. RRF converts ranks to scores and sums them:

```
score(doc) = 1/(60 + rank_bm25) + 1/(60 + rank_dense)
```

A doc ranked #3 by BM25 and #8 by dense beats a doc ranked #1 by only one retriever. `k=60` is a standard smoothing constant.

## Key Files

| File | What it does |
|------|-------------|
| `implementation/bm25_retriever.py` | `BM25Index` — builds at startup from all chunk texts |
| `implementation/csv_tool.py` | `detect_intent()` + `run_query()` — 8 intent patterns, Pandas execution |
| `implementation/pipeline.py` | `Step07RAG` — BM25 + dense + RRF + graph + CSV → LLM |

## Results

| Step | PASS | PARTIAL | FAIL | Pass Rate |
|------|------|---------|------|-----------|
| 06 Graph RAG | 20 | 1 | 1 | 91% |
| **07 RAG Fusion** | **21** | **1** | **0** | **95%** |

Q20 (Q3 2023 revenue) now passes via the structured CSV tool. Q18 remains partial — the two-Phoenix disambiguation still needs an agent with a multi-query strategy (Step 09).

## Run It

```bash
uv run python step_07_rag_fusion/evaluation/run_eval.py
```
