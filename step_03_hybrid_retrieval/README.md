# Step 04 — Hybrid Retrieval (BM25 + Dense)

## What it adds
Adds a BM25 lexical retriever alongside the dense vector retriever and fuses the two ranked lists with Reciprocal Rank Fusion (RRF). Dense retrieval covers semantic paraphrase; BM25 anchors on exact tokens — version strings, finding IDs, vendor names. Newly handles Tier 4 questions (Q08-Q10) where the answer hinges on a rare keyword.

## Design
- **Class:** `Step03HybridRAG` in `step_03_hybrid_retrieval/implementation/pipeline.py`
- **Inherits from:** does not subclass, but reuses Step 02's ChromaDB collection and Step 03's CSV tool
- **Key components:**
  - `step_03_hybrid_retrieval/implementation/bm25_retriever.py` — `BM25Index` built over the same chunks stored in ChromaDB
  - `_rrf_fuse()` in `step_03_hybrid_retrieval/implementation/pipeline.py` — reciprocal rank fusion with `k_rrf=60`
  - Step 03's CSV tool (`detect_intent` / `run_query`)

## How it works
On `build()`, the pipeline loads the `vertexia_smart` ChromaDB collection and constructs a BM25 index over the same chunk texts. At query time it runs dense cosine search and BM25 in parallel, each returning roughly `2k` candidates. The two ranked lists are merged with RRF: each chunk earns `1 / (60 + rank)` from each list it appears in, scores are summed, and the top-`k` chunks are kept. The fused chunks plus any CSV tool result are formatted into the LLM context. Step 04 also becomes the retrieval substrate that Steps 05-10 build on.

## Run
```bash
uv run python evaluation/run_eval.py --step step_03_hybrid_retrieval
```

## Results
See `step_03_hybrid_retrieval/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Dense embeddings normalise away the distinctive tokens that some questions depend on. Q08 ("which NexusFlow endpoint was deprecated in v2.1") needs the chunk that literally contains the string `v2.1`; cosine similarity treats `v2.0`, `v2.1`, and `v2.2` as near-equivalent. Q09 ("audit finding M-2") and Q10 ("Snowflake contract") have the same shape. BM25 ranks those rare tokens highly and RRF lets them rise without throwing away the semantic recall that dense retrieval still provides.
