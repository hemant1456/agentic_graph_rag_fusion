# Step 10 — Context Engineering

> **Problem**: The LLM receives up to 20 retrieved chunks in raw form — many are near-duplicates, many contain irrelevant sentences, and there's no token budget enforcement.  
> **Fix**: Run a four-stage pipeline between retrieval and synthesis: rerank → deduplicate → compress → format within a token budget.

## How It Works

```
Step 09 retrieval  (k=20 raw chunks, ~45,000 chars)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  1. RERANK  (reranker.py)                             │
│     CrossEncoder ms-marco-MiniLM-L-6-v2              │
│     scores each (question, chunk) pair                │
│     keeps top-8 by cross-encoder score               │
│     45,000 chars → ~18,000 chars                     │
├───────────────────────────────────────────────────────┤
│  2. DEDUPLICATE  (deduplicator.py)                    │
│     6-gram Jaccard similarity between all chunk pairs │
│     drops any chunk with Jaccard > 0.72 vs a keeper  │
│     removes paraphrases and near-copy paragraphs     │
├───────────────────────────────────────────────────────┤
│  3. COMPRESS  (compressor.py)                         │
│     extractive sentence scoring: key-term overlap    │
│     between question and each sentence               │
│     keeps top 60% of sentences per chunk             │
│     CSV data and graph context are NEVER compressed  │
├───────────────────────────────────────────────────────┤
│  4. FORMAT  (formatter.py)                            │
│     structured XML with source attribution:          │
│     <csv_data>, <graph_context>,                     │
│     <passage rank="1" src="..." score="0.92">        │
│     enforces 24,000-char budget (priority order)     │
└────────────────────┬──────────────────────────────────┘
                     │  context_xml  +  ce_metrics{}
                     ▼
              Synthesis + Critic  (Step 09 agents)
```

## Context Engineering Metrics

Every query reports:

| Metric | Meaning |
|--------|---------|
| `raw_chars` | Total chars of all retrieved chunks before CE |
| `engineered_chars` | Chars sent to LLM after CE |
| `compression_ratio` | engineered / raw (lower = more aggressive) |
| `chunks_before` | Chunks after retrieval |
| `chunks_after_dedup` | After Jaccard deduplication |
| `chunks_final` | After CrossEncoder top-k selection |

Typical: 40 chunks (45,000 chars) → 5,000 chars (11%) with answer quality preserved.

## Key Files

| File | What it does |
|------|-------------|
| `implementation/reranker.py` | CrossEncoder singleton, `rerank(question, chunks, k)` |
| `implementation/deduplicator.py` | `_jaccard_6gram()` + `deduplicate(scored_chunks, threshold)` |
| `implementation/compressor.py` | Extractive sentence filter by key-term overlap |
| `implementation/formatter.py` | XML builder with char-budget enforcement |
| `implementation/context_engineer.py` | Orchestrates all 4 stages, returns `(xml, metrics)` |
| `implementation/pipeline.py` | `Step08RAG` — wide retrieval → CE → synthesis |

## Install

The CrossEncoder requires sentence-transformers:
```bash
uv sync --extra step-07
```

## Run It

```bash
uv run python step_08_context_engineering/evaluation/run_eval.py
```
