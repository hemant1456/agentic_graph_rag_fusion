# Step 08 — Context Engineering

## What it adds
Inserts a four-stage context-engineering layer between retrieval and synthesis: a CrossEncoder reranker reorders a wide candidate set, a Jaccard-based deduplicator drops near-identical chunks, an extractive compressor keeps only the sentences most relevant to the question, and an XML formatter packs the result into a token budget. The goal is higher signal-to-noise in the LLM prompt, not new capabilities.

## Design
- **Class:** `Step08RAG` in `step_08_context_engineering/implementation/pipeline.py`
- **Inherits from:** composes `Step04HybridRAG` (k=20 wide candidates) and reuses the Step 07 agents
- **Key components:**
  - `step_08_context_engineering/implementation/reranker.py` — CrossEncoder rerank to top `rerank_k=8`
  - `step_08_context_engineering/implementation/deduplicator.py` — Jaccard near-duplicate removal
  - `step_08_context_engineering/implementation/compressor.py` — sentence-level extractive compression to `compress_ratio=0.60`
  - `step_08_context_engineering/implementation/formatter.py` — XML envelope with explicit token budget
  - `step_08_context_engineering/implementation/context_engineer.py` — `engineer_context()` orchestrates the four stages and emits `ce_metrics`

## How it works
QueryAnalyst runs first to produce sub-questions. RetrievalSpecialist fetches 20 chunks for the main question and 10 more for each of up to four sub-questions. GraphNavigator and StructuredData run unconditionally so exact graph and CSV facts are preserved verbatim. The raw chunks then enter `engineer_context()`: the reranker reorders them by question relevance, the deduplicator removes near-duplicates, the compressor keeps the top sentences from each survivor, and the formatter wraps everything (CSV data, graph context, compressed chunks) inside an XML envelope sized to the token budget. The engineered XML is the only context sent to Synthesis; Critic then reviews. CE metrics (raw chars, engineered chars, compression ratio) are returned alongside the answer.

## Run
```bash
uv run python evaluation/run_eval.py --step step_08_context_engineering
```

## Results
See `step_08_context_engineering/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Step 07 sends up to 20 chunks plus graph and CSV blocks straight to the LLM. Many chunks overlap, many contain only one or two relevant sentences, and the prompt edges out of the model's effective attention window. CrossEncoder reranking promotes the chunks the bi-encoder underweighted; dedup cuts redundancy; compression trims filler sentences; XML formatting gives the model a clearly delimited budget. The exact graph and CSV outputs bypass compression so deterministic facts are never lossy.
