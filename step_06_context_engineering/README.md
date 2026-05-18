# Step 06 — Context Engineering + VSA

## What it adds
Combines two layers in a single step. First, a keyword-scored Vertical Slice Architecture (VSA) router dispatches each question to a domain slice (Finance / HR / Engineering / General) that owns its own system prompt, retrieval-query augmentation, rerank top-k, and compression ratio. Second, every slice runs the same four-stage context-engineering stack: a CrossEncoder reranker reorders a wide candidate set, a Jaccard-based deduplicator drops near-identical chunks, an extractive compressor keeps only the sentences most relevant to the question, and an XML formatter packs the result into a token budget.

## Design
- **Class:** `Step06RAG` in `step_06_context_engineering/implementation/pipeline.py`
- **Inherits from:** composes `Step03HybridRAG` (k=20 wide candidates) and reuses the Step 05 agents
- **Key components:**
  - `step_06_context_engineering/implementation/router.py` — `dispatch()` scores slices by keyword overlap and returns the winning slice plus confidence
  - `step_06_context_engineering/implementation/slices/base.py` — shared `SliceConfig` contract + `run_with_config()` that executes the CE + synthesis pipeline with slice-specific overrides
  - `step_06_context_engineering/implementation/slices/finance_slice.py` — finance-tuned prompt, exact-number formatting rules, CSV-forced
  - `step_06_context_engineering/implementation/slices/hr_slice.py` — HR-tuned prompt, graph-forced, employee/org query augmentation
  - `step_06_context_engineering/implementation/slices/engineering_slice.py` — engineering-tuned prompt, graph-forced, product-name keyword expansion
  - `step_06_context_engineering/implementation/slices/general_slice.py` — fallback for low-confidence routes
  - `step_06_context_engineering/implementation/reranker.py` — CrossEncoder rerank to top `rerank_k` (default 8)
  - `step_06_context_engineering/implementation/deduplicator.py` — Jaccard near-duplicate removal
  - `step_06_context_engineering/implementation/compressor.py` — sentence-level extractive compression to `compress_ratio` (default 0.60)
  - `step_06_context_engineering/implementation/formatter.py` — XML envelope with explicit token budget
  - `step_06_context_engineering/implementation/context_engineer.py` — `engineer_context()` orchestrates the four stages and emits `ce_metrics`

## How it works
At query time, `router.dispatch()` scores the question against each slice's keyword lexicon and returns the highest-scoring slice along with a confidence in `[0, 1]`. The chosen slice runs the context-engineered pipeline using its own configuration: which system prompt, which query augmentation, which `rerank_k`, which `compress_ratio`. QueryAnalyst runs first to produce sub-questions. RetrievalSpecialist fetches 20 chunks for the (possibly augmented) main question and 10 more for each of up to four sub-questions. GraphNavigator and StructuredData run unconditionally so exact graph and CSV facts are preserved verbatim. The raw chunks then enter `engineer_context()`: the reranker reorders them by question relevance, the deduplicator removes near-duplicates, the compressor keeps the top sentences from each survivor, and the formatter wraps everything (CSV data, graph context, compressed chunks) inside an XML envelope sized to the token budget. The engineered XML is the only context sent to Synthesis using the slice's system prompt; Critic then reviews. The returned `Step06Result` exposes the slice name, router confidence, and engineering metrics.

## Run
```bash
uv run python evaluation/run_eval.py --step step_06_context_engineering
```

## Results
See `step_06_context_engineering/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Step 05 sends up to 20 chunks plus graph and CSV blocks straight to the LLM with one generic system prompt. Many chunks overlap, many contain only one or two relevant sentences, and one prompt cannot be optimal for every domain — finance answers must be exact numbers, HR answers must respect reporting chains, engineering answers must use canonical product names. CrossEncoder reranking promotes the chunks the bi-encoder underweighted; dedup cuts redundancy; compression trims filler sentences; XML formatting gives the model a clearly delimited budget. The VSA router then routes each question to a domain-specific prompt and configuration so each slice can be tuned independently. Pure-keyword routing keeps the dispatch cost near zero, and the exact graph and CSV outputs bypass compression so deterministic facts are never lossy.
