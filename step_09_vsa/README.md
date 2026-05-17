# Step 09 — Vertical Slice Architecture

## What it adds
Routes each question to a domain slice (Finance, HR, Engineering, or General) that owns its own system prompt, retrieval overrides, and keyword augmentation. Routing is pure keyword scoring with a confidence value — no extra LLM call. The pipeline beneath each slice is the Step 08 context-engineered stack.

## Design
- **Class:** `Step09RAG` in `step_09_vsa/implementation/pipeline.py`
- **Inherits from:** composes `Step04HybridRAG` (k=20) and the Step 05 knowledge graph; reuses Step 08's `engineer_context`
- **Key components:**
  - `step_09_vsa/implementation/router.py` — `dispatch()` scores slices by keyword overlap and returns the winning slice plus confidence
  - `step_09_vsa/implementation/slices/finance_slice.py` — finance-tuned prompt, CSV-first context order
  - `step_09_vsa/implementation/slices/hr_slice.py` — HR-tuned prompt, graph-first context order
  - `step_09_vsa/implementation/slices/engineering_slice.py` — engineering-tuned prompt, product-name keyword expansion
  - `step_09_vsa/implementation/slices/general_slice.py` — fallback for low-confidence routes
  - `step_09_vsa/implementation/slices/base.py` — shared slice contract

## How it works
At query time, `router.dispatch()` scores the question against each slice's keyword lexicon and returns the highest-scoring slice along with a confidence in `[0, 1]`. The chosen slice runs the context-engineered pipeline using its own configuration: which retrievers to weight more heavily, which agents to include, which system prompt to use for Synthesis. If no slice clears the confidence threshold, `general_slice` runs the unmodified Step 08 path. The returned `Step09Result` exposes the slice name, router confidence, and engineering metrics.

## Run
```bash
uv run python evaluation/run_eval.py --step step_09_vsa
```

## Results
See `step_09_vsa/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
One system prompt cannot be optimal for every domain. Finance answers must be exact numbers; HR answers must respect reporting chains; engineering answers must use canonical product names. A single prompt that tries to cover all three either over-hedges or under-specifies. Per-slice prompts and retrieval weights let each domain be tuned independently, and pure-keyword routing keeps the dispatch cost near zero.
