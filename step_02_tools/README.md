# Step 03 — CSV Tools

## What it adds
Introduces a structured-data tool: a pandas-backed CSV query function that runs alongside dense retrieval. When the question matches an aggregate intent (total ARR, Q3 revenue, headcount by city), the tool computes the exact answer from the source CSV and prepends the result to the LLM context. Newly handles Tier 3 questions (Q05-Q07) where vector retrieval can find the relevant rows but cannot sum them.

## Design
- **Class:** `Step02ToolsRAG` in `step_02_tools/implementation/pipeline.py`
- **Inherits from:** composes `BaselineRAG`'s ChromaDB collection (`vertexia_smart`), retrieval, and generation
- **Key components:**
  - `step_02_tools/implementation/csv_tool.py` — `detect_intent()` regex/keyword router and `run_query()` pandas executor
  - Step 02's `vertexia_smart` ChromaDB collection (no re-ingest)

## How it works
At query time, the pipeline runs the Step 02 dense retrieval to produce a vector context. In parallel, `detect_intent(question)` pattern-matches the question against a small registry of aggregate intents (total ARR, quarterly revenue sum, employee count by location, etc.). If an intent matches, `run_query(intent)` loads the relevant CSV from `dataset/company_data/` with pandas, computes the aggregate, and renders it as a labelled text block. The CSV result is prepended to the vector context, then both are sent to the LLM. If no intent matches, the pipeline degrades to pure dense retrieval.

## Run
```bash
uv run python evaluation/run_eval.py --step step_02_tools
```

## Results
See `step_02_tools/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Dense retrieval can locate the customer-list CSV chunks for Q05 but cannot sum 20 ARR values. The LLM can sometimes add small numbers it sees in context, but it routinely truncates the CSV, hallucinates rows, or drops precision. A deterministic pandas call is both correct and traceable. This step makes the structured side of the corpus first-class instead of treating CSVs as if they were prose.
