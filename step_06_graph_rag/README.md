# Step 06 — Graph RAG (Aliases + Blast-Radius BFS)

## What it adds
Strengthens the Step 05 graph layer with two upgrades: an alias dictionary that maps colloquial product names to canonical IDs (e.g. `LENS` → `InsightLens`), and a full blast-radius BFS that expands beyond one-hop neighbours to all transitively reachable nodes. The retrieval and tool stack is unchanged from Step 05.

## Design
- **Class:** `Step06RAG` in `step_06_graph_rag/implementation/pipeline.py`
- **Inherits from:** `Step05RAG` (only `query()` is overridden)
- **Key components:**
  - `step_06_graph_rag/implementation/aliases.py` — keyword/alias table for products, teams, and customers
  - `step_06_graph_rag/implementation/graph_query.py` — `build_graph_context()` performs alias-resolved entity matching and BFS traversal
  - The graph file written by Step 05 (`step_05_knowledge_graph/results/graph.json`)

## How it works
`Step06RAG.query()` runs hybrid retrieval and the CSV tool exactly as Step 05 does, but swaps `get_graph_context()` for `build_graph_context()`. The new function first resolves aliases against the question and chunk texts (so `LENS`, `Lens dashboard`, and `InsightLens` all map to the same node), then performs a breadth-first walk over the dependency edges to collect every directly or indirectly affected service. The expanded set is rendered into the prompt as a labelled "blast radius" block. The vector and CSV contexts are concatenated as before.

## Run
```bash
uv run python evaluation/run_eval.py --step step_06_graph_rag
```

## Results
See `step_06_graph_rag/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Step 05's one-hop traversal answers direct relations but misses transitive ones. Q12 ("if NexusFlow goes down, which services are affected") requires the full reachable set, not just immediate neighbours. Step 05 also fails when the question uses a colloquial name that does not match the canonical node ID. Aliases fix the entry point into the graph; BFS fixes the depth of traversal. Together they turn fragile graph lookups into robust ones.
