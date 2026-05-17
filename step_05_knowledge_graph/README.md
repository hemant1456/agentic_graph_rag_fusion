# Step 05 — Knowledge Graph

## What it adds
Builds an entity knowledge graph from the structured CSVs (employees, accounts, API dependencies, departures) and uses multi-hop traversal to answer relational questions. Edges include `reports_to`, `manages_account`, `depends_on`, and `uses`. Newly handles Tier 5 questions (Q11-Q13) that require joining facts across multiple CSV rows or files.

## Design
- **Class:** `Step05RAG` in `step_05_knowledge_graph/implementation/pipeline.py`
- **Inherits from:** `Step04HybridRAG` (extends hybrid retrieval with graph context)
- **Key components:**
  - `step_05_knowledge_graph/implementation/builder.py` — reads HR, sales, engineering, and finance CSVs and emits `networkx.DiGraph` nodes/edges
  - `step_05_knowledge_graph/implementation/graph_store.py` — `load_or_build()` caches the graph to `step_05_knowledge_graph/results/graph.json`
  - `step_05_knowledge_graph/implementation/query.py` — `get_graph_context()` extracts a per-question subgraph and renders it as text

## How it works
On `build()`, the pipeline calls `Step04HybridRAG.build()` and then loads (or rebuilds) the graph from `dataset/company_data/`. Nodes are people, products, customers, and vendors; edges encode reports-to, manages-account, depends-on, and uses relations. At query time, hybrid retrieval runs first and produces the top-k chunks. The chunk texts are passed to `get_graph_context()` as seeds: it extracts named entities, walks the graph one or two hops out, and emits a textual block like `Aisha Johnson reports_to Tomás García; Tomás García reports_to Sarah Chen`. The CSV-tool result, the graph context, and the vector context are concatenated and sent to the LLM.

## Run
```bash
uv run python evaluation/run_eval.py --step step_05_knowledge_graph
```

## Results
See `step_05_knowledge_graph/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Hybrid retrieval can find the right CSV rows but it cannot follow a relation across them. Q11 ("CSM for Phoenix Corp, and their manager") needs two joins: `customer → CSM` then `CSM → manager`. Q12 ("if NexusFlow goes down, what is affected") needs a BFS over the dependency graph. Q13 ("two-hop reporting chain for Aisha Johnson") is a pure graph walk. Encoding relations as first-class edges makes these answers deterministic instead of relying on the LLM to chain implicit references.
