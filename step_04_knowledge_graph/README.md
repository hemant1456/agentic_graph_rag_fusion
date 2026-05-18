# Step 05 — Knowledge Graph

## What it adds
Builds an entity knowledge graph from the structured CSVs (employees, accounts, API dependencies, departures) and uses multi-hop traversal to answer relational questions. Edges include `reports_to`, `manages_account`, `depends_on`, and `uses`. Newly handles Tier 5 questions (Q11-Q13) that require joining facts across multiple CSV rows or files.

## Design
- **Class:** `Step04RAG` in `step_04_knowledge_graph/implementation/pipeline.py`
- **Inherits from:** `Step03HybridRAG` (extends hybrid retrieval with graph context)
- **Key components:**
  - `step_04_knowledge_graph/implementation/builder.py` — reads HR, sales, engineering, and finance CSVs and emits `networkx.DiGraph` nodes/edges
  - `step_04_knowledge_graph/implementation/graph_store.py` — `load_or_build()` caches the graph to `step_04_knowledge_graph/results/graph.json`
  - `step_04_knowledge_graph/implementation/query.py` — `get_graph_context()` extracts a per-question subgraph and renders it as text

## How it works
On `build()`, the pipeline calls `Step03HybridRAG.build()` and then loads (or rebuilds) the graph from `dataset/company_data/`. Nodes are people, products, customers, and vendors; edges encode reports-to, manages-account, depends-on, and uses relations. At query time, hybrid retrieval runs first and produces the top-k chunks. The chunk texts are passed to `get_graph_context()` as seeds: it extracts named entities, walks the graph one or two hops out, and emits a textual block like `Aisha Johnson reports_to Tomás García; Tomás García reports_to Sarah Chen`. The CSV-tool result, the graph context, and the vector context are concatenated and sent to the LLM.

## Run
```bash
uv run python evaluation/run_eval.py --step step_04_knowledge_graph
```

## Results
See `step_04_knowledge_graph/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Hybrid retrieval can find the right CSV rows but it cannot follow a relation across them. Q11 ("CSM for Phoenix Corp, and their manager") needs two joins: `customer → CSM` then `CSM → manager`. Q12 ("if NexusFlow goes down, what is affected") needs a BFS over the dependency graph. Q13 ("two-hop reporting chain for Aisha Johnson") is a pure graph walk. Encoding relations as first-class edges makes these answers deterministic instead of relying on the LLM to chain implicit references.

<!-- RESULTS_DETAIL_START -->

## Eval results

**Run summary** — 10 PASS · 3 PARTIAL · 1 FAIL out of 14 questions (71% pass rate).

RAGAS averages:

| answer_correctness | faithfulness | answer_relevancy | context_precision | context_recall |
|---:|---:|---:|---:|---:|
| 0.807 | 0.986 | 0.900 | 0.286 | 0.757 |

### Per-question detail

| ID | Grade | correctness | Fixed-by step | Notes |
|---|---|---:|---|---|
| **Q01** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q02** | PASS | 1.00 | `step_01_baseline_rag` | Continues to PASS from an earlier tier — capability still works. |
| **Q03** | PARTIAL | 0.50 | `step_01_baseline_rag` | Regression: was solvable at step 1; now only PARTIAL. The retrieved context lacks the specific runbook instructions for the alert, leading to a hallucination of the first action, though the owner was correctly ide… |
| **Q04** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q05** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q06** | PASS | 1.00 | `step_02_tools` | Continues to PASS from an earlier tier — capability still works. |
| **Q07** | PASS | 1.00 | `step_03_hybrid_retrieval` | Continues to PASS from an earlier tier — capability still works. |
| **Q08** | PASS | 0.80 | `step_03_hybrid_retrieval` | Continues to PASS from an earlier tier — capability still works. |
| **Q09** | PASS | 1.00 | `step_03_hybrid_retrieval` | Continues to PASS from an earlier tier — capability still works. |
| **Q10** | PASS | 1.00 | `step_04_knowledge_graph` | Pass-tier hits as designed — the step's new capability surfaces the required fact(s). |
| **Q11** | PARTIAL | 0.60 | `step_04_knowledge_graph` | Should PASS at this tier but only PARTIAL. The actual answer correctly identifies all three services but adds a non-existent 'DataCraft (Integration)' and misidentifies endpoints; it is mostly grounded but includes minor h… |
| **Q12** | PARTIAL | 0.40 | `step_04_knowledge_graph` | Should PASS at this tier but only PARTIAL. The answer correctly identifies Priya Nair but misses four other employees; all claims are grounded in context, though the retrieved context contains enough information to infer t… |
| **Q13** | FAIL | 0.00 | `step_05_multi_agent` | Expected FAIL — required capability arrives at step 5. |
| **Q14** | PASS | 1.00 | `step_05_multi_agent` | Unexpected PASS — question targets step 5's capability, but retrieved context happened to contain enough signal. |

> Each question's text + reference answer lives in `step_01_baseline_rag/evaluation/golden_questions.py`. The full per-question JSON (including the judge's reasoning) is in `results/eval_results.json`.

<!-- RESULTS_DETAIL_END -->
