# Step 05 — Knowledge Graph

> **Problem**: Vector similarity can't answer "who does X report to?" or "which products depend on Y?" — there's no semantic signal to bridge cross-row CSV joins.  
> **Fix**: Build a directed graph from the CSV data. Relationships become first-class edges that can be traversed in one hop.

## How It Works

```
company_data/
  hr/employee_directory.csv ──────────────────► Person nodes  ──► reports_to edges
  engineering/api_dependencies.csv ──────────► Service nodes ──► depends_on edges
  sales/customer_list.csv + csm_account_history ► Customer nodes ► uses / manages_account edges
  finance/vendor_contracts_summary.csv ────────► Vendor nodes  ──► owns_contract edges
                    │
                    ▼ build_graph()
            graph.json (NetworkX DiGraph, persisted)
                    │
          ┌─────────┴──────────┐
          │  query.py          │
          │  1. entity_match() │  exact name → partial alias → keyword fallback
          │  2. expand()       │  BFS up to depth-2 from seed nodes
          │  3. format_context │  human-readable "Person X reports to Y..."
          └─────────┬──────────┘
                    │
              graph_context (str) injected into LLM prompt alongside vector chunks
```

## Node & Edge Types

| Node | Source | Example ID |
|------|--------|-----------|
| Person | employee_directory.csv | `E001` |
| Product | api_dependencies.csv | `NexusFlow` |
| ExternalService | api_dependencies.csv | `external_pulsar` |
| Customer | customer_list.csv | `Phoenix Corp` |
| Vendor | vendor_contracts_summary.csv | `vendor:Snowflake` |

| Edge | Meaning |
|------|---------|
| `reports_to` | Person → Person (org hierarchy) |
| `depends_on` | Service → Service (with criticality attr) |
| `uses` | Customer → Product |
| `manages_account` | Person → Customer (CSM) |
| `owns_contract` | Person → Vendor |

## Key Files

| File | What it does |
|------|-------------|
| `implementation/builder.py` | Parses all CSVs, creates nodes/edges, returns `nx.DiGraph` |
| `implementation/graph_store.py` | `load_or_build()` — caches graph to `results/graph.json` |
| `implementation/query.py` | Entity matching + BFS expansion + text formatting |
| `implementation/pipeline.py` | `Step07RAG` — vector chunks + graph context → LLM |

## Results

| Step | PASS | PARTIAL | FAIL | Pass Rate |
|------|------|---------|------|-----------|
| 01 Baseline | 6 | 4 | 12 | 27% |
| 04 Chunking | 13 | 5 | 4 | 59% |
| **05 Graph** | **18** | **2** | **2** | **82%** |

Questions fixed by the graph: Q13, Q14 (cross-CSV ID joins via `reports_to` edges), Q20, Q22 (dependency chain traversal).

## Run It

```bash
# Build graph + run evaluation (requires GOOGLE_API_KEY)
uv run python step_07_knowledge_graph/evaluation/run_eval.py
```
