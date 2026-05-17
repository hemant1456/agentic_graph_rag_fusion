# Step 06 — Graph RAG + Alias Resolution

> **Problem**: "InsightLens" and "analytics dashboard" refer to the same product — vector search misses this. Blast-radius questions also require traversing the full dependency chain, not just the seed node.  
> **Fix**: Add an alias dictionary and recursive dependency expansion on top of Step 05's graph.

## How It Works

```
User question: "Which customers use the analytics dashboard?"
        │
        ▼
  aliases.py  ──────────────► "analytics dashboard" → "InsightLens"
        │                      (also handles: "data platform" → "DataCraft",
        │                       "messaging" → "NexusFlow", etc.)
        ▼
  graph_query.py
  ┌────────────────────────────────────────────────────────┐
  │  1. seed_nodes  = entity_match(question, aliases)      │
  │  2. neighbours  = expand(seed, depth=2)                │
  │  3. dependency  = full_chain(seed)  ← NEW in Step 06   │
  │     • follows depends_on edges recursively             │
  │     • labels criticality: "critical" / "non-critical"  │
  │  4. aggregate   = compute inline (e.g. enterprise ARR%)│
  └──────────────┬─────────────────────────────────────────┘
                 │  graph_context (str)
                 ▼
  Step04 vector chunks + graph_context → LLM
```

## What's New vs Step 05

| Feature | Step 05 | Step 06 |
|---------|---------|---------|
| Alias resolution | ❌ exact name only | ✅ 40+ alias mappings |
| Dependency traversal | depth-2 BFS | full chain, all paths |
| Aggregate graph queries | ❌ | ✅ e.g. enterprise ARR % |
| Disambiguation | ❌ | ✅ "Phoenix" = product + customer |

## Key Files

| File | What it does |
|------|-------------|
| `implementation/aliases.py` | Static alias dict + `resolve(term)` function |
| `implementation/graph_query.py` | `build_graph_context()` — alias → seed → expand → format |
| `implementation/pipeline.py` | `Step06RAG` — wraps vector retrieval + graph context |

## Results

| Step | PASS | PARTIAL | FAIL | Pass Rate |
|------|------|---------|------|-----------|
| 05 Graph | 18 | 2 | 2 | 82% |
| **06 Graph RAG** | **20** | **1** | **1** | **91%** |

Q18 (two "Phoenix" disambiguation), Q22 (blast-radius full chain) now pass. Q20 remains partial — needs a structured CSV query tool (fixed in Step 07).

## Run It

```bash
uv run python step_06_graph_rag/evaluation/run_eval.py
```
