# Step 09 — Multi-Agent System

> **Problem**: The Step 08 agent is a single LLM deciding everything — query classification, retrieval strategy, graph traversal, synthesis, and verification all in one prompt. It's hard to debug and improve.  
> **Fix**: Split responsibilities across 6 specialised agents, each with a typed input/output contract. An orchestrator routes between them based on query classification.

## How It Works

```
Query
  │
  ▼
┌──────────────────┐
│  QueryAnalyst    │  LLM classifies query type:
│                  │  simple_lookup | aggregation | multi_hop | graph | comparative
│  → QueryAnalysis │  + needs_vector, needs_graph, needs_csv, sub_questions[]
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Orchestrator                                │
│                                                                  │
│   always ──► RetrievalSpecialist   k=10, + sub-question queries  │
│                                                                  │
│   if needs_graph ──► GraphNavigator  entity match + BFS expand  │
│                                                                  │
│   if needs_csv   ──► StructuredData  Pandas intent + query      │
└──────────────┬───────────────────────────────────────────────────┘
               │  (all contexts merged)
               ▼
        ┌──────────────┐
        │  Synthesis   │  builds answer from merged context
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │   Critic     │  LLM validates: approved? confidence? issue?
        │              │  if low-confidence → 1 revision attempt
        └──────┬───────┘
               │
               ▼
           final answer  +  AgentTrace[] (one per agent, for observability)
```

## Agent Contracts (`contracts.py`)

```python
@dataclass
class QueryAnalysis:
    query_type: str        # "simple_lookup" | "aggregation" | "multi_hop" | ...
    needs_vector: bool
    needs_graph:  bool
    needs_csv:    bool
    sub_questions: list[str]
    primary_entities: list[str]

@dataclass
class AgentTrace:
    agent_id: str          # "query_analyst" | "retrieval_specialist" | ...
    input_summary: str
    output_summary: str
    latency_ms: float
    status: str            # "ok" | "error" | "fallback"
```

## Key Files

| File | What it does |
|------|-------------|
| `implementation/agents/contracts.py` | Typed dataclasses shared by all agents |
| `implementation/agents/query_analyst.py` | LLM JSON classification + heuristic fallback |
| `implementation/agents/retrieval_specialist.py` | Dense vector retrieval wrapper |
| `implementation/agents/graph_navigator.py` | Entity match + graph expansion |
| `implementation/agents/structured_data.py` | Pandas CSV query tool |
| `implementation/agents/synthesis.py` | Context → answer generation |
| `implementation/agents/critic.py` | Answer validation + optional revision |
| `implementation/orchestrator.py` | Routes between agents, merges context |
| `implementation/pipeline.py` | `Step09RAG` with standard `.query()` interface |

## Run It

```bash
uv run python step_09_multi_agent/evaluation/run_eval.py
```

> Each question triggers 3–5 LLM calls. Expect ~15–30 s per question.
