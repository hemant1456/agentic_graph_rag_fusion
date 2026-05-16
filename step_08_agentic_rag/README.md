# Step 08 — Agentic RAG via LLM Gateway V2

> **Problem**: Step 07 retrieves everything upfront — it can't decide mid-answer "I need to look up X" or choose a different tool based on what the first retrieval returned.  
> **Fix**: Give the LLM a tool-use loop. It reads pre-retrieved context, then calls tools to fill gaps — up to 3 additional rounds.

## How It Works

```
Query
  │
  ▼
Step07RAG.query()  ──────────────────────────► PRE-RETRIEVED CONTEXT
  (BM25+dense+graph+CSV, full pipeline)         (grounds agent before first LLM call)
  │
  ▼
LLM Gateway V2 (port 8100)
  ┌────────────────────────────────────────────────────────┐
  │  Round 0:  read pre-retrieved context → answer or...  │
  │  Round 1:  tool_call → retrieve("specific query")     │  max 3 rounds
  │  Round 2:  tool_call → csv_query("SUM arr WHERE ...")  │
  │  Round 3:  synthesise final answer                     │
  └────────────────────────────────────────────────────────┘
  │
  ▼
Fallback: Gemini direct (if gateway unreachable)
  uses pre-retrieved context, no tool calls
```

## LLM Gateway V2

A local FastAPI server that routes to free LLM providers with automatic failover:

```
http://localhost:8100
  ├── Gemini 3.1 Flash-Lite Preview  (primary)
  ├── NVIDIA NIM                     (fallback 1)
  ├── Groq                           (fallback 2)
  └── Cerebras                       (fallback 3)
```

All providers are free-tier. The gateway handles rate-limit cooldowns and provider selection transparently. Every step from 08 onward uses `from llm_gatewayV2.client import LLM`.

## Tools Available to the Agent

| Tool | Description |
|------|-------------|
| `retrieve(query, k)` | Dense vector search against Step 04 ChromaDB index |
| `csv_query(intent, filters)` | Pandas-backed structured CSV arithmetic |
| `graph_lookup(entity)` | Graph traversal for org/dependency relationships |

## Key Files

| File | What it does |
|------|-------------|
| `implementation/agent.py` | Tool-calling loop (max 3 rounds), gateway + Gemini fallback |
| `implementation/tools.py` | Tool implementations called by the agent |
| `implementation/pipeline.py` | `Step08RAG` — pre-retrieves then hands off to agent |
| `llm_gatewayV2/main.py` | Gateway FastAPI server |
| `llm_gatewayV2/client.py` | `LLM` client used by all steps 08–12 |

## Results

| Step | PASS | PARTIAL | FAIL | Pass Rate |
|------|------|---------|------|-----------|
| 07 RAG Fusion | 21 | 1 | 0 | 95% |
| **08 Agentic** | **22** | **0** | **0** | **100%** |

Q18 (two-Phoenix disambiguation) now passes — the agent issues two targeted tool calls, one for the engineering migration and one for the sales deal, then synthesises both into a single answer.

## Run It

```bash
# Start gateway first (free, runs locally)
cd llm_gatewayV2 && uv run uvicorn main:app --port 8100 &

uv run python -m step_08_agentic_rag.evaluation.run_eval
```
