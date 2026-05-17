# Step 11 — Vertical Slice Architecture (VSA)

> **Problem**: One system prompt and one retrieval config can't be optimal for every domain — a finance question needs exact CSV numbers, an HR question needs graph traversal, an engineering question needs product name precision.  
> **Fix**: Route each query to a domain slice that owns its own system prompt, retrieval overrides, and keyword augmentation. Zero extra LLM calls — routing is pure keyword scoring.

## How It Works

```
Query: "What is the total ARR?"
        │
        ▼
  Router  (router.py)
  ┌─────────────────────────────────────────────────────────┐
  │  for each slice: score = keyword_hits / word_count * 4  │
  │                  capped at 1.0                          │
  │  finance_slice.can_handle()     → 1.00                  │
  │  hr_slice.can_handle()          → 0.15  (floor)         │
  │  engineering_slice.can_handle() → 0.15  (floor)         │
  │  general_slice.can_handle()     → 0.15  (floor)         │
  │                                                         │
  │  winner: finance (score 1.00)                           │
  └──────────────────────────────┬──────────────────────────┘
                                 │  SliceConfig
                                 ▼
                    run_with_config(question, config)
                    (base.py — the universal runner)
                         │
                    Step09 agents + Step10 CE pipeline
                    with slice-specific overrides applied
                         │
                         ▼
                      final answer
```

## The Four Slices

| Slice | force_csv | force_graph | Key Prompt Rules |
|-------|-----------|-------------|-----------------|
| **Finance** | ✅ | ❌ | Comma-format numbers, exact labels (closed-won, ARR, etc.) |
| **HR/People** | ❌ | ✅ | "voluntary" / "departed", on-call single-engineer rule |
| **Engineering** | ❌ | ✅ | Canonical product names, full blast-radius chain, SLO precision |
| **General** | ❌ | ❌ | Fallback; always scores ≥ 0.15 so it can win any unmatched query |

## SliceConfig Fields

```python
@dataclass
class SliceConfig:
    name: str               # "finance" | "hr" | "engineering" | "general"
    system_prompt: str      # domain-tuned synthesis instructions
    keywords: list[str]     # routing vocabulary
    force_csv: bool         # activate CSV tool regardless of query_analyst
    force_graph: bool       # activate graph tool regardless of query_analyst
    query_augmentation: str # extra terms appended to retrieval query
    rerank_k: int           # CrossEncoder top-k (default 8)
    compress_ratio: float   # extractive compression retention (default 0.60)
    owns_questions: list[str]  # golden Q IDs owned by this slice (for eval breakdown)
```

## Adding a New Slice

1. Create `step_09_vsa/implementation/slices/myslice.py` with `CONFIG: SliceConfig` and `can_handle(question) -> float`
2. Register it in `router.py` — one line in `_SLICE_MODULES`

Nothing else changes.

## Key Files

| File | What it does |
|------|-------------|
| `implementation/slices/base.py` | `SliceConfig` + `run_with_config()` universal runner |
| `implementation/slices/finance_slice.py` | Finance domain config + keyword scorer |
| `implementation/slices/hr_slice.py` | HR domain config + keyword scorer |
| `implementation/slices/engineering_slice.py` | Engineering domain config + keyword scorer |
| `implementation/slices/general_slice.py` | General fallback config |
| `implementation/router.py` | `route()` + `dispatch()` + `all_slice_configs()` |
| `implementation/pipeline.py` | `Step09RAG` with `query_extended()` → `Step09Result` |

## Run It

```bash
uv run python step_09_vsa/evaluation/run_eval.py
```
