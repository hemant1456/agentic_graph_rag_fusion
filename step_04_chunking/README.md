# Step 04 — Format-aware Parsing & Chunking

## Goal

The baseline (Step 01) fails all 12 aggregation questions because `k=5` cosine retrieval returns only 5 rows from a 20-row CSV — the model never sees the remaining rows and cannot compute totals.

Step 04 fixes this by adding **computed-aggregate summary chunks**: before embedding, each CSV file gets an extra chunk that pre-computes sums, group breakdowns, date-period aggregations, and per-row ratios over the _entire table_. When the LLM retrieves this single chunk it has everything needed to answer aggregation questions.

## Architecture

```
step_04_chunking/
  implementation/
    types.py           SmartChunk dataclass (row | aggregate | section | prose)
    parsers/
      csv_parser.py    Row chunks + 1 aggregate summary chunk per CSV
      markdown_parser.py  H1/H2/H3 section splits
      text_parser.py   ALL-CAPS / rule-line section detection, paragraph fallback
    chunker.py         Format dispatcher, load_and_chunk()
    ingestor.py        Gemini embedding + ChromaDB "vertexia_step04"
    pipeline.py        Step04RAG — same .build()/.query() interface
  evaluation/
    run_eval.py        22 golden questions → eval_results.json
  tests/
    test_parsers.py    10 unit tests (no API calls)
  results/             Created at runtime
```

## Chunking Strategies

### CSV — `csv_parser.py`

Every CSV produces two kinds of chunk:

1. **Row chunks** (`chunk_type="row"`) — one per data row, identical format to the baseline. Preserved for precise single-row lookups.

2. **Aggregate chunk** (`chunk_type="aggregate"`) — one per file, built by a general algorithm:
   - Detects column types (numeric, date, categorical, name, location) by trying to parse each value at runtime — no hardcoded logic.
   - Computes numeric column totals.
   - For categorical columns (cardinality 2–15): group counts + group sums + percentages of total.
   - For date × numeric pairs: year totals, half-year (H1/H2), and quarterly breakdowns.
   - For 2+ numeric columns: per-row ratios identifying the highest-ratio row (answers "which department has the highest budget-per-headcount?" questions).
   - Lists all values for "name" columns (all-unique, <30 rows).
   - Special location column: counts employees per office.

**Why this fixes the FAIL questions:**
- Q07 (total ARR $11M): aggregate chunk contains `arr_usd: 11,000,000`
- Q08 (total vendor spend $956,400): aggregate chunk contains `annual_value_usd: 956,400`
- Q09 (Berlin headcount = 5): aggregate chunk contains `Berlin: 5 employees`
- Q10 (highest budget/HC = Platform Engineering): aggregate chunk lists all ratios + `Highest: Platform Engineering at 195238/unit`
- Q11 (Q3 Closed-Won deals = 7, $1,692K): aggregate chunk contains stage breakdown + ARR totals
- Q12 (enterprise % = 65%): aggregate chunk contains segment breakdown with percentages
- Q15 (engineering budget % = 59%): aggregate chunk has total budget denominator
- Q16 (H2 2023 ARR = $3,120K): aggregate chunk has H2 date-period aggregations
- Q19 (total headcount = 181): aggregate chunk has `headcount: 181`
- Q20 (Q3 2023 revenue = $4,120K): aggregate chunk has Q3 2023 totals

### Markdown — `markdown_parser.py`

Splits on `## Heading` / `### Subheading` lines. Each heading + following content = one `section` chunk. Sections exceeding 3000 chars are sub-chunked paragraph-by-paragraph (2000 chars, 200 overlap).

**Why this is better:** The baseline splits markdown at 2000-char boundaries that break mid-section. Section-aware chunks keep related content together and avoid retrieving half a procedure.

### Plain Text — `text_parser.py`

Detects section boundaries using:
- Lines of `===`, `---`, or `***` (visual rules)
- ALL-CAPS lines (typical section headers in .txt reports)
- Short lines ending with `:` (sub-section labels)

Falls back to paragraph-aware chunking (2000 chars, 200 overlap) when no structure is detected.

## Expected Improvements vs Baseline

| Question | Baseline | Step 04 | Why |
|---|---|---|---|
| Q07 total ARR | FAIL | PASS | Aggregate chunk has sum |
| Q08 vendor spend | FAIL | PASS | Aggregate chunk has sum |
| Q09 Berlin headcount | FAIL | PASS | Aggregate chunk has location breakdown |
| Q10 budget/HC ratio | FAIL | PASS | Aggregate chunk has per-row ratios |
| Q11 Q3 Closed-Won | FAIL | PASS | Aggregate chunk has stage breakdown + ARR |
| Q12 enterprise % | FAIL | PASS | Aggregate chunk has segment % |
| Q15 eng budget % | FAIL | PASS | Aggregate chunk has total budget |
| Q16 H2 2023 ARR | FAIL | PASS | Aggregate chunk has H2 date-period |
| Q19 total HC | FAIL | PASS | Aggregate chunk has HC sum |
| Q20 Q3 2023 revenue | FAIL | PASS | Aggregate chunk has Q3 date-period |

Q13/Q14 remain FAIL (graph traversal, not chunking). Q17/Q18/Q21/Q22 remain PARTIAL (multi-hop reasoning).

Expected: ~16 PASS / 4 PARTIAL / 2 FAIL (vs baseline 6 PASS / 4 PARTIAL / 12 FAIL)

## How to Run

### Unit tests (no API keys required)
```bash
uv run pytest step_04_chunking/tests/ -v
```

### Verify chunk counts
```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from step_04_chunking.implementation.chunker import load_and_chunk
corpus = Path('step_00_dataset/company_data')
chunks = load_and_chunk(corpus)
types = {}
for c in chunks:
    types[c.chunk_type] = types.get(c.chunk_type, 0) + 1
print('Total chunks:', len(chunks))
print('By type:', types)
"
```

### Build index (requires GOOGLE_API_KEY)
```bash
uv run python step_04_chunking/implementation/ingestor.py
```

### Run full evaluation (requires GOOGLE_API_KEY + ANTHROPIC_API_KEY)
```bash
uv run python step_04_chunking/evaluation/run_eval.py
```
