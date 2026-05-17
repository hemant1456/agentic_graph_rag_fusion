# Step 02 — Format-Aware Chunking

## What it adds
Replaces the baseline paragraph splitter with format-specific parsers: markdown is split by H1/H2 sections, each prefixed with a `[FILE | DOC | SECTION]` contextual header, and CSVs are emitted one row per chunk with column-name context. This is the only change vs Step 01 — same embedder, same dense retrieval. It newly handles Tier 2 questions (Q03-Q04) where the right answer lives inside one section that the baseline chunker shredded.

## Design
- **Class:** `Step02RAG` in `step_02_chunking/implementation/pipeline.py`
- **Inherits from:** reuses `BaselineRAG`'s `generate_answer`, `format_context`, and `RAGResult`
- **Key components:**
  - `step_02_chunking/implementation/chunker.py` — dispatcher that routes files by extension
  - `step_02_chunking/implementation/parsers/markdown_parser.py` — H1/H2 splitter with contextual headers
  - `step_02_chunking/implementation/parsers/csv_parser.py` — row-level chunker
  - `step_02_chunking/implementation/parsers/text_parser.py` — fallback paragraph splitter
  - `step_02_chunking/implementation/ingestor.py` — writes to `chroma_db/` collection `vertexia_smart`

## How it works
Ingestion walks `dataset/company_data/` and dispatches by file extension. Markdown files are parsed into one chunk per H1/H2 section; each chunk is prefixed with `[FILE: <path> | DOC: <doc-title> | SECTION: <heading>]` so the section identity is embedded into the vector itself. CSVs become one chunk per row with column names interleaved. Text files use the baseline paragraph splitter. Retrieval is unchanged: query is embedded with MiniLM, top-k cosine search runs against ChromaDB. Step 02+ all share the `vertexia_smart` collection.

## Run
```bash
uv run python evaluation/run_eval.py --step step_02_chunking
```

## Results
See `step_02_chunking/results/eval_results.json` for the latest RAGAS scores.

## Why this step exists
Step 01's fixed-size paragraph splitter merges unrelated markdown sections into the same chunk. Q03 ("PulseConnect webhook delivery failure first action") returns a chunk that mixes several alert sections, and Q04 ("Datadog sub-processors and retention") mixes Datadog with Snowflake/Stripe rows. Cutting on H1/H2 boundaries and tagging each chunk with its section path isolates the right answer and gives the retriever a much cleaner embedding to match against.
