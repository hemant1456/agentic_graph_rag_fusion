"""
Format dispatcher for Step 04 — load_and_chunk() entry point.

Dispatches each file to the appropriate parser based on file extension:
  .csv  → csv_parser  (row chunks + aggregate summary chunk)
  .md   → markdown_parser (section-aware H1/H2/H3 splits)
  .txt  → text_parser (structure-aware, falls back to paragraph chunking)
  .json → pretty-print text, then markdown_parser
  .py   → text_parser (treat as plain text)
"""

import json
from pathlib import Path

from step_04_chunking.implementation.parsers.csv_parser import parse_csv
from step_04_chunking.implementation.parsers.markdown_parser import parse_markdown
from step_04_chunking.implementation.parsers.text_parser import parse_text
from step_04_chunking.implementation.types import SmartChunk


def load_and_chunk(corpus_path: Path) -> list[SmartChunk]:
    """
    Walk corpus directory, parse every recognised file, return all SmartChunks.

    Recognised extensions: .csv, .md, .txt, .json, .py
    Files with other extensions are skipped.
    """
    all_chunks: list[SmartChunk] = []

    for file_path in sorted(corpus_path.rglob("*")):
        if not file_path.is_file():
            continue

        department = file_path.parent.name
        source = str(file_path.relative_to(corpus_path))
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            all_chunks.extend(parse_csv(file_path, source, department))

        elif suffix == ".md":
            all_chunks.extend(parse_markdown(file_path, source, department))

        elif suffix == ".txt":
            all_chunks.extend(parse_text(file_path, source, department))

        elif suffix == ".json":
            # Pretty-print JSON as text, then parse as markdown (sections often appear)
            try:
                data = json.loads(file_path.read_text(errors="replace"))
                json_text = json.dumps(data, indent=2)
            except (json.JSONDecodeError, OSError):
                json_text = file_path.read_text(errors="replace")

            # Write to a temp-like in-memory parse by creating a temporary path wrapper
            # We reuse text_parser since JSON pretty-prints are prose-like
            # But we need a Path object — use a helper that treats the string as text
            _chunks = _chunk_text_string(
                text=json_text,
                source=source,
                department=department,
                fmt="json",
            )
            all_chunks.extend(_chunks)

        elif suffix == ".py":
            all_chunks.extend(parse_text(file_path, source, department))

        # Skip unknown formats silently

    return all_chunks


def _chunk_text_string(
    text: str,
    source: str,
    department: str,
    fmt: str,
) -> list[SmartChunk]:
    """
    Paragraph-aware chunking for in-memory text strings (used for JSON).
    Mirrors the baseline's _chunk_text logic but returns SmartChunks.
    """
    CHUNK_SIZE = 2000
    OVERLAP = 200

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[SmartChunk] = []
    current = ""
    idx = 0

    for para in paragraphs:
        if len(current) + len(para) + 2 <= CHUNK_SIZE:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(SmartChunk(
                    text=current,
                    source=source,
                    department=department,
                    format=fmt,
                    chunk_type="prose",
                    chunk_index=idx,
                ))
                idx += 1
                current = current[-OVERLAP:].strip() + "\n\n" + para
                current = current.strip()
            else:
                for i in range(0, len(para), CHUNK_SIZE - OVERLAP):
                    piece = para[i:i + CHUNK_SIZE]
                    chunks.append(SmartChunk(
                        text=piece,
                        source=source,
                        department=department,
                        format=fmt,
                        chunk_type="prose",
                        chunk_index=idx,
                    ))
                    idx += 1
                current = ""

    if current:
        chunks.append(SmartChunk(
            text=current,
            source=source,
            department=department,
            format=fmt,
            chunk_type="prose",
            chunk_index=idx,
        ))

    return chunks
