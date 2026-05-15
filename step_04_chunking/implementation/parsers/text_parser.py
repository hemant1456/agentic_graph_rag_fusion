"""
Plain-text parser for Step 04 — structure-aware chunking.

Detects section boundaries using visual cues:
  - Lines of repeated =, -, or * (3+ chars)
  - ALL-CAPS lines (look like section headers)
  - Lines ending with : that are short (<60 chars)

Falls back to paragraph-aware chunking (2000 chars, 200 overlap) when no
structure is detected — same as the Step 01 baseline.
"""

import re
from pathlib import Path

from step_04_chunking.implementation.types import SmartChunk

# Regex patterns for section separators
_RULE_RE = re.compile(r"^[=\-*]{3,}\s*$")          # === or --- or ***
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z0-9 \-_/&,\.]{4,}$")  # SECTION HEADER
_COLON_HEADER_RE = re.compile(r"^.{1,59}:\s*$")    # Short line ending with :

CHUNK_SIZE = 2000
OVERLAP = 200
MIN_SECTION_LINES = 3   # a "section" must have at least this many content lines


def _is_section_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(
        _RULE_RE.match(stripped) or
        _ALLCAPS_RE.match(stripped) or
        (len(stripped) < 60 and _COLON_HEADER_RE.match(stripped))
    )


def _paragraph_chunk(
    text: str,
    source: str,
    department: str,
    start_index: int,
    section_title: str = "",
) -> list[SmartChunk]:
    """Standard paragraph-aware chunker (same as Step 01 baseline)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[SmartChunk] = []
    current = ""
    idx = start_index

    for para in paragraphs:
        if len(current) + len(para) + 2 <= CHUNK_SIZE:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(SmartChunk(
                    text=current,
                    source=source,
                    department=department,
                    format="txt",
                    chunk_type="prose",
                    chunk_index=idx,
                    extra={"section_title": section_title} if section_title else {},
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
                        format="txt",
                        chunk_type="prose",
                        chunk_index=idx,
                        extra={"section_title": section_title} if section_title else {},
                    ))
                    idx += 1
                current = ""

    if current:
        chunks.append(SmartChunk(
            text=current,
            source=source,
            department=department,
            format="txt",
            chunk_type="prose",
            chunk_index=idx,
            extra={"section_title": section_title} if section_title else {},
        ))

    return chunks


def parse_text(path: Path, source: str, department: str) -> list[SmartChunk]:
    """
    Parse a plain-text file into SmartChunks.

    Algorithm:
      1. Scan lines for section-separator cues.
      2. If >= 2 separators found → split into sections, chunk each.
      3. Otherwise → fall back to paragraph-aware chunking.
    """
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    # Detect section separators
    sep_indices = [i for i, line in enumerate(lines) if _is_section_separator(line)]

    if len(sep_indices) >= 2:
        return _section_split(lines, sep_indices, source, department)

    # Fallback: paragraph-aware chunking
    return _paragraph_chunk(text, source, department, start_index=0)


def _section_split(
    lines: list[str],
    sep_indices: list[int],
    source: str,
    department: str,
) -> list[SmartChunk]:
    """Split document into sections at detected separator lines."""
    chunks: list[SmartChunk] = []
    chunk_index = 0

    # Build list of (start, end) line ranges for each section
    boundaries = [-1] + sep_indices + [len(lines)]

    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1
        end = boundaries[i + 1]
        section_lines = lines[start:end]

        if not section_lines:
            continue

        # Try to detect a section title: line immediately after separator (if short + non-empty)
        section_title = ""
        content_lines = section_lines
        if section_lines:
            first_nonempty = next((l for l in section_lines if l.strip()), "")
            if first_nonempty and len(first_nonempty.strip()) < 80:
                section_title = first_nonempty.strip()

        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue

        sub = _paragraph_chunk(
            text=section_text,
            source=source,
            department=department,
            start_index=chunk_index,
            section_title=section_title,
        )
        chunks.extend(sub)
        chunk_index += len(sub)

    return chunks
