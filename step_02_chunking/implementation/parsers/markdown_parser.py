import re
from pathlib import Path

from step_02_chunking.implementation.types import SmartChunk

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
MAX_SECTION_CHARS = 3000
FALLBACK_CHUNK_SIZE = 2000
FALLBACK_OVERLAP = 200


def _paragraph_chunks(
    text: str,
    source: str,
    department: str,
    start_index: int,
    section_title: str,
) -> list[SmartChunk]:
    """Split a long section into paragraph-aware sub-chunks. Used when a section exceeds MAX_SECTION_CHARS."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[SmartChunk] = []
    current = ""
    idx = start_index

    for para in paragraphs:
        if len(current) + len(para) + 2 <= FALLBACK_CHUNK_SIZE:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(SmartChunk(
                    text=current,
                    source=source,
                    department=department,
                    format="md",
                    chunk_type="section",
                    chunk_index=idx,
                    extra={"section_title": section_title},
                ))
                idx += 1
                current = current[-FALLBACK_OVERLAP:].strip() + "\n\n" + para
                current = current.strip()
            else:
                for i in range(0, len(para), FALLBACK_CHUNK_SIZE - FALLBACK_OVERLAP):
                    piece = para[i:i + FALLBACK_CHUNK_SIZE]
                    chunks.append(SmartChunk(
                        text=piece,
                        source=source,
                        department=department,
                        format="md",
                        chunk_type="section",
                        chunk_index=idx,
                        extra={"section_title": section_title},
                    ))
                    idx += 1
                current = ""

    if current:
        chunks.append(SmartChunk(
            text=current,
            source=source,
            department=department,
            format="md",
            chunk_type="section",
            chunk_index=idx,
            extra={"section_title": section_title},
        ))

    return chunks


def parse_markdown(path: Path, source: str, department: str) -> list[SmartChunk]:
    """
    Parse a Markdown file into section-based SmartChunks.

    Algorithm:
      1. Walk lines looking for H1/H2/H3 headings.
      2. Accumulate content from heading until the next heading.
      3. If a section > MAX_SECTION_CHARS, sub-chunk it paragraph-by-paragraph.
    """
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    sections: list[tuple[str, list[str]]] = []
    current_heading = "__preamble__"
    current_lines: list[str] = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            sections.append((current_heading, current_lines))
            current_heading = m.group(2).strip()
            current_lines = [line]  # include the heading line itself
        else:
            current_lines.append(line)

    sections.append((current_heading, current_lines))

    chunks: list[SmartChunk] = []
    chunk_index = 0

    for heading, content_lines in sections:
        section_text = "\n".join(content_lines).strip()
        if not section_text:
            continue

        if len(section_text) <= MAX_SECTION_CHARS:
            chunks.append(SmartChunk(
                text=section_text,
                source=source,
                department=department,
                format="md",
                chunk_type="section",
                chunk_index=chunk_index,
                extra={"section_title": heading},
            ))
            chunk_index += 1
        else:
            sub = _paragraph_chunks(
                text=section_text,
                source=source,
                department=department,
                start_index=chunk_index,
                section_title=heading,
            )
            chunks.extend(sub)
            chunk_index += len(sub)

    return chunks
