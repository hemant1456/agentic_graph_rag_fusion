import re
from pathlib import Path

from step_01_baseline_rag.implementation.types import SmartChunk

HEADING_RE = re.compile(r"^(#{1,2})\s+(.+)$")
H1_RE = re.compile(r"^#\s+(.+)$")
MAX_SECTION_CHARS = 3000
FALLBACK_CHUNK_SIZE = 1000
FALLBACK_OVERLAP = 200


def _contextualize(source: str, doc_title: str, section_title: str, body: str) -> str:
    """Prepend document/section context so small chunks carry retrieval signal."""
    header_parts = [f"[FILE: {source}"]
    if doc_title:
        header_parts.append(f"DOC: {doc_title}")
    if section_title and section_title != "__preamble__":
        header_parts.append(f"SECTION: {section_title}")
    header = " | ".join(header_parts) + "]"
    return f"{header}\n\n{body}"


def _paragraph_chunks(
    text: str,
    source: str,
    department: str,
    start_index: int,
    section_title: str,
    doc_title: str,
) -> list[SmartChunk]:
    """Split a long section into paragraph-aware sub-chunks. Used when a section exceeds MAX_SECTION_CHARS."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[SmartChunk] = []
    current = ""
    idx = start_index

    def make_chunk(body: str, i: int) -> SmartChunk:
        return SmartChunk(
            text=_contextualize(source, doc_title, section_title, body),
            source=source,
            department=department,
            format="md",
            chunk_type="section",
            chunk_index=i,
            extra={"section_title": section_title, "doc_title": doc_title},
        )

    for para in paragraphs:
        if len(current) + len(para) + 2 <= FALLBACK_CHUNK_SIZE:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(make_chunk(current, idx))
                idx += 1
                current = current[-FALLBACK_OVERLAP:].strip() + "\n\n" + para
                current = current.strip()
            else:
                for i in range(0, len(para), FALLBACK_CHUNK_SIZE - FALLBACK_OVERLAP):
                    chunks.append(make_chunk(para[i:i + FALLBACK_CHUNK_SIZE], idx))
                    idx += 1
                current = ""

    if current:
        chunks.append(make_chunk(current, idx))

    return chunks


def parse_markdown(path: Path, source: str, department: str) -> list[SmartChunk]:
    """Parse a Markdown file into section-based SmartChunks, prepending doc/section context."""
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    # Extract document title from first H1
    doc_title = ""
    for line in lines:
        m = H1_RE.match(line)
        if m:
            doc_title = m.group(1).strip()
            break

    sections: list[tuple[str, list[str]]] = []
    current_heading = "__preamble__"
    current_lines: list[str] = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            sections.append((current_heading, current_lines))
            current_heading = m.group(2).strip()
            current_lines = [line]
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
                text=_contextualize(source, doc_title, heading, section_text),
                source=source,
                department=department,
                format="md",
                chunk_type="section",
                chunk_index=chunk_index,
                extra={"section_title": heading, "doc_title": doc_title},
            ))
            chunk_index += 1
        else:
            sub = _paragraph_chunks(
                text=section_text,
                source=source,
                department=department,
                start_index=chunk_index,
                section_title=heading,
                doc_title=doc_title,
            )
            chunks.extend(sub)
            chunk_index += len(sub)

    return chunks
