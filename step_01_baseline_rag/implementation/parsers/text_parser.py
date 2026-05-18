import re
from pathlib import Path

from step_01_baseline_rag.implementation.types import SmartChunk

_RULE_RE = re.compile(r"^[=\-*]{3,}\s*$")
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z0-9 \-_/&,\.]{4,}$")
_COLON_HEADER_RE = re.compile(r"^.{1,59}:\s*$")

CHUNK_SIZE = 1000
OVERLAP = 200


def _filename_to_title(source: str) -> str:
    """vendor_contracts_summary.csv → 'Vendor Contracts Summary'."""
    stem = Path(source).stem
    return stem.replace("_", " ").replace("-", " ").title()


def _contextualize(source: str, doc_title: str, section_title: str, body: str) -> str:
    """Prepend document/section context so small chunks carry retrieval signal."""
    header_parts = [f"[FILE: {source}", f"DOC: {doc_title}"]
    if section_title:
        header_parts.append(f"SECTION: {section_title}")
    header = " | ".join(header_parts) + "]"
    return f"{header}\n\n{body}"


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
    doc_title: str,
    section_title: str = "",
) -> list[SmartChunk]:
    """Standard paragraph-aware chunker, with prepended context header."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[SmartChunk] = []
    current = ""
    idx = start_index

    def make_chunk(body: str, i: int) -> SmartChunk:
        return SmartChunk(
            text=_contextualize(source, doc_title, section_title, body),
            source=source,
            department=department,
            format="txt",
            chunk_type="prose",
            chunk_index=i,
            extra={"section_title": section_title, "doc_title": doc_title} if section_title else {"doc_title": doc_title},
        )

    for para in paragraphs:
        if len(current) + len(para) + 2 <= CHUNK_SIZE:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(make_chunk(current, idx))
                idx += 1
                current = current[-OVERLAP:].strip() + "\n\n" + para
                current = current.strip()
            else:
                for i in range(0, len(para), CHUNK_SIZE - OVERLAP):
                    chunks.append(make_chunk(para[i:i + CHUNK_SIZE], idx))
                    idx += 1
                current = ""

    if current:
        chunks.append(make_chunk(current, idx))

    return chunks


def parse_text(path: Path, source: str, department: str) -> list[SmartChunk]:
    """Parse a plain-text file into SmartChunks with contextual headers."""
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    doc_title = _filename_to_title(source)

    sep_indices = [i for i, line in enumerate(lines) if _is_section_separator(line)]

    if len(sep_indices) >= 2:
        return _section_split(lines, sep_indices, source, department, doc_title)

    return _paragraph_chunk(text, source, department, start_index=0, doc_title=doc_title)


def _section_split(
    lines: list[str],
    sep_indices: list[int],
    source: str,
    department: str,
    doc_title: str,
) -> list[SmartChunk]:
    """Split document into sections at detected separator lines."""
    chunks: list[SmartChunk] = []
    chunk_index = 0

    boundaries = [-1] + sep_indices + [len(lines)]

    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1
        end = boundaries[i + 1]
        section_lines = lines[start:end]

        if not section_lines:
            continue

        section_title = ""
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
            doc_title=doc_title,
            section_title=section_title,
        )
        chunks.extend(sub)
        chunk_index += len(sub)

    return chunks
