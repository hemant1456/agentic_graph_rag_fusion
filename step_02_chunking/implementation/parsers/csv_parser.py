import csv
from pathlib import Path

from step_02_chunking.implementation.types import SmartChunk


def _filename_to_title(source: str) -> str:
    """vendor_contracts_summary.csv → 'Vendor Contracts Summary'."""
    stem = Path(source).stem
    return stem.replace("_", " ").replace("-", " ").title()


def parse_csv(path: Path, source: str, department: str) -> list[SmartChunk]:
    """Parse a CSV file into one SmartChunk per data row, with table context header."""
    chunks: list[SmartChunk] = []
    doc_title = _filename_to_title(source)

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = [dict(row) for row in reader]

    for idx, row in enumerate(rows):
        header = f"[FILE: {source} | TABLE: {doc_title} | ROW: {idx + 1}]"
        row_text = header + "\n" + " | ".join(f"{k}: {v}" for k, v in row.items())
        chunks.append(SmartChunk(
            text=row_text,
            source=source,
            department=department,
            format="csv",
            chunk_type="row",
            chunk_index=idx,
            extra={"row_number": idx, "doc_title": doc_title},
        ))

    return chunks
