import csv
from pathlib import Path

from step_04_chunking.implementation.types import SmartChunk


def parse_csv(path: Path, source: str, department: str) -> list[SmartChunk]:
    """Parse a CSV file into one SmartChunk per data row."""
    chunks: list[SmartChunk] = []

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = [dict(row) for row in reader]

    for idx, row in enumerate(rows):
        row_text = f"[{source}]\n" + " | ".join(f"{k}: {v}" for k, v in row.items())
        chunks.append(SmartChunk(
            text=row_text,
            source=source,
            department=department,
            format="csv",
            chunk_type="row",
            chunk_index=idx,
            extra={"row_number": idx},
        ))

    return chunks
