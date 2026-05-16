import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SmartChunk:
    text: str
    source: str        # relative path from corpus root
    department: str    # engineering, hr, finance, etc.
    format: str        # txt, md, csv, json
    chunk_type: str    # "row" | "aggregate" | "section" | "prose"
    chunk_index: int
    extra: dict = field(default_factory=dict)
    chunk_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.chunk_id:
            key = f"{self.source}::{self.chunk_type}::{self.chunk_index}"
            digest = hashlib.md5(key.encode()).hexdigest()[:8]
            safe_source = self.source.replace("/", "_").replace(".", "_")
            self.chunk_id = f"{safe_source}__{self.chunk_type}__{self.chunk_index}__{digest}"

    def to_metadata(self) -> dict[str, Any]:
        """
        Return a flat metadata dict suitable for ChromaDB.
        ChromaDB only accepts str / int / float / bool values.
        """
        meta: dict[str, Any] = {
            "source": self.source,
            "department": self.department,
            "format": self.format,
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
        }
        for k, v in self.extra.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
        return meta
