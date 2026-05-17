import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

CHUNK_SIZE_CHARS = 1000   # ~256 tokens
CHUNK_OVERLAP_CHARS = 200
CHROMA_COLLECTION = "vertexia_baseline"

_embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@dataclass
class Chunk:
    text: str
    source: str       # relative path from corpus root
    department: str   # engineering, hr, finance, etc.
    format: str       # txt, md, csv, json
    chunk_index: int
    chunk_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.chunk_id:
            digest = hashlib.md5(f"{self.source}::{self.chunk_index}".encode()).hexdigest()[:8]
            self.chunk_id = f"{self.source.replace('/', '_').replace('.', '_')}__{self.chunk_index}__{digest}"

    def to_metadata(self) -> dict:
        return {
            "source": self.source,
            "department": self.department,
            "format": self.format,
            "chunk_index": self.chunk_index,
        }


def _chunk_text(text: str, source: str, department: str, fmt: str) -> list[Chunk]:
    """Paragraph-aware chunker. Splits on blank lines, accumulates to CHUNK_SIZE_CHARS."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    idx = 0

    for para in paragraphs:
        if len(current) + len(para) + 2 <= CHUNK_SIZE_CHARS:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(Chunk(current, source, department, fmt, idx))
                idx += 1
                # overlap: carry last CHUNK_OVERLAP_CHARS of previous chunk
                current = current[-CHUNK_OVERLAP_CHARS:].strip() + "\n\n" + para
                current = current.strip()
            else:
                # paragraph itself is larger than chunk size — hard split
                for i in range(0, len(para), CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS):
                    piece = para[i:i + CHUNK_SIZE_CHARS]
                    chunks.append(Chunk(piece, source, department, fmt, idx))
                    idx += 1
                current = ""

    if current:
        chunks.append(Chunk(current, source, department, fmt, idx))

    return chunks


def _chunk_csv(path: Path, source: str, department: str) -> list[Chunk]:
    """Each CSV row becomes its own chunk — key for structured data retrieval."""
    chunks: list[Chunk] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row_text = f"[{source}]\n" + " | ".join(f"{k}: {v}" for k, v in row.items())
            chunks.append(Chunk(row_text, source, department, "csv", idx))
    return chunks


def _chunk_json(path: Path, source: str, department: str) -> list[Chunk]:
    with open(path) as f:
        data = json.load(f)
    text = json.dumps(data, indent=2)
    return _chunk_text(text, source, department, "json")


def load_and_chunk(corpus_path: Path) -> list[Chunk]:
    """Walk corpus directory, parse every file, return all chunks."""
    all_chunks: list[Chunk] = []
    for file_path in sorted(corpus_path.rglob("*")):
        if not file_path.is_file():
            continue
        department = file_path.parent.name
        source = str(file_path.relative_to(corpus_path))
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            all_chunks.extend(_chunk_csv(file_path, source, department))
        elif suffix == ".json":
            all_chunks.extend(_chunk_json(file_path, source, department))
        elif suffix in (".txt", ".md"):
            text = file_path.read_text(errors="replace")
            all_chunks.extend(_chunk_text(text, source, department, suffix.lstrip(".")))

    return all_chunks


def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    return _embedder.embed_documents([c.text for c in chunks])


def embed_query(query: str) -> list[float]:
    return _embedder.embed_query(query)


def get_chroma_collection(persist_dir: Path, reset: bool = False) -> chromadb.Collection:
    db = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        try:
            db.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
    return db.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    collection: chromadb.Collection,
    batch_size: int = 100,
) -> None:
    """Upsert chunks + embeddings into ChromaDB in batches."""
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeds = embeddings[i:i + batch_size]
        collection.upsert(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=batch_embeds,
            documents=[c.text for c in batch_chunks],
            metadatas=[c.to_metadata() for c in batch_chunks],
        )


def build_index(corpus_path: Path, persist_dir: Path, reset: bool = False) -> chromadb.Collection:
    """Full ingestion pipeline: load → chunk → embed → store."""
    print(f"Loading and chunking documents from {corpus_path}...")
    chunks = load_and_chunk(corpus_path)
    print(f"  {len(chunks)} chunks from {len(set(c.source for c in chunks))} files")

    collection = get_chroma_collection(persist_dir, reset=reset)
    if not reset and collection.count() > 0:
        print(f"  Index already has {collection.count()} chunks — skipping re-embedding.")
        print("  Pass reset=True to force rebuild.")
        return collection

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)
    print(f"  Done. Embedding dim: {len(embeddings[0])}")

    print("Storing in ChromaDB...")
    store_chunks(chunks, embeddings, collection)
    print(f"  Stored {collection.count()} chunks.")

    return collection


if __name__ == "__main__":
    corpus = Path(__file__).parent.parent.parent / "dataset" / "company_data"
    db_dir = Path(__file__).parent.parent / "results" / "chroma_db"
    build_index(corpus, db_dir, reset=True)
