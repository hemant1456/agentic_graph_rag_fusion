from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from step_01_baseline_rag.implementation.types import SmartChunk

CHROMA_COLLECTION = "vertexia_smart"

# Legacy constant kept for backward compat (used by some tests).
CHUNK_SIZE_CHARS = 2000

_embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def embed_chunks(chunks: list[SmartChunk]) -> list[list[float]]:
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
    chunks: list[SmartChunk],
    embeddings: list[list[float]],
    collection: chromadb.Collection,
    batch_size: int = 100,
) -> None:
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeds = embeddings[i:i + batch_size]
        collection.upsert(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=batch_embeds,
            documents=[c.text for c in batch_chunks],
            metadatas=[c.to_metadata() for c in batch_chunks],
        )


def build_index(
    corpus_path: Path,
    persist_dir: Path,
    reset: bool = False,
) -> chromadb.Collection:
    from step_01_baseline_rag.implementation.chunker import load_and_chunk

    print(f"Loading and chunking {corpus_path}...")
    chunks = load_and_chunk(corpus_path)

    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    print(f"  {len(chunks)} chunks from {len(set(c.source for c in chunks))} files")
    print(f"  By type: {by_type}")

    collection = get_chroma_collection(persist_dir, reset=reset)
    if not reset and collection.count() > 0:
        print(f"  Index already has {collection.count()} chunks — skipping re-embedding.")
        return collection

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)
    print(f"  Done. Dim: {len(embeddings[0])}")

    print("Storing in ChromaDB...")
    store_chunks(chunks, embeddings, collection)
    print(f"  Stored {collection.count()} chunks.")

    return collection


# Re-export load_and_chunk so callers can `from step_01_baseline_rag.implementation.ingest import load_and_chunk`.
def load_and_chunk(corpus_path: Path) -> list[SmartChunk]:
    from step_01_baseline_rag.implementation.chunker import load_and_chunk as _impl
    return _impl(corpus_path)


if __name__ == "__main__":
    corpus = Path(__file__).parent.parent.parent / "dataset" / "company_data"
    db_dir = Path(__file__).parent.parent.parent / "chroma_db"
    build_index(corpus, db_dir, reset=True)
