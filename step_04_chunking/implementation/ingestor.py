"""
Embedding and ChromaDB storage for Step 04.

Uses the same Gemini embedding model as Step 01 (gemini-embedding-2).
Collection name: "vertexia_step04"
Persist dir: step_04_chunking/results/chroma_db

Embedding functions are self-contained (not imported from step_01) to
avoid tight coupling, but use identical API calls.
"""

import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

from step_04_chunking.implementation.types import SmartChunk

load_dotenv()

GEMINI_EMBED_MODEL = "gemini-embedding-2"
CHROMA_COLLECTION = "vertexia_step04"
EMBED_BATCH_PAUSE_EVERY = 50   # pause every N chunks to avoid rate limits


def embed_chunks(chunks: list[SmartChunk]) -> list[list[float]]:
    """
    Embed all SmartChunks using Google gemini-embedding-2.
    One API call per chunk (same as baseline). Returns list of embedding vectors.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set")
    client = genai.Client(api_key=api_key)

    embeddings: list[list[float]] = []
    for i, chunk in enumerate(chunks):
        response = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=chunk.text,
        )
        embeddings.append(response.embeddings[0].values)
        if (i + 1) % EMBED_BATCH_PAUSE_EVERY == 0:
            print(f"    {i + 1}/{len(chunks)} embedded...")
            time.sleep(0.5)

    return embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set")
    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(model=GEMINI_EMBED_MODEL, contents=query)
    return response.embeddings[0].values


def get_chroma_collection(persist_dir: Path, reset: bool = False) -> chromadb.Collection:
    """Get or create the step04 ChromaDB collection."""
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


def build_index(
    corpus_path: Path,
    persist_dir: Path,
    reset: bool = False,
) -> chromadb.Collection:
    """Full ingestion pipeline: load → chunk → embed → store."""
    from step_04_chunking.implementation.chunker import load_and_chunk

    print(f"Loading and chunking documents from {corpus_path}...")
    chunks = load_and_chunk(corpus_path)

    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    print(f"  {len(chunks)} chunks from {len(set(c.source for c in chunks))} files")
    print(f"  By type: {by_type}")

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
    corpus = Path(__file__).parent.parent.parent / "step_00_dataset" / "company_data"
    db_dir = Path(__file__).parent.parent / "results" / "chroma_db"
    build_index(corpus, db_dir, reset=True)
