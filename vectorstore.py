"""
vectorstore.py — shared config for Milestone 4 (embedding + retrieval).

Both the embedding step (embed.py) and the query step (retrieve.py) import
from here so they use the *same* model and the *same* distance space. If the
index were built with one model and queried with another, every result would
be wrong — keeping this in one place prevents that class of bug.

  - Embedding model: all-MiniLM-L6-v2 (sentence-transformers, 384-dim, local)
  - Vector store:    ChromaDB persistent collection on disk
  - Distance:        cosine (vectors are L2-normalized at encode time)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_db"          # gitignored; safe to delete and rebuild
COLLECTION_NAME = "uvm_chunks"
CHUNKS_PATH = "chunks.jsonl"       # output of ingest.py


@lru_cache(maxsize=1)
def get_model():
    """Load (and cache) the sentence-transformers model. First call downloads
    ~80 MB to the local HF cache; subsequent calls and runs reuse it."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed(texts: List[str]) -> List[List[float]]:
    """Encode texts into L2-normalized vectors.

    Normalizing means cosine similarity == dot product, which is what the
    collection's cosine space expects. We embed both documents and queries
    through this one function so they live in the same normalized space.
    """
    model = get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vecs.tolist()


@lru_cache(maxsize=1)
def get_client():
    """Persistent ChromaDB client rooted at CHROMA_PATH."""
    import chromadb

    return chromadb.PersistentClient(path=str(Path(CHROMA_PATH)))


def get_collection(create: bool = False):
    """Return the chunk collection.

    create=False (default): fetch the existing collection; raises if the index
    hasn't been built yet (i.e. run embed.py first).
    create=True: get-or-create it, configured for cosine distance.
    """
    client = get_client()
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=COLLECTION_NAME)
