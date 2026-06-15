"""
embed.py — Milestone 4 (embedding): build the vector store.

Loads the chunks produced by ingest.py (chunks.jsonl), embeds each chunk with
all-MiniLM-L6-v2, and writes them into a persistent ChromaDB collection along
with source metadata (document name + position) needed for attribution later.

Usage:
    python embed.py                 # build/rebuild the index from chunks.jsonl
    python embed.py --batch 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import vectorstore as vs


def load_chunks(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"{path} not found. Run ingest.py first to produce it.")
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks into ChromaDB.")
    parser.add_argument("--chunks", default=vs.CHUNKS_PATH, help="Chunks JSONL.")
    parser.add_argument("--batch", type=int, default=128, help="Embedding batch size.")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks from {args.chunks}")

    # Rebuild from scratch so re-running is idempotent (no stale/duplicate rows).
    client = vs.get_client()
    try:
        client.delete_collection(vs.COLLECTION_NAME)
        print(f"Dropped existing collection '{vs.COLLECTION_NAME}'")
    except Exception:
        pass  # collection didn't exist yet
    collection = vs.get_collection(create=True)

    print(f"Embedding with {vs.MODEL_NAME} and adding to ChromaDB...")
    for start in range(0, len(chunks), args.batch):
        batch = chunks[start : start + args.batch]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=vs.embed([c["text"] for c in batch]),
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "source": c["source"],
                    "title": c["title"],
                    "chunk_index": c["chunk_index"],
                    "n_tokens": c["n_tokens"],
                }
                for c in batch
            ],
        )
        print(f"  embedded {min(start + args.batch, len(chunks))}/{len(chunks)}")

    print(
        f"\nDone. Collection '{vs.COLLECTION_NAME}' now holds "
        f"{collection.count()} vectors at '{vs.CHROMA_PATH}'."
    )


if __name__ == "__main__":
    main()
