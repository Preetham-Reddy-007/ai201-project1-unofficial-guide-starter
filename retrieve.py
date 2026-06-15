"""
retrieve.py — Milestone 4 (retrieval): query the vector store.

Embeds a query with the same model used for indexing and returns the top-k
most similar chunks with their source metadata. Run it directly to sanity-test
retrieval against the 5 evaluation questions from planning.md *before* wiring
in generation (Milestone 5) — most retrieval bugs surface here.

Usage:
    python retrieve.py                       # run the 5 eval questions
    python retrieve.py "how do I end a test cleanly?"
    python retrieve.py --k 8 "what is a sequencer?"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import vectorstore as vs

# The 5 evaluation questions from planning.md (plain-language phrasing).
EVAL_QUESTIONS = [
    "What is the UVM factory used for, and how do you register a class with it?",
    "How does a UVM driver get a handle to the DUT's signals?",
    "What is the role of objections in ending a UVM test?",
    "What is a UVM sequence and how does it relate to a sequence item and a sequencer?",
    "What does the UVM register abstraction layer (RAL) provide, and what is the adapter for?",
]


@dataclass
class Result:
    rank: int
    score: float        # cosine similarity in [-1, 1]; higher = more similar
    source: str
    title: str
    chunk_index: int
    text: str


def retrieve(query: str, k: int = 5) -> List[Result]:
    """Return the top-k chunks most similar to `query`."""
    collection = vs.get_collection()
    res = collection.query(
        query_embeddings=vs.embed([query]),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    results: List[Result] = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        results.append(
            Result(
                rank=i,
                score=1.0 - dist,  # cosine distance -> similarity
                source=meta["source"],
                title=meta["title"],
                chunk_index=meta["chunk_index"],
                text=doc,
            )
        )
    return results


def _print_results(query: str, results: List[Result], preview: int = 600) -> None:
    print("\n" + "=" * 88)
    print(f"QUERY: {query}")
    print("=" * 88)
    for r in results:
        text = " ".join(r.text.split())  # collapse whitespace for tidy display
        if preview and len(text) > preview:
            text = text[:preview] + "..."  # mark truncation only when it happens
        print(f"\n#{r.rank}  sim={r.score:.3f}  {r.title}  ({r.source} #{r.chunk_index})")
        print(f"    {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-k chunks for a query.")
    parser.add_argument("query", nargs="*", help="Query text (omit to run eval set).")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks (default 5).")
    parser.add_argument(
        "--preview",
        type=int,
        default=600,
        help="Chars of each chunk to show (0 = full text). Default 600.",
    )
    args = parser.parse_args()

    if args.query:
        q = " ".join(args.query)
        _print_results(q, retrieve(q, k=args.k), preview=args.preview)
    else:
        print(f"Running {len(EVAL_QUESTIONS)} evaluation questions (k={args.k})...")
        for q in EVAL_QUESTIONS:
            _print_results(q, retrieve(q, k=args.k), preview=args.preview)


if __name__ == "__main__":
    main()
