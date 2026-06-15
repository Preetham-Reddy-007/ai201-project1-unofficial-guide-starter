"""
query.py — Milestone 5 (grounded generation): retrieval -> LLM -> cited answer.

Ties the pipeline together: retrieve top-k chunks (Milestone 4), feed them to
Groq's llama-3.3-70b-versatile as the ONLY allowed source of truth, and return
an answer plus the source documents it was drawn from.

Grounding is enforced in two independent ways, so a single weak link can't
produce an ungrounded answer:

  1. Programmatic relevance gate — if retrieval finds nothing similar enough
     (best cosine similarity below RELEVANCE_FLOOR), we return the "not enough
     information" message WITHOUT calling the LLM. The model never gets a chance
     to answer an out-of-domain question from its training knowledge.
  2. Strict system prompt — the model is told to answer ONLY from the provided
     context and to return a fixed refusal string otherwise.

Source attribution is built from the retrieved chunks in code (see `ask`), so
it is guaranteed regardless of whether the model remembers to cite.

Usage:
    python query.py "what is the UVM factory for?"
    python query.py            # runs a few demo queries incl. an out-of-domain one
"""

from __future__ import annotations

import os
import sys
from typing import List

from retrieve import Result, retrieve

MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
# Below this best-match cosine similarity we treat the corpus as not covering
# the question and decline. Tuned against observed scores: on-topic DV queries
# land ~0.5-0.78; unrelated questions fall far below this.
RELEVANCE_FLOOR = 0.30
REFUSAL = "I don't have enough information on that."

SYSTEM_PROMPT = (
    "You are a precise assistant for UVM / SystemVerilog design verification. "
    "Answer the user's question using ONLY the numbered source excerpts provided "
    "in the context. Follow these rules strictly:\n"
    "1. Use only facts stated in the context. Do NOT use any outside or prior "
    "knowledge, and do NOT invent class names, macros, methods, or APIs that do "
    "not appear in the context.\n"
    f"2. If the context does not contain enough information to answer, reply with "
    f'EXACTLY this sentence and nothing else: "{REFUSAL}"\n'
    "3. Cite the source title(s) you used, e.g. (Source: UVM Basics). Cite only "
    "titles that appear in the context.\n"
    "4. Be concise and technical."
)


def build_context(chunks: List[Result]) -> str:
    """Render retrieved chunks as a numbered, source-labeled context block."""
    blocks = []
    for c in chunks:
        blocks.append(
            f"[{c.rank}] (Source: {c.title} | file: {c.source})\n{c.text}"
        )
    return "\n\n".join(blocks)


def _unique_sources(chunks: List[Result]) -> List[str]:
    """De-duplicated, rank-ordered source labels for programmatic attribution."""
    seen, out = set(), []
    for c in chunks:
        label = f"{c.title} ({c.source})"
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def _get_client():
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key or key == "your_key_here":
        sys.exit(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get a free one at https://console.groq.com)."
        )
    return Groq(api_key=key)


def ask(question: str, k: int = TOP_K) -> dict:
    """Answer `question` grounded in the retrieved corpus.

    Returns {"answer": str, "sources": [str], "chunks": [Result], "grounded": bool}.
    `sources` always reflects the chunks actually retrieved (code-guaranteed
    attribution); it is empty when the question is declined as out-of-scope.
    """
    chunks = retrieve(question, k=k)

    # Gate 1: nothing relevant retrieved -> decline without invoking the LLM.
    if not chunks or chunks[0].score < RELEVANCE_FLOOR:
        return {"answer": REFUSAL, "sources": [], "chunks": chunks, "grounded": True}

    client = _get_client()
    user_msg = (
        f"CONTEXT:\n{build_context(chunks)}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the context above."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,  # deterministic, no creative drift away from the context
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = resp.choices[0].message.content.strip()

    declined = answer.strip().rstrip(".").lower() == REFUSAL.rstrip(".").lower()
    sources = [] if declined else _unique_sources(chunks)
    return {"answer": answer, "sources": sources, "chunks": chunks, "grounded": True}


def _demo() -> None:
    queries = [
        "What is the UVM factory used for, and how do you register a class with it?",
        "What is the role of objections in ending a UVM test?",
        "What is the best pizza topping?",  # out-of-domain: should be declined
    ]
    for q in queries:
        res = ask(q)
        print("\n" + "=" * 88)
        print(f"Q: {q}")
        print("-" * 88)
        print(res["answer"])
        if res["sources"]:
            print("\nRetrieved from:")
            for s in res["sources"]:
                print(f"  - {s}")
        else:
            print("\n(no sources — question declined as out-of-scope)")


def main() -> None:
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        res = ask(q)
        print(res["answer"])
        if res["sources"]:
            print("\nRetrieved from:")
            for s in res["sources"]:
                print(f"  - {s}")
    else:
        _demo()


if __name__ == "__main__":
    main()
