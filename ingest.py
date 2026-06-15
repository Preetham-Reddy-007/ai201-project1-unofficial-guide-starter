"""
ingest.py — Milestone 3: Document Ingestion + Chunking

Loads every PDF in ./documents/, cleans the extracted text, and splits each
document into overlapping chunks that respect a configurable chunk size and
overlap. Writes the chunks (with source metadata) to a JSONL file that the
embedding/vector-store stage (Milestone 4) can consume directly.

This follows the plan in planning.md:
  - Ingestion:  pdfplumber -> raw text + source metadata
  - Chunking:   structure-aware *recursive* split, ~500-token chunks,
                ~100-token (~20%) overlap, keeping source/chapter metadata.

Font-encoding repair
--------------------
Most of the UVM Cookbook PDFs embed *subsetted fonts with no ToUnicode map*, so
text extractors emit raw glyph codes as `(cid:N)` placeholders instead of real
characters. For these fonts the glyph code is a constant offset below the true
Unicode codepoint (the Cookbook chapters use -29; e.g. glyph 36 -> 'A'=65).
`extract_pdf_text` detects that offset per document by trying candidate shifts
and keeping the one that yields the most English/DV-domain text, then rewrites
each `(cid:N)` to `chr(N + shift)`. Documents that already extract cleanly
(no cid tokens, e.g. the Siemens white paper) pass through unchanged.

Chunk size and overlap are measured in TOKENS using the same tokenizer as the
planned embedding model (all-MiniLM-L6-v2) when `transformers` is installed, so
the numbers match what the embedder actually sees. If that tokenizer is not
available, it falls back to a character-based approximation and warns.

Usage:
    python ingest.py                          # defaults: 500 / 100 tokens
    python ingest.py --chunk-size 256 --overlap 50
    python ingest.py --docs-dir documents --out chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List

# ---------------------------------------------------------------------------
# Human-readable titles for the known corpus (falls back to the filename stem).
# Keeps a clean "source chapter" label on every chunk for citation later.
# ---------------------------------------------------------------------------
TITLE_MAP = {
    "UVM_Basics": "UVM Basics",
    "Testbench_architecture": "Testbench Architecture",
    "DUT_Testbench_Connection": "DUT-Testbench Connection",
    "Configure_a_test_environment": "Configure a Test Environment",
    "Sequences": "Sequences",
    "Other_stimulus_techniques": "Other Stimulus Techniques",
    "Analysis_Componenets_and_techniques": "Analysis Components and Techniques",
    "Register_abstraction_level": "Register Abstraction Level",
    "End_of_test_mechanisms": "End of Test Mechanisms",
    "The_UFM_messaging_system": "The UVM Messaging System",
    "Debug_of_SV_and_UVM": "Debug of SV and UVM",
    "Testbench_acceleration_through_Co_Emulation": "Testbench Acceleration through Co-Emulation",
    "UVM_Connect_SV_SystemC_interoperobility": "UVM Connect - SV/SystemC Interoperability",
    "uvm-connect-2.3.4-primer-guide-verification-academy": "UVM Connect 2.3.4 Primer Guide",
    "siemens-eda-wp_the-life-of-a-systemverilog-variable_drich": "The Life of a SystemVerilog Variable (Siemens EDA WP)",
}

# Recursive split boundaries, tried in order: paragraph -> line -> sentence ->
# word -> character. This keeps a heading + prose + code example together when
# it fits, only falling back to finer boundaries for oversized blocks.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ---------------------------------------------------------------------------
# Length function (token-aware with a character fallback)
# ---------------------------------------------------------------------------
def build_length_fn() -> tuple[Callable[[str], int], str]:
    """Return (length_fn, unit_label).

    Prefers the all-MiniLM-L6-v2 tokenizer so chunk sizes match what the
    embedding model actually encodes. Falls back to a ~4-chars-per-token
    character estimate if `transformers` / the tokenizer is unavailable
    (e.g. offline with no model cache).
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
        from transformers.utils import logging as hf_logging  # type: ignore

        hf_logging.set_verbosity_error()  # silence "sequence length > 512" notices
        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        # We use the tokenizer only to COUNT tokens, not to feed the model, so
        # raise the max-length cap to avoid spurious truncation warnings.
        tok.model_max_length = 1_000_000_000

        def token_len(text: str) -> int:
            # add_special_tokens=False: we measure content, not [CLS]/[SEP].
            return len(tok.encode(text, add_special_tokens=False))

        return token_len, "tokens"
    except Exception as exc:  # ImportError or offline download failure
        print(
            f"[warn] MiniLM tokenizer unavailable ({exc.__class__.__name__}); "
            "falling back to a character-based length estimate (~4 chars/token).",
            file=sys.stderr,
        )

        def char_len(text: str) -> int:
            return max(1, round(len(text) / 4))

        return char_len, "tokens(~est)"


# ---------------------------------------------------------------------------
# Font-encoding repair (see module docstring)
# ---------------------------------------------------------------------------
_CID_RE = re.compile(r"\(cid:(\d+)\)")

# Common English + DV-domain substrings used to score a decode attempt. Counted
# as substrings, not whole words, because the broken fonts often drop spaces.
_SCORE_WORDS = (
    "the", "and", "for", "with", "that", "this", "uvm", "sequence", "class",
    "phase", "test", "register", "driver", "verification", "method", "object",
    "transaction", "component", "interface", "signal", "design", "value",
)
# Range of candidate glyph->Unicode offsets to try when detecting the shift.
_SHIFT_RANGE = range(0, 64)


def _score_text(text: str) -> int:
    """Cheap 'how English/DV is this?' score: total domain-word substring hits."""
    low = text.lower()
    return sum(low.count(w) for w in _SCORE_WORDS)


def _decode_cids(text: str, shift: int) -> str:
    """Rewrite every `(cid:N)` token to `chr(N + shift)`.

    Out-of-range results collapse to a space. Text with no cid tokens is
    returned unchanged, so cleanly-extracted documents are never altered.
    """
    if "(cid:" not in text:
        return text

    def repl(m: "re.Match[str]") -> str:
        v = int(m.group(1)) + shift
        return chr(v) if 0x20 <= v < 0x110000 else " "

    return _CID_RE.sub(repl, text)


def _detect_shift(sample: str) -> tuple[int, int]:
    """Return (best_shift, score) — the offset that makes `sample` most readable."""
    best_shift, best_score = 0, -1
    for k in _SHIFT_RANGE:
        s = _score_text(_decode_cids(sample, k))
        if s > best_score:
            best_shift, best_score = k, s
    return best_shift, best_score


@dataclass
class Extraction:
    text: str        # cleaned-of-encoding, page-joined document text
    shift: int       # glyph offset applied to repair cid fonts (0 = none needed)
    n_pages: int
    n_cid_pages: int  # pages that contained (cid:N) glyph placeholders


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def extract_pdf_text(path: Path) -> Extraction:
    """Extract text from a PDF page-by-page (pdfplumber) and repair cid fonts."""
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        sys.exit(
            "pdfplumber is required. Install it with:\n"
            "    python -m pip install pdfplumber\n"
            "(also uncomment pdfplumber in requirements.txt)"
        )

    raw_pages: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            raw_pages.append(page.extract_text() or "")

    cid_pages = [t for t in raw_pages if "(cid:" in t]
    # Detect the glyph offset once per document from a capped sample of the
    # cid-bearing pages; the broken font is uniform within a document.
    shift = 0
    if cid_pages:
        sample = "\n".join(cid_pages)[:40000]
        shift, _ = _detect_shift(sample)

    decoded = [_decode_cids(t, shift) for t in raw_pages]
    kept = [t for t in decoded if t.strip()]
    # Join pages with a blank line so page breaks act as paragraph boundaries.
    return Extraction(
        text="\n\n".join(kept),
        shift=shift,
        n_pages=len(raw_pages),
        n_cid_pages=len(cid_pages),
    )


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
# A page number standing alone on its own line.
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
# Common boilerplate footers in the Verification Academy / Siemens cookbook.
_BOILERPLATE_RE = re.compile(
    r"(verification\s*academy|siemens|copyright|all rights reserved|www\.|https?://)",
    re.IGNORECASE,
)
# Control / non-printable characters (keep \n and \t).
_CTRL_RE = re.compile(r"[^\x09\x0a\x20-\x7e]")
# Word split across a line break by hyphenation: "config-\nuration".
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
# Table-of-contents / index dot leaders, e.g. "47.1 packet........80".
# Requires 4+ dots so it never matches a "// ..." ellipsis in code comments.
_TOC_RE = re.compile(r"\.{4,}")


def clean_text(raw: str) -> str:
    """Normalize messy PDF text without destroying code structure.

    - reconnects hyphenated line breaks
    - drops standalone page numbers, TOC/index dot-leader lines, and obvious
      boilerplate footer lines
    - strips control/non-printable chars and trailing whitespace
    - collapses 3+ blank lines down to a single blank line
    """
    text = raw.replace("\f", "\n")  # form feed -> newline
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _CTRL_RE.sub("", text)

    kept: List[str] = []
    for line in text.split("\n"):
        line = line.rstrip()
        stripped = line.strip()
        if _PAGE_NUM_RE.match(line):
            continue
        # Drop table-of-contents / index entries (dot leaders to a page number).
        if _TOC_RE.search(stripped):
            continue
        # Drop short boilerplate lines (footers), but keep longer prose that
        # merely happens to contain a URL/keyword.
        if _BOILERPLATE_RE.search(stripped) and len(stripped) < 60:
            continue
        kept.append(line)

    text = "\n".join(kept)
    # Collapse runs of blank lines to a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Recursive, token-aware splitter (LangChain-style, dependency-free)
# ---------------------------------------------------------------------------
def _merge_splits(
    splits: List[str],
    separator: str,
    chunk_size: int,
    overlap: int,
    length_fn: Callable[[str], int],
) -> List[str]:
    """Greedily merge small pieces into chunks <= chunk_size, with overlap.

    Mirrors the merge step of LangChain's RecursiveCharacterTextSplitter but
    measures length with `length_fn` (tokens) instead of characters.
    """
    sep_len = length_fn(separator) if separator else 0
    docs: List[str] = []
    current: List[str] = []
    total = 0

    for piece in splits:
        piece_len = length_fn(piece)
        # Would adding this piece overflow the chunk?
        if total + piece_len + (sep_len if current else 0) > chunk_size and current:
            doc = separator.join(current).strip()
            if doc:
                docs.append(doc)
            # Slide the window: pop from the front until we're back under the
            # overlap budget (and small enough for the next piece to fit).
            while current and (
                total > overlap
                or total + piece_len + (sep_len if current else 0) > chunk_size
            ):
                total -= length_fn(current[0]) + (sep_len if len(current) > 1 else 0)
                current.pop(0)

        current.append(piece)
        total += piece_len + (sep_len if len(current) > 1 else 0)

    doc = separator.join(current).strip()
    if doc:
        docs.append(doc)
    return docs


def recursive_split(
    text: str,
    separators: List[str],
    chunk_size: int,
    overlap: int,
    length_fn: Callable[[str], int],
) -> List[str]:
    """Recursively split `text` on the first usable separator, merging the
    resulting pieces into overlapping chunks of at most `chunk_size`."""
    final: List[str] = []

    # Choose the first separator that appears in the text.
    separator = separators[-1]
    remaining = separators[-1:]
    for i, sep in enumerate(separators):
        if sep == "":
            separator = sep
            remaining = []
            break
        if sep in text:
            separator = sep
            remaining = separators[i + 1 :]
            break

    splits = list(text) if separator == "" else text.split(separator)

    good: List[str] = []
    for piece in splits:
        if not piece:
            continue
        if length_fn(piece) < chunk_size:
            good.append(piece)
        else:
            # Flush the buffered small pieces first.
            if good:
                final.extend(
                    _merge_splits(good, separator, chunk_size, overlap, length_fn)
                )
                good = []
            # The piece is still too big: recurse with finer separators.
            if not remaining:
                final.append(piece)
            else:
                final.extend(
                    recursive_split(piece, remaining, chunk_size, overlap, length_fn)
                )

    if good:
        final.extend(_merge_splits(good, separator, chunk_size, overlap, length_fn))

    return final


def _join_dedup(a: str, b: str) -> str:
    """Concatenate `a` and `b`, dropping any overlap `b` repeats from `a`'s tail.

    Adjacent chunks share an overlap region, so `b` typically *starts* with the
    last few lines of `a`. We find the largest line-aligned overlap and append
    only `b`'s non-duplicated remainder, so merging doesn't duplicate text.
    """
    a_lines = a.split("\n")
    b_lines = b.split("\n")
    k = 0
    for cand in range(min(len(a_lines), len(b_lines)), 0, -1):
        if a_lines[-cand:] == b_lines[:cand]:
            k = cand
            break
    merged = a_lines + b_lines[k:]
    return "\n".join(merged).strip()


def _is_boilerplate(text: str) -> bool:
    """True for chunks that are overwhelmingly `//` comment scaffolding.

    Targets the repeated Apache-license headers and separator rulers that the
    UVM Connect primer reproduces in every example -- near-duplicate noise with
    no retrievable content. Real code/teaching chunks mix comments with code and
    prose, so they stay well under this threshold.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True
    comment_lines = sum(1 for l in lines if l.startswith("//"))
    return comment_lines / len(lines) > 0.7


def _merge_small_chunks(
    chunks: List[str], min_chunk_size: int, length_fn: Callable[[str], int]
) -> List[str]:
    """Fold chunks below `min_chunk_size` tokens into the preceding chunk.

    Eliminates orphaned fragments (e.g. a document's trailing code tail) that
    are too small to answer a question on their own. A leading small chunk with
    no predecessor is kept as-is.
    """
    if min_chunk_size <= 0:
        return chunks
    merged: List[str] = []
    for body in chunks:
        if merged and length_fn(body) < min_chunk_size:
            merged[-1] = _join_dedup(merged[-1], body)
        else:
            merged.append(body)
    return merged


# ---------------------------------------------------------------------------
# Chunk record
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    id: str
    source: str          # original filename
    title: str           # human-readable chapter title
    chunk_index: int     # position of this chunk within its document
    n_tokens: int
    n_chars: int
    text: str


def chunk_document(
    text: str,
    *,
    source: str,
    title: str,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
    length_fn: Callable[[str], int],
) -> List[Chunk]:
    raw_chunks = recursive_split(text, SEPARATORS, chunk_size, overlap, length_fn)
    raw_chunks = _merge_small_chunks(raw_chunks, min_chunk_size, length_fn)
    raw_chunks = [c for c in raw_chunks if not _is_boilerplate(c)]
    stem = Path(source).stem
    chunks: List[Chunk] = []
    for i, body in enumerate(raw_chunks):
        body = body.strip()
        if not body:
            continue
        chunks.append(
            Chunk(
                id=f"{stem}::{i}",
                source=source,
                title=title,
                chunk_index=i,
                n_tokens=length_fn(body),
                n_chars=len(body),
                text=body,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs and produce overlapping text chunks for RAG."
    )
    parser.add_argument("--docs-dir", default="documents", help="Folder of PDFs.")
    parser.add_argument("--out", default="chunks.jsonl", help="Output JSONL path.")
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Max chunk length (tokens)."
    )
    parser.add_argument(
        "--overlap", type=int, default=100, help="Overlap between chunks (tokens)."
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=100,
        help="Fold trailing chunks smaller than this (tokens) into the previous "
        "chunk. 0 disables. Default 100.",
    )
    args = parser.parse_args()

    if args.overlap >= args.chunk_size:
        sys.exit("--overlap must be smaller than --chunk-size.")
    if args.min_chunk_size >= args.chunk_size:
        sys.exit("--min-chunk-size must be smaller than --chunk-size.")

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        sys.exit(f"Documents folder not found: {docs_dir.resolve()}")

    pdf_paths = sorted(docs_dir.glob("*.pdf"))
    if not pdf_paths:
        sys.exit(f"No PDFs found in {docs_dir.resolve()}")

    length_fn, unit = build_length_fn()
    print(
        f"Ingesting {len(pdf_paths)} PDFs from '{docs_dir}' "
        f"(chunk={args.chunk_size} {unit}, overlap={args.overlap} {unit}, "
        f"min={args.min_chunk_size} {unit})\n"
    )

    all_chunks: List[Chunk] = []
    for path in pdf_paths:
        stem = path.stem
        title = TITLE_MAP.get(stem, stem.replace("_", " "))
        try:
            extraction = extract_pdf_text(path)
        except Exception as exc:
            print(f"  [skip] {path.name}: extraction failed ({exc})")
            continue

        cleaned = clean_text(extraction.text)
        if not cleaned:
            print(f"  [skip] {path.name}: no extractable text (scanned image?)")
            continue

        chunks = chunk_document(
            cleaned,
            source=path.name,
            title=title,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            min_chunk_size=args.min_chunk_size,
            length_fn=length_fn,
        )
        all_chunks.extend(chunks)
        toks = [c.n_tokens for c in chunks]
        avg = round(sum(toks) / len(toks)) if toks else 0
        # Note the font repair when one was applied, so it's visible per document.
        if extraction.n_cid_pages:
            repair = f" [cid-repair shift={extraction.shift} on {extraction.n_cid_pages}/{extraction.n_pages} pp]"
        else:
            repair = ""
        print(f"  {path.name:<55} -> {len(chunks):>4} chunks (avg {avg} {unit}){repair}")

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    over_limit = sum(1 for c in all_chunks if c.n_tokens > args.chunk_size)
    print(
        f"\nDone. Wrote {len(all_chunks)} chunks from {len(pdf_paths)} PDFs "
        f"-> {out_path.resolve()}"
    )
    if over_limit:
        print(
            f"[note] {over_limit} chunk(s) exceed {args.chunk_size} {unit} "
            "(an indivisible block, e.g. a long unbroken code line)."
        )


if __name__ == "__main__":
    main()
