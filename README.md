# The Unofficial Guide — Project 1

A retrieval-augmented question-answering system over the Siemens / Verification
Academy **UVM Cookbook** and supporting white papers. Ask a question about UVM /
SystemVerilog design verification and get an answer grounded **only** in the
indexed documents, with the source chapter(s) cited.

**Pipeline:** PDF ingestion (`ingest.py`) → chunking → embedding into ChromaDB
(`embed.py`) → retrieval (`retrieve.py`) → grounded generation (`query.py`) →
Gradio UI (`app.py`).

**Run it:**
```bash
pip install -r requirements.txt
python ingest.py        # documents/*.pdf  -> chunks.jsonl   (966 chunks)
python embed.py         # chunks.jsonl     -> ChromaDB index (966 vectors)
python app.py           # web UI at http://localhost:7860
# or, headless:  python query.py "what is the role of objections in ending a test?"
```

---

## Domain

The domain is **Design Verification (DV)** — specifically building UVM
(Universal Verification Methodology) and SystemVerilog testbenches to verify a
Design Under Test (DUT). The corpus is a set of UVM Cookbook chapters plus
supporting white papers on SystemVerilog and tool interoperability.

This knowledge is valuable because DV is a narrow, closed industry domain.
General-purpose LLMs perform noticeably worse here than on mainstream
programming topics: there is far less public training data, the methodology is
detail-heavy (factory registration, phasing, TLM connections, sequences, the
register model), and small mistakes in boilerplate silently break a testbench.
Most material lives behind vendor logins, licensed tools, and PDF cookbooks
rather than open Q&A sites, so a beginner can rarely find a single correct,
cited answer to "how do I do X in UVM." A grounded, citing RAG system over the
cookbook gives a trustworthy answer *and* tells you which chapter to read next.

---

## Document Sources

All 15 sources are local PDFs in `./documents/` — mostly Siemens / Verification
Academy UVM Cookbook chapters (prose + SystemVerilog code + tables), plus two
supporting documents on UVM Connect and SystemVerilog.

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | UVM Basics | PDF (cookbook chapter) | `documents/UVM_Basics.pdf` |
| 2 | Testbench Architecture | PDF (cookbook chapter) | `documents/Testbench_architecture.pdf` |
| 3 | DUT–Testbench Connection | PDF (cookbook chapter) | `documents/DUT_Testbench_Connection.pdf` |
| 4 | Configure a Test Environment | PDF (cookbook chapter) | `documents/Configure_a_test_environment.pdf` |
| 5 | Sequences | PDF (cookbook chapter) | `documents/Sequences.pdf` |
| 6 | Other Stimulus Techniques | PDF (cookbook chapter) | `documents/Other_stimulus_techniques.pdf` |
| 7 | Analysis Components and Techniques | PDF (cookbook chapter) | `documents/Analysis_Componenets_and_techniques.pdf` |
| 8 | Register Abstraction Level | PDF (cookbook chapter) | `documents/Register_abstraction_level.pdf` |
| 9 | End of Test Mechanisms | PDF (cookbook chapter) | `documents/End_of_test_mechanisms.pdf` |
| 10 | The UVM Messaging System | PDF (cookbook chapter) | `documents/The_UFM_messaging_system.pdf` |
| 11 | Debug of SV and UVM | PDF (cookbook chapter) | `documents/Debug_of_SV_and_UVM.pdf` |
| 12 | Testbench Acceleration through Co-Emulation | PDF (cookbook chapter) | `documents/Testbench_acceleration_through_Co_Emulation.pdf` |
| 13 | UVM Connect — SV/SystemC Interoperability | PDF (white paper) | `documents/UVM_Connect_SV_SystemC_interoperobility.pdf` |
| 14 | UVM Connect 2.3.4 Primer Guide | PDF (reference manual) | `documents/uvm-connect-2.3.4-primer-guide-verification-academy.pdf` |
| 15 | The Life of a SystemVerilog Variable (Siemens EDA WP) | PDF (white paper) | `documents/siemens-eda-wp_the-life-of-a-systemverilog-variable_drich.pdf` |

---

## Chunking Strategy

**Chunk size:** 500 tokens (measured with the `all-MiniLM-L6-v2` tokenizer, so
the count matches what the embedder actually sees), with a 100-token minimum.

**Overlap:** 100 tokens (~20%) between adjacent chunks.

**Why these choices fit your documents:**
These are long, structured technical chapters, not short reviews. A single idea
(e.g. "how a factory override works") is typically a heading + a paragraph of
prose + a SystemVerilog code snippet. I used a **structure-aware recursive
splitter** that splits first on paragraph boundaries (`\n\n`), then lines,
sentences, words, and only finally characters — so a chunk tends to hold one
concept plus its code. The ~100-token overlap keeps a definition and the code
that uses it from landing in different chunks with no shared context. 500 tokens
is an upper bound; the embedding model truncates at ~256, so larger chunks would
silently lose their tails — but retrieval quality was good at 500 (see
Evaluation), so I kept it rather than tuning down.

**Preprocessing before chunking** (the most involved part of this project):
- **Font-encoding repair.** Most cookbook PDFs embed subsetted fonts with no
  `ToUnicode` map, so extractors emit raw glyph codes as `(cid:NN)`. The glyph
  codes are a constant offset below the true Unicode codepoint (the cookbook
  fonts use −29). `ingest.py` auto-detects that offset per document by trying
  candidate shifts and keeping the one that yields the most English/DV text,
  then rewrites each `(cid:NN)` to `chr(N + shift)`. Documents that extract
  cleanly (the white paper, the primer body) are left untouched.
- **Cleaning:** removes standalone page numbers, table-of-contents/index
  dot-leader lines, repeated Apache-license/separator comment blocks, and
  boilerplate footers; reconnects hyphenated line breaks.
- **Min-chunk merge:** folds orphaned trailing fragments (e.g. a dangling
  `endclass` code tail) into the previous chunk so every chunk is self-contained.

**Final chunk count:** **966 chunks** across the 15 PDFs (avg 368 tokens).

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dimensional,
runs locally on CPU, no API key, no rate limits). Vectors are L2-normalized and
stored in a **persistent ChromaDB** collection configured for **cosine**
similarity (`hnsw:space=cosine`). Retrieval returns **top-k = 5**.

**Production tradeoff reflection:**
If this were a real product for verification engineers and cost weren't a
constraint, the tradeoffs I'd weigh:
- **Domain accuracy:** MiniLM is trained on general web text and doesn't place
  UVM jargon ("sequencer", "objection", "RAL") as close to the right concepts as
  a domain-tuned model would. A larger model (`bge-large`, `e5-large`) or one
  fine-tuned on DV text would likely improve retrieval the most for this closed
  domain — and would directly help the failure case below.
- **Context length:** MiniLM truncates at ~256 tokens, which forces small chunks
  and can split a code example from its explanation. A longer-context embedding
  model (512–8k tokens) would let me keep code + explanation in one chunk.
- **Accuracy vs. latency/size:** 768/1024-dim models retrieve better but are
  slower and use more index memory; for an interactive guide I'd benchmark
  whether the quality gain justifies the latency.
- **Multilingual:** not needed — the corpus is English-only — so I wouldn't pay
  for multilingual capability.

---

## Grounded Generation

Generation uses Groq's `llama-3.3-70b-versatile` at `temperature=0`. Grounding is
enforced by **two independent mechanisms** so a single weak link can't leak an
ungrounded answer (`query.py`):

**1. Programmatic relevance gate (structural).** Before any LLM call,
`ask()` checks the best retrieved chunk's cosine similarity. If it is below a
floor (`RELEVANCE_FLOOR = 0.30`), the function returns the refusal string
**without ever invoking the model** — so a wildly off-topic question can't be
answered from training knowledge at all.

**2. System prompt grounding instruction (the actual instruction given):**
> "You are a precise assistant for UVM / SystemVerilog design verification.
> Answer the user's question using ONLY the numbered source excerpts provided in
> the context. … 1. Use only facts stated in the context. Do NOT use any outside
> or prior knowledge, and do NOT invent class names, macros, methods, or APIs
> that do not appear in the context. 2. If the context does not contain enough
> information to answer, reply with EXACTLY this sentence and nothing else: 'I
> don't have enough information on that.' 3. Cite the source title(s) you used,
> e.g. (Source: UVM Basics). …"

The retrieved chunks are passed as a **numbered, source-labeled context block**
(`[1] (Source: UVM Basics | file: UVM_Basics.pdf) …`), which both anchors the
model and lets it cite by title.

**How source attribution is surfaced in the response:** Attribution is
**guaranteed in code, not left to the LLM.** After retrieval, `_unique_sources()`
builds a deduplicated, rank-ordered list of the source documents that were
actually retrieved, and `ask()` returns it as `result["sources"]`. The Gradio UI
shows it in a separate "Retrieved from" box. So even if the model forgets an
inline citation, the response still names the documents the answer was drawn
from. (The model is *also* asked to cite inline, and does — see the examples —
but the displayed source list does not depend on it.)

---

## Evaluation Report

All 5 questions run through the live system (`python query.py`). "System
response" is summarized; retrieval quality reflects the top-5 chunks; accuracy is
judged against the expected answer.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What is the UVM factory used for, and how do you register a class with it? | Factory creates components/objects by type so they can be overridden at runtime; classes register via `` `uvm_component_utils ``/`` `uvm_object_utils `` macros, built with `type_id::create()`. | Correctly explains the factory's purpose (create-by-type + override). For registration it cites the `uvm_component_registry` `create` mechanism and constructor conventions but **does not name the `` `uvm_component_utils `` macro**, and explicitly notes "the exact steps to register a class are not fully described in the provided context." Cited UVM Basics. | Partially relevant (right chapter #1–2, but the specific registration-macro chunk not surfaced) | **Partially accurate** |
| 2 | How does a UVM driver get a handle to the DUT's signals? | Through a **virtual interface** set into `uvm_config_db` at the top and `get()`-retrieved by the driver, which drives/samples the signals. | "Through a virtual interface handle, typically passed using `uvm_config_db`; the driver uses it to reference a static interface and drive/sample DUT signals." Cited DUT–Testbench Connection + UVM Basics. | Relevant | **Accurate** |
| 3 | What is the role of objections in ending a UVM test? | A component raises an objection before stimulus, drops it when done; the phase/sim ends only after all objections drop. | "Objections control the end of each phase via the `uvm_objection` shared counter; participants raise/drop asynchronously, and when the count hits zero the 'all dropped' condition ends the phase." Cited End of Test Mechanisms. | Relevant (4 of 5 from the right chapter) | **Accurate** |
| 4 | What is a UVM sequence and how does it relate to a sequence item and a sequencer? | A sequence generates a stream of sequence_items; it runs on a sequencer that arbitrates and hands items to the driver via the `get_next_item`/`item_done` handshake. | "A sequence is an OO transaction-level stimulus mechanism that sends sequence_items to a driver via a sequencer; the sequencer arbitrates; a sequence_item carries the info for a pin-level transaction." (Relationship correct; TLM handshake not named.) Cited UVM Basics + Sequences. | Relevant (sim up to 0.78) | **Accurate** |
| 5 | What does the UVM RAL provide, and what is the adapter for? | RAL models the DUT's registers/fields for abstract read/write; the adapter converts abstract register ops into bus transactions (predictor keeps the model in sync). | "RAL tracks register content and is a convenience layer for accessing registers/memory; the adapter bidirectionally translates generic register sequence_items to/from VIP bus items and extends `uvm_reg_adapter`." (Predictor not mentioned.) Cited Register Abstraction Level. | Relevant (top 3 from the right chapter) | **Accurate** |
| 6 | What is UVM? *(extra probe — short/generic query, see Failure Case Analysis)* | A definition: UVM is a standardized SystemVerilog class library / methodology for building reusable, modular, transaction-level testbenches. | **"I don't have enough information on that."** — declined with no sources, even though the entire corpus is about UVM. The top-5 were all UVM Connect Primer fragments (tool requirements, command API, license text); no definitional chunk was retrieved. | **Off-target** (best sim 0.588, but every hit was the over-represented primer, not UVM Basics) | **Inaccurate** (unable to answer) |

**Summary:** Of the 5 planning questions, 4 accurate and 1 partially accurate;
every answer was grounded and correctly attributed, and no answer contained
invented APIs. Row 6 is an additional probe I added to surface a concrete
failure: a short, generic query that retrieval handles poorly (analyzed in full
below). The partial result (Q1) and this failure case are the honest limits of
the system.

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:** `What is UVM` (a short, generic definitional query).

**What the system returned:** *"I don't have enough information on that."* — no
answer and no sources, even though the corpus is entirely about UVM.

**Root cause (tied to a specific pipeline stage):** This is a **retrieval**
failure (embedding + corpus composition), not a generation failure — and the
relevance gate did *not* trigger (top similarity was 0.588, above the 0.30
floor), so the query did reach the LLM. The problem is *what* was retrieved. The
top-5 (and even top-12) chunks were almost all from the **UVM Connect Primer** —
tool-platform requirements, the UVMC command API, even a license-text fragment —
none of which *define* what UVM is. The actual UVM Basics overview chunk did not
appear in the top 12. Two pipeline factors combine to cause this:
1. **The query is too short.** "What is UVM" embeds to a vector dominated by the
   single token "UVM." The header string "Introduction to UVM Connect" appears on
   hundreds of primer chunks, so they all match the keyword superficially — a
   3-word query gives MiniLM almost no semantic signal to separate "UVM the
   methodology" from "UVM Connect the library."
2. **Corpus imbalance.** The UVM Connect Primer is ~340 of 966 chunks (≈35%), so
   it floods top-k for any UVM-ish query. With no source-diversity constraint,
   one over-represented document can occupy all 5 slots and crowd out UVM Basics.

Notably, grounding behaved *correctly*: given context that didn't define UVM, the
model declined rather than reciting a definition from training data.

**What you would change to fix it:** Add **per-source diversification (MMR-style)**
to `retrieve()` — fetch ~3× the candidates, then cap how many chunks come from a
single document (e.g. ≤2) — so the primer can't monopolize top-k and the UVM
Basics overview can surface. Longer-term, a domain-tuned embedding model would
give short queries more semantic signal, and rebalancing/Capping the primer's
share of the corpus at ingestion would reduce its dominance. (A simple
workaround that already works today: ask a fuller question, e.g. "What is the
Universal Verification Methodology used for?", which gives the embedder enough
signal to retrieve the right chapter.)

---

## Spec Reflection

**One way the spec helped you during implementation:**
The *Chunking Strategy* section gave concrete, testable parameters — 500-token
chunks, 100-token overlap, structure-aware recursive splitting, and the explicit
caveat that MiniLM truncates at ~256 tokens. That turned an open-ended "split the
PDFs somehow" task into a precise spec I could implement and verify directly: I
built the recursive splitter to those numbers and measured length with the actual
MiniLM tokenizer. The spec's reasoning ("keep code with its explanation")
also justified design decisions downstream — it's why I added the overlap *and*
the min-chunk merge when a checkpoint review found an orphaned code-tail chunk.

**One way your implementation diverged from the spec, and why:**
The Architecture diagram planned **Claude (opus/sonnet)** as the generation
model, but I used **Groq's `llama-3.3-70b-versatile`** because the Milestone 5
requirement was a free, OpenAI-compatible tier with no card required — the
grounding mechanism is model-agnostic, so the switch didn't affect the design. A
second, larger divergence: the spec anticipated that "PDF extraction mangles
code, tables, and diagrams," but the real corruption was worse and more specific
— subsetted fonts with no `ToUnicode` map produced `(cid:NN)` glyph codes. I had
to add an entire **per-document font-shift auto-detection and repair stage** to
`ingest.py` that the plan never envisioned; without it, every cookbook chunk
would have been unreadable garbage.

---

## AI Usage

**Instance 1 — Ingestion & chunking (and debugging the font corruption)**

- *What I gave the AI:* my `planning.md` *Chunking Strategy* and *Architecture*
  sections, plus the requirement "load every PDF in `documents/`, clean the text,
  and produce chunks matching the 500-token size and 100-token overlap, keeping
  source metadata."
- *What it produced:* `ingest.py` with `pdfplumber` extraction and a token-aware
  recursive splitter writing `chunks.jsonl`.
- *What I changed or overrode:* The first run produced `(cid:NN)` garbage. Rather
  than accept it, I had the AI diagnose the cause (subsetted fonts, no
  `ToUnicode`) and verify that the glyph codes were a constant offset below
  Unicode. I **rejected a hardcoded shift** and directed it to implement
  *per-document* shift auto-detection, because the white paper and primer don't
  need a shift while the cookbook chapters need −29. After reviewing 5 random
  chunks at a checkpoint, I further directed it to add a TOC dot-leader filter, a
  repeated-license-block filter, and a min-chunk merge — none of which were in my
  original plan but all of which the sampled data showed were necessary.

**Instance 2 — Grounded generation**

- *What I gave the AI:* the grounding requirement ("answer from retrieved context
  only, cite sources"), the Groq `llama-3.3-70b-versatile` model, and the Gradio
  Blocks skeleton.
- *What it produced:* a first version that passed the chunks as context and a
  system prompt telling the model to answer only from them.
- *What I changed or overrode:* A prompt instruction alone *suggests* grounding;
  I wanted it *enforced*. I directed the AI to add a **programmatic relevance
  gate** that returns the refusal string before the LLM is even called when
  retrieval finds nothing similar enough, and to build the **source list in code**
  from the retrieved chunks (`_unique_sources`) instead of trusting the model to
  cite. This is what makes the out-of-domain questions ("best pizza topping")
  decline reliably and makes attribution independent of the model's behavior.
