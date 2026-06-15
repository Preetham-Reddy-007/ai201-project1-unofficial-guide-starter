# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->
The domain is **Design Verification (DV)** — specifically building UVM (Universal Verification Methodology) and SystemVerilog testbenches to verify the functionality of a Design Under Test (DUT). The database is a set of UVM Cookbook chapters plus supporting white papers and guides on SystemVerilog and tool interoperability.

This knowledge is valuable because DV is a narrow, closed industry domain. General-purpose LLMs perform noticeably worse here than on mainstream programming topics: there is far less public training data, the methodology is detail-heavy (factory registration, phasing, TLM connections, sequences, register model), and small mistakes in boilerplate silently break a testbench. Most material lives behind vendor logins, licensed tools, and PDF cookbooks rather than open Q&A sites, so a beginner can rarely find a single correct, cited answer to "how do I do X in UVM."

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

All sources are local PDFs stored in `./documents/`. They are mostly Siemens/Verification Academy UVM Cookbook chapters (prose + SystemVerilog code + tables + diagrams), plus two supporting documents on SystemVerilog and UVM Connect.

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | UVM Basics | Core UVM concepts: components, objects, factory, phasing, config DB — the foundation chapter | `documents/UVM_Basics.pdf` |
| 2 | Testbench Architecture | How a UVM testbench is structured: agents, drivers, monitors, sequencers, env, test | `documents/Testbench_architecture.pdf` |
| 3 | DUT–Testbench Connection | Connecting the testbench to the DUT via virtual interfaces and the config DB | `documents/DUT_Testbench_Connection.pdf` |
| 4 | Configure a Test Environment | Building/configuring the env and test, overrides, and configuration objects | `documents/Configure_a_test_environment.pdf` |
| 5 | Sequences | Writing UVM sequences and sequence items; stimulus generation patterns | `documents/Sequences.pdf` |
| 6 | Other Stimulus Techniques | Additional stimulus approaches beyond basic sequences | `documents/Other_stimulus_techniques.pdf` |
| 7 | Analysis Components and Techniques | Scoreboards, subscribers, coverage, and analysis ports for checking results | `documents/Analysis_Componenets_and_techniques.pdf` |
| 8 | Register Abstraction Level | The UVM register model (RAL): register modeling, adapters, predictors | `documents/Register_abstraction_level.pdf` |
| 9 | End of Test Mechanisms | Objections and ending a simulation cleanly | `documents/End_of_test_mechanisms.pdf` |
| 10 | The UVM Messaging System | `uvm_info`/`uvm_error` reporting, verbosity, and message control | `documents/The_UFM_messaging_system.pdf` |
| 11 | Debug of SV and UVM | Debugging SystemVerilog and UVM testbenches | `documents/Debug_of_SV_and_UVM.pdf` |
| 12 | Testbench Acceleration through Co-Emulation | Accelerating testbenches with co-emulation | `documents/Testbench_acceleration_through_Co_Emulation.pdf` |
| 13 | UVM Connect — SV/SystemC Interoperability | Connecting SystemVerilog and SystemC via UVM Connect | `documents/UVM_Connect_SV_SystemC_interoperobility.pdf` |
| 14 | UVM Connect 2.3.4 Primer Guide | Verification Academy primer guide for UVM Connect | `documents/uvm-connect-2.3.4-primer-guide-verification-academy.pdf` |
| 15 | The Life of a SystemVerilog Variable (Siemens EDA WP) | White paper on SystemVerilog variable semantics and lifetime | `documents/siemens-eda-wp_the-life-of-a-systemverilog-variable_drich.pdf` |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** ~500 tokens (roughly 2,000 characters) as a starting point.

**Overlap:** ~100 tokens (~20%) between adjacent chunks.

**Reasoning:**
These are long, structured technical chapters, not short reviews. A single idea (e.g. "how the factory override works") is usually explained across a heading, a paragraph of prose, and an accompanying SystemVerilog code snippet. If chunks are too small, a code example gets separated from the sentence that explains it; if too large, retrieval returns a lot of irrelevant text and dilutes the embedding.

- **Structure-aware splitting first:** split on the document's natural boundaries (headings/sections) before falling back to a fixed character/token size, so a chunk tends to hold one concept plus its code example. A recursive splitter (e.g. LangChain `RecursiveCharacterTextSplitter` on `\n\n`, `\n`, then spaces) is a good beginner default.
- **Keep code with its explanation:** the ~100-token overlap reduces the chance that a definition and the code that uses it land in different chunks with no shared context.
- **Caveat on the embedding model limit:** the planned embedding model (`all-MiniLM-L6-v2`) only encodes ~256 tokens; anything past that in a chunk is silently truncated. So ~500-token chunks are an *upper bound to tune down* — I will start at 500, measure retrieval quality on my 5 test questions, and likely reduce toward ~256 tokens if answers are getting cut off. (If I switch to a longer-context embedding model, I can raise the chunk size.)
- **Tables and images:** PDF extraction will turn tables into messy text and drop diagrams entirely (see Anticipated Challenges). I will not rely on table/figure content for correctness and will favor the prose + code that survives extraction.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dimensional, fast, runs locally on CPU, good general-purpose baseline). Vector store: FAISS (or Chroma) for similarity search with cosine/inner-product.

**Top-k:** 5 chunks per query. Enough to cover a concept spread across a few paragraphs/code blocks without flooding the generation prompt with off-topic text. I'll tune this (try 3 and 8) against my test questions.

**Production tradeoff reflection:**
If this were a real product for verification engineers and cost weren't a constraint, the main tradeoffs I'd weigh:

- **Domain accuracy vs. general models:** `all-MiniLM-L6-v2` is trained on general web text and may not place UVM jargon (e.g. "sequencer", "objection", "phasing", "RAL") close to the right concepts. A larger or domain-tuned embedding model (or fine-tuning on DV text) would likely improve retrieval most for this closed domain.
- **Context length:** MiniLM truncates at ~256 tokens, which forces small chunks and can split code from explanation. A longer-context embedding model (e.g. `bge-large`, `e5-large`, or an API embedding model with 512–8k token windows) would let me keep a code example and its explanation in one chunk.
- **Accuracy vs. latency/size:** larger models (768/1024-dim) retrieve better but are slower and use more memory/disk for the index; for an interactive guide I'd benchmark whether the quality gain justifies the latency.
- **Multilingual:** not needed here — the corpus is English-only technical writing — so I would not pay for multilingual capability.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

<!-- These are draft Q/A pairs based on the chapter topics. Verify the expected answers against the
     actual PDF text before using them to grade the system — edit as needed. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What is the UVM factory used for, and how do you register a class with it? | The factory lets you create components/objects by type so they can be overridden at runtime without editing the original code. Classes register with the `\`uvm_component_utils` / `\`uvm_object_utils` macros, and objects are built via `type_id::create()` instead of `new()`. (Source: UVM Basics) |
| 2 | How does a UVM driver get a handle to the DUT's signals? | Through a **virtual interface** that is set into the `uvm_config_db` at the top level and `get()`-retrieved by the driver (typically in `build_phase`/`connect_phase`); the driver drives/samples the interface signals. (Source: DUT–Testbench Connection / Configure a Test Environment) |
| 3 | What is the role of objections in ending a UVM test? | A component raises an objection before stimulus and drops it when done; the run phase (and simulation) ends only after all raised objections are dropped, preventing the test from finishing prematurely. (Source: End of Test Mechanisms) |
| 4 | What is a UVM sequence and how does it relate to a sequence item and a sequencer? | A sequence is a class that generates a stream of sequence items (transactions); it runs on a sequencer, which arbitrates and hands items to the driver via the TLM `get_next_item`/`item_done` handshake. (Source: Sequences) |
| 5 | What does the UVM register abstraction layer (RAL) provide, and what is the adapter for? | RAL provides a model of the DUT's registers/fields so tests can read/write registers abstractly; a register adapter converts the abstract register operations into the bus transactions the driver understands (with a predictor keeping the model in sync). (Source: Register Abstraction Level) |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. **PDF extraction mangles code, tables, and diagrams.** The cookbook relies heavily on SystemVerilog code blocks, tables, and figures. A plain text extractor will scramble table columns into run-on text, drop images/diagrams entirely, and may reflow code (losing indentation/line breaks). Retrieved chunks may therefore contain garbled code. *Mitigation:* use a layout-aware extractor (e.g. `pdfplumber`/`PyMuPDF`), clean up obvious artifacts, and treat figures as unavailable — design test questions around prose + code, not diagrams.

2. **Key information gets split across chunk boundaries.** Because a concept = heading + explanation + code, a naive fixed-size split can put the code in a different chunk than the sentence that explains it, so retrieval returns one without the other. *Mitigation:* structure-aware (recursive) chunking with ~20% overlap, and tune chunk size against the 5 test questions.

3. **Jargon / acronym mismatch hurts retrieval.** A user might ask "how do I stop my test from ending too early?" while the document says "objection". A general embedding model may not connect the two, causing off-topic or empty retrieval. *Mitigation:* evaluate with realistic, plain-language questions; consider a domain-tuned embedding model (see Production tradeoff reflection) if recall is poor.

4. **Overlapping topics cause off-topic retrieval.** Many chapters touch the same components (sequencer, driver, config DB), so a query can pull chunks from the wrong chapter. *Mitigation:* keep source/chapter metadata on each chunk, cite sources in the answer, and review top-k results during evaluation.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

```
┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│ 1. Document         │     │ 2. Chunking      │     │ 3. Embedding + Vector    │
│    Ingestion        │ ──> │                  │ ──> │    Store                 │
│                     │     │ structure-aware  │     │                          │
│ 15 UVM/SV PDFs in   │     │ recursive split  │     │ all-MiniLM-L6-v2         │
│ ./documents/        │     │ ~500 tok chunks  │     │ (sentence-transformers)  │
│ pdfplumber/PyMuPDF  │     │ ~100 tok overlap │     │ -> FAISS / Chroma index  │
│ -> raw text +       │     │ + keep source/   │     │ (vectors + metadata)     │
│    source metadata  │     │   chapter metadata│    │                          │
└─────────────────────┘     └──────────────────┘     └────────────┬─────────────┘
                                                                   │
                                                                   v
                          ┌──────────────────────┐     ┌──────────────────────────┐
                          │ 5. Generation        │     │ 4. Retrieval             │
   user question ───────> │                      │ <── │                          │
                          │ Claude (claude-      │     │ embed query -> top-k=5    │
   grounded, cited <───── │ opus/sonnet) given   │     │ nearest chunks from index │
   answer                 │ query + retrieved    │     │ (cosine similarity)       │
                          │ chunks as context    │     │                           │
                          └──────────────────────┘     └──────────────────────────┘
```

**Stage → tool summary**
1. **Ingestion:** `pdfplumber` or `PyMuPDF` to extract text from the 15 PDFs; attach `source`/chapter metadata to each document.
2. **Chunking:** LangChain `RecursiveCharacterTextSplitter` (or equivalent), ~500-token chunks, ~100-token overlap.
3. **Embedding + store:** `sentence-transformers` (`all-MiniLM-L6-v2`) → FAISS (or Chroma) vector index storing vectors + metadata.
4. **Retrieval:** embed the user query with the same model, return top-k=5 nearest chunks.
5. **Generation:** Claude API, prompted with the query + retrieved chunks, instructed to answer only from context and cite the source chapter.

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

- **Tool:** Claude Code (Claude Opus), driven by this `planning.md`.
- **Input I gave it:** the *Chunking Strategy* and *Architecture* sections above, plus the requirement: "load every PDF in `documents/`, clean the text, and produce chunks matching the specified 500-token size and 100-token overlap, keeping source/chapter metadata."
- **What it produced:** `ingest.py` — `pdfplumber` extraction, text cleaning, a token-aware recursive splitter (using the `all-MiniLM-L6-v2` tokenizer so chunk sizes match what the embedder sees), and `chunks.jsonl` output with `source`/`title`/`chunk_index`/`n_tokens` metadata.
- **What I changed / directed differently:** The first run produced `(cid:NN)` garbage. I diagnosed it as subsetted fonts with no `ToUnicode` map and directed the AI to add **per-document glyph-shift auto-detection** (the Cookbook fonts are offset by 29) rather than hardcoding a shift, since the white paper and primer don't need one. After a checkpoint review of random chunks I also had it add: a **minimum-chunk merge** (the smallest chunk was an orphaned 38-token code tail), a **TOC dot-leader filter** (≈114 table-of-contents chunks from the primer), and a **repeated-license/comment-block filter** (9 near-duplicate Apache headers).
- **How I verified it:** ran on all 15 PDFs (→ 966 chunks), printed 5 random chunks for readability/self-containment, and ran a corpus QA pass (0 empty, 0 residual cid, 0 TOC leaders, readable-char ratio = 1.0 per source).

**Milestone 4 — Embedding and retrieval:**

- **Tool:** Claude Code, driven by the *Retrieval Approach* section and the pipeline diagram.
- **Input I gave it:** "embed `chunks.jsonl` with `all-MiniLM-L6-v2` via `sentence-transformers`, store in a persistent ChromaDB collection with source metadata, use cosine similarity, and write a `retrieve(query, k=5)` function that returns the top-k chunks with their source info."
- **What it produced:** `vectorstore.py` (shared model + cosine-space config so indexing and querying can't diverge), `embed.py` (batch-embeds and rebuilds the collection idempotently), and `retrieve.py` (`retrieve()` + a CLI that runs the 5 eval questions).
- **What I changed / directed differently:** I had it factor the model/embedding config into one shared module rather than duplicating `SentenceTransformer(...)` in both scripts — the classic RAG bug is indexing with one config and querying with another. I also had it set `hnsw:space=cosine` at collection creation and L2-normalize vectors, and convert Chroma's returned *distance* into a *similarity* (`1 - distance`) for readable scores.
- **How I verified it:** built the index (966 vectors) and ran all 5 eval questions; retrieval returned relevant, correctly-attributed chunks (cosine similarity 0.50–0.78), with the right chapter ranking #1 or #2 for 4 of 5.

**Milestone 5 — Generation and interface:**

- **Tool:** Claude Code.
- **Input I gave it:** the grounding requirement ("answer from retrieved context only, with source attribution"), the desired output format (answer + source list), the Groq `llama-3.3-70b-versatile` model from `.env`, and the Gradio Blocks skeleton.
- **What it produced:** `query.py` — `ask(question, k=5)` that retrieves chunks, builds a numbered source-labeled context, calls Groq at `temperature=0`, and returns `{answer, sources, chunks}`; and `app.py` — a minimal Gradio UI (question → answer + retrieved sources).
- **What I changed / directed differently:** I insisted grounding be **enforced, not suggested**, so I had it add two independent guards: (1) a **programmatic relevance gate** that returns the refusal string *without calling the LLM* when the best retrieved similarity is below a floor, and (2) a strict system prompt with a fixed refusal sentence. I also had source attribution built **in code** from the retrieved chunks (`_unique_sources`) instead of trusting the model to cite. I switched display separators to ASCII after the Windows console mojibached the em-dash/bullet, and pinned `huggingface-hub<1.0` to resolve a gradio/transformers conflict.
- **How I verified it:** tested in-domain queries (grounded, cited answers), out-of-domain queries ("best pizza topping", "bake sourdough" → declined with no sources), and confirmed the Gradio server serves on `http://localhost:7860`. A short query ("What is UVM") exposed a retrieval limitation (the oversized UVM Connect primer floods top-k), which I recorded as a failure case for the README.
