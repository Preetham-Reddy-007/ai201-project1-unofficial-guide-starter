"""
app.py — Milestone 5 interface: a Gradio web UI over the grounded RAG pipeline.

Run:
    python app.py
then open http://localhost:7860

The UI shows the grounded answer and, separately, the source document(s) the
answer was retrieved from — so a viewer can see attribution at a glance.
"""

from __future__ import annotations

import gradio as gr

from query import ask

EXAMPLES = [
    "What is the UVM factory used for, and how do you register a class with it?",
    "How does a UVM driver get a handle to the DUT's signals?",
    "What is the role of objections in ending a UVM test?",
    "What does the UVM register abstraction layer (RAL) provide?",
]


def handle_query(question: str):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""
    result = ask(question)
    sources = "\n".join(f"• {s}" for s in result["sources"])
    if not sources:
        sources = "(no sources — the documents don't cover this question)"
    return result["answer"], sources


with gr.Blocks(title="The Unofficial UVM Guide") as demo:
    gr.Markdown(
        "# The Unofficial UVM Guide\n"
        "Ask about UVM / SystemVerilog design verification. Answers are grounded "
        "**only** in the indexed UVM Cookbook documents and cite their sources. "
        "If the documents don't cover your question, the system will say so."
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. What is the role of objections in ending a UVM test?",
        lines=2,
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    gr.Examples(examples=EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
