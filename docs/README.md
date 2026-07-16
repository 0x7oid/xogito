# Xogito

> **Computational reasoning for decisions that matter.**

<p align="center">
  <img src="LOGO.png" height="550">
</p>
Here is the updated README, with a new section added right after the introduction to clearly define the kinds of complex problems and decision-making scenarios Xogito is built to handle.

---

# Xogito

**A second brain for large language models.**

Ask a frontier model a hard question, and it will often reason correctly at every step yet still land on the wrong answer, stated with full confidence—leaving you with no way to tell which of its assumptions you should trust.

Xogito keeps that from happening by doing the model's bookkeeping externally, in code, instead of trusting the model to track its own beliefs.

Powered by a structured reasoning engine, Xogito processes queries through an orchestrated system where logic and verification are handled independently. By relying on a strict shared state, it ensures the model cannot easily hallucinate or overwrite facts to fit a flawed hypothesis.

**Goal:** Given an open-ended, ambiguous question, arrive at a conclusion you can actually rely on — and if the evidence doesn't support one, say so, instead of guessing with confidence.

Concretely, it:

* **Separates** observations from assumptions.
* **Evaluates** competing hypotheses.
* **Identifies** contradictions instead of quietly picking a side.
* **Tracks** uncertainty explicitly, rather than hiding it behind a confident tone.
* **Respects** user-defined constraints instead of reinterpreting them.
* **Checks** whether its own conclusion actually follows from the evidence.
* **Produces** a fully traceable reasoning chain.

---

## What It's Built For

Xogito is not meant for simple text generation or basic Q&A. It is engineered specifically for **complex problem handling and high-stakes decision-making**. It thrives in scenarios where standard models lose the plot:

* **Rigorous Decision Support:** Providing traceable, evidence-backed reasoning for critical choices where being "confidently wrong" is an unacceptable risk.
* **Navigating Ambiguity:** Resolving deep research queries where data is conflicting or incomplete, highlighting contradictions rather than forcing a clean (but false) narrative.
* **Multi-Step Logic & Constraints:** Solving problems with rigid, compounding rules—such as strategic analysis, auditing, or systems planning—without the model "forgetting" constraints halfway through.
* **Epistemic Truth-Seeking:** Situations where tracking the *source* of a belief and explicitly stating "there isn't enough evidence to know" is just as valuable as the answer itself.

---

## Getting Started

### Install

```bash
pip install -r requirements.txt   # google-genai, python-dotenv, jinja2

```

Add your API key to a `.env` file:

```env
GEMINI_API_KEY=your-key-here

```

### Run

```bash
python main.py

```

You'll be asked to describe your problem in plain language. From there, Xogito:

1. **Formalizes** your input into a strict specification and asks you to confirm it.
2. **Executes** an autonomous investigation cycle until the success criteria are met, the remaining uncertainty can't be reduced further, or a safety limit trips for user review.
3. **Generates** a comprehensive report to `reports/` as a self-contained HTML file.

---

## Repository Layout

```text
xogito/
├── main.py          # Entry point
├── orchestrator.py  # Run sequencer and lifecycle manager
├── workspace.py     # Shared state manager
├── parametres.py    # Framework configuration and constants
├── intake/          # Input parsing and specification ratification
├── agents/          # Specialized processing and verification modules
├── core/            # Core logic engine and system constraints
├── llm/             # Model provider integration layer
├── model/           # Internal data structures and message types
├── reporting/       # Output generation and formatting
└── docs/            # High-level design documentation

```

---

## Design Philosophy

Every component in the `xogito/` architecture exists to answer a specific failure mode common in naive LLM pipelines. That reasoning is written up in `docs/DESIGN.html`.

*Xogito — from the same root as cogito, "I think." The difference is what happens after the thinking.*
