# Xogito

> **A second brain for large language models.**

<p align="center">
  <img src="docs/LOGO.png" height="400">
</p>

Ask a frontier model a hard question and it will often produce a chain of reasoning in which every individual step is competent — and the conclusion is still wrong. Worse: *confidently* wrong, with nothing in the output telling you which of its premises deserves your trust.

Xogito exists because of what those failures have in common.

## The problem isn't intelligence. It's bookkeeping.

Inspect enough failed reasoning chains and a pattern emerges. The model rarely fails at any given step. It fails because it loses track of its own epistemic state:

- It treats a guess made in paragraph two as an established fact by paragraph nine.
- It resolves a contradiction between two of its own statements by silently forgetting one of them.
- It cites its own earlier speculation as support for later speculation.
- It cannot tell you, at the end, which conclusions rest on evidence and which rest on momentum.

These are **bookkeeping failures** — failures of the machinery that should track what is believed, on what basis, with what confidence, and what conflicts with what. A human analyst with a notebook, a ledger of claims, and the discipline to never promote a hunch to a fact without writing down why will outperform a smarter colleague who keeps everything in their head.

LLMs keep everything in their head. That's the whole problem.

## What Xogito does about it

Xogito is that notebook — a second brain that does the model's bookkeeping *for* it, externally, explicitly, and mechanically. The model supplies the judgment; Xogito supplies the ledger, and the ledger cannot be talked out of its rules.

The bet is simple: **if the bookkeeping is made external and mechanically enforced, ordinary LLM intelligence becomes sufficient for extraordinary reasoning reliability.**

So the product is not answers. It is *justified, auditable conclusions*. Every claim in a Xogito report carries a confidence label that was earned through gated transitions, backed by evidence that exists and can be traced, produced by a process whose every state change was recorded. Sometimes the most honest output is *"the available evidence isn't sufficient"* — and Xogito treats that as a valid result, not a failure.

## Why this is different

Plenty of systems chain LLM calls together. Xogito's distinguishing rule is a strict division of labor at every boundary:

> **The LLM proposes; the code disposes.**

Model outputs are never allowed to directly mutate the system's state. They are always structured *proposals* — candidate tasks, candidate verdicts, candidate problem framings — and every proposal passes through deterministic validation before it can touch anything. The model holds all of the judgment and none of the authority. The code holds all of the authority and exercises no judgment.

Doubt is free; confidence is expensive. A claim can be contested at any moment for nothing, but climbing from *unverified* to *verified* requires evidence, one rung at a time, checked in code. A system built this way fails safe: its errors are under-confidence (which costs efficiency), never over-confidence (which costs correctness).

## How a run works

1. **You describe your problem** in plain language, optionally declaring fixed facts ("the budget is $40k") that the system must never reinterpret.
2. **Xogito formalizes it** — and asks *you* to ratify the formal framing before anything else happens. If two framings genuinely tie, it asks you to choose rather than flipping a coin behind your back.
3. **The reasoning loop runs**: plan a few next steps, investigate them, judge what the findings actually establish, detect contradictions, repeat — until the success criteria are met, the remaining uncertainty is irreducible, or a safety fuse halts the run for human review.
4. **You get a report** that doesn't summarize the reasoning — it *exhibits* it: every conclusion labeled with earned confidence, every label traceable to evidence, every unresolved conflict shown with both sides.

## Running it

```bash
pip install -r requirements.txt   # google-genai, python-dotenv, jinja2
# put your API key in .env
python main.py
```

Reports are written to `reports/` as self-contained HTML.

## Repository overview

```
xogito/
├── main.py            # single entry point
├── orchestrator.py    # sequences the run; makes no judgments of its own
├── workspace.py       # the shared memory every module reads and writes through
├── parametres.py      # every constant in the framework, in one place
├── intake/            # turning your words into a ratified problem specification
├── agents/            # planner, scheduler, executor, evaluator, adjudicator, checkpoint
├── core/              # kernel (the rules of belief), calibration, compression
├── llm/               # the only module that talks to a model provider
├── model/             # message types passed between modules
├── reporting/         # turns a finished run into a human-first report
└── docs/              # design documentation
```

## Learn more

The architecture is the interesting part — every component exists because a specific failure mode demanded it. The full story, told from first principles:

**→ [docs/DESIGN.html](docs/DESIGN.html)** — *Xogito: The Architecture of Auditable Reasoning*

---

> **Xogito** — from the same root as *cogito*, "I think."
> The difference is what happens **after** the thinking.
