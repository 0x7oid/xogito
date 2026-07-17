# Xogito

> **A second brain for large language models.**

<p align="center">
  <img src="docs/LOGO.png" height="400">
</p>

LLMs are unreliable on hard questions because they treat reasoning as transient text. A premise assumed in one paragraph is indistinguishable, three paragraphs later, from a fact that was established. Nothing in the output records what rests on what, so errors compound silently and the final answer arrives with uniform confidence regardless of what's underneath it.

Xogito is a multi-agent system which makes that reasoning state explicit and external. A run maintains a ledger the model cannot bypass: every claim is recorded individually, linked to the evidence produced for it, and assigned a belief state that changes only through transitions the code enforces. The model supplies judgment. The ledger decides what that judgment is allowed to establish. The machinery behind this — contextual anchors, the belief ladder, contested claims and adjudication — is laid out in the [design document](docs/DESIGN.html).

## Problem classes

- Literature review and evidence synthesis, where sources genuinely disagree and the disagreement is the point
- Due diligence and fact verification: whether a repeated statistic, "best practice," or vendor claim is actually backed by anything
- Policy and strategy recommendations that will be defended in front of a committee
- Risk and compliance assessments that need a traceable basis for each finding
- Technology, vendor, or procurement evaluations where the marketing outruns the evidence
- Root-cause investigations, where the tempting explanation and the supported one often differ
- Scientific and technical research questions with contested or partial evidence
- Legal or regulatory analysis, as preparation for human review rather than a substitute for it

The common shape: a decision that needs an auditable chain of reasoning, not a persuasive answer.

A chat can't serve these cases, because a conversation loses its own structure as it scrolls: assumptions blend into conclusions, dropped threads disappear, and nothing preserves which statement rested on which source. The report format exists to keep assumptions, evidence, provenance, and unresolved disputes intact after the run ends.

## What a run looks like

Expect a dialogue before the work starts. The intake asks for four things, in plain language:

1. **Your problem** — what you want figured out or decided. The only required field.
2. **Scope** — what to include and what to leave out ("open-source options only", "ignore cost").
3. **Fixed facts** — things that are simply true for you, taken as given and never second-guessed ("budget is $2M", "the deadline is March").
4. **Reasoning rules** — how you want disagreements settled ("prefer recent sources", "official statistics beat blog posts").

Xogito then shows you the guesses it had to make about your situation and asks you to confirm, correct, or hand each one over for investigation. Once you ratify the framing, it works on its own and writes the report to `reports/` as self-contained HTML.

Runs are not instant. Depending on the question and the model, expect minutes to tens of minutes. Every model call is cached to disk, so an interrupted run replays its finished work instantly on restart instead of paying for it twice.

## Setup

```bash
pip install -r requirements.txt   # google-genai, python-dotenv, jinja2
python main.py
```

- **Provider** — set `LLM_PROVIDER` in `.env` to `gemini`, `openai`, or `claude`. If unset, Gemini is used when its key is present, OpenAI otherwise.
  - `gemini` — put your key in `.env` as `API_KEY`.
  - `openai` — put your key in `.env` as `OPENAI_API_KEY`, and `pip install openai`.
  - `claude` — no key needed; uses your locally installed, logged-in [Claude Code](https://claude.com/claude-code) CLI, so runs bill your subscription instead of an API account.
- **Models** — per provider in `parametres.py`: `DEFAULT_MODEL` (Gemini), `OPENAI_MODEL`, `CLAUDE_MODEL`. A stronger model raises the quality of judgment-heavy steps at higher cost.
- **Call timeout** — `CALL_TIMEOUT_SECONDS` in `parametres.py` (`CLAUDE_CALL_TIMEOUT_SECONDS` for the CLI, whose round-trips are slower).
- **Windows** — run with UTF-8 enabled (`PYTHONUTF8=1`); console encodings narrower than UTF-8 can choke on model output.

## Learn more

The full architecture, component by component, each traced to the failure mode that demanded it:

**→ [docs/DESIGN.html](docs/DESIGN.html)** — *Xogito: The Architecture of Auditable Reasoning*

---

