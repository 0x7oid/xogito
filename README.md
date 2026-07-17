# Xogito

> **A second brain for large language models.**

<p align="center">
  <img src="docs/LOGO.png" height="400">
</p>

LLMs are unreliable on hard questions because they treat reasoning as transient text. A premise assumed in one paragraph is indistinguishable, three paragraphs later, from a fact that was established. Nothing in the output records what rests on what, so errors compound silently and the final answer arrives with uniform confidence regardless of what's underneath it.

Xogito makes that reasoning state explicit and external. A run maintains a ledger the model cannot bypass: every claim is recorded individually, linked to the evidence produced for it, and assigned a belief state that changes only through transitions the code enforces. The model supplies judgment. The ledger decides what that judgment is allowed to establish.

## The mechanism, briefly

A run starts by turning your question into a ratified problem specification. Facts you declare are **contextual anchors**, carried verbatim and never reinterpreted; anything the system must assume is surfaced to you as an assumption before work begins, so guesses can't hide inside the framing.

The investigation loop generates claims. Every claim enters the ledger separately, as unverified. It gains standing on the **belief ladder** only when supporting evidence justifies a promotion, and those promotions are validated by deterministic code rather than the language model. When two claims can't both be true, the pair is marked **contested** and sent to **adjudication**: the conflict is resolved with a recorded rationale, or kept open. It is never silently dropped in favor of whichever claim came last.

One invariant holds throughout: model output never mutates state directly. Everything a model produces is a proposal that passes deterministic validation before it touches the ledger.

The report is a projection of that ledger. Every conclusion links back to the claims that support it. Every claim links to its evidence. Assumptions remain visibly separate from established facts. If independent evidence conflicts, both positions appear in the report instead of being merged into one confident answer. And when the evidence doesn't suffice, the report states that plainly — an unsupported conclusion is the exact failure the system exists to prevent, so it is never manufactured to fill the space.

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

## Usage

```bash
pip install -r requirements.txt   # google-genai, python-dotenv, jinja2
# put your Gemini API key in .env as API_KEY
python main.py
```

The intake asks four questions and only the problem statement is required. Reports land in `reports/` as self-contained HTML.

On Windows, run with UTF-8 enabled (`PYTHONUTF8=1`).

## Learn more

The full architecture, component by component, each traced to the failure mode that demanded it:

**→ [docs/DESIGN.html](docs/DESIGN.html)** — *Xogito: The Architecture of Auditable Reasoning*

---

> **Xogito** — from the same root as *cogito*, "I think."
> The difference is what happens **after** the thinking.
