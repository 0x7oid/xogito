# Xogito

> **A second brain for large language models.**

<p align="center">
  <img src="docs/LOGO.png" height="400">
</p>

LLMs are unreliable on hard questions because they treat reasoning as transient text. A premise assumed in one paragraph is indistinguishable, three paragraphs later, from a fact that was established. Nothing in the output records what rests on what, so errors compound silently and the final answer arrives with uniform confidence regardless of what's underneath it.

Xogito makes that reasoning state explicit and external. A run maintains a ledger the model cannot bypass: every claim is recorded individually, linked to the evidence produced for it, and assigned a belief state that changes only through transitions the code enforces. The model supplies judgment. The ledger decides what that judgment is allowed to establish.

## The mechanism, briefly

A run starts by turning your question into a ratified problem specification. Facts you declare are **contextual anchors**, carried verbatim and never reinterpreted; anything the system must assume is surfaced to you as an assumption before work begins, so guesses can't hide inside the framing.

The investigation loop then generates claims, and every claim climbs a **belief ladder**: it enters as unverified and gains standing only when linked evidence justifies the promotion, with each transition validated in code rather than asserted by the model. When two claims can't both be true, contradiction detection marks the pair **contested** and routes it to **adjudication** — the conflict is resolved with a recorded rationale or kept open, never silently dropped in favor of whichever claim came last.

One invariant holds throughout: model output never mutates state directly. Everything a model produces is a proposal that passes deterministic validation before it touches the ledger.

The report is a projection of that ledger. Every conclusion links back to the claims that support it. Every claim links to its evidence. Assumptions remain visibly separate from established facts. If independent evidence conflicts, both positions appear in the report instead of being merged into one confident answer. And when the evidence doesn't suffice, the report states that plainly — an unsupported conclusion is the exact failure the system exists to prevent, so it is never manufactured to fill the space.

## When a five-minute run is worth it

- A literature review or scientific synthesis where sources genuinely disagree and the disagreement is the point.
- Due diligence: checking whether a widely repeated statistic, "best practice," or vendor claim is actually backed by anything before building on it.
- A policy or spending recommendation that will be defended in front of a committee, where the reasoning needs an audit trail rather than a persuasive summary.
- Any decision where a wrong answer is expensive and "the model sounded sure" won't survive scrutiny.

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
