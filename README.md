# Xogito

> **A second brain for large language models.**

<p align="center">
  <img src="docs/LOGO.png" height="400">
</p>

Are you tired of asking a model a hard question and getting a confident answer built on a premise it never checked? Of watching a guess made in paragraph two get treated as an established fact by paragraph nine? Of long reasoning chains where every individual step looks competent and the conclusion is still wrong — with nothing in the output telling you *which* premise to distrust?

Those are not intelligence failures. They are bookkeeping failures: the model loses track of what it believes, on what basis, and what conflicts with what. Xogito does that bookkeeping externally and mechanically, so the model can't lose track.

## Who this is for

People making a decision that is expensive to get wrong, who need the reasoning to be *checkable* rather than merely persuasive:

- an analyst or strategist weighing contested evidence ("should we spend this budget on a finding that might not replicate?")
- a researcher or student who needs every conclusion traced back to the claims it actually rests on
- an engineer who wants to see what a reasoning pipeline looks like when the LLM proposes and deterministic code disposes

It is a batch reasoning tool, not a chat replacement. A run takes minutes and produces a report, not a conversation.

## What it actually does

- **Separates your facts from its guesses.** Facts you declare are anchors — never reworded, never second-guessed. Anything the system had to assume is surfaced instead of silently baked in.
- **Makes confidence expensive.** A claim starts unverified and climbs a belief ladder only with evidence, checked in code. Doubt, by contrast, is free: any claim can be contested at any time. The system's errors run toward under-confidence, never over-confidence.
- **Detects and adjudicates contradictions** between its own findings instead of silently keeping whichever came last.
- **Votes on judgment calls.** Label decisions are sampled multiple times and majority-decided, with the tie-break rule stated in code rather than left to chance.
- **Stops honestly.** If no progress is being made, or a component behaves abnormally, a fuse halts the run and the report says so. "The available evidence isn't sufficient" is a valid result, not a failure state.
- **Shows its work.** The report exhibits the reasoning — every conclusion labeled with earned confidence, every label traceable to evidence, every unresolved conflict shown with both sides. Every LLM call is also cached to disk, so an interrupted run replays instantly instead of re-paying for finished work.

## Usage

```bash
pip install -r requirements.txt   # google-genai, python-dotenv, jinja2
# put your Gemini API key in .env as API_KEY
python main.py
```

The intake asks four questions — only the problem statement is required. Declare your fixed facts and any reasoning rules you want followed; the run ratifies its formal framing with you before doing anything else. Reports land in `reports/` as self-contained HTML.

On Windows, run with UTF-8 enabled (`PYTHONUTF8=1`) — console encodings narrower than UTF-8 can choke on model output.

## The design rule everything follows

> **The LLM proposes; the code disposes.**

Model outputs never mutate state directly. They are structured proposals — candidate tasks, candidate verdicts, candidate framings — and every one passes deterministic validation before it touches anything. The model holds all of the judgment and none of the authority.

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
