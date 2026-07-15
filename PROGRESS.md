# Xogito progress log

Resumable checklist. Update as sections complete, so an interrupted run
(usage limit, fuse trip, crash) restarts from the next unchecked box.

## Done in prior sessions
- [x] Core framework: intake, formalization, kernel, workspace, planner,
      scheduler (inner dispatch loop), executor, evaluator (gap-list
      dual-pass), adjudicator, contested stub, checkpoint, calibration
- [x] llm/client.py: native timeout, call cache, self-consistency voting
      wired at dual-pass + entailment gate, vote_split calibration kind
- [x] parametres.py unifying all constants (prompt-invisible)
- [x] orchestrator full loop wiring, fuse handling, halted-run reports
- [x] reporting/report.py + jinja2 template (v1, audit-first order)
- [x] Logic fixes: kernel evidence_ids, workspace actors + provenance
      format, deep-copy getters, Task field order, executor timeout,
      Verdict.is_negative, model.verdict rename, intake module restored
- [x] Committed on branch claude/framework-completion

## Current work
- [x] 1. Anchor tracing: trace_anchors in formalization detects dropped
      anchors at the source; assumptions labeled load_bearing/peripheral
      by the spec call (code-validated); review_assumptions_with_user asks
      the user about every load-bearing guess before ratification and
      promotes confirmations/corrections to anchors
- [x] 2+3. Human-first report rebuild: answer + plain summary + stat tiles
      + dropped-anchor/load-bearing warnings always visible; independent
      <details> panels with one-line teasers; decision map as nested
      collapsible tree; dependency map with per-claim summaries + inline
      SVG diagram; severity-distinct banners; jargon translated via
      closed maps in report.py; still one offline self-contained file
- [x] 4. core/compressor.py: statistics, surviving vs dropped claims,
      llm synthesis over established claims only, honest extractive
      fallback when synthesis is unavailable
- [x] 5. Token-efficiency review: findings written up in the session
      summary; safe fix applied (executor retry causes now printed).
      Flagged for sign-off: worker prompt sends the full claim table;
      no local gemma endpoint is actually configured yet
- [x] 6. scripts/eval_gpqa.py written. NOT RUN: the dataset is gated on
      Hugging Face (needs accepted terms + login + a download approval)
      and no gemma-3n-e4b endpoint is configured in this repo. Run with:
      pip install datasets && python scripts/eval_gpqa.py gemma-3n-e4b
- [x] Smoke test updated + passing (compressor fallback, dropped anchors
      visible, halted-run plain wording, SVG present)
- [ ] PR: branch pushed; PR creation blocked - gh CLI not installed on
      this machine (see session notes for the compare URL)
