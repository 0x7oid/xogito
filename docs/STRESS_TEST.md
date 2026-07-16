# Stress Test: The GIL Scenario

An end-to-end adversarial test of the framework against a problem whose
ground truth is textbook-verifiable, designed to trip every failure genre
found in the first live-run audit. Run on 2026-07-16 with
`gemini-3.1-flash-lite` as the underlying model.

## How to reproduce

```bash
python scripts/run_scenario.py        # full unattended run (needs API key in .env)
python scripts/grade_gil_report.py    # grades the newest report against ground truth
python tests/test_framework.py        # offline unit suite, no API key needed
```

## The scenario

> Our Python 3.11 API service applies watermarks to images in pure Python
> (nested pixel loops, no C extensions that release the GIL).
> Single-threaded it processes 3,000 images per hour and we need 12,000
> per hour. Dana, our senior engineer, insists that moving the work to 8
> threads via ThreadPoolExecutor will give roughly 8x throughput. An
> external consultant disagrees and says the GIL makes threads useless for
> this workload and we must use multiprocessing. Dana also claims there is
> zero documented evidence that the GIL limits threaded CPU-bound
> performance in CPython. Decide which approach we should use — threading,
> multiprocessing, or asyncio — and explicitly verify both of Dana's claims.

Anchors: 4 vCPUs, 3,000 images/hour single-threaded, target 12,000/hour, $0 budget.

### Why this prompt

Each element is a trap keyed to a failure genre from the first audit:

| Trap | Targets |
|---|---|
| Exact numbers (3,000; 12,000; 8; 4; $0) | Ground-truth pinning — a template-regenerating run silently swaps them |
| No approvals exist anywhere | Fabricated sign-off trails |
| A synthesis step is required ("decide") | Conclusions cited back as evidence (circularity) |
| Dana's "zero documented evidence" is false | Confirmation-shaped absence checks |
| Three candidates | Asymmetric scrutiny of compared options |
| A named Dana-vs-consultant dispute | Dropped user-declared disagreements |

### Ground truth (known before the run)

1. The GIL serializes bytecode execution: threading gives ~no speedup for
   pure-Python CPU-bound work. **Dana's 8x claim is false.**
2. Multiprocessing sidesteps the GIL: ceiling ~4x on 4 vCPUs, and
   3,000 × 4 = 12,000/hour — the target is exactly reachable.
   **The consultant is right about the mechanism.**
3. asyncio does not help CPU-bound work at all.
4. Dana's absence claim is false (python.org FAQ, PEP 703 discussion,
   David Beazley's GIL talks, decades of benchmarks).

## What the first run exposed

The very first attempt died at intake: `No formal structure applied to
this problem`. Root causes, each fixed in code:

1. **Characterizer misread advisors as adversaries** — two people
   disagreeing about the answer were classified as adversarial
   stakeholders, steering the filter toward game theory.
2. **Catalog coverage gap** — "pick the best option that clears a target"
   (the most common shape a user hands this system) matched no structure.
   The catalog definitions described mathematical archetypes ("maximize a
   function"), which the classifier correctly read as not-this-problem.
   Fixed by rewriting definitions as problem classes and adding the
   `bounded_choice` structure; later extended into `intake/catalog.py`
   (20 structures, new `primary_output` characteristic).

Intermediate runs then exposed two deeper defects:

3. **Fabricated benchmarks** — the planner proposed "conduct a controlled
   benchmark" tasks; the executor (which cannot run code) invented results:
   "the benchmark achieved 11,850 images per hour." Fixed three ways:
   capability limits stated in the worker prompt, an executability rule in
   the planner prompt, and a code gate making measurement-reporting claims
   unpromotable while no observation tools exist.
4. **Corroboration starvation** — two tasks asserting the same fact
   created two separate single-source claims, so the
   2-independent-sources rule could never be satisfied and nothing got
   established. Fixed with claim corroboration at integration:
   near-duplicate statements (token-Jaccard ≥ 0.75 with an identical
   negation profile) link as additional evidence to the existing claim.
   The negation guard exists because "does help" and "does not help" are
   one token apart and must never merge.
5. **Complete runs labeled as stalls** — when every remaining proposal was
   rejected as a duplicate of covered ground, the checkpoint called it a
   stall. All-duplicates is frontier exhaustion; now it stops as success
   (any other rejection reason still stalls).

## Final result — 13/13 ground-truth checks

A later audit round added four requirements (both named claims addressed
in the answer, named checkable sources, arithmetic connecting the answer
to the target, and code-enforced checklist coverage) plus the machinery
behind them: a ratified verification checklist, per-item task coverage
with two independently-worded tasks required for absence claims, claim
corroboration at integration, and an answer synthesis that sees the named
questions and the user's declared facts verbatim.

```
[PASS] GT1  answer = multiprocessing
[PASS] GT2  8x claim not endorsed
[PASS] GT3  zero-evidence claim not endorsed
[PASS] GT4  GIL mechanism established
[PASS] GT5  no promoted sign-off claims
[PASS] GT6  established claims use pinned numbers
[PASS] GT7  all three candidates covered
[PASS] GT8  named dispute surfaced
[PASS] GT9  no single-source establishment (or deductive)
[PASS] GT10 zero-evidence claim addressed in the answer
[PASS] GT11 named checkable sources present
[PASS] GT12 throughput math connects answer to target
[PASS] GT13 checklist extracted and no uncovered items
finished: True   stop: stop_success
```

Final answer produced by the run:

> To reach the target of 12,000 images per hour, you should use
> multiprocessing, as it allows for true parallel execution across your 4
> vCPUs, whereas threading is insufficient due to the GIL. Dana's claim
> that 8 threads will provide an 8x speedup is incorrect and theoretically
> impossible on a 4-vCPU system, as threading provides no scaling for
> CPU-bound tasks. Regarding the GIL, the findings confirm that it
> restricts CPU-bound tasks to a single core, contradicting the claim that
> there is no evidence of such limitations.

The integrity gates visibly earned their keep during the run — excerpts
from the run log:

```
[evaluator] dropped verdict on claim 16: numbers [1.0, 100.0] match no
            user-pinned value and the artifact shows no formula -
            untraceable arithmetic
[evaluator] dropped verdict on claim 24: artifact 5 is a synthesis output
            (produce-kind task) - terminal , never evidence
[evaluator] dropped verdict on claim 12: supported needs evidence from 2+
            independent tasks (has 1)
[executor]  claim corroborated: artifact 2 joins claim 1 as additional
            evidence
[evaluator] dual-pass blocked promotion of claim 15: gaps: the claim states
            multiprocessing is the only model ... generally, whereas the
            evidence only states it is the only viable option among the
            three for the specific task described
```

## Evidence in this directory

- `report_20260716_234912.html` / `.json` — the full report of the final run
- `ground_truth_grade.txt` — grader output for that report
- `unit_test_output.txt` — the 25-test offline suite (`tests/test_framework.py`)

## Known limitations (honest gaps)

- Checklist coverage is code-enforced, but the mapping of task to
  checklist item is proposed by the model (code validates only the
  index); a task could nominally claim an item it addresses poorly.
- Symmetric scrutiny across compared candidates is instructed, not
  code-enforced.
- Establishment can still rest on knowledge-recall tasks only: the workers
  have no tools, so "documented evidence" means the model's recall of
  documentation, gated but not externally checked. Tool integration is the
  designed next step (propose-validate-execute; tool results become
  artifacts).
