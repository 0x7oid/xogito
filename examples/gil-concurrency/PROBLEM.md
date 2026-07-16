# Example: Python Concurrency Decision

A real end-to-end run. The four intake answers below were given to Xogito
exactly as written; `report.html` / `report.json` in this folder are the
report it returned, unedited.

## Problem (required)

> Our Python 3.11 API service applies watermarks to images in pure Python
> (nested pixel loops, no C extensions that release the GIL).
> Single-threaded it processes 3,000 images per hour and we need 12,000
> per hour. Dana, our senior engineer, insists that moving the work to 8
> threads via ThreadPoolExecutor will give roughly 8x throughput. An
> external consultant disagrees and says the GIL makes threads useless for
> this workload and we must use multiprocessing. Dana also claims there is
> zero documented evidence that the GIL limits threaded CPU-bound
> performance in CPython. Decide which approach we should use - threading,
> multiprocessing, or asyncio - and explicitly verify both of Dana's
> claims: the 8x speedup claim and the zero-documented-evidence claim.

## Scope

> CPython 3.11 only. No rewriting in another language, no C extensions,
> no external processing services. The decision is only between threading,
> multiprocessing, and asyncio.

## Fixed facts (anchors)

> the server has 4 vCPUs ; current single-threaded throughput is 3,000
> images per hour ; the target is 12,000 images per hour ; the budget for
> new infrastructure is $0

## Reasoning rules (policy)

> *(skipped)*

## Output

- [`report.html`](report.html) - the report Xogito returned (open in a browser)
- [`report.json`](report.json) - the same report as data

To reproduce: `python scripts/run_scenario.py` (requires an API key in `.env`).
