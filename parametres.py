'''
parametres.py - every constant of the framework , in one place .

constants belong to CODE and are invisible to every prompt . no value in
this file may ever be interpolated into prompt text - a prompt that knows
a cap starts aiming for it (Goodhart) , and a prompt that knows a
threshold starts arguing with it . code validates , prompts never see .

grouped by owning module . plain module-level constants , no config
classes , no env parsing .
'''

import os

# the project root , anchored to THIS file's location , NEVER the current
# working directory - a cwd-relative path would silently split cross-run
# history into multiple files depending on where python was launched from
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# shared vocabularies (used by more than one module)
# ===========================================================================

VALID_BELIEF_LABELS = ("unverified", "supported", "verified", "contested")

BELIEF_ORDER = {"unverified": 0, "supported": 1, "verified": 2}
# contested is more of a flag than a belief order so we exlcude it in this table

VALID_EVIDENCE_TYPES = ("empirical", "deductive", "testimonial")

# evidence type strength ordering for the lexicographic tiebreaker .
# a label ordering , not a score : deductive beats empirical beats
# testimonial . this mirrors the taxonomy's ceilings (a derivation can
# verify alone ; a pointer never can)
EVIDENCE_TYPE_ORDER = {"testimonial": 0, "empirical": 1, "deductive": 2}


# ===========================================================================
# llm/client.py
# ===========================================================================

# FIX: gemini-2.5-flash was retired for new api keys (404 as of july
# 2026) . gemini-3-flash-preview kept collapsing under 503 overload plus
# 429 rate limits mid-run ; the lite tier probes stable and carries much
# higher throughput quotas , which matters more here than raw capability -
# a run that finishes beats a smarter run that cannot
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# per-provider defaults . a model name only exists on its own api , so
# the active provider picks which of these applies at call time
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
OPENAI_MODEL = "gpt-4o-mini"

CALL_TIMEOUT_SECONDS = 300
CLAUDE_CALL_TIMEOUT_SECONDS = 600   # cli round-trips are slower than raw api

# self-consistency voting (Wang et al. 2023) : how many samples per voted
# judgment , and the sampling temperature . votes are on LABELS only ,
# never averaged . N is odd on purpose - a tie between two labels cannot
# happen with an odd sample count over a binary label
VOTING_N = 3
VOTING_TEMPERATURE = 0.7

# transport retry (client-level , distinct from the executor's task
# retry) : the preview models 503 in spikes that outlast a short backoff ,
# so the client waits longer between attempts - 3s , 9s , 27s , then
# capped at 60s per wait (60s also rides out per-minute rate-limit
# windows , which is what a 429 usually is on this api)
TRANSPORT_RETRIES = 6
TRANSPORT_BACKOFF_BASE_SECONDS = 3
TRANSPORT_MAX_BACKOFF_SECONDS = 60

# the llm call cache : json-lines , append-only , same file discipline as
# the calibration log
LLM_CACHE_PATH = os.path.join(PROJECT_ROOT, "llm_cache.jsonl")


# ===========================================================================
# core/calibration.py
# ===========================================================================

# one file , json-lines format : one entry per line , append and forget .
CALIBRATION_LOG_PATH = os.path.join(PROJECT_ROOT, "calibration_log.jsonl")


# ===========================================================================
# intake/formalization.py
# ===========================================================================

# scope fidelity (live stress run , round 2) : a named sub-question
# ("verify Dana's zero-documented-evidence claim") was dropped silently
# even though the user asked for it EXPLICITLY . the fix is structural :
# every named question or claim-to-verify is extracted into a checklist
# at formalization , ratified with the spec , and the checkpoint refuses
# stop_success while any item lacks a completed covering task . the cap
# is a fuse : more extracted items than this halts for a human
MAX_CHECKLIST_ITEMS = 10


# ===========================================================================
# agents/planner.py
# ===========================================================================

MAX_TASKS_PER_ITERATION = 10
MAX_TOTAL_TASKS = 30
# The llm doesn't know about these fuses, and the planner never proposes more than the minimum
# this is a mechanism to prevent the llm for generating tasks for sake of generating tasks , no given benefit in return.


# ===========================================================================
# agents/executor.py
# ===========================================================================

MAX_PARALLEL_WORKERS = 4          # how many tasks run at once

# claim corroboration at integration (live stress run) : two tasks
# asserting the same fact used to create two separate single-source
# claims , so no claim could ever satisfy the 2-independent-tasks rule -
# the independence gate was structurally unsatisfiable . a new statement
# whose token overlap with an existing claim clears this jaccard bound
# AND whose negation profile matches exactly is treated as corroboration :
# the new artifact links to the EXISTING claim instead of spawning a twin .
# the negation guard exists because "asyncio does help" vs "asyncio does
# not help" are one token apart and must never merge
CLAIM_MATCH_JACCARD = 0.75
NEGATION_TOKENS = ("not", "no", "never", "cannot", "without", "zero", "none")
MAX_RETRIES = 3                   # attempts per task before it is "failed"
BACKOFF_BASE_SECONDS = 2          # wait 2s, 4s, 8s between attempts
BATCH_BUDGET_SECONDS = 1200        #  20 min wall-clock fuse for a whole batch


# ===========================================================================
# agents/evaluator.py
# ===========================================================================

MAX_VERDICTS_PER_ITERATION = 40

# epistemic-integrity gates , added after a live-run audit found four
# failure genres the gauntlet let through : fabricated approvals ,
# single-source "supported" , synthesis outputs cited back as evidence ,
# and numbers in claims that never appeared in the user's input .
# gauntlet policy , not kernel law - these are quality standards

# non-deductive support needs evidence from 2+ DIFFERENT tasks before a
# claim may leave unverified (citation volume from one source is an echo ,
# not corroboration) . deductive evidence is exempt : a derivation is
# checked by re-tracing , not by source-counting
MIN_SOURCE_TASKS_FOR_SUPPORTED = 2

# outputs of these task kinds are TERMINAL : they may cite evidence but
# never serve as evidence for another claim in the same run . this is the
# no-circular-evidence rule - a synthesis feeding its own conclusion back
# into the evidence graph is bootstrapping with extra steps
TERMINAL_TASK_KINDS = ("produce",)

# claims asserting real-world events the system cannot observe (sign-offs ,
# approvals , meetings , verifications by named people) can never be
# promoted from run-internal artifacts : no llm text is evidence that a
# human approved something . matching claims stay unverified with the
# reason recorded
EXTERNAL_CONFIRMATION_MARKERS = (
    "sign-off", "signed off", "sign off", "signoff",
    "approval", "approved by", "approves",
    "confirmed by", "verified by", "validated by",
    "agreed to", "stakeholder", "formally accepted",
)

# a computed number in a claim is only promotable if the artifact shows
# its working - any of these markers counts as "the formula is visible" .
# crude v1 presence check , same spirit as SCOPE_MARKERS
FORMULA_MARKERS = ("=", "formula", "computed as", "calculation", "derived from")

# v1 has NO observation tools - no code execution , no web , no
# measurement . a claim reporting a benchmark or measurement is therefore
# always invented (the live stress run produced "the benchmark achieved
# 11,850 images per hour" from a worker that cannot run code) . such
# claims are unpromotable until tools exist ; when they do , a tool-call
# citation becomes the exemption
UNOBSERVED_MEASUREMENT_MARKERS = (
    "benchmark", "benchmarked", "measured", "we observed",
    "achieved a throughput", "was executed", "test run", "profiling showed",
)


# ===========================================================================
# agents/adjudicator.py
# ===========================================================================

MAX_PAIRS_PER_ITERATION = 20   # runaway-fight fuse


# ===========================================================================
# agents/contested.py
# ===========================================================================

# the investigate-first threshold for the contested decision function .
# DISABLED : no evidence justifies a value yet (generalize on the second
# instance) . the v1 stub routes every pair to "collapse" ; when fan-out
# data from real runs exists , this becomes the fan-out level above which
# a pair earns a verify task instead of an immediate resolution
INVESTIGATE_FANOUT_THRESHOLD = None   # disabled on purpose


# ===========================================================================
# agents/checkpoint.py
# ===========================================================================

MAX_ITERATIONS = 2   # hard budget fuse on the plan->execute->evaluate loop
