# the executor takes the tasks from the shceduler , executes them and returns artifacts and claims to the workspace , and logs them ofc
# what are the steps of the executor ?
'''
the executor should get the tasks from  the scheduler (those are the raedy  tasks from the worskapce)
the tasks must be gotten as a batch
the tasks with no  dependencies are executed in paralel , the tasks with dependencies are serialized
the main  thread here will insert the return of each task into the execution_return object in serial order.
so we need to start by creating the execution_return dataclass
'''

"""
    scheduler -> execute_batch(tasks, spec, workspace) -> [ExecutionResult]
    scheduler -> integrate_results(results, workspace) -> workspace updated

- Parallelism: up to MAX_PARALLEL_WORKERS at once (4). Workers finish
  in any order; results are collected as they come and sorted ONCE by
  task id before integration, so runs stay deterministic.
- A worker failure is a RESULT, not a crash: retries with exponential
  backoff, then status="failed" with the error recorded. One task dying
  never kills the batch.
- Fuses: per-call timeout, per-batch wall-clock budget. Tripping the
  budget = halt and show the human, never "good enough, continue".
- The executor NEVER creates tasks. Missing work goes into
  proposed_tasks on the result; the next planner iteration sees it.
- Claims born here enter the workspace as "unverified" and get linked
  to the artifact that asserted them. Belief changes are the
  evaluator's job - no belief ever changes in this file.

REQUIRED WORKSPACE EDIT (the tripwire, do it once):
    import threading
    # in Workspace, on construction (e.g. __post_init__):
    self._owner_thread_id = threading.get_ident()
    # first line of _log_provenance:
    assert threading.get_ident() == self._owner_thread_id, \
        "workspace write from a worker thread - integration must be serial"
"""

import json
import re
import time
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from llm.client import ask_llm
from workspace import Workspace, Task, Claim, Artifact


# ===========================================================================
# PARAMETRES
# ===========================================================================

# the fuses moved to parametres.py (their explanations moved with them) -
# one file owns every constant , and no prompt can ever see them
from parametres import (
    MAX_PARALLEL_WORKERS,
    MAX_RETRIES,
    BACKOFF_BASE_SECONDS,
    BATCH_BUDGET_SECONDS,
    CLAIM_MATCH_JACCARD,
    NEGATION_TOKENS,
)


class ExecutorFuseTripped(Exception):
    # raised to HALT the run and show the human. never caught silently.
    pass


# ===========================================================================

# tHE workers takes a task + execution context , executes it and returns an execution result object
# the execution result object is a report of the exectuion of the task
# ===========================================================================
POSSIBLE_STATUS = Literal["completed", "failed"]

@dataclass
class ExecutionResult:
    task_id: int
    status: POSSIBLE_STATUS                            # "completed" or "failed"
    content: str = ""                              # the actual work product
    summary: str = ""                              # one line, for future planning
    claim_statements: list = field(default_factory=list)  # asserted facts (strings)
    proposed_tasks: list = field(default_factory=list)    # missing work, whispered to the planner
    error: str = ""                                # filled when status == "failed"
    attempts: int = 0
    elapsed_seconds: float = 0.0


# ===========================================================================
# STEP 3 - the execution context of the task (pure code, no LLM).
#
# A worker sees ONLY what its one task needs: the task itself, the spec
# essentials, the belief table, and the FULL content of the tasks it
# depends on (that content is its input - a summary would not be enough
# to build on). It never sees rejected tasks, pending tasks, or anything
# Reads come from snapshot() -> deep copies, so workers hold no live state.
# ===========================================================================

def build_execution_context(task, spec, snapshot):
    dependency_lines = []
    # for each dependency, find the artifact produced by that task and include its summary and full content in the context
    for dependency_id in task.depends_on:
        for artifact in snapshot["artifacts"]:
            if artifact.task_id == dependency_id:
                if artifact.summary:
                    dependency_lines.append(f"- {artifact.summary}")
                else:
                    dependency_lines.append("- (dependency produced no summary)")
                dependency_lines.append(f"  full output: {artifact.content}")

    belief_lines = []
    # for each claim, include its belief and statement in the context
    for claim in snapshot["claims"]:
        belief_lines.append(f"- [{claim.belief}] {claim.statement}")

    ps = spec["problem_specification"]
    # finally we return the execution context 
    return {
        "task_id": task.id,
        "task_description": task.description,
        "task_kind": task.kind,
        "done_when": task.done_when,
        "goal": ps["goal"],
        "constraints": ps["constraints"],
        "scope": ps["scope"],
        "anchors": ps["contextual_anchors"],       # verbatim, never reworded
        "dependencies": "\n".join(dependency_lines) or "(this task has no dependencies)",
        "beliefs": "\n".join(belief_lines) or "(no claims established yet)",
    }


# ===========================================================================
# STEP 4 - the worker prompt and schema (LLM boundary).
# ===========================================================================

EXECUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": "The full work product of this task.",
        },
        "summary": {
            "type": "string",
            "description": "ONE line stating what this task found or produced. "
                           "This is what future planning sees.",
        },
        "claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Factual assertions this work makes, one per string, "
                           "each checkable on its own. Empty list if none.",
        },
        "proposed_tasks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Work you discovered is MISSING but is not yours to do. "
                           "One clause each. Empty list if none.",
        },
    },
    "required": ["content", "summary", "claims", "proposed_tasks"],
}


def _build_worker_prompt(context):
    return (
        "You are an execution worker. Do ONE task, completely, and report.\n\n"
        f"THE TASK ({context['task_kind']}): {context['task_description']}\n"
        f"It counts as done when: {context['done_when']}\n\n"
        "BACKGROUND\n"
        f"Overall goal: {context['goal']}\n"
        f"Constraints: {context['constraints']}\n"
        f"Scope: {context['scope']}\n"
        f"Fixed facts (treat as given): {context['anchors']}\n\n"
        f"OUTPUT OF THE TASKS THIS ONE DEPENDS ON\n{context['dependencies']}\n\n"
        f"CLAIMS ESTABLISHED SO FAR (with confidence labels)\n{context['beliefs']}\n\n"
        "RULES\n"
        "- Do the task itself. Do not plan, do not delegate, do not describe "
        "what you would do - produce the actual output.\n"
        "- Aim squarely at the done-when condition above.\n"
        "- 'claims': list the factual assertions your output makes, each one "
        "self-contained and checkable. If your output asserts nothing "
        "factual, return an empty list. Do not inflate.\n"
        "- Treat claims labeled [unverified] or [contested] with caution - "
        "do not build on them as if they were certain.\n"
        "- 'proposed_tasks': if you notice work that is missing but outside "
        "this task, name it there. Do NOT attempt it yourself.\n"
        "- 'summary': one line, concrete, stating the result (not the effort).\n"
        # FIX: this line was a plain string , not an f-string , so the raw
        # text "{context['scope']}" leaked verbatim into the prompt . the
        # scope is already shown under BACKGROUND above - the broken
        # interpolation is simply removed
        "- If your task is an investigation, state WHERE and HOW you searched, "
        "and report any evidence AGAINST the main finding if you "
        "encountered it. Do not invent counter-evidence.\n"
        "- When you verify or refute a QUANTITATIVE claim, state the "
        "expected observable magnitude, never a bare 'possible/impossible' "
        "(e.g. 'expect roughly 1x, potentially slower under contention' "
        "beats 'a speedup is impossible').\n"
        "- Ground factual assertions in NAMED, checkable sources where they "
        "exist: official documentation sections, PEPs, published analyses - "
        "named specifically enough that a reader could look them up. A "
        "source you recall but cannot fully cite is still worth naming as "
        "such; it will be treated as testimonial evidence.\n"
        "- When your output involves the user's stated numbers, show the "
        "arithmetic explicitly as a formula connecting your conclusion to "
        "those numbers (e.g. '3,000/hr x 4 processes = 12,000/hr, meeting "
        "the target') - never assert a derived number without its "
        "derivation.\n"
        "- CAPABILITY LIMITS: you cannot run code, execute benchmarks, "
        "measure anything, access the web, or contact anyone. Your only "
        "capabilities are recalling documented knowledge and deriving "
        "conclusions step by step. If the task requires an action you "
        "cannot perform (a benchmark, a measurement, an approval), say so "
        "plainly in the content, name the missing capability in "
        "proposed_tasks, and assert NO claims about results you did not "
        "actually obtain. Reporting 'this cannot be done from here' is a "
        "correct and complete answer.\n"
    )


# ===========================================================================
# STEP 5 - call llm with worker
# ===========================================================================


def execute_task(task, spec, snapshot):
    context = build_execution_context(task, spec, snapshot)
    prompt = _build_worker_prompt(context)

    started = time.monotonic()
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = ask_llm(prompt, EXECUTE_SCHEMA)
            output = json.loads(raw)

            # never trust, always verify - minimal shape checks
            content = output["content"].strip()
            if not content:
                raise ValueError("worker returned empty content")

            summary = output["summary"].strip()
            if not summary:
                raise ValueError("worker returned empty summary")

            claim_statements = []
            for statement in output["claims"]:
                cleaned = statement.strip()
                if cleaned:
                    claim_statements.append(cleaned)

            proposed_tasks = []
            for proposal in output["proposed_tasks"]:
                cleaned = proposal.strip()
                if cleaned:
                    proposed_tasks.append(cleaned)

            return ExecutionResult(
                task_id=task.id,
                status="completed",
                content=content,
                summary=summary,
                claim_statements=claim_statements,
                proposed_tasks=proposed_tasks,
                attempts=attempt,
                elapsed_seconds=time.monotonic() - started,
            )

        except Exception as error:
            # FIX: the old tuple (FutureTimeout, ValueError, KeyError,
            # JSONDecodeError) let the google client's own error types
            # (network failures , http timeouts , rate limits) escape the
            # retry loop , crash the worker thread and kill the whole
            # batch - violating "a worker failure is a RESULT, not a
            # crash" . a hung call is already bounded by the native http
            # timeout in llm/client.py , so catching broadly here only
            # converts failures into results , never hides a hang
            last_error = f"attempt {attempt}: {type(error).__name__}: {error}"
            # every retry burns a full llm call , so its cause is never
            # silent - if schema violations keep showing up here , that is
            # a quiet token sink worth fixing at the prompt/schema level
            print(f"[executor] task {task.id} retrying after {last_error}")
            if attempt < MAX_RETRIES:
                # exponential backoff: 2s, 4s, 8s
                time.sleep(BACKOFF_BASE_SECONDS ** attempt)

    # all retries exhausted - the fallback: report failure as a result
    return ExecutionResult(
        task_id=task.id,
        status="failed",
        error=last_error,
        attempts=MAX_RETRIES,
        elapsed_seconds=time.monotonic() - started,
    )


# ===========================================================================
# STEP 6 - the batch runner (the parallel part).
#
# Contexts are built from snapshots, workers share nothing mutable, so
# up to MAX_PARALLEL_WORKERS tasks run at once safely. Results are
# collected as they finish ("like they are") and sorted ONCE by task id
# before returning - deterministic integration order for free.
# ===========================================================================

def _check_batch_budget(batch_started):
    elapsed = time.monotonic() - batch_started
    if elapsed > BATCH_BUDGET_SECONDS:
        raise ExecutorFuseTripped(
            f"batch exceeded its wall-clock budget "
            f"({elapsed:.0f}s > {BATCH_BUDGET_SECONDS}s). Halting for human review."
        )


def execute_batch(tasks, spec, workspace):
    if not tasks:
        return []
    snapshot = workspace.snapshot()
    batch_started = time.monotonic()
    results = []

    # FIX: the old `with ThreadPoolExecutor(...) as pool:` form made the
    # budget fuse cosmetic - future.result() blocked with no timeout , and
    # even on a fuse trip the context-manager exit waited for every hung
    # worker to return . now each wait is bounded by the REMAINING budget
    # and the pool is shut down with wait=False in a finally , so tripping
    # the fuse actually frees the main thread
    pool = ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS)
    try:
        futures = []
        for task in tasks:
            future = pool.submit(execute_task, task, spec, snapshot)
            futures.append(future)

        for future in futures:
            _check_batch_budget(batch_started)
            remaining_seconds = BATCH_BUDGET_SECONDS - (time.monotonic() - batch_started)
            try:
                result = future.result(timeout=remaining_seconds)
            except FutureTimeout:
                raise ExecutorFuseTripped(
                    f"batch exceeded its wall-clock budget "
                    f"({BATCH_BUDGET_SECONDS}s) while waiting on a worker. "
                    "Halting for human review."
                )
            results.append(result)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # workers finish in any order; sort ONCE here so integration - and
    # therefore artifact ids and provenance order - is deterministic
    results.sort(key=_result_task_id)
    return results


def _result_task_id(result):
    return result.task_id


# ===========================================================================
# STEP 7 - integration (SERIAL, main thread only the single door).
# task execution is parallel, but integration is serial. the workspace is not thread-safe, so we must integrate results one by one in the main thread.
# ===========================================================================

# ---------------------------------------------------------------------------
# claim corroboration (live stress run) : two tasks asserting the same
# fact used to spawn two single-source twins , so the evaluator's
# 2-independent-tasks rule could never be met and nothing got established .
# a near-identical statement is now treated as CORROBORATION of the
# existing claim - the new artifact links to it as additional evidence .
# matching is conservative : token jaccard above a code-owned bound AND an
# identical negation profile ("does help" / "does not help" are one token
# apart and must never merge)
# ---------------------------------------------------------------------------

_CLAIM_TOKEN_PATTERN = re.compile(r"[a-z0-9.]+")


def _claim_profile(statement):
    tokens = set(_CLAIM_TOKEN_PATTERN.findall(statement.lower()))
    negations = tuple(sorted(t for t in tokens if t in NEGATION_TOKENS))
    return tokens, negations


def _find_corroborated_claim(statement, known_profiles):
    # known_profiles : claim_id -> (token_set , negation_profile) .
    # returns the id of the matching existing claim , or None
    tokens, negations = _claim_profile(statement)
    if not tokens:
        return None
    for claim_id in known_profiles:
        known_tokens, known_negations = known_profiles[claim_id]
        if negations != known_negations:
            continue
        union = tokens | known_tokens
        overlap = tokens & known_tokens
        if union and len(overlap) / len(union) >= CLAIM_MATCH_JACCARD:
            return claim_id
    return None


def integrate_results(results, workspace):
    integrated_artifact_ids = []

    # profiles of every claim already in the workspace , extended as this
    # batch adds new ones - so corroboration works within a batch too
    known_profiles = {}
    evidence_by_claim = {}
    for existing in workspace.snapshot()["claims"]:
        known_profiles[existing.id] = _claim_profile(existing.statement)
        evidence_by_claim[existing.id] = set(existing.evidence_ids)

    for result in results:

        if result.status == "failed":
            workspace.update_task_status(result.task_id, "failed", actor="executor")
            print(f"[executor] task {result.task_id} failed: {result.error}")
            continue

        # 1. the artifact enters through the front door
        artifact = Artifact(
            id=0,                      # workspace assigns the real id
            task_id=result.task_id,
            content=result.content,
            summary=result.summary,
        )
        workspace.add_artifact(artifact, actor="executor")

        # 2. each asserted claim enters UNVERIFIED and gets linked to the
        #    artifact that asserted it - unless it corroborates an existing
        #    claim , in which case the artifact becomes ADDITIONAL evidence
        #    for that claim instead of spawning a single-source twin
        for statement in result.claim_statements:
            matched_id = _find_corroborated_claim(statement, known_profiles)

            if matched_id is not None:
                if artifact.id in evidence_by_claim[matched_id]:
                    continue   # same artifact asserting the same fact twice
                try:
                    workspace.link_evidence(matched_id, artifact.id, actor="executor")
                    evidence_by_claim[matched_id].add(artifact.id)
                    print(f"[executor] claim corroborated: artifact {artifact.id} "
                          f"joins claim {matched_id} as additional evidence")
                except ValueError as error:
                    print(f"[executor] corroboration link failed on claim "
                          f"{matched_id}: {error}")
                continue

            claim = Claim(
                id=0,                  # workspace assigns the real id
                statement=statement,
                belief="unverified",
            )
            workspace.add_claim(claim, actor="executor")
            workspace.link_evidence(claim.id, artifact.id, actor="executor")
            known_profiles[claim.id] = _claim_profile(statement)
            evidence_by_claim[claim.id] = {artifact.id}

        # 3. proposed tasks are NOT inserted - the planner is the single
        #    door for the task graph. surfaced here for the record; the
        #    next planner iteration sees completed summaries anyway.
        if result.proposed_tasks:
            joined = "; ".join(result.proposed_tasks)
            print(f"[executor] task {result.task_id} proposes further work: {joined}")

        # 4. only now does the task count as done
        workspace.update_task_status(result.task_id, "completed", actor="executor")
        integrated_artifact_ids.append(artifact.id)

    return integrated_artifact_ids

