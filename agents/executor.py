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
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field

from llm.client import ask_llm
from graph.workspace import Workspace, Task, Claim, Artifact


# ===========================================================================
# STEP 1 - knobs and fuses. code-owned, never appear in any prompt.
# ===========================================================================

MAX_PARALLEL_WORKERS = 4          # how many tasks run at once
CALL_TIMEOUT_SECONDS = 300       # one LLM call may not hang forever , 5min max
MAX_RETRIES = 3                   # attempts per task before it is "failed"
BACKOFF_BASE_SECONDS = 2          # wait 2s, 4s, 8s between attempts
BATCH_BUDGET_SECONDS = 1200        #  20 min wall-clock fuse for a whole batch


class ExecutorFuseTripped(Exception):
    # raised to HALT the run and show the human. never caught silently.
    pass


# ===========================================================================
# STEP 2 - the result dataclass. the ONLY thing a worker produces.
#
# Task is the assignment (lives in the workspace, part of the graph).
# ExecutionResult is the report of one attempt (a transient message:
# travels worker -> integration, then dissolves into artifacts, claims,
# status and provenance). Entities get one dataclass, messages another.
# ===========================================================================

@dataclass
class ExecutionResult:
    task_id: int
    status: str                                    # "completed" or "failed"
    content: str = ""                              # the actual work product
    summary: str = ""                              # one line, for future planning
    claim_statements: list = field(default_factory=list)  # asserted facts (strings)
    proposed_tasks: list = field(default_factory=list)    # missing work, whispered to the planner
    error: str = ""                                # filled when status == "failed"
    attempts: int = 0
    elapsed_seconds: float = 0.0


# ===========================================================================
# STEP 3 - the execution context (pure code, no LLM).
#
# A worker sees ONLY what its one task needs: the task itself, the spec
# essentials, the belief table, and the FULL content of the tasks it
# depends on (that content is its input - a summary would not be enough
# to build on). It never sees rejected tasks, pending tasks, or anything
# global - that is the planner's context, a different animal.
# Reads come from snapshot() -> deep copies, so workers hold no live state.
# ===========================================================================

def build_execution_context(task, spec, workspace):
    snapshot = workspace.snapshot()

    dependency_lines = []
    for dependency_id in task.depends_on:
        for artifact in snapshot["artifacts"]:
            if artifact.task_id == dependency_id:
                if artifact.summary:
                    dependency_lines.append(f"- {artifact.summary}")
                else:
                    dependency_lines.append("- (dependency produced no summary)")
                dependency_lines.append(f"  full output: {artifact.content}")

    belief_lines = []
    for claim in snapshot["claims"]:
        belief_lines.append(f"- [{claim.belief}] {claim.statement}")

    ps = spec["problem_specification"]
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
    )


# ===========================================================================
# STEP 5 - one worker run: LLM call under a timeout, retries with
# backoff, output shape-checked. Failure is a RESULT. Only programming
# errors escape this function.
# ===========================================================================

def _call_llm_with_timeout(prompt, schema):
    # ask_llm is synchronous; bound it by running in a single-use thread
    # and capping the wait.
    with ThreadPoolExecutor(max_workers=1) as single:
        future = single.submit(ask_llm, prompt, schema)
        return future.result(timeout=CALL_TIMEOUT_SECONDS)


def execute_task(task, spec, workspace):
    context = build_execution_context(task, spec, workspace)
    prompt = _build_worker_prompt(context)

    started = time.monotonic()
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _call_llm_with_timeout(prompt, EXECUTE_SCHEMA)
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

        except (FutureTimeout, ValueError, KeyError, json.JSONDecodeError) as error:
            last_error = f"attempt {attempt}: {type(error).__name__}: {error}"
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

    batch_started = time.monotonic()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as pool:
        futures = []
        for task in tasks:
            future = pool.submit(execute_task, task, spec, workspace)
            futures.append(future)

        for future in futures:
            _check_batch_budget(batch_started)
            result = future.result()
            results.append(result)

    # workers finish in any order; sort ONCE here so integration - and
    # therefore artifact ids and provenance order - is deterministic
    results.sort(key=_result_task_id)
    return results


def _result_task_id(result):
    return result.task_id


# ===========================================================================
# STEP 7 - integration (SERIAL, main thread only - the single door).
#
# The one place execution results become workspace state, through the
# workspace's normal guarded methods. Claims enter as "unverified":
# the executor asserts, the evaluator judges later.
# Called by the scheduler after execute_batch.
# ===========================================================================

def integrate_results(results, workspace):
    integrated_artifact_ids = []

    for result in results:

        if result.status == "failed":
            workspace.update_task_status(result.task_id, "failed")
            print(f"[executor] task {result.task_id} failed: {result.error}")
            continue

        # 1. the artifact enters through the front door
        artifact = Artifact(
            id=0,                      # workspace assigns the real id
            task_id=result.task_id,
            content=result.content,
            summary=result.summary,
        )
        workspace.add_artifact(artifact)

        # 2. each asserted claim enters UNVERIFIED and gets linked to the
        #    artifact that asserted it
        for statement in result.claim_statements:
            claim = Claim(
                id=0,                  # workspace assigns the real id
                statement=statement,
                belief="unverified",
            )
            workspace.add_claim(claim)
            workspace.link_evidence(claim.id, artifact.id)

        # 3. proposed tasks are NOT inserted - the planner is the single
        #    door for the task graph. surfaced here for the record; the
        #    next planner iteration sees completed summaries anyway.
        if result.proposed_tasks:
            joined = "; ".join(result.proposed_tasks)
            print(f"[executor] task {result.task_id} proposes further work: {joined}")

        # 4. only now does the task count as done
        workspace.update_task_status(result.task_id, "completed")
        integrated_artifact_ids.append(artifact.id)

    return integrated_artifact_ids

