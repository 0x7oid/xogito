"""
planner.py - turns the ratified spec + current workspace state into the
NEXT few tasks. Frontier planning, not blueprint planning:

    Specification -> Planner -> a few uncertainty-reducing tasks
    -> Execution -> new knowledge -> Planner again -> ...

Design decisions (settled):
- The planner proposes only tasks workable RIGHT NOW. The full plan
  emerges from the loop. No task is ever generated at position N of a
  long list (quality decays down long lists).
- Flexible width, quality-gated: no truncation by position. The judge
  removes tasks only for being vague / duplicate / structurally invalid,
  NEVER for seeming unpromising (protects exploration).
- done_when is mandatory: a commitment at generation (filters mush),
  a contract at evaluation later. Judge criteria never appear in the
  generator prompt (graders' rubrics invisible to generators).
- Caps are FUSES, not targets, and never appear in any prompt:
  tripping one means "halt and show the human", never "plan complete".
- Rejection is a state, not deletion: rejected tasks are stored with a
  reason label, excluded from execution, shown to future iterations so
  the planner doesn't re-propose them.
- Two LLM calls per iteration (propose, judge). Everything else is code.
- Cycles: impossible by construction. add_task only accepts existing
  dependency ids, and we insert proposals in order - a cyclic proposal
  fails as "dependency does not exist". No DFS needed.

REQUIRED WORKSPACE EDITS (3 small ones, do them first):
  1. Task gains:  done_when: str = ""   why_now: str = ""
                  rejection_reason: str = ""
  2. task_statuses gains "rejected"
  3. (already there) Artifact.summary - used for planner context
"""

import json

from llm.client import ask_llm
from core.kernel import assert_spec_ratified
from graph.workspace import Workspace, Task


# ---------------------------------------------------------------------------
# Fuses. Code-owned. NEVER mentioned in any prompt (a model told "up to
# 10" drifts toward producing 10). Tripping a fuse is an abnormal halt.
# Derivations: healthy frontier observed to be 1-5 tasks -> ceiling 2x = 10.
# Healthy full run ~ 5 iterations x ~3 tasks = 15 -> budget 2x = 30.
# Revisit both after logging real runs.
# ---------------------------------------------------------------------------

MAX_TASKS_PER_ITERATION = 10   # runaway-generation fuse
MAX_TOTAL_TASKS = 30           # runaway-loop / API-cost fuse


class PlannerFuseTripped(Exception):
    # raised to HALT the run and show the human. never caught silently.
    pass


# ---------------------------------------------------------------------------
# Labels the judge may use. Qualitative only - no numbers, ever.
# ---------------------------------------------------------------------------

JUDGE_LABELS = {"sound", "vague", "duplicate", "invalid_dependency"}

VALID_KINDS = {"investigate", "produce"}   # planner never proposes "verify"
                                           # (verify tasks are contradiction-
                                           # triggered, evaluator's job later)


# ---------------------------------------------------------------------------
# Phase A - build the planning context (pure code, no LLM)
#
# The planner sees STRUCTURED state, never full artifact prose:
# belief table + one-line summaries. This is the light version of the
# Compressor - full carry-over compression comes much later.
# ---------------------------------------------------------------------------

def _build_planning_context(spec, workspace):
    snapshot = workspace.snapshot()

    completed_lines = []
    rejected_lines = []
    for task in snapshot["tasks"]:
        if task.status == "completed":
            # find this task's artifact summaries (executor wrote them)
            summaries = [
                a.summary for a in snapshot["artifacts"] if a.task_id == task.id
            ]
            joined = " | ".join(s for s in summaries if s) or "no summary"
            completed_lines.append(f"- [{task.kind}] {task.description} -> {joined}")
        elif task.status == "rejected":
            rejected_lines.append(
                f"- {task.description} (rejected: {task.rejection_reason})"
            )

    belief_lines = [
        f"- [{c.belief}] {c.statement}" for c in snapshot["claims"]
    ]

    pending_lines = [
        f"- [{t.kind}] {t.description}"
        for t in snapshot["tasks"]
        if t.status in ("pending", "in_progress")
    ]

    ps = spec["problem_specification"]
    return {
        "goal": ps["goal"],
        "constraints": ps["constraints"],
        "success_criteria": ps["success_criteria"],
        "scope": ps["scope"],
        "anchors": ps["contextual_anchors"],   # verbatim, never reworded
        "assumptions": ps["assumptions"],
        "completed": "\n".join(completed_lines) or "(nothing yet - first iteration)",
        "rejected": "\n".join(rejected_lines) or "(none)",
        "pending": "\n".join(pending_lines) or "(none)",
        "beliefs": "\n".join(belief_lines) or "(no claims yet)",
        "total_tasks_so_far": len(snapshot["tasks"]),
    }


# ---------------------------------------------------------------------------
# Phase B - propose (LLM call #1)
#
# The prompt asks for the MINIMUM frontier. It never sees the fuses,
# never sees the judge's criteria. Exploration is permitted, not forced.
# ---------------------------------------------------------------------------

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "description": "One of: investigate, produce",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Indices into THIS list (0-based), only "
                                       "where a task genuinely consumes another's "
                                       "output. Order alone is not a dependency.",
                    },
                    "done_when": {
                        "type": "string",
                        "description": "A concrete, checkable condition that tells "
                                       "an evaluator this task is finished.",
                    },
                    "why_now": {
                        "type": "string",
                        "description": "Which uncertainty this task reduces, in one clause.",
                    },
                },
                "required": ["description", "kind", "depends_on", "done_when", "why_now"],
            },
        },
        "frontier_empty_reason": {
            "type": "string",
            "description": "If you propose zero tasks, say why. Otherwise empty string.",
        },
    },
    "required": ["tasks", "frontier_empty_reason"],
}


def _propose_tasks(context):
    prompt = (
        "You are a planning component. Your job is NOT to produce a complete "
        "plan. Propose only the next few tasks that are workable RIGHT NOW "
        "and would most reduce uncertainty about reaching the goal.\n\n"
        "PROBLEM SPECIFICATION\n"
        f"Goal: {context['goal']}\n"
        f"Constraints: {context['constraints']}\n"
        f"Success criteria: {context['success_criteria']}\n"
        f"Scope: {context['scope']}\n"
        f"Fixed facts (treat as given, never re-investigate): {context['anchors']}\n"
        f"Assumptions so far: {context['assumptions']}\n\n"
        "CURRENT STATE\n"
        f"Completed tasks and what they produced:\n{context['completed']}\n\n"
        f"Tasks still pending or running (do not duplicate them):\n{context['pending']}\n\n"
        f"Previously rejected proposals (do not re-propose them):\n{context['rejected']}\n\n"
        f"Claims and their current belief labels:\n{context['beliefs']}\n\n"
        "RULES\n"
        "- Propose the MINIMUM set of tasks workable immediately. If one task "
        "is enough, propose one. Do not pad.\n"
        "- kind: 'investigate' (produces information) or 'produce' (produces a "
        "deliverable from existing information). Never anything else.\n"
        "- Every task must have done_when: a concrete, checkable condition. "
        "If you cannot state one, do not propose the task.\n"
        "- depends_on: indices into THIS list, only where a task genuinely "
        "consumes another's output. Order alone is not a dependency.\n"
        "- Each task must be completable by one focused worker in one "
        "sitting. If it decomposes further, propose only its first workable piece.\n"
        "- Before proposing, ask yourself: is there an angle on the goal that "
        "no completed task has touched? If yes, you may propose one task for "
        "it.\n"
        "- If the success criteria appear satisfiable with current knowledge, "
        "or remaining uncertainty cannot be reduced by investigation, return "
        "an empty task list and explain why in frontier_empty_reason.\n"
    )
    result = json.loads(ask_llm(prompt, PROPOSE_SCHEMA))
    return result["tasks"], result["frontier_empty_reason"]


# ---------------------------------------------------------------------------
# Phase C - validate (pure code, raises loudly)
# ---------------------------------------------------------------------------

def _validate_proposals(proposals, workspace):
    # fuse 1: runaway generation
    if len(proposals) > MAX_TASKS_PER_ITERATION:
        raise PlannerFuseTripped(
            f"planner proposed {len(proposals)} tasks in one iteration "
            f"(fuse: {MAX_TASKS_PER_ITERATION}). Abnormal - halting for human review."
        )

    # fuse 2: global budget (cost protection, should never trip normally)
    existing = len(workspace.snapshot()["tasks"])
    if existing + len(proposals) > MAX_TOTAL_TASKS:
        raise PlannerFuseTripped(
            f"total task count would reach {existing + len(proposals)} "
            f"(fuse: {MAX_TOTAL_TASKS}). Possible runaway loop - halting for human review."
        )

    for position, p in enumerate(proposals):
        if p["kind"] not in VALID_KINDS:
            raise ValueError(
                f"[planner] task {position} has kind {p['kind']!r} - "
                f"only {sorted(VALID_KINDS)} allowed (verify tasks are "
                "contradiction-triggered later, not planned)."
            )

        done = p["done_when"].strip()
        if len(done) < 15 or done.lower() in ("the task is complete", "task is done"):
            raise ValueError(
                f"[planner] task {position} has a vacuous done_when: {done!r}"
            )

        for dep in p["depends_on"]:
            if not (0 <= dep < position):
                # forward or self reference: either reorderable garbage or a
                # cycle. reject loudly instead of guessing.
                raise ValueError(
                    f"[planner] task {position} depends on index {dep} - "
                    "dependencies may only point to EARLIER tasks in the proposal."
                )

    return proposals


# ---------------------------------------------------------------------------
# Phase D - judge (LLM call #2, blind to the proposer's reasoning)
#
# The judge is a building inspector, not an investor: it rejects only
# for vagueness, duplication, or invalid structure - NEVER for a task
# seeming unpromising. That sentence is the exploration firewall.
# ---------------------------------------------------------------------------

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "label": {
                        "type": "string",
                        "description": "One of: sound, vague, duplicate, invalid_dependency",
                    },
                    "reason": {"type": "string"},
                },
                "required": ["index", "label", "reason"],
            },
        }
    },
    "required": ["judgments"],
}


def _judge_proposals(proposals, context):
    listing = "\n".join(
        f"{i}. [{p['kind']}] {p['description']} "
        f"(done when: {p['done_when']}; depends on: {p['depends_on']})"
        for i, p in enumerate(proposals)
    )
    prompt = (
        "You review proposed tasks against a problem specification and the "
        "work already completed. Judge every task exactly once, by index.\n\n"
        "Labels (use only these, no numbers):\n"
        "- 'sound': concrete, executable, not already done\n"
        "- 'vague': cannot tell what doing it would actually involve, or its "
        "completion condition is not checkable\n"
        "- 'duplicate': the same investigation or output already exists in "
        "the completed work below\n"
        "- 'invalid_dependency': its stated dependency does not actually "
        "feed it anything\n\n"
        "IMPORTANT: do NOT reject tasks for being unusual, indirect, or "
        "seemingly unpromising. Unconventional angles are allowed as long "
        "as they are concrete. You judge executability and redundancy, "
        "not promise.\n\n"
        f"Goal: {context['goal']}\n"
        f"Success criteria: {context['success_criteria']}\n\n"
        f"Completed work:\n{context['completed']}\n\n"
        f"Proposed tasks:\n{listing}\n"
    )
    judgments = json.loads(ask_llm(prompt, JUDGE_SCHEMA))["judgments"]

    # code checks the judge too - never trust, always verify
    seen = set()
    for j in judgments:
        j["label"] = j["label"].strip().lower()
        if j["label"] not in JUDGE_LABELS:
            raise ValueError(f"[planner-judge] bad label: {j['label']!r}")
        if not (0 <= j["index"] < len(proposals)):
            raise ValueError(f"[planner-judge] bad index: {j['index']}")
        if j["index"] in seen:
            raise ValueError(f"[planner-judge] duplicate judgment for index {j['index']}")
        seen.add(j["index"])
    if seen != set(range(len(proposals))):
        raise ValueError(f"[planner-judge] judge skipped tasks: {set(range(len(proposals))) - seen}")

    return judgments


# ---------------------------------------------------------------------------
# Phase E - insert (pure code, through the single door)
#
# Accepted tasks go in as "pending". Rejected ones ALSO go in, as
# "rejected" with their reason - rejection is a state, not deletion.
# A rejected task's dependents are rejected too (they'd dangle otherwise).
# ---------------------------------------------------------------------------

def _insert_tasks(proposals, judgments, workspace):
    label_by_index = {j["index"]: j for j in judgments}
    position_to_real_id = {}
    accepted_ids = []

    for position, p in enumerate(proposals):
        judgment = label_by_index[position]

        # a task depending on a rejected task can't run - reject it too
        depends_on_rejected = any(
            label_by_index[dep]["label"] != "sound" or dep not in position_to_real_id
            for dep in p["depends_on"]
        ) if p["depends_on"] else False

        rejected = judgment["label"] != "sound" or depends_on_rejected
        reason = (
            judgment["reason"] if judgment["label"] != "sound"
            else ("depends on a rejected task" if depends_on_rejected else "")
        )

        task = Task(
            id=0,  # workspace assigns the real id
            description=p["description"],
            kind=p["kind"],
            depends_on=[],  # filled below for accepted tasks
            status="rejected" if rejected else "pending",
        )
        task.done_when = p["done_when"]
        task.why_now = p["why_now"]
        task.rejection_reason = reason

        if not rejected:
            # translate proposal indices -> real workspace ids
            task.depends_on = [position_to_real_id[dep] for dep in p["depends_on"]]

        workspace.add_task(task)
        # add_task assigned the real id; cycles are impossible here because
        # dependencies can only reference already-inserted tasks (DAG by
        # construction - this is why we need no DFS).

        if not rejected:
            position_to_real_id[position] = task.id
            accepted_ids.append(task.id)

    return accepted_ids


# ---------------------------------------------------------------------------
# The one public function. The orchestrator calls this each iteration.
#
# Returns the list of newly accepted task ids.
# An EMPTY list is a signal, not a failure: the frontier is empty and the
# Checkpoint should consider stopping.
# ---------------------------------------------------------------------------

def plan_next_tasks(spec, workspace):
    # every entry point re-checks ratification. cheap, and forgetting is silent.
    assert_spec_ratified(spec)

    context = _build_planning_context(spec, workspace)

    proposals, empty_reason = _propose_tasks(context)

    if not proposals:
        # empty frontier: legitimate stop signal for the checkpoint
        print(f"[planner] empty frontier: {empty_reason}")
        return []

    proposals = _validate_proposals(proposals, workspace)
    judgments = _judge_proposals(proposals, context)
    accepted_ids = _insert_tasks(proposals, judgments, workspace)

    if not accepted_ids:
        # everything was rejected - also a stall signal for the checkpoint,
        # visible in the workspace as a batch of rejected tasks with reasons
        print("[planner] all proposals rejected this iteration - possible stall")

    return accepted_ids