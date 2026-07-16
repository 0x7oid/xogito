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
from workspace import Workspace, Task

# the fuses moved to parametres.py (their explanation moved with them) -
# one file owns every constant , and no prompt can ever see them
from parametres import MAX_TASKS_PER_ITERATION, MAX_TOTAL_TASKS

class PlannerFuseTripped(Exception):
    # raised to HALT the run and show the human. never caught silently.
    pass


JUDGE_LABELS = {"sound", "vague", "duplicate", "invalid_dependency"}

VALID_KINDS = {"investigate", "produce"} 
# the kinds of tasks are of threee : investivate , produce , verify
# verify is for the contradictions only .


# ---------------------------------------------------------------------------
# Phase A - build the planning context (pure code, no LLM)

# this is one of the good reasons we introduced the summary field in the Artifact class: to reduce context costs for the planner
# the context : summaries of completed tasks, information on pending tasks, information on rejected tasks, and the current belief state of the claims
# ---------------------------------------------------------------------------

def _build_planning_context(spec, workspace):
    snapshot = workspace.snapshot()

    completed_lines = []
    rejected_lines = []
    pending_lines = []

    for task in snapshot["tasks"]:

        if task.status == "completed":
            # collect this task's artifact summaries (the executor wrote them)
            summaries = []
            for artifact in snapshot["artifacts"]:
                if artifact.task_id == task.id and artifact.summary:
                    summaries.append(artifact.summary)

            if summaries:
                joined = " | ".join(summaries)
            else:
                joined = "no summary"

            completed_lines.append(f"- [{task.kind}] {task.description} -> {joined}")

        elif task.status == "rejected":
            rejected_lines.append(
                f"- {task.description} (rejected: {task.rejection_reason})"
            )

        elif task.status in ("pending", "in_progress"):
            pending_lines.append(f"- [{task.kind}] {task.description}")

    belief_lines = []
    for claim in snapshot["claims"]:
        belief_lines.append(f"- [{claim.belief}] {claim.statement}")

    # finally this is the context that will be sent to the planner llm
    ps = spec["problem_specification"]
    return {
        "goal": ps["goal"],
        "constraints": ps["constraints"],
        "success_criteria": ps["success_criteria"],
        "scope": ps["scope"],
        "anchors": ps["contextual_anchors"],
        "assumptions": ps["assumptions"],
        "completed": "\n".join(completed_lines) or "(nothing yet - first iteration)",
        "rejected": "\n".join(rejected_lines) or "(none)",
        "pending": "\n".join(pending_lines) or "(none)",
        "beliefs": "\n".join(belief_lines) or "(no claims yet)",
        "total_tasks_so_far": len(snapshot["tasks"]),
    }


# ---------------------------------------------------------------------------
# Phase B - propose (LLM call #1)
# The prompt asks for the MINIMUM frontier. It never sees the fuses,
# never sees the judge's criteria. Exploration is permitted, not forced.
# Goodhart's law applied here
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
        "- Write done_when outcome-neutrally: 'the presence or absence of X is established'"
        " - never presupposing the answer.\n"
        "- If the problem statement names a disagreement between people, a "
        "specific claim someone made, or an explicit sub-question, propose a "
        "task that checks each one directly. User-named disputes are work "
        "items, not background.\n"
        "- When candidates are being compared, apply the same investigation "
        "to every candidate: a task like 'find known reliability issues of X' "
        "must exist once per candidate, never only for one of them.\n"
        "- To check a claim of the form 'no X exists / zero incidents', "
        "propose searches worded to FIND counter-evidence (different phrasings, "
        "different angles) - never a single search that mirrors the claim.\n"
    )
    result = json.loads(ask_llm(prompt, PROPOSE_SCHEMA))
    return result["tasks"], result["frontier_empty_reason"]


# ---------------------------------------------------------------------------
# Phase C - validate (pure code, raises loudly)
# validation in a nutshell includes : checking for max tasks per iteration, checking for max total tasks, checking for valid kinds, checking for vacuous done_when, checking for invalid dependencies
# ---------------------------------------------------------------------------

VACUOUS_DONE_WHENS = ("the task is complete", "task is done")

'''
TODO  : no code check yet (detecting leading phrasing in code is unreliable) — instead let the judge catch it: add to the judge prompt's vague label description: "or its done_when presupposes the answer." Zero new labels, reuses existing machinery.
'''
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

    for position, proposal in enumerate(proposals):

        if proposal["kind"] not in VALID_KINDS:
            raise ValueError(
                f"[planner] task {position} has kind {proposal['kind']!r} - "
                f"only {sorted(VALID_KINDS)} allowed (verify tasks are "
                "contradiction-triggered later, not planned)."
            )

        done_when = proposal["done_when"].strip()
        if len(done_when) < 15 or done_when.lower() in VACUOUS_DONE_WHENS:
            raise ValueError(
                f"[planner] task {position} has a vacuous done_when: {done_when!r}"
            )

        for dependency_index in proposal["depends_on"]:
            if dependency_index < 0 or dependency_index >= position:
                # forward or self reference: either reorderable garbage or a
                # cycle. reject loudly instead of guessing.
                raise ValueError(
                    f"[planner] task {position} depends on index {dependency_index} - "
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
    listing_lines = []
    for index, proposal in enumerate(proposals):
        listing_lines.append(
            f"{index}. [{proposal['kind']}] {proposal['description']} "
            f"(done when: {proposal['done_when']}; depends on: {proposal['depends_on']})"
        )
    listing = "\n".join(listing_lines)

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
    seen_indices = set()
    for judgment in judgments:

        judgment["label"] = judgment["label"].strip().lower()
        if judgment["label"] not in JUDGE_LABELS:
            raise ValueError(f"[planner-judge] bad label: {judgment['label']!r}")

        index = judgment["index"]
        if index < 0 or index >= len(proposals):
            raise ValueError(f"[planner-judge] bad index: {index}")

        if index in seen_indices:
            raise ValueError(f"[planner-judge] duplicate judgment for index {index}")

        seen_indices.add(index)

    expected_indices = set(range(len(proposals)))
    missing_indices = expected_indices - seen_indices
    if missing_indices:
        raise ValueError(f"[planner-judge] judge skipped tasks: {missing_indices}")

    return judgments


# ---------------------------------------------------------------------------
# Phase E - insert (pure code, through the single door)
#
# Accepted tasks go in as "pending". Rejected ones ALSO go in, as
# "rejected" with their reason - rejection is a state, not deletion.
# A rejected task's dependents are rejected too (they'd dangle otherwise).
# ---------------------------------------------------------------------------

def _insert_tasks(proposals, judgments, workspace):
    label_by_index = {}
    for judgment in judgments:
        label_by_index[judgment["index"]] = judgment

    position_to_real_id = {}
    accepted_ids = []

    for position, proposal in enumerate(proposals):
        judgment = label_by_index[position]

        # a task depending on a rejected task can't run - reject it too.
        # a dependency was rejected if it never made it into the
        # position_to_real_id map.
        depends_on_rejected = False
        for dependency_index in proposal["depends_on"]:
            if dependency_index not in position_to_real_id:
                depends_on_rejected = True
                break

        rejected = (judgment["label"] != "sound") or depends_on_rejected

        if judgment["label"] != "sound":
            reason = judgment["reason"]
        elif depends_on_rejected:
            reason = "depends on a rejected task"
        else:
            reason = ""

        if rejected:
            status = "rejected"
        else:
            status = "pending"

        task = Task(
            id=0,  # workspace assigns the real id
            description=proposal["description"],
            kind=proposal["kind"],
            depends_on=[],  # filled below for accepted tasks
            status=status,
        )
        task.done_when = proposal["done_when"]
        task.why_now = proposal["why_now"]
        task.rejection_reason = reason

        if not rejected:
            # translate proposal indices -> real workspace ids
            real_dependency_ids = []
            for dependency_index in proposal["depends_on"]:
                real_dependency_ids.append(position_to_real_id[dependency_index])
            task.depends_on = real_dependency_ids

        workspace.add_task(task, actor="planner")
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

# FIX: the checkpoint distinguishes "empty frontier with a stated reason"
# from "all proposals rejected" , but the stated reason was only printed
# and lost . the function now returns a small dict carrying both the
# accepted ids and the frontier-empty reason ; the empty accepted list is
# still the stall/stop signal it always was
def plan_next_tasks(spec, workspace):
    # every entry point re-checks ratification. cheap, and forgetting is silent.
    assert_spec_ratified(spec)

    context = _build_planning_context(spec, workspace)

    proposals, empty_reason = _propose_tasks(context)

    if not proposals:
        # empty frontier: legitimate stop signal for the checkpoint
        print(f"[planner] empty frontier: {empty_reason}")
        return {"accepted_task_ids": [], "frontier_empty_reason": empty_reason}

    proposals = _validate_proposals(proposals, workspace)
    judgments = _judge_proposals(proposals, context)
    accepted_ids = _insert_tasks(proposals, judgments, workspace)

    if not accepted_ids:
        # everything was rejected - also a stall signal for the checkpoint,
        # visible in the workspace as a batch of rejected tasks with reasons
        print("[planner] all proposals rejected this iteration - possible stall")

    return {"accepted_task_ids": accepted_ids, "frontier_empty_reason": ""}

# the other iterations are planned through the orchestrator