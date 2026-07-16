'''
checkpoint.py - the loop's stopping brain . pure code , NO llm calls .

the orchestrator calls decide_checkpoint at the end of every iteration
with the workspace and the run log (the per-iteration records the
orchestrator accumulates - structure documented in reporting/report.py) .

the decision vocabulary is CLOSED - four labels , nothing else :
- "continue"     : keep looping
- "stop_success" : the success heuristic is satisfied
- "stop_stall"   : the loop is spinning without progress
- "stop_budget"  : the iteration fuse tripped

confidence is expensive , doubt is free - so stop_success has the
strictest conditions and stall/budget stop cheaply .

the checkpoint also collects WARNING SIGNALS for the report : it never
acts on them (acting would make it a judge) , it only carries them .
'''

from core.calibration import read_calibration_summary
from parametres import MAX_ITERATIONS, BELIEF_ORDER, NEGATION_TOKENS


CHECKPOINT_DECISIONS = ("continue", "stop_success", "stop_stall", "stop_budget")


# ===========================================================================
# the decision . checked strictest-first : success , then stall , then
# budget , then continue . a run that satisfies its success heuristic at
# the same moment its budget runs out deserves the success label
# ===========================================================================

def decide_checkpoint(spec, workspace, run_log):
    iteration_record = run_log[-1]
    snapshot = workspace.snapshot()

    decision = "continue"
    reason = "work remains and progress was made this iteration"

    # scope fidelity : named sub-questions the user explicitly asked for .
    # stop_success is refused while any lacks a COMPLETED covering task -
    # an empty frontier that skipped a named question is not done , it is
    # incomplete , and the report says which items were dropped
    uncovered = _uncovered_checklist_items(spec, snapshot)

    if uncovered and _success_heuristic_met(iteration_record, snapshot):
        decision = "stop_stall"
        reason = ("frontier exhausted but the run is INCOMPLETE - named "
                  "checklist items were never covered by a completed task: "
                  + "; ".join(uncovered))
    elif _success_heuristic_met(iteration_record, snapshot):
        decision = "stop_success"
        reason = (
            "frontier exhausted (empty, or every proposal was a duplicate "
            "of covered ground), no contested claims remain, and at least "
            "one supported-or-verified claim exists"
        )
    elif _stalled(iteration_record):
        decision = "stop_stall"
        reason = _stall_reason(iteration_record)
    elif len(run_log) >= MAX_ITERATIONS:
        decision = "stop_budget"
        reason = f"iteration count reached the budget fuse ({MAX_ITERATIONS})"

    warnings = _collect_warning_signals(run_log)
    if uncovered and decision != "continue":
        warnings.append({
            "signal": "uncovered_checklist_items",
            "detail": "the user explicitly asked for these and no completed "
                      "task covered them: " + "; ".join(uncovered),
        })

    return {
        "decision": decision,
        "reason": reason,
        "warnings": warnings,
    }


def required_coverage(item):
    # an absence-type item ("zero documented evidence" , "no incidents")
    # needs TWO independently-worded completed tasks : a single search
    # that found nothing closes no question , and two searches sharing a
    # wording share blind spots . detection is the same negation-token
    # heuristic the corroboration matcher uses
    lowered = f" {item.lower()} "
    for token in NEGATION_TOKENS:
        if f" {token} " in lowered:
            return 2
    return 1


def _uncovered_checklist_items(spec, snapshot):
    # pure code : an item is covered when enough COMPLETED tasks claim its
    # index (two for absence-type items , one otherwise) . pending or
    # in-progress does not count - the work must be done , and a failed
    # task un-covers its item again
    checklist = spec["problem_specification"].get("verification_checklist", [])
    uncovered = []
    for index, item in enumerate(checklist):
        completed_covering = 0
        for task in snapshot["tasks"]:
            if (getattr(task, "covers_checklist_item", -1) == index
                    and task.status == "completed"):
                completed_covering += 1
        if completed_covering < required_coverage(item):
            uncovered.append(item)
    return uncovered


def _success_heuristic_met(iteration_record, snapshot):
    # v1 HEURISTIC , marked as such : the honest version - "every success
    # criterion from the ratified spec has at least one supported-or-
    # verified claim relevant to it" - needs an llm to judge relevance ,
    # which the checkpoint must not have . the richer version would need :
    # a per-criterion relevance judgment (llm , code-validated) , plus a
    # coverage table criteria -> claim ids in the report . until then :
    # frontier empty AND no contested claims AND some established claim
    frontier_exhausted = (
        _frontier_was_empty(iteration_record)
        or _all_rejected_as_duplicates(iteration_record, snapshot)
    )
    if not frontier_exhausted:
        return False

    has_established_claim = False
    for claim in snapshot["claims"]:
        if claim.belief == "contested":
            return False
        if claim.belief in BELIEF_ORDER and BELIEF_ORDER[claim.belief] >= 1:
            # supported or verified
            has_established_claim = True

    return has_established_claim


def _all_rejected_as_duplicates(iteration_record, snapshot):
    # live stress run : a complete investigation ended labeled "stall"
    # because the planner re-proposed covered ground and the judge
    # rejected every proposal as a duplicate . all-duplicates IS frontier
    # exhaustion - nothing new remains to propose . any other rejection
    # reason (vague , invalid_dependency) stays a stall : those signal
    # malfunction , not completion . pure code : reads the rejection
    # reasons the planner recorded on this iteration's tasks
    created = iteration_record["tasks_created_by_planner"]
    if created == 0 or iteration_record["accepted_task_ids"]:
        return False
    recent_tasks = snapshot["tasks"][-created:]
    for task in recent_tasks:
        if task.status != "rejected":
            return False
        if "duplicate" not in task.rejection_reason.lower():
            return False
    return True


def _frontier_was_empty(iteration_record):
    # the planner created no tasks at all this iteration (an all-rejected
    # batch is NOT an empty frontier - it is a stall , handled below)
    return iteration_record["tasks_created_by_planner"] == 0


def _all_proposals_rejected(iteration_record):
    created = iteration_record["tasks_created_by_planner"]
    accepted = iteration_record["accepted_task_ids"]
    return created > 0 and len(accepted) == 0


def _no_progress(iteration_record):
    # no belief changed and no task completed across the full iteration
    beliefs_changed = len(iteration_record["evaluator"]["applied"])
    tasks_integrated = len(iteration_record["integrated_artifact_ids"])
    adjudications = len(iteration_record["adjudicator"]["resolved"])
    return beliefs_changed == 0 and tasks_integrated == 0 and adjudications == 0


def _stalled(iteration_record):
    if _frontier_was_empty(iteration_record):
        return True
    if _all_proposals_rejected(iteration_record):
        return True
    if _no_progress(iteration_record):
        return True
    return False


def _stall_reason(iteration_record):
    if _frontier_was_empty(iteration_record):
        stated = iteration_record["frontier_empty_reason"]
        if stated:
            return f"planner returned an empty frontier: {stated}"
        return "planner returned an empty frontier (no reason stated)"
    if _all_proposals_rejected(iteration_record):
        return "every task the planner proposed this iteration was rejected"
    return "no belief changed and no task completed across a full iteration"


# ===========================================================================
# warning signals for the report . collected , never acted on . labels and
# counts only - no arithmetic on calibration data happens here
# ===========================================================================

def _collect_warning_signals(run_log):
    warnings = []

    # accumulating done_when failures across the whole run
    total_done_when_failures = 0
    for record in run_log:
        total_done_when_failures += len(record["evaluator"]["done_when_failures"])
    if total_done_when_failures > 0:
        warnings.append({
            "signal": "done_when_failures",
            "detail": f"{total_done_when_failures} task(s) did not satisfy "
                      "their done_when condition across the run",
        })

    # contested pairs the adjudicator parked (overlap , bad classification)
    total_parked = 0
    for record in run_log:
        total_parked += len(record["adjudicator"]["parked"])
    if total_parked > 0:
        warnings.append({
            "signal": "parked_contested_pairs",
            "detail": f"{total_parked} contested pair(s) were parked "
                      "unresolved and appear with both sides in the report",
        })

    # propagation flags the adjudicator raised . v1 : no component spawns
    # tasks from flags yet , so every raised flag counts as unaddressed
    total_flags = 0
    for record in run_log:
        total_flags += len(record["adjudicator"]["propagation_flags"])
    if total_flags > 0:
        warnings.append({
            "signal": "unaddressed_propagation_flags",
            "detail": f"{total_flags} propagation flag(s) from adjudication "
                      "have not been addressed by any task",
        })

    # vote splits from calibration - label tiers only , never rates
    summary = read_calibration_summary()
    for tier in summary["vote_splits_by_tier"]:
        count = summary["vote_splits_by_tier"][tier]
        warnings.append({
            "signal": "vote_split",
            "detail": f"{count} voted judgment(s) were non-unanimous "
                      f"(tier: {tier})",
        })

    return warnings


# ===========================================================================
# criteria revision - DEFERRED . the channel is : the system PROPOSES a
# revision of the success criteria (when evidence shows they are
# unreachable or mis-stated) and ONLY THE USER ratifies it - same posture
# as spec ratification , the system never moves its own goalposts .
# not implemented : no run data yet shows what a revision proposal needs
# to contain . the stub exists so the orchestrator has a named seam .
# ===========================================================================

def propose_criteria_revision(spec, workspace, run_log):
    raise NotImplementedError(
        "criteria revision is designed but deferred - the system may "
        "propose , only the user ratifies . build when a real run "
        "produces a criteria dead-end to learn from ."
    )
