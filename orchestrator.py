'''
LAYOUT DECISION (reconciled across every file) : the project runs from
the project ROOT . workspace.py , parametres.py and orchestrator.py are
top-level modules ; agents/ , core/ , intake/ , llm/ , model/ and
reporting/ are packages addressed by folder path (agents.evaluator ,
core.kernel , intake.intake , llm.client , model.verdict ,
reporting.report) . flat imports like `from workspace import ...` are
CORRECT under this layout because workspace.py sits at the root ; the
package-relative style (`from intake import UserQuery`) was the
inconsistency and is gone . model.veridict was renamed model.verdict .

the orchestrator SEQUENCES and PASSES DATA . it makes no judgments ,
contains no llm calls and no voting logic . every fuse exception halts
the loop - and a halted run still produces its report , because an audit
trail that disappears exactly when things go wrong is worthless .
'''

from intake.intake import collect_user_query

'''
intake.py
formalisation.py
planner.py
scheduler.py
executors.py
evaluator.py
checkpoint.py
│
├── finished? ── Yes ──► reporter.py ──► End
│
└── No
      │
      └──────────── back to planner.py
'''

from intake.formalization import (
    formalize,
    build_problem_specification,
    ratify_with_user,
)
from core.kernel import assert_spec_ratified
from workspace import Workspace
from agents.planner import plan_next_tasks, PlannerFuseTripped
from agents.scheduler import monitor_and_schedule_tasks
from agents.executor import ExecutorFuseTripped
from agents.evaluator import evaluate_iteration, EvaluatorFuseTripped
from agents.contested import triage_contested_pairs
from agents.adjudicator import adjudicate_contested_pairs, AdjudicatorFuseTripped
from agents.checkpoint import decide_checkpoint
from reporting.report import generate_report


def run_orchestrator():
    user_query = collect_user_query()
    print("\nThank you for providing your input.")
    print("Here is a summary of your query:")
    print(user_query.summary())

    # formalization : the user's problem becomes a formal spec , or the
    # run stops right here - never proceed on a tie or a non-fit
    record = formalize(user_query, max_fields=4)

    if record["decision"]["contested"]:
        print("\nTied formalizations - ask the user to choose before proceeding.")
        return
    if record["decision"]["selected_structure_id"] is None:
        print("\nNo formal structure applied to this problem.")
        return

    spec = build_problem_specification(user_query, record)
    spec = ratify_with_user(spec)          # does the asking
    if not spec.get("ratified"):
        print("Stopping - the spec was not ratified.")
        return
    assert_spec_ratified(spec)             # refuses to proceed if not ratified

    workspace = Workspace(spec=spec)

    # the run log : one record per iteration , structure documented at the
    # top of reporting/report.py . the checkpoint reads it , the report
    # renders it
    run_log = []
    halted_by_fuse = ""

    try:
        while True:
            # first line of every entry into the loop , always
            assert_spec_ratified(spec)

            iteration_number = len(run_log) + 1
            print(f"\n[orchestrator] --- iteration {iteration_number} ---")

            # 1. planner : the next few uncertainty-reducing tasks
            tasks_before_planner = len(workspace.snapshot()["tasks"])
            planner_output = plan_next_tasks(spec, workspace)
            tasks_after_planner = len(workspace.snapshot()["tasks"])

            # 2. scheduler : dispatch ready tasks , inner loop included
            scheduler_output = monitor_and_schedule_tasks(workspace, spec)

            # 3. evaluator : judge only what THIS iteration integrated
            evaluator_output = evaluate_iteration(
                spec, workspace,
                new_artifact_ids=scheduler_output["integrated_artifact_ids"],
            )

            # 4. contested triage (v1 stub : everything collapses)
            routed_pairs = triage_contested_pairs(
                workspace, evaluator_output["contested_pairs"],
            )

            # 5. adjudicator : resolve the collapsed fights
            adjudicator_output = adjudicate_contested_pairs(
                workspace, routed_pairs["collapse"],
            )

            # 6. checkpoint : the stopping brain reads the full run log
            iteration_record = {
                "iteration": iteration_number,
                "tasks_created_by_planner": tasks_after_planner - tasks_before_planner,
                "accepted_task_ids": planner_output["accepted_task_ids"],
                "frontier_empty_reason": planner_output["frontier_empty_reason"],
                "dispatched_task_ids": scheduler_output["dispatched_task_ids"],
                "integrated_artifact_ids": scheduler_output["integrated_artifact_ids"],
                "evaluator": evaluator_output,
                "adjudicator": adjudicator_output,
                "checkpoint": None,   # filled right below
            }
            run_log.append(iteration_record)

            checkpoint_result = decide_checkpoint(spec, workspace, run_log)
            iteration_record["checkpoint"] = checkpoint_result

            print(f"[checkpoint] {checkpoint_result['decision']}: "
                  f"{checkpoint_result['reason']}")

            if checkpoint_result["decision"] != "continue":
                break

    except (PlannerFuseTripped, ExecutorFuseTripped,
            EvaluatorFuseTripped, AdjudicatorFuseTripped) as fuse:
        # a fuse is a halt-and-show-the-human event , never a silent
        # degradation . the report still gets generated below - marked as
        # halted - because the audit trail matters MOST on abnormal runs
        halted_by_fuse = f"{type(fuse).__name__}: {fuse}"
        print(f"\n[orchestrator] FUSE TRIPPED - halting the loop: {halted_by_fuse}")

    json_path, html_path = generate_report(
        workspace, spec, run_log, halted_by_fuse=halted_by_fuse,
    )
    print(f"\n[orchestrator] report written: {json_path}")
    print(f"[orchestrator] report written: {html_path}")


if __name__ == "__main__":
    run_orchestrator()
