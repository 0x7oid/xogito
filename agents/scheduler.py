# this should call the function get_ready_tasks
# then call the log on eeach setting of the ready tasks
# for the scheduling policy we will stick with the fifo , the planner.py already did most of the work , they inserted them in order

'''
the scheduler stays THIN : dispatch only . it decides WHEN ready tasks go
to the executor , never WHAT they mean - no judgments , no llm calls .

the inner while-loop : after a batch integrates , tasks whose dependencies
just completed become ready . dispatching them in the SAME iteration
(instead of waiting a full planner cycle) keeps dependency chains moving .
the loop is bounded by the existing batch budget fuse , so a chain of
batches can never run past the wall-clock budget .
'''

import time

from agents.executor import execute_batch, integrate_results, ExecutorFuseTripped
from parametres import BATCH_BUDGET_SECONDS


def monitor_and_schedule_tasks(workspace, spec):
    loop_started = time.monotonic()
    dispatched_task_ids = []
    integrated_artifact_ids = []

    while True:

        # the budget fuse bounds the WHOLE dispatch loop , not just one
        # batch - chained batches share the same wall-clock allowance
        elapsed = time.monotonic() - loop_started
        if elapsed > BATCH_BUDGET_SECONDS:
            raise ExecutorFuseTripped(
                f"dispatch loop exceeded its wall-clock budget "
                f"({elapsed:.0f}s > {BATCH_BUDGET_SECONDS}s). Halting for human review."
            )

        ready_tasks = workspace.get_ready_tasks()
        if not ready_tasks:
            # nothing ready : either everything is done or the remaining
            # tasks wait on incomplete dependencies - the planner's problem
            break

        for task in ready_tasks:
            print(f"[scheduler] Task {task.id} is ready to be executed. Scheduling...")
            # Here you would add the logic to actually schedule the task for execution
            # For example, you might add it to a queue or send it to an executor

            # TODO : Implement the evaluatior call here
            # the state of the task should be altered after this call
            # (the evaluator call lives in the orchestrator now - the
            # scheduler only flips pending -> in_progress , through the
            # single door , attributed to itself)
            workspace.update_task_status(task.id, "in_progress", actor="scheduler")
            dispatched_task_ids.append(task.id)

        # execute the batch , integrate serially , then loop : integration
        # may have completed dependencies that unlock the next wave
        results = execute_batch(ready_tasks, spec, workspace)
        batch_artifact_ids = integrate_results(results, workspace)
        for artifact_id in batch_artifact_ids:
            integrated_artifact_ids.append(artifact_id)

    return {
        "dispatched_task_ids": dispatched_task_ids,
        "integrated_artifact_ids": integrated_artifact_ids,
    }
