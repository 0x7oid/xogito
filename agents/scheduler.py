# this should call the function get_ready_tasks
# then call the log on eeach setting of the ready tasks
# for the scheduling policy we will stick with the fifo , the planner.py already did most of the work , they inserted them in order

def monitor_and_schedule_tasks(workspace):
    ready_tasks = workspace.get_ready_tasks()
    for task in ready_tasks:
        print(f"[scheduler] Task {task.id} is ready to be executed. Scheduling...")
        # Here you would add the logic to actually schedule the task for execution
        # For example, you might add it to a queue or send it to an executor

        # TODO : Implement the evaluatior call here
        # the state of the task should be altered after this call

        pass
    