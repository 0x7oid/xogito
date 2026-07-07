from intake import collect_user_query

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

def run_orchestrator():
    user_query = collect_user_query()
    print("\nThank you for providing your input.")
    print("Here is a summary of your query:")
    print(user_query.summary())