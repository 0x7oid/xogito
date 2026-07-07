"""
Collect information from the user before the reasoning process begins.

Required:
- Problem

Optional:
- Scope
- Contextual anchors
- Policy base
"""

from dataclasses import dataclass

MAX_ITERATIONS = 5


@dataclass
class UserQuery:
    prompt: str
    scope: str | None = None
    contextual_anchors: str | None = None
    policy_base: str | None = None
    max_iterations: int = MAX_ITERATIONS

    def summary(self):
        skipped = "Skipped"

        return (
            "\n========== USER QUERY ==========\n"
            f"Problem:\n{self.prompt}\n\n"
            f"Scope:\n{self.scope if self.scope else skipped}\n\n"
            f"Anchors:\n{self.contextual_anchors if self.contextual_anchors else skipped}\n\n"
            f"Policy:\n{self.policy_base if self.policy_base else skipped}\n"
            "===============================\n"
        )


def get_user_prompt():
    print("\nProblem (Required)")
    print("Describe the problem or question you want me to solve.")

    while True:
        prompt = input("> ").strip()

        if prompt:
            return prompt

        print("Please describe your problem.")


def get_user_scope():
    print("\nScope (Optional)")
    print("Example: only open-source tools, ignore cost, Algerian market only.")

    scope = input("> (Press Enter to skip) ").strip()

    return scope if scope else None


def get_user_contextual_anchors():
    print("\nFixed Facts (Optional)")
    print("Example: budget is $10k, deadline is March, team size is 3.")

    anchors = input("> (Press Enter to skip) ").strip()

    return anchors if anchors else None


def get_user_policy_base():
    print("\nReasoning Preferences (Optional)")
    print("Example: prefer official sources, be conservative, report disagreements.")

    policy = input("> (Press Enter to skip) ").strip()

    return policy if policy else None


def collect_user_query():
    print(
        "I'll ask four questions.\n"
        "Only the first is required.\n"
        "Press Enter to skip the others."
    )

    query = UserQuery(
        prompt=get_user_prompt(),
        scope=get_user_scope(),
        contextual_anchors=get_user_contextual_anchors(),
        policy_base=get_user_policy_base(),
    )

    return query


if __name__ == "__main__":
    query = collect_user_query()

    print(f"Framework iteration limit: {MAX_ITERATIONS}")