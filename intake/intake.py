'''
intake.py - the front door . collects the user's problem , verbatim .

NOTE: this file previously held an early draft of formalization.py by
mistake (preserved in git history) . the actual intake module lives here
now : the UserQuery dataclass and collect_user_query() , which the
orchestrator and formalization.py import .

three fields , all straight from the user :
- prompt             : the problem itself , in the user's own words
- scope              : where the answer is allowed to look , optional
- contextual_anchors : facts the user DECLARES as fixed . these are
                       treated as given , never re-investigated , and
                       never reworded by any model - the formalization
                       splits them into a list but copies each verbatim .

no llm here . intake records ; formalization interprets .
'''

from dataclasses import dataclass


@dataclass
class UserQuery:
    prompt: str
    scope: str = ""
    contextual_anchors: str = ""

    def summary(self):
        # a short human-readable echo , used by the orchestrator to play
        # the query back before formalization starts
        lines = []
        lines.append(f"Problem: {self.prompt}")
        lines.append(f"Scope: {self.scope or 'not specified'}")
        lines.append(f"Fixed facts: {self.contextual_anchors or 'none'}")
        return "\n".join(lines)


def collect_user_query():
    # plain terminal i/o , same register as ratify_with_user in
    # formalization.py - intake is the only other place the system talks
    # to the human directly
    print("===== XOGITO INTAKE =====")
    prompt = input("Describe the problem > ").strip()
    while not prompt:
        prompt = input("The problem cannot be empty. Describe the problem > ").strip()

    scope = input("Scope - where may the answer look? (enter to skip) > ").strip()
    contextual_anchors = input(
        "Fixed facts - anything to treat as given? (enter to skip) > "
    ).strip()

    return UserQuery(
        prompt=prompt,
        scope=scope,
        contextual_anchors=contextual_anchors,
    )
