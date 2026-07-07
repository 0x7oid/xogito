"""
Turn a UserQuery into a formal problem structure.

Five steps:
1. Characterize  - figure out what kind of problem this is
2. Filter        - keep only structures that could apply     (code, no LLM)
3. Classify fit  - label each candidate: fits / partially fits / doesn't fit
4. Generate      - write the actual formalization for each selected structure
5. Evaluate      - label each formalization: strong / adequate / weak

No numbers anywhere. Only labels. The cap on how many structures we try
is a plain function argument (max_fields) - the LLM never sees this
number, so it never nudges itself toward hitting it.
"""

import json
from dataclasses import dataclass

from intake import UserQuery


# ---------------------------------------------------------------------------
# The catalog of formal structures we know about.
# This is plain data + plain python functions. No LLM involved here.
# ---------------------------------------------------------------------------

def _fits_optimization(c):
    return c["success_type"] == "graded" and c["solution_uniqueness"] == "unique_best_required"


def _fits_csp(c):
    return c["solution_uniqueness"] == "any_qualifying_ok"


def _fits_satisficing(c):
    return c["solution_uniqueness"] == "any_qualifying_ok" or c["success_type"] == "threshold"


def _fits_game_theory(c):
    return c["stakeholders"] == "multiple_adversarial"


def _fits_control_loop(c):
    return c["temporal_structure"] == "continuous_feedback"


STRUCTURE_CATALOG = [
    {
        "id": "optimization",
        "name": "Objective Optimization",
        "definition": "Find the input that maximizes or minimizes some function, given constraints.",
        "applies_if": _fits_optimization,
    },
    {
        "id": "csp",
        "name": "Constraint Satisfaction",
        "definition": "Find any solution that satisfies a set of requirements. No 'best' needed, just 'good enough'.",
        "applies_if": _fits_csp,
    },
    {
        "id": "satisficing",
        "name": "Satisficing",
        "definition": "Stop as soon as a solution clears a threshold, instead of searching for the best one.",
        "applies_if": _fits_satisficing,
    },
    {
        "id": "game_theory",
        "name": "Game-Theoretic",
        "definition": "Several parties with conflicting interests, each reacting to the others.",
        "applies_if": _fits_game_theory,
    },
    {
        "id": "control_loop",
        "name": "Control / Feedback Loop",
        "definition": "Not a one-shot decision - an ongoing process that keeps adjusting over time.",
        "applies_if": _fits_control_loop,
    },
]


# ---------------------------------------------------------------------------
# Placeholder for the actual model call.
# Replace this with a real call to your LLM API. It must return a string
# that is valid JSON (or JSON wrapped in normal prose - your call).
# ---------------------------------------------------------------------------

def call_llm(system_prompt, user_prompt):
    raise NotImplementedError("Hook this up to your LLM API")


# ---------------------------------------------------------------------------
# Phase 1 - Characterize
# ---------------------------------------------------------------------------

def characterize_problem(query):
    system_prompt = (
        "Extract structural characteristics of a problem. You have no "
        "knowledge of what formal frameworks might apply - that is a "
        "separate stage you cannot see. Return JSON only, with these keys: "
        "cardinality, stakeholders, temporal_structure, success_type, "
        "solution_uniqueness, measurability, uncertainty_present."
    )
    user_prompt = (
        f"Goal: {query.prompt}\n"
        f"Scope: {query.scope or 'not specified'}\n"
        f"Fixed facts: {query.contextual_anchors or 'none'}\n"
    )
    response = call_llm(system_prompt, user_prompt)
    return json.loads(response)


# ---------------------------------------------------------------------------
# Phase 2 - Filter (pure code, no LLM call)
# ---------------------------------------------------------------------------

def filter_structures(characteristics):
    return [s for s in STRUCTURE_CATALOG if s["applies_if"](characteristics)]


# ---------------------------------------------------------------------------
# Phase 3 - Classify fit, then select up to max_fields (code decides the cut)
# ---------------------------------------------------------------------------

def classify_fit(query, candidates):
    listing = "\n".join(f"- {s['id']}: {s['name']} - {s['definition']}" for s in candidates)
    system_prompt = (
        "For each listed structure, judge how well its assumptions actually "
        "hold for this problem. Use only these labels, no numbers: "
        "'fits', 'partially fits', 'doesn't fit'. Give one clause of "
        "justification for each. Return JSON: a list of objects with "
        "structure_id, label, rationale."
    )
    user_prompt = f"Problem: {query.prompt}\n\nStructures:\n{listing}"
    response = call_llm(system_prompt, user_prompt)
    return json.loads(response)


def select_structures(classified, max_fields):
    tiers = {"fits": [], "partially fits": []}
    for c in classified:
        if c["label"] in tiers:
            tiers[c["label"]].append(c)

    ranked = tiers["fits"] + tiers["partially fits"]
    selected = ranked[:max_fields]

    boundary_tension = False
    if len(ranked) > max_fields:
        if ranked[max_fields - 1]["label"] == ranked[max_fields]["label"]:
            boundary_tension = True

    return selected, boundary_tension


# ---------------------------------------------------------------------------
# Phase 4 - Generate a formalization for each selected structure
# ---------------------------------------------------------------------------

def generate_formalizations(query, selected_structures):
    schemas = "\n".join(
        f"- {s['id']}: {s['name']} - {s['definition']}" for s in selected_structures
    )
    system_prompt = (
        "Formalize the problem separately under each structure listed "
        "below. Treat each one as if it were the only option - do not let "
        "one formalization influence another. If part of the problem "
        "doesn't map cleanly onto a structure, say so instead of forcing "
        "it. Return JSON: a list of objects with structure_id, "
        "formal_statement, preserved_from_original, dropped_or_altered."
    )
    user_prompt = f"Problem: {query.prompt}\n\nStructures to use:\n{schemas}"
    response = call_llm(system_prompt, user_prompt)
    return json.loads(response)


# ---------------------------------------------------------------------------
# Phase 5 - Evaluate the formalizations, labels only
# ---------------------------------------------------------------------------

def evaluate_candidates(query, candidates):
    system_prompt = (
        "For each candidate formalization, judge four things using only "
        "these labels, no numbers: 'strong', 'adequate', 'weak'. The four "
        "things: similarity (does solving this feel like solving the "
        "original?), exactness (precise enough to derive consequences?), "
        "fruitfulness (produces something actionable?), simplicity "
        "(leanest structure that still qualifies?). One clause of "
        "justification per label. Do not declare an overall winner. "
        "Return JSON: a list of objects with structure_id, labels, justification."
    )
    user_prompt = f"Problem: {query.prompt}\n\nCandidates:\n{json.dumps(candidates, indent=2)}"
    response = call_llm(system_prompt, user_prompt)
    return json.loads(response)


def pick_best(evaluations):
    def label_count(evaluation):
        strong = sum(1 for v in evaluation["labels"].values() if v == "strong")
        adequate = sum(1 for v in evaluation["labels"].values() if v == "adequate")
        return (strong, adequate)

    ranked = sorted(evaluations, key=label_count, reverse=True)
    best = ranked[0]

    contested = len(ranked) > 1 and label_count(ranked[0]) == label_count(ranked[1])

    return best, contested


# ---------------------------------------------------------------------------
# Put it all together
# ---------------------------------------------------------------------------

def formalize(query, max_fields=4):
    characteristics = characterize_problem(query)
    candidates = filter_structures(characteristics)

    classified = classify_fit(query, candidates)
    selected, boundary_tension = select_structures(classified, max_fields)
    selected_ids = [c["structure_id"] for c in selected]
    selected_structures = [s for s in candidates if s["id"] in selected_ids]

    formalizations = generate_formalizations(query, selected_structures)
    evaluations = evaluate_candidates(query, formalizations)
    best, contested = pick_best(evaluations)

    record = {
        "problem": query.prompt,
        "characteristics": characteristics,
        "selection": {
            "candidates_scored": [c["structure_id"] for c in classified],
            "labels": {c["structure_id"]: c["label"] for c in classified},
            "max_structures_cap": max_fields,
            "selected": selected_ids,
            "boundary_tension": boundary_tension,
        },
        "candidates": formalizations,
        "evaluations": evaluations,
        "decision": {
            "mode": "user_ratified" if contested else "auto_selected",
            "selected_structure_id": best["structure_id"],
            "contested": contested,
        },
    }

    return record


if __name__ == "__main__":
    from intake import collect_user_query

    query = collect_user_query()
    result = formalize(query, max_fields=4)
    print(json.dumps(result, indent=2))