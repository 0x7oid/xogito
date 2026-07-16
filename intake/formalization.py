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

from llm.client import ask_llm
# FIX: was `from intake import UserQuery` - that resolves to the intake
# PACKAGE , not the module , and only worked by accident from inside the
# folder . the chosen layout (see orchestrator.py header) runs everything
# from the project root , so the module path is intake.intake
from intake.intake import UserQuery


# ---------------------------------------------------------------------------
# The catalog of formal structures lives in intake/catalog.py - plain data
# plus pure predicates , importable without touching pipeline logic .
# ---------------------------------------------------------------------------

from intake.catalog import CHARACTERISTIC_VALUES, STRUCTURE_CATALOG



# ---------------------------------------------------------------------------
# Id resolution.
#
# The model writes whatever id it wants; code maps it back to the
# catalog. Matching goes from strict to loose, stopping at the first
# tier that resolves:
#
#   1. exact     - normalized string equals the id, name, or an alias
#   2. tokens    - the id/name/alias words are a subset of the returned
#                  words: "mathematical optimization" -> "optimization",
#                  "constraint satisfaction problem" -> "csp"
#   3. fuzzy     - difflib similarity, catches typos like "optimizaton"
#
# If a tier produces MORE than one catalog match, that's ambiguous ->
# raise (never guess). If no tier matches anything -> raise. Silent
# drops are impossible either way.
# ---------------------------------------------------------------------------

from difflib import SequenceMatcher


def _normalize(text):
    return text.strip().lower().replace("-", " ").replace("_", " ").replace("/", " ")


def _match_one(raw, allowed_structures, phase):
    text = _normalize(raw)
    words = set(text.split())

    # each catalog entry gets a set of normalized "known ways to say it"
    variants = {
        s["id"]: {_normalize(s["id"]), _normalize(s["name"])}
        | {_normalize(a) for a in s.get("aliases", [])}
        for s in allowed_structures
    }

    # tier 1: exact
    hits = [sid for sid, vs in variants.items() if text in vs]
    # tier 2: token containment ("mathematical optimization" superset of "optimization")
    if not hits:
        hits = [sid for sid, vs in variants.items() if any(set(v.split()) <= words for v in vs)]
    # tier 3: fuzzy (typos)
    if not hits:
        hits = [
            sid for sid, vs in variants.items()
            if any(SequenceMatcher(None, text, v).ratio() >= 0.85 for v in vs)
        ]

    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(f"[{phase}] ambiguous structure_id {raw!r}: matches {hits}")
    raise ValueError(f"[{phase}] unrecognized structure_id: {raw!r}")


def resolve_ids(items, allowed_structures, phase):
    seen = set()
    for item in items:
        item["structure_id"] = _match_one(item["structure_id"], allowed_structures, phase)
        if item["structure_id"] in seen:
            raise ValueError(f"[{phase}] duplicate structure_id: {item['structure_id']!r}")
        seen.add(item["structure_id"])

    expected = {s["id"] for s in allowed_structures}
    if expected - seen:
        raise ValueError(f"[{phase}] model skipped: {expected - seen}")

    return items


# ---------------------------------------------------------------------------
# Phase 1 - Characterize
#
# The filter functions compare exact strings, so the allowed values are
# spelled out in the schema descriptions and checked in code after.
# ---------------------------------------------------------------------------


CHARACTERIZE_SCHEMA = {
    "type": "object",
    "properties": {
        **{
            key: {"type": "string", "description": f"One of: {', '.join(values)}"}
            for key, values in CHARACTERISTIC_VALUES.items()
        },
        "uncertainty_present": {"type": "boolean"},
    },
    "required": list(CHARACTERISTIC_VALUES) + ["uncertainty_present"],
}


def characterize_problem(query):
    prompt = (
        "Extract structural characteristics of a problem. You have no "
        "knowledge of what formal frameworks might apply - that is a "
        "separate stage you cannot see. For each field, pick exactly one "
        "of the allowed values written in its description.\n\n"
        "Operational definitions:\n"
        "- stakeholders: parties whose INTERESTS conflict inside the "
        "problem itself (competitors, negotiating sides). People who "
        "merely disagree about what the answer is - advisors, experts, "
        "colleagues arguing for different options - are NOT adversarial "
        "stakeholders.\n"
        "- solution_uniqueness: 'unique_best_required' only when the "
        "asker needs the single best answer; 'any_qualifying_ok' when "
        "anything clearing the stated success bar would satisfy them.\n"
        "- primary_output: what the asker wants handed back. 'decision' "
        "(which option to take), 'explanation' (why something is "
        "happening), 'forecast' (what will happen), 'estimate' (how big "
        "or how many something is right now), 'artifact' (a designed "
        "thing: a plan, a document, an architecture), 'verdict_on_claims' "
        "(whether stated claims are true), 'ranking' (many items "
        "ordered), 'risk_profile' (what could go wrong, how likely, how "
        "bad). Pick what dominates; a decision that also needs claims "
        "checked is still 'decision'.\n\n"
        f"Goal: {query.prompt}\n"
        f"Scope: {query.scope or 'not specified'}\n"
        f"Fixed facts: {query.contextual_anchors or 'none'}\n"
    )
    characteristics = json.loads(ask_llm(prompt, CHARACTERIZE_SCHEMA))

    for key, values in CHARACTERISTIC_VALUES.items():
        canonical = {_normalize(v): v for v in values}
        got = _normalize(characteristics[key])
        if got not in canonical:
            raise ValueError(f"[characterize] bad value for {key}: {characteristics[key]!r}")
        characteristics[key] = canonical[got]  # store the exact string filters expect

    return characteristics


# ---------------------------------------------------------------------------
# Phase 2 - Filter (pure code, no LLM call)
# ---------------------------------------------------------------------------

def filter_structures(characteristics):
    return [s for s in STRUCTURE_CATALOG if s["applies_if"](characteristics)]


# ---------------------------------------------------------------------------
# Phase 3 - Classify fit, then select up to max_fields (code decides the cut)
# ---------------------------------------------------------------------------

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "structure_id": {"type": "string"},
                    "label": {
                        "type": "string",
                        "description": "One of: fits, partially fits, doesn't fit",
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["structure_id", "label", "rationale"],
            },
        }
    },
    "required": ["judgments"],
}

FIT_LABELS = {"fits", "partially fits", "doesn't fit"}


def classify_fit(query, candidates):
    listing = "\n".join(
        f"- {s['id']}: {s['name']} - {s['definition']}"
        + (f" (example problem: {s['example']})" if s.get("example") else "")
        for s in candidates
    )
    # FIX (live stress run) : classify used to see the prompt ONLY - no
    # scope , no anchors . a threshold stated in the anchors ("target is
    # 12,000/hour") is exactly what makes satisficing fit , so hiding the
    # anchors from this call starved it of the fit signal . label
    # definitions are operational now for the same reason the evaluator's
    # are : a bare word invites vibes , a definition invites checking
    prompt = (
        "For each listed structure, judge how well its assumptions actually "
        "hold for this problem. Judge every structure exactly once. Use only "
        "these labels, no numbers:\n"
        "- 'fits': solving the problem as this structure defines it would "
        "directly answer what the asker needs.\n"
        "- 'partially fits': the structure captures the central decision "
        "but drops or distorts a secondary aspect.\n"
        "- 'doesn't fit': solving the structure would not answer the "
        "problem.\n"
        "It is perfectly fine to label every structure 'doesn't fit' - do "
        "not force a match. Give one clause of justification for each.\n\n"
        f"Problem: {query.prompt}\n"
        f"Scope: {query.scope or 'not specified'}\n"
        f"Fixed facts: {query.contextual_anchors or 'none'}\n\n"
        f"Structures:\n{listing}"
    )
    judgments = json.loads(ask_llm(prompt, CLASSIFY_SCHEMA))["judgments"]
    judgments = resolve_ids(judgments, candidates, "classify_fit")

    for j in judgments:
        j["label"] = j["label"].strip().lower()
        if j["label"] not in FIT_LABELS:
            raise ValueError(f"[classify_fit] bad label: {j['label']!r}")

    return judgments


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

GENERATE_SCHEMA = {
    "type": "object",
    "properties": {
        "formalizations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "structure_id": {"type": "string"},
                    "formal_statement": {"type": "string"},
                    "preserved_from_original": {"type": "string"},
                    "dropped_or_altered": {"type": "string"},
                },
                "required": [
                    "structure_id",
                    "formal_statement",
                    "preserved_from_original",
                    "dropped_or_altered",
                ],
            },
        }
    },
    "required": ["formalizations"],
}


def generate_formalizations(query, selected_structures):
    schemas = "\n".join(
        f"- {s['id']}: {s['name']} - {s['definition']}" for s in selected_structures
    )
    prompt = (
        "Formalize the problem separately under each structure listed "
        "below, exactly one formalization per structure. Treat each one as "
        "if it were the only option - do not let one formalization "
        "influence another. If part of the problem doesn't map cleanly "
        "onto a structure, say so in dropped_or_altered instead of forcing "
        "it.\n\n"
        f"Problem: {query.prompt}\n\nStructures to use:\n{schemas}"
    )
    formalizations = json.loads(ask_llm(prompt, GENERATE_SCHEMA))["formalizations"]
    return resolve_ids(formalizations, selected_structures, "generate_formalizations")


# ---------------------------------------------------------------------------
# Phase 5 - Evaluate the formalizations, labels only
# ---------------------------------------------------------------------------

EVALUATE_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "structure_id": {"type": "string"},
                    "labels": {
                        "type": "object",
                        "properties": {
                            "similarity": {"type": "string"},
                            "exactness": {"type": "string"},
                            "fruitfulness": {"type": "string"},
                            "simplicity": {"type": "string"},
                        },
                        "required": ["similarity", "exactness", "fruitfulness", "simplicity"],
                        "description": "Each one of: strong, adequate, weak",
                    },
                    "justification": {"type": "string"},
                },
                "required": ["structure_id", "labels", "justification"],
            },
        }
    },
    "required": ["evaluations"],
}

QUALITY_LABELS = {"strong", "adequate", "weak"}


def evaluate_candidates(query, candidates):
    prompt = (
        "For each candidate formalization, judge four things using only "
        "these labels, no numbers: 'strong', 'adequate', 'weak'. The four "
        "things: similarity (does solving this feel like solving the "
        "original?), exactness (precise enough to derive consequences?), "
        "fruitfulness (produces something actionable?), simplicity "
        "(leanest structure that still qualifies?). Evaluate every "
        "candidate exactly once. One clause of justification per "
        "candidate. Do not declare an overall winner.\n\n"
        f"Problem: {query.prompt}\n\nCandidates:\n{json.dumps(candidates, indent=2)}"
    )
    evaluations = json.loads(ask_llm(prompt, EVALUATE_SCHEMA))["evaluations"]
    allowed = [s for s in STRUCTURE_CATALOG if s["id"] in {c["structure_id"] for c in candidates}]
    evaluations = resolve_ids(evaluations, allowed, "evaluate_candidates")

    for e in evaluations:
        for key, value in e["labels"].items():
            e["labels"][key] = value.strip().lower()
            if e["labels"][key] not in QUALITY_LABELS:
                raise ValueError(f"[evaluate] bad label for {key}: {value!r}")

    return evaluations


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

    classified = classify_fit(query, candidates) if candidates else []
    selected, boundary_tension = select_structures(classified, max_fields)

    if not selected:
        return {
            "problem": query.prompt,
            "characteristics": characteristics,
            "selection": {
                "candidates_scored": [c["structure_id"] for c in classified],
                "labels": {c["structure_id"]: c["label"] for c in classified},
                "max_structures_cap": max_fields,
                "selected": [],
                "boundary_tension": False,
            },
            "candidates": [],
            "evaluations": [],
            "decision": {
                "mode": "no_structure_applies",
                "selected_structure_id": None,
                "contested": False,
            },
        }

    selected_ids = [c["structure_id"] for c in selected]
    by_id = {s["id"]: s for s in candidates}
    selected_structures = [by_id[sid] for sid in selected_ids]

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
            # when contested, no structure is chosen yet - a human must ratify.
            # we do NOT pretend a winner was picked.
            "mode": "needs_user_ratification" if contested else "auto_selected",
            "selected_structure_id": None if contested else best["structure_id"],
            "contested": contested,
        },
    }

    return record


# ---------------------------------------------------------------------------
# Hand off the actual problem specification.
#
# The record above is the full paper trail. This is the ONE sanctioned way
# to pull out the formalization that the rest of the pipeline should use.
# It refuses to hand anything over when the choice is contested or when no
# structure applied - so a caller can never silently proceed on a tie.
# ---------------------------------------------------------------------------

def get_final_formalization(record):
    decision = record["decision"]

    if decision["contested"]:
        raise ValueError(
            "formalization is contested - two structures tied. "
            "Ask the user to pick one before continuing."
        )

    if decision["selected_structure_id"] is None:
        raise ValueError("no formal structure applied to this problem.")

    chosen_id = decision["selected_structure_id"]
    for c in record["candidates"]:
        if c["structure_id"] == chosen_id:
            # the problem specification the Planner will consume
            return {
                "problem": record["problem"],
                "characteristics": record["characteristics"],
                "structure_id": chosen_id,
                "formal_statement": c["formal_statement"],
                "preserved_from_original": c["preserved_from_original"],
                "dropped_or_altered": c["dropped_or_altered"],
            }

    raise ValueError(f"chosen structure {chosen_id!r} not found in candidates.")


# ---------------------------------------------------------------------------
# Build the final Problem Specification.
#
# Independent function. Takes the formalization record (the reasoning
# trail) + the original UserQuery, and produces the document the rest of
# the pipeline consumes:
#
#   {
#     "problem_specification": { goal, constraints, success_criteria,
#                                scope, contextual_anchors, assumptions },
#     "characteristics": {...},   <- kept: the reasoning trail
#     "selection": {...},
#     "candidates": [...],
#     "evaluations": [...],
#     "decision": {...}
#   }
#
# One LLM call extracts goal / constraints / success_criteria /
# assumptions under the WINNING structure only. Scope and anchors are
# copied verbatim from the user - the model never gets to reword what
# the user declared as fixed.
# ---------------------------------------------------------------------------

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "description": "The goal stated precisely under the chosen formal structure.",
        },
        "constraints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Hard limits the solution must respect. Empty list if none.",
        },
        "success_criteria": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Checkable conditions that tell us the goal was reached. Empty list if none.",
        },
        "assumptions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Things taken for granted that were NOT declared by the user. Empty list if none.",
        },
        "assumption_impacts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "assumption": {"type": "string"},
                    "impact": {
                        "type": "string",
                        "description": "One of: load_bearing, peripheral. "
                                       "load_bearing means the answer would "
                                       "change if this assumption were wrong.",
                    },
                    "why": {"type": "string"},
                },
                "required": ["assumption", "impact", "why"],
            },
            "description": "One entry per assumption, same order.",
        },
    },
    "required": ["goal", "constraints", "success_criteria", "assumptions",
                 "assumption_impacts"],
}

IMPACT_LABELS = ("load_bearing", "peripheral")


def _split_anchors(raw_anchors):
    # user anchors stay verbatim - code only splits them into a list,
    # never rewords them. Splits on newlines and semicolons and commas.
    # FIX: comma-splitting sheared numeric anchors in half ("peak load is
    # 12,000 msg/s" became "peak load is 12" + "000 msg/s") . a comma
    # BETWEEN TWO DIGITS is part of a number , never a separator - it is
    # shielded before the split and restored after , so the anchor stays
    # verbatim
    if not raw_anchors:
        return []

    shielded = ""
    for position in range(len(raw_anchors)):
        character = raw_anchors[position]
        is_numeric_comma = (
            character == ","
            and position > 0
            and position + 1 < len(raw_anchors)
            and raw_anchors[position - 1].isdigit()
            and raw_anchors[position + 1].isdigit()
        )
        if is_numeric_comma:
            shielded += "\x00"
        else:
            shielded += character

    pieces = []
    for line in shielded.replace(";", "\n").replace(",", "\n").split("\n"):
        line = line.strip().replace("\x00", ",")
        if line:
            pieces.append(line)
    return pieces


# ---------------------------------------------------------------------------
# Anchor tracing (pure code, no LLM).
#
# FIX: user-declared anchors were copied into the spec verbatim and then
# silently ignored - nothing checked whether the formalization actually
# CARRIED them. The "dropped_or_altered" honesty field only ever covered
# the free-form parts of the prompt, never the anchors. This traces every
# anchor into the extracted spec text; an anchor whose content words
# mostly fail to appear anywhere in goal/constraints/criteria/assumptions
# is a DROPPED anchor and is recorded as such.
#
# crude v1 heuristic on purpose (same posture as the evaluator's scope
# markers): majority content-word overlap checks PRESENCE, not meaning.
# ---------------------------------------------------------------------------

ANCHOR_TRACE_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "to", "in",
    "on", "for", "and", "or", "not", "no", "it", "this", "that", "with",
}


def _anchor_content_words(anchor):
    words = []
    for word in _normalize(anchor).split():
        if word not in ANCHOR_TRACE_STOPWORDS:
            words.append(word)
    return words


def _anchor_is_carried(anchor, spec_text):
    words = _anchor_content_words(anchor)
    if not words:
        return True   # an anchor with no content words cannot be traced
    hits = 0
    for word in words:
        if word in spec_text:
            hits += 1
    # majority of the anchor's content words must appear somewhere
    return hits * 2 >= len(words)


def trace_anchors(anchors, extracted):
    joined_parts = [extracted["goal"]]
    for part_list in (extracted["constraints"], extracted["success_criteria"],
                      extracted["assumptions"]):
        for part in part_list:
            joined_parts.append(part)
    spec_text = _normalize(" ".join(joined_parts))

    carried = []
    dropped = []
    for anchor in anchors:
        if _anchor_is_carried(anchor, spec_text):
            carried.append(anchor)
        else:
            dropped.append(anchor)
    return {"carried": carried, "dropped": dropped}


def _validate_assumption_impacts(extracted):
    # code validates the llm's labels , always . an assumption the model
    # forgot to judge defaults to LOAD-BEARING - an unjudged guess can
    # never be quietly treated as harmless
    impacts_by_assumption = {}
    for entry in extracted["assumption_impacts"]:
        label = entry["impact"].strip().lower()
        if label not in IMPACT_LABELS:
            raise ValueError(f"[spec] bad impact label: {entry['impact']!r}")
        impacts_by_assumption[entry["assumption"]] = {
            "impact": label,
            "why": entry["why"],
        }

    reviewed = []
    for assumption in extracted["assumptions"]:
        entry = impacts_by_assumption.get(assumption)
        if entry is None:
            entry = {"impact": "load_bearing",
                     "why": "the system did not judge this assumption's "
                            "impact - treated as load-bearing by default"}
        reviewed.append({
            "assumption": assumption,
            "impact": entry["impact"],
            "why": entry["why"],
            "user_response": "not asked yet",
        })
    return reviewed


def build_problem_specification(query, record):
    # this raises on contested / no-structure, so we can't build a spec
    # from an unratified decision - the door stays locked.
    final = get_final_formalization(record)

    prompt = (
        "Extract a problem specification from the problem below, using "
        "the chosen formal framing. Do not invent constraints or criteria "
        "the problem doesn't imply. Anything you had to take for granted "
        "goes in assumptions.\n"
        "For each assumption, also judge its impact: 'load_bearing' if the "
        "answer to the problem would change were the assumption wrong, "
        "'peripheral' otherwise. One impact entry per assumption, with one "
        "clause of why.\n\n"
        f"Problem: {query.prompt}\n"
        f"Scope: {query.scope or 'not specified'}\n"
        f"Chosen framing ({final['structure_id']}): {final['formal_statement']}\n"
        f"Parts that didn't map cleanly: {final['dropped_or_altered']}\n"
    )
    extracted = json.loads(ask_llm(prompt, SPEC_SCHEMA))

    anchors = _split_anchors(query.contextual_anchors)

    return {
        "problem_specification": {
            "goal": extracted["goal"],
            "constraints": extracted["constraints"],
            "success_criteria": extracted["success_criteria"],
            "scope": query.scope or "not specified",
            "contextual_anchors": anchors,
            "assumptions": extracted["assumptions"],
        },
        # anchors are user-declared GROUND TRUTH ; assumptions are the
        # system's own guesses . the two must never be presented with the
        # same confidence , so they are traced and labeled separately here
        "anchor_trace": trace_anchors(anchors, extracted),
        "assumption_review": _validate_assumption_impacts(extracted),
        # FIX: the user's ORIGINAL prompt was dropped here - only the
        # formalized goal survived , so the final report could not show
        # what the user actually asked (honesty about framing requires
        # both) . copied verbatim , never reworded
        "problem": query.prompt,
        "structure_id": final["structure_id"],
        "characteristics": record["characteristics"],
        "selection": record["selection"],
        "candidates": record["candidates"],
        "evaluations": record["evaluations"],
        "decision": record["decision"],
    }

def review_assumptions_with_user(spec):
    # a load-bearing assumption is a DECISION POINT , not a footnote : the
    # answer flips if the guess is wrong , so the system asks instead of
    # silently filling the gap . lives here with ratify_with_user because
    # it is I/O , and it runs BEFORE ratification so the user confirms a
    # spec whose guesses they have already seen .
    # dropped anchors are surfaced here too - the user declared them as
    # fixed facts , so losing one in the framing must never be silent .
    for dropped in spec["anchor_trace"]["dropped"]:
        print(f"\nWARNING - your declared fact was NOT carried into the "
              f"formalization: \"{dropped}\"")
        print("It will be flagged as a dropped anchor in the final report.")

    for entry in spec["assumption_review"]:
        if entry["impact"] != "load_bearing":
            entry["user_response"] = "not asked (peripheral)"
            continue

        print(f"\nThe system had to GUESS: \"{entry['assumption']}\"")
        print(f"Why it matters: {entry['why']}")
        answer = input("Is this guess correct? (yes / no / unsure) > ").strip().lower()

        if answer == "yes":
            entry["user_response"] = "confirmed by user"
            # a confirmed guess is a user-declared fact now - promoted to
            # the anchors , verbatim , so downstream treats it as given
            spec["problem_specification"]["contextual_anchors"].append(
                entry["assumption"])
        elif answer == "no":
            correction = input("What is actually true? (one line) > ").strip()
            entry["user_response"] = f"rejected by user; correction: {correction}"
            if correction:
                spec["problem_specification"]["contextual_anchors"].append(correction)
        else:
            # "unsure" (or anything else) : the guess stays a guess ,
            # visibly labeled - never silently upgraded to fact
            entry["user_response"] = "user unsure - remains an unconfirmed guess"

    return spec


def ratify_with_user(spec):
    # does the actual asking - lives here, not in kernel, because this
    # is I/O (print + input), and kernel never touches I/O.
    print("\n===== PROBLEM SPECIFICATION =====")
    print(json.dumps(spec["problem_specification"], indent=2))
    print("==================================")
 
    answer = input("Confirm this spec? (yes / no) > ").strip().lower()
    spec["ratified"] = (answer == "yes")
 
    if not spec["ratified"]:
        print("Not ratified - send back for repair or ask the user to edit.")
 
    return spec
 
 
if __name__ == "__main__":
    from intake.intake import collect_user_query
    from core.kernel import assert_spec_ratified
 
    query = collect_user_query()
    result = formalize(query, max_fields=4)
    print(json.dumps(result, indent=2))
 
    if result["decision"]["contested"]:
        print("\nTied formalizations - ask the user to choose before proceeding.")
 
    elif result["decision"]["selected_structure_id"] is None:
        print("\nNo formal structure applied to this problem.")
 
    else:
        spec = build_problem_specification(query, result)
        spec = ratify_with_user(spec)          # does the asking
 
        assert_spec_ratified(spec)             # refuses to proceed if not ratified
        print("\nSpec ratified - ready for the Planner.")