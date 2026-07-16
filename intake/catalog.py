'''
catalog.py - the catalog of formal problem structures , in its own file .

WHY A SIDE FILE
    the catalog is DATA plus pure predicates - no llm , no state , no io .
    formalization.py consumes it ; nothing else should have to import
    formalization just to read the catalog . growing the catalog must
    never mean touching pipeline logic .

WHAT LIVES HERE
    1. CHARACTERISTIC_VALUES - the closed vocabulary the characterizer
       must answer in . code validates every extraction against it ,
       because the predicates below compare exact strings .
    2. one predicate per structure - plain python over the
       characteristics . filtering is set intersection , done in code ,
       so the model can never answer it inconsistently .
    3. STRUCTURE_CATALOG - the entries themselves .

ENTRY CONVENTIONS (definitions feed prompts , so prompt-clarity rules
apply even though this file is not a prompt) :
    - "definition" states the TASK first , then the SUCCESS CONDITION ,
      in words a classifier can check a problem against . no jargon
      that only means something after you already know the structure .
    - "example" is one concrete user-shaped sentence . classifiers match
      real queries against real queries better than against theory .
    - describe the problem CLASS , never the mathematical archetype -
      a live stress run showed "maximize a function" reads as literal
      calculus and rejects every real decision problem .
    - overlap between structures is fine and expected . the filter
      surfaces candidates ; the classify phase weeds them ; selection
      never trusts this file to be disjoint .

ADMISSION TEST for a new structure : it must have a DISTINCT success
condition - it generates different tasks and a different kind of answer .
two entries whose success conditions coincide are one entry .
'''


# ===========================================================================
# the closed characteristic vocabulary
# ===========================================================================

CHARACTERISTIC_VALUES = {
    "cardinality": ["single_decision", "batch", "stream"],
    "stakeholders": ["single", "multiple_aligned", "multiple_adversarial"],
    "temporal_structure": ["one_shot", "sequential", "continuous_feedback"],
    "success_type": ["binary", "threshold", "graded"],
    "solution_uniqueness": ["unique_best_required", "any_qualifying_ok"],
    "measurability": ["directly_measurable", "proxy_only", "unmeasurable"],
    # WHAT the asker wants handed back . added with the catalog extension :
    # the six characteristics above cannot tell "pick an option" apart
    # from "explain a failure" or "check these claims"
    "primary_output": [
        "decision",           # which option to take
        "explanation",        # why something is happening
        "forecast",           # what will happen
        "estimate",           # how big / how many X is right now
        "artifact",           # a designed thing : plan , document , architecture
        "verdict_on_claims",  # whether stated claims are true
        "ranking",            # many items , ordered
        "risk_profile",       # what could go wrong , how likely , how bad
    ],
}


# ===========================================================================
# applicability predicates . pure functions over the characteristics .
# one per structure , named _fits_<id> , kept adjacent for auditability
# ===========================================================================

def _fits_optimization(c):
    # find-the-single-best is an argmax whether success is graded or
    # thresholded (live stress run : requiring graded AND unique-best
    # matched nothing and killed the run at intake)
    return c["solution_uniqueness"] == "unique_best_required"


def _fits_csp(c):
    return c["solution_uniqueness"] == "any_qualifying_ok"


def _fits_satisficing(c):
    # binary and threshold success are the same signal here : both mean
    # "clears a stated bar" . the characterizer legitimately wavers
    # between the two for a numeric target , and the filter must not
    # starve satisficing over that adjacent-value coin flip
    return (c["solution_uniqueness"] == "any_qualifying_ok"
            or c["success_type"] in ("threshold", "binary"))


def _fits_game_theory(c):
    return c["stakeholders"] == "multiple_adversarial"


def _fits_control_loop(c):
    return c["temporal_structure"] == "continuous_feedback"


def _fits_bounded_choice(c):
    # "pick the best of these named options" - the most common shape a
    # user hands this system (added from live-run evidence)
    return (c["cardinality"] == "single_decision"
            and c["solution_uniqueness"] == "unique_best_required")


def _fits_diagnosis(c):
    return c["primary_output"] == "explanation"


def _fits_claim_verification(c):
    return c["primary_output"] == "verdict_on_claims"


def _fits_forecasting(c):
    return c["primary_output"] == "forecast"


def _fits_estimation(c):
    return c["primary_output"] == "estimate"


def _fits_design_synthesis(c):
    return c["primary_output"] == "artifact"


def _fits_prioritization(c):
    return c["primary_output"] == "ranking" or c["cardinality"] == "batch"


def _fits_risk_assessment(c):
    return c["primary_output"] == "risk_profile"


def _fits_tradeoff_analysis(c):
    return c["primary_output"] == "decision" and c["success_type"] == "graded"


def _fits_negotiation_strategy(c):
    return (c["stakeholders"] == "multiple_adversarial"
            and c["primary_output"] in ("decision", "artifact"))


def _fits_anomaly_detection(c):
    return c["cardinality"] == "stream"


def _fits_resource_allocation(c):
    return (c["cardinality"] == "batch"
            and c["primary_output"] in ("decision", "artifact"))


def _fits_scheduling(c):
    return (c["primary_output"] == "artifact"
            and c["temporal_structure"] == "sequential")


def _fits_policy_design(c):
    return (c["primary_output"] == "artifact"
            and c["temporal_structure"] == "continuous_feedback")


def _fits_categorization(c):
    return (c["cardinality"] == "batch"
            and c["primary_output"] in ("decision", "verdict_on_claims"))


# ===========================================================================
# the catalog
# ===========================================================================

STRUCTURE_CATALOG = [
    # --- decision-shaped ---------------------------------------------------
    {
        "id": "optimization",
        "name": "Objective Optimization",
        "aliases": ["mathematical optimization", "optimization problem",
                    "mathematical programming"],
        "definition": "Choose, among candidate options or inputs, the one that "
                      "best serves a stated objective under constraints. The "
                      "objective does not need to be a literal mathematical "
                      "function - 'best against the stated goal' is enough. "
                      "Success: the chosen option demonstrably beats every "
                      "considered alternative on the objective.",
        "example": "Which caching strategy minimizes our cloud bill without "
                   "raising p95 latency?",
        "applies_if": _fits_optimization,
    },
    {
        "id": "csp",
        "name": "Constraint Satisfaction",
        "aliases": ["constraint satisfaction problem", "constraint programming"],
        "definition": "Find any solution that satisfies every stated "
                      "requirement - no 'best' needed. Success: one solution "
                      "with each requirement checked off against it.",
        "example": "Find a deployment window that avoids all three teams' "
                   "freeze periods and the marketing launch.",
        "applies_if": _fits_csp,
    },
    {
        "id": "satisficing",
        "name": "Satisficing",
        "aliases": ["good-enough search", "threshold search"],
        "definition": "Accept any option that clears the stated bar (a target, "
                      "a threshold, a deadline) instead of hunting for the "
                      "theoretical best. Success: an option shown to clear the "
                      "bar, with the check visible.",
        "example": "We need any queue setup that sustains 5,000 messages per "
                   "second - first one that works wins.",
        "applies_if": _fits_satisficing,
    },
    {
        "id": "bounded_choice",
        "name": "Bounded Choice",
        "aliases": ["decision analysis", "option selection",
                    "multi-criteria decision"],
        "definition": "Decide among a fixed set of named options by "
                      "establishing the decision-relevant facts about each one, "
                      "verifying any disputed claims, and selecting the option "
                      "the established facts favor. Success: a selection whose "
                      "every load-bearing fact carries evidence.",
        "example": "Should we build this service in Django, Rails, or "
                   "Laravel, given a two-person team and a March deadline?",
        "applies_if": _fits_bounded_choice,
    },
    {
        "id": "tradeoff_analysis",
        "name": "Trade-off Analysis",
        "aliases": ["pareto analysis", "cost-benefit analysis"],
        "definition": "Map how the candidate options trade competing criteria "
                      "against each other - what each choice buys and what it "
                      "sacrifices. Success: dominated options eliminated, and "
                      "every surviving option annotated with its explicit "
                      "trade-off, so the asker can apply their own weights.",
        "example": "Lay out what we gain and lose between strong consistency "
                   "and multi-region availability for the session store.",
        "applies_if": _fits_tradeoff_analysis,
    },
    {
        "id": "game_theory",
        "name": "Game-Theoretic",
        "aliases": ["game theory", "strategic interaction"],
        "definition": "Several parties with genuinely conflicting interests, "
                      "each reacting to the others' moves. Success: a strategy "
                      "stated together with the counter-moves it anticipates "
                      "and survives.",
        "example": "If we cut prices 20%, our two competitors will respond - "
                   "what pricing move is best against their likely reactions?",
        "applies_if": _fits_game_theory,
    },
    {
        "id": "negotiation_strategy",
        "name": "Negotiation Strategy",
        "aliases": ["bargaining", "deal structuring"],
        "definition": "Prepare one side of a negotiation: establish each "
                      "party's interests, alternatives, and walk-away points, "
                      "then derive offers. Success: a position paper grounding "
                      "every proposed term in an established interest or "
                      "alternative.",
        "example": "We are renegotiating our cloud contract - what should we "
                   "ask for, concede, and refuse, given their renewal "
                   "incentives?",
        "applies_if": _fits_negotiation_strategy,
    },

    # --- explanation-shaped ------------------------------------------------
    {
        "id": "diagnosis",
        "name": "Diagnosis / Root-Cause Analysis",
        "aliases": ["root cause analysis", "abductive inference",
                    "troubleshooting", "fault diagnosis", "debugging"],
        "definition": "Explain why an observed situation is happening: "
                      "enumerate candidate causes, test each against the "
                      "observed symptoms, and select the explanation that "
                      "accounts for the observations with the fewest "
                      "unsupported assumptions. Success: one explanation that "
                      "fits every symptom, with the rejected candidates and "
                      "why they fail shown.",
        "example": "Our API p99 doubled last Tuesday and nothing was "
                   "deployed - why?",
        "applies_if": _fits_diagnosis,
    },

    # --- verification-shaped -----------------------------------------------
    {
        "id": "claim_verification",
        "name": "Claim Verification",
        "aliases": ["fact checking", "claim audit", "due diligence"],
        "definition": "Take a set of stated claims and determine, one by one, "
                      "whether each is true, false, or undeterminable - with "
                      "the evidence for the verdict shown. Success: every "
                      "named claim carries a justified verdict; no claim is "
                      "silently skipped.",
        "example": "A vendor says their database does 1M writes/sec, "
                   "zero-downtime upgrades, and SOC2 - check each claim.",
        "applies_if": _fits_claim_verification,
    },

    # --- quantity-shaped ---------------------------------------------------
    {
        "id": "forecasting",
        "name": "Forecasting",
        "aliases": ["prediction", "projection", "scenario analysis"],
        "definition": "Estimate what will happen: identify the drivers of the "
                      "outcome, state the scenarios they produce, and attach "
                      "explicit conditions under which each scenario holds. "
                      "Success: a projection whose every assumption is "
                      "labeled, so the asker can see what would change it.",
        "example": "If signups keep growing 12% monthly, when does our "
                   "single-node Postgres stop keeping up?",
        "applies_if": _fits_forecasting,
    },
    {
        "id": "estimation",
        "name": "Estimation",
        "aliases": ["sizing", "fermi estimation", "quantification"],
        "definition": "Quantify something unknown as it is now - a size, a "
                      "count, a cost, a duration. Success: a value or range "
                      "with the derivation shown step by step and every input "
                      "either sourced or labeled as an assumption.",
        "example": "Roughly how much would it cost per month to store and "
                   "serve 40TB of user images ourselves versus S3?",
        "applies_if": _fits_estimation,
    },
    {
        "id": "risk_assessment",
        "name": "Risk Assessment",
        "aliases": ["failure mode analysis", "premortem", "threat modeling"],
        "definition": "Enumerate what could go wrong with a plan or system, "
                      "label each risk's likelihood and impact (labels, never "
                      "invented probabilities), and propose mitigations for "
                      "the severe ones. Success: a risk register where every "
                      "entry names its trigger and its consequence, and no "
                      "known failure mode is missing.",
        "example": "We are migrating the billing database over a weekend - "
                   "what can go wrong and what do we do about each?",
        "applies_if": _fits_risk_assessment,
    },

    # --- artifact-shaped ---------------------------------------------------
    {
        "id": "design_synthesis",
        "name": "Design / Synthesis",
        "aliases": ["design problem", "architecture design",
                    "plan construction"],
        "definition": "Produce an artifact (a plan, an architecture, a "
                      "document, a process) that satisfies stated "
                      "requirements. Success: the artifact exists and each "
                      "requirement is demonstrably addressed by some named "
                      "part of it.",
        "example": "Design an ingestion pipeline for 2,000 events/sec that "
                   "survives a region outage and costs under $3k/month.",
        "applies_if": _fits_design_synthesis,
    },
    {
        "id": "scheduling",
        "name": "Scheduling / Sequencing",
        "aliases": ["project scheduling", "task ordering", "planning"],
        "definition": "Order a set of actions over time under dependencies, "
                      "deadlines, and capacity limits. Success: a sequence in "
                      "which no action precedes its prerequisites, every "
                      "deadline is either met or explicitly flagged as "
                      "unmeetable, and the binding constraint is named.",
        "example": "Sequence the six workstreams of our data-center exit so "
                   "we are out before the lease ends in November.",
        "applies_if": _fits_scheduling,
    },
    {
        "id": "policy_design",
        "name": "Policy / Rule Design",
        "aliases": ["governance design", "process design", "playbook"],
        "definition": "Write the rules that will govern repeated future "
                      "decisions, rather than making one decision now. "
                      "Success: a rule set that covers the recurring cases, "
                      "states its exceptions explicitly, and can be applied "
                      "by someone who was not in the room.",
        "example": "Define our incident severity levels and who may declare, "
                   "escalate, and close each level.",
        "applies_if": _fits_policy_design,
    },

    # --- many-items-shaped -------------------------------------------------
    {
        "id": "prioritization",
        "name": "Prioritization / Ranking",
        "aliases": ["ranking", "triage", "portfolio selection"],
        "definition": "Order many items by stated criteria rather than pick "
                      "one: establish the criterion values per item, then "
                      "rank. Success: a defensible ordering, with ties and "
                      "near-ties acknowledged instead of hidden.",
        "example": "Rank these 15 tech-debt items by user impact and "
                   "engineering cost for next quarter.",
        "applies_if": _fits_prioritization,
    },
    {
        "id": "resource_allocation",
        "name": "Resource Allocation",
        "aliases": ["budgeting", "portfolio allocation", "capacity planning"],
        "definition": "Distribute a limited resource (money, people, compute, "
                      "time) across competing uses. Success: an allocation "
                      "that exhausts or deliberately reserves the budget, "
                      "with the binding constraint identified and the cost "
                      "of the last unit granted to each use visible.",
        "example": "Split 6 engineers across the mobile rewrite, the API "
                   "v2, and on-call so nothing critical starves.",
        "applies_if": _fits_resource_allocation,
    },
    {
        "id": "categorization",
        "name": "Categorization / Triage",
        "aliases": ["classification", "labeling", "bucketing"],
        "definition": "Assign each of many items to one of a fixed set of "
                      "categories by stated criteria. Success: every item "
                      "labeled, the criteria applied uniformly, and the "
                      "items that genuinely straddle categories flagged "
                      "rather than forced.",
        "example": "Sort these 40 customer complaints into bug, missing "
                   "feature, docs gap, or user error.",
        "applies_if": _fits_categorization,
    },

    # --- stream-shaped -----------------------------------------------------
    {
        "id": "anomaly_detection",
        "name": "Anomaly Detection",
        "aliases": ["outlier detection", "monitoring design"],
        "definition": "Decide which items in an ongoing stream deviate from "
                      "an established norm. Success: each flagged item names "
                      "the norm it violates and by how much; the norm itself "
                      "is stated, not implicit.",
        "example": "Which of these daily spend records are suspicious, and "
                   "against what baseline?",
        "applies_if": _fits_anomaly_detection,
    },
    {
        "id": "control_loop",
        "name": "Control / Feedback Loop",
        "aliases": ["control theory", "feedback control",
                    "closed loop control"],
        "definition": "Not a one-shot decision - an ongoing process that "
                      "keeps adjusting over time toward a setpoint. Success: "
                      "a sensing-deciding-acting cycle where each part is "
                      "named, plus the correction rule when the process "
                      "drifts.",
        "example": "Keep our ad spend hitting a $40 cost-per-acquisition "
                   "week after week as auction prices move.",
        "applies_if": _fits_control_loop,
    },
]
