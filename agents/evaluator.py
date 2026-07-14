'''
now that the executor integerate the artifcates as unverified , the evaluator will be responsible to:
-compare the delievered artifacts with the done_when in the task
- belief proposal : an example of that is claim 5 deserves to be supported based on artifact 3 and 4
    we introduce a new metric here for the evidence , that is the evidence_type since not all evidence are of the same weight
    for example a theorem is not of the same weight as a saying .
- finally contradiction detection between claims , the resolution of contradiction will be handled by the adjucator (detector vs judge analogy)
'''
'''
now that the executor integrates the artifacts as unverified , the evaluator is responsible for:
- comparing the delivered artifacts with the done_when in the task
- belief proposal : an example of that is claim 5 deserves to be supported based on artifact 3 and 4
    we introduce a new metric here for the evidence , that is the evidence_type since not all evidence are of the same weight
    for example a theorem is not of the same weight as a saying .
- finally contradiction detection between claims , the resolution of contradiction will be handled by the adjudicator (detector vs judge analogy)

the evaluator PROPOSES , the kernel PERMITS , the workspace RECORDS.
every verdict must terminate in something code can check :
- evidence ids are assigned by CODE (the artifact under evaluation) , never asked from the llm
- verdicts cite ARTIFACT ids only -> self-testimony chains terminate in executor artifacts BY CONSTRUCTION
  (a verdict physically cannot cite another claim as evidence , so belief cannot bootstrap on belief)
- promotions pass a DUAL-PASS : a second blind read of just the claim + evidence , no first-pass reasoning visible
- tempo rule : one rung up per claim per iteration maximum . confidence is expensive , doubt is free .
- bias defense is fields not attitudes : the gauntlet checks evidence PROPERTIES
  (scope stated , 2+ tasks for verified , quantitative caps) , it never asks the model to "be skeptical"
'''

import json
from dataclasses import dataclass, field

from llm.client import ask_llm
from workspace import Workspace, belief_ladder
from verdict import Verdict
from calibration import log_dual_pass_disagreement


# ===========================================================================
# PARAMETRES
# ===========================================================================

MAX_VERDICTS_PER_ITERATION = 40   # runaway-judgment fuse (~2x healthy max)

BELIEF_ORDER = {"unverified": 0, "supported": 1, "verified": 2}
# contested is absent on purpose : it is not a rung on the ladder but a flag
# state - reachable from anywhere , comparable to nothing

VALID_EVIDENCE_TYPES = {"empirical", "deductive", "testimonial"}

# heuristic scope markers for negative evidence - the artifact must say
# WHERE/HOW it searched . crude on purpose : v1 checks presence not quality
SCOPE_MARKERS = (
    "searched", "search scope", "sources consulted", "looked in",
    "checked", "queried", "reviewed", "examined sources",
)


class EvaluatorFuseTripped(Exception):
    # raised to HALT the run and show the human . never caught silently
    pass


# ===========================================================================
# the evaluation context - the trial packet for ONE completed task .
# it is a MESSAGE like ExecutionResult and Verdict : built , consumed , gone
# ===========================================================================

@dataclass
class EvaluationContext:
    # v1 : one artifact per task (the executor produces exactly one)
    task_id: int
    done_when: str
    artifact_id: int
    artifact_content: str
    linked_claims_and_beliefs: dict     # claim_id -> current belief
    linked_claims_statements: dict      # claim_id -> statement (the llm must read them to judge them)


def _build_evaluation_contexts(workspace):
    # one context per completed task . pure code , no llm
    snapshot = workspace.snapshot()
    contexts = []

    for task in snapshot["tasks"]:
        if task.status != "completed":
            continue

        for artifact in snapshot["artifacts"]:
            if artifact.task_id != task.id:
                continue

            # claims linked to this artifact (via the synced evidence lists)
            linked_beliefs = {}
            linked_statements = {}
            for claim in snapshot["claims"]:
                if artifact.id in claim.evidence_ids:
                    linked_beliefs[claim.id] = claim.belief
                    linked_statements[claim.id] = claim.statement

            contexts.append(EvaluationContext(
                task_id=task.id,
                done_when=task.done_when,
                artifact_id=artifact.id,
                artifact_content=artifact.content,
                linked_claims_and_beliefs=linked_beliefs,
                linked_claims_statements=linked_statements,
            ))

    return contexts


def _render_belief_table(workspace):
    # iteration-level view : ALL claims with ids , for awareness and for
    # contradiction detection . the ids matter - the evaluator must
    # reference claims in its output
    snapshot = workspace.snapshot()
    belief_lines = []
    for claim in snapshot["claims"]:
        belief_lines.append(f"- [{claim.id}][{claim.belief}] {claim.statement}")
    return "\n".join(belief_lines) or "(no claims yet)"


# ===========================================================================
# verdict proposal (llm call #1 , one per context)
# the prompt never sees the gauntlet rules , the kernel thresholds or the
# fuses . graders' rubrics stay invisible to generators
# ===========================================================================

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "integer"},
                    "proposed_belief": {
                        "type": "string",
                        "description": "One of: unverified, supported, verified, contested",
                    },
                    "evidence_type": {
                        "type": "string",
                        "description": "One of: empirical, deductive, testimonial",
                    },
                    "is_negative": {
                        "type": "boolean",
                        "description": "True ONLY for empirical evidence of the form "
                                       "'searched and found nothing'.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One or two sentences: why this evidence "
                                       "entitles this belief.",
                    },
                },
                "required": ["claim_id", "proposed_belief", "evidence_type",
                             "is_negative", "rationale"],
            },
        },
        "done_when_satisfied": {"type": "boolean"},
        "done_when_reason": {"type": "string"},
    },
    "required": ["verdicts", "done_when_satisfied", "done_when_reason"],
}


def _propose_verdicts(context, belief_table):
    claim_lines = []
    for claim_id in context.linked_claims_statements:
        belief = context.linked_claims_and_beliefs[claim_id]
        statement = context.linked_claims_statements[claim_id]
        claim_lines.append(f"- [{claim_id}][{belief}] {statement}")
    claims_listing = "\n".join(claim_lines) or "(this artifact asserted no claims)"

    prompt = (
        "You are an evaluation component. Judge whether ONE task's output "
        "entitles its claims to a change of belief label. Judge only from "
        "the evidence shown - not from plausibility or your own knowledge.\n\n"
        f"It counts as done when: {context.done_when}\n\n"
        f"THE ARTIFACT (id {context.artifact_id}) THIS TASK PRODUCED:\n"
        f"{context.artifact_content}\n\n"
        f"CLAIMS LINKED TO THIS ARTIFACT:\n{claims_listing}\n\n"
        f"ALL CLAIMS IN THE SYSTEM (for awareness, judge only the linked ones):\n"
        f"{belief_table}\n\n"
        "LABELS\n"
        "- unverified: the evidence does not establish the claim\n"
        "- supported: the evidence genuinely backs the claim\n"
        "- verified: the evidence is strong enough that the claim can be "
        "relied on (be conservative with this)\n"
        "- contested: the evidence CONTRADICTS the claim\n\n"
        "RULES\n"
        "- Judge each linked claim exactly once, by its id.\n"
        "- evidence_type: 'empirical' if the artifact reports observations "
        "or search results; 'deductive' if it derives the claim by "
        "calculation or logic; 'testimonial' if it relays what a source "
        "or authority states.\n"
        "- is_negative: true ONLY when the evidence is of the form "
        "'searched and found nothing'.\n"
        "- If the evidence is thin, say unverified. Do not reward effort, "
        "volume, or confident wording - only what the evidence shows.\n"
        "- Also answer whether the artifact satisfies the done-when "
        "condition, with a one-line reason.\n"
    )
    result = json.loads(ask_llm(prompt, VERDICT_SCHEMA))
    return result


# ===========================================================================
# the gauntlet (pure code) . every proposal passes or dies HERE ,
# individually . a dropped verdict is logged , never fatal .
# order matters : cheap existence checks first , judgment-adjacent last .
# checks 6 , 8 , 9 DEMOTE instead of drop - the information survives ,
# the overreach does not
# ===========================================================================

def _looks_quantitative(statement):
    # crude v1 heuristic : digits or percent signs in the claim
    if "%" in statement:
        return True
    for character in statement:
        if character.isdigit():
            return True
    return False


def _artifact_states_scope(artifact_content):
    lowered = artifact_content.lower()
    for marker in SCOPE_MARKERS:
        if marker in lowered:
            return True
    return False


def _run_gauntlet(raw_verdicts, context, workspace, promoted_this_iteration):
    snapshot = workspace.snapshot()

    claims_by_id = {}
    for claim in snapshot["claims"]:
        claims_by_id[claim.id] = claim

    artifacts_by_id = {}
    for artifact in snapshot["artifacts"]:
        artifacts_by_id[artifact.id] = artifact

    survivors = []

    for raw in raw_verdicts:

        claim = claims_by_id.get(raw["claim_id"])

        # check 1 - the claim must exist
        if claim is None:
            print(f"[evaluator] dropped verdict: claim {raw['claim_id']} does not exist")
            continue

        # check 2 - label sanity (code validates the llm's enums , always)
        proposed = raw["proposed_belief"].strip().lower()
        if proposed not in ("unverified", "supported", "verified", "contested"):
            print(f"[evaluator] dropped verdict on claim {claim.id}: bad label {proposed!r}")
            continue

        evidence_type = raw["evidence_type"].strip().lower()
        if evidence_type not in VALID_EVIDENCE_TYPES:
            print(f"[evaluator] dropped verdict on claim {claim.id}: bad evidence_type {evidence_type!r}")
            continue

        # the verdict's evidence is the artifact under evaluation - code
        # decides what counts as evidence , the model only judges
        evidence_ids = [context.artifact_id]

        # check 3 - the cited evidence must actually be linked to the claim
        if context.artifact_id not in claim.evidence_ids:
            print(f"[evaluator] dropped verdict on claim {claim.id}: artifact "
                  f"{context.artifact_id} is not linked to it")
            continue

        # check 4 - no-op transitions carry no information
        if proposed == claim.belief:
            continue

        # check 5 - tempo : one rung at a time , once per iteration .
        # contested and downward moves always allowed (doubt is free)
        if proposed in BELIEF_ORDER and claim.belief in BELIEF_ORDER:
            step = BELIEF_ORDER[proposed] - BELIEF_ORDER[claim.belief]
            if step > 1:
                print(f"[evaluator] dropped verdict on claim {claim.id}: "
                      f"{claim.belief} -> {proposed} skips a rung")
                continue
            if step == 1 and claim.id in promoted_this_iteration:
                print(f"[evaluator] dropped verdict on claim {claim.id}: "
                      "already promoted this iteration (tempo rule)")
                continue

        # check 6 - testimonial evidence never verifies . pointers alone
        # cannot make a claim reliable
        if proposed == "verified" and evidence_type == "testimonial":
            proposed = "supported"
            raw["rationale"] += " [capped: testimonial evidence cannot verify]"

        # check 7 - negative evidence must state its search scope or it
        # supports nothing (absence of evidence != evidence of absence)
        if raw["is_negative"] and proposed in ("supported", "verified"):
            if not _artifact_states_scope(context.artifact_content):
                print(f"[evaluator] dropped verdict on claim {claim.id}: negative "
                      "evidence with no stated search scope")
                continue

        # check 8 - quantitative trap : numbers without methodology cap at
        # supported (sample size / base rate checking is judgment ; the cap is code)
        if proposed == "verified" and _looks_quantitative(claim.statement):
            proposed = "supported"
            raw["rationale"] += " [capped: quantitative claim, single-source]"

        # check 9 - framing independence : verified needs evidence from at
        # least 2 DIFFERENT tasks (same method = same blind spots)
        if proposed == "verified":
            source_task_ids = set()
            for linked_artifact_id in claim.evidence_ids:
                linked_artifact = artifacts_by_id.get(linked_artifact_id)
                if linked_artifact is not None:
                    source_task_ids.add(linked_artifact.task_id)
            if len(source_task_ids) < 2:
                proposed = "supported"
                raw["rationale"] += " [capped: verified needs evidence from 2+ tasks]"

        # survived - build the real message
        survivors.append(Verdict(
            claim_id=claim.id,
            proposed_belief=proposed,
            evidence_ids=evidence_ids,
            evidence_type=evidence_type,
            is_negative=raw["is_negative"],
            rationale=raw["rationale"].strip(),
        ))

    return survivors


# ===========================================================================
# dual-pass (llm call #2 , blind) . PROMOTIONS ONLY .
# the second pass sees the claim and the evidence - nothing else . not the
# first pass's rationale , not the proposed label . one question .
# disagreement kills the promotion and is logged to calibration
# ===========================================================================

DUAL_PASS_SCHEMA = {
    "type": "object",
    "properties": {
        "entailed": {
            "type": "boolean",
            "description": "true if the evidence actually states or directly "
                           "entails the claim, false otherwise",
        },
        "reason": {"type": "string"},
    },
    "required": ["entailed", "reason"],
}


def _dual_pass_confirms(verdict, workspace):
    snapshot = workspace.snapshot()

    claim_statement = ""
    for claim in snapshot["claims"]:
        if claim.id == verdict.claim_id:
            claim_statement = claim.statement
            break

    evidence_texts = []
    for artifact in snapshot["artifacts"]:
        if artifact.id in verdict.evidence_ids:
            evidence_texts.append(artifact.content)
    evidence_block = "\n---\n".join(evidence_texts)

    prompt = (
        "Read the evidence, then the claim. Answer one question: does the "
        "evidence actually state or directly entail the claim? Judge only "
        "what is written - not plausibility, not your own knowledge.\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        f"CLAIM: {claim_statement}\n"
    )
    result = json.loads(ask_llm(prompt, DUAL_PASS_SCHEMA))
    return result["entailed"], result["reason"]


# ===========================================================================
# contradiction handling . the detector NEVER resolves - it finds pairs ,
# moves both to contested through the single door , and hands the pairs to
# the adjudicator (tomorrow's file) . detector vs judge
# ===========================================================================

CONTRADICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id_a": {"type": "integer"},
                    "claim_id_b": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["claim_id_a", "claim_id_b", "reason"],
            },
        }
    },
    "required": ["pairs"],
}


def _detect_contradictions(belief_table):
    prompt = (
        "Below is a table of claims with ids and belief labels. List every "
        "PAIR of claims that cannot both be true at the same time.\n\n"
        "- Report only genuine logical or factual conflicts. Different "
        "emphasis, different scope, or paraphrase is NOT a contradiction.\n"
        "- If there are none, return an empty list. Do not invent pairs.\n\n"
        f"CLAIMS:\n{belief_table}\n"
    )
    result = json.loads(ask_llm(prompt, CONTRADICTION_SCHEMA))
    return result["pairs"]


def handle_contradictions(workspace):
    # detect -> validate -> contest both members through the single door ->
    # return the pairs for the adjudicator . never resolves anything .
    snapshot = workspace.snapshot()
    existing_claim_ids = set()
    for claim in snapshot["claims"]:
        existing_claim_ids.add(claim.id)

    belief_table = _render_belief_table(workspace)
    raw_pairs = _detect_contradictions(belief_table)

    contested_pairs = []
    for pair in raw_pairs:

        # code checks the detector too - never trust , always verify
        if pair["claim_id_a"] not in existing_claim_ids:
            print(f"[evaluator] dropped contradiction pair: claim {pair['claim_id_a']} does not exist")
            continue
        if pair["claim_id_b"] not in existing_claim_ids:
            print(f"[evaluator] dropped contradiction pair: claim {pair['claim_id_b']} does not exist")
            continue
        if pair["claim_id_a"] == pair["claim_id_b"]:
            print("[evaluator] dropped contradiction pair: a claim cannot contradict itself")
            continue

        # both members go to contested . the verdict rationale records who
        # the opponent is , so the adjudicator can reconstruct the fight
        for claim_id, opponent_id in ((pair["claim_id_a"], pair["claim_id_b"]),
                                      (pair["claim_id_b"], pair["claim_id_a"])):
            pair_verdict = {
                "claim_id": claim_id,
                "proposed_belief": "contested",
                "evidence_ids": [],
                "evidence_type": "deductive",
                "is_negative": False,
                "rationale": f"contradiction with claim {opponent_id}: {pair['reason']}",
            }
            try:
                workspace.update_belief_of_claim(
                    claim_id, "contested", pair_verdict, actor="evaluator",
                )
            except ValueError as error:
                # already contested is the common benign case here
                print(f"[evaluator] could not contest claim {claim_id}: {error}")

        contested_pairs.append(pair)

    return contested_pairs


# ===========================================================================
# the one public function . the orchestrator calls this after each batch
# integrates . returns everything the checkpoint will want to see
# ===========================================================================

def evaluate_iteration(spec, workspace):
    contexts = _build_evaluation_contexts(workspace)

    if not contexts:
        print("[evaluator] nothing to evaluate - no completed-task artifacts")
        return {"applied": [], "contested_pairs": [], "done_when_failures": []}

    belief_table = _render_belief_table(workspace)

    applied = []
    done_when_failures = []
    promoted_this_iteration = set()
    total_raw_verdicts = 0

    for context in contexts:

        proposal = _propose_verdicts(context, belief_table)
        total_raw_verdicts += len(proposal["verdicts"])

        # fuse : runaway judgment
        if total_raw_verdicts > MAX_VERDICTS_PER_ITERATION:
            raise EvaluatorFuseTripped(
                f"evaluator produced over {MAX_VERDICTS_PER_ITERATION} verdicts "
                "in one iteration. Abnormal - halting for human review."
            )

        # done_when failures are RECORDED not enforced (v1) - the planner
        # sees the gap in the next iteration's context anyway
        if not proposal["done_when_satisfied"]:
            done_when_failures.append({
                "task_id": context.task_id,
                "reason": proposal["done_when_reason"],
            })
            print(f"[evaluator] task {context.task_id} did not satisfy its "
                  f"done_when: {proposal['done_when_reason']}")

        survivors = _run_gauntlet(
            proposal["verdicts"], context, workspace, promoted_this_iteration
        )

        for verdict in survivors:

            # dual-pass gate on promotions only - demotions and contested
            # always pass through (doubt is free)
            old_belief = context.linked_claims_and_beliefs.get(verdict.claim_id, "")

            is_promotion = (
                verdict.proposed_belief in BELIEF_ORDER
                and old_belief in BELIEF_ORDER
                and BELIEF_ORDER[verdict.proposed_belief] > BELIEF_ORDER[old_belief]
            )

            if is_promotion:
                confirmed, reason = _dual_pass_confirms(verdict, workspace)
                if not confirmed:
                    log_dual_pass_disagreement(verdict, reason)
                    print(f"[evaluator] dual-pass blocked promotion of claim "
                          f"{verdict.claim_id}: {reason}")
                    continue

            # the single door . the workspace re-checks legality via the
            # kernel and logs provenance - belt and braces by design
            verdict_record = {
                "claim_id": verdict.claim_id,
                "proposed_belief": verdict.proposed_belief,
                "evidence_ids": verdict.evidence_ids,
                "evidence_type": verdict.evidence_type,
                "is_negative": verdict.is_negative,
                "rationale": verdict.rationale,
            }
            try:
                workspace.update_belief_of_claim(
                    verdict.claim_id, verdict.proposed_belief,
                    verdict_record, actor="evaluator",
                )
            except ValueError as error:
                # kernel said no - drop this one verdict , keep the batch
                print(f"[evaluator] kernel rejected verdict on claim "
                      f"{verdict.claim_id}: {error}")
                continue

            if is_promotion:
                promoted_this_iteration.add(verdict.claim_id)
            applied.append(verdict_record)

    # contradiction pass runs AFTER verdicts so it sees the updated table
    contested_pairs = handle_contradictions(workspace)

    return {
        "applied": applied,
        "contested_pairs": contested_pairs,
        "done_when_failures": done_when_failures,
    }