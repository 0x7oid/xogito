'''
adjudicator.py - the judge . resolves the fights the evaluator's detector froze .

the detector found pairs and moved both members to contested (detector vs
judge : the component that finds a problem never resolves it) . this file
is the judge : it consumes contested_pairs , examines both sides' evidence
trails , and rules - through the single door , logged , reopenable .

the thirteen principles this file implements (full reasoning in DESIGN.md) :

 1. fairness = identical procedure for both sides . asymmetry only from
    stated rules , never per-case discretion .
 2. entailment gate FIRST , entrenchment LATER . both sides' evidence passes
    the same blind check before any history is allowed to matter . a
    battle-tested claim with failed evidence loses to a fresh claim with
    sound evidence .
 3. lexicographic comparison , never numeric scoring . the ruling states
    WHICH tier decided it ("lost on evidence type") , not a score .
 4. minimal change . retract the smallest component that dissolves the
    conflict - sometimes that is an evidence link , not a claim .
 5. evidence invalidation is its own event . a claim whose evidence is
    discredited is RECOMPUTED from remaining support , not decreed dead .
 6. entrenchment = pre-contest rung + evidence type + contest record .
    dependency fan-out is deliberately NOT entrenchment (sunk-cost trap) -
    fan-out lives in the contested decision function , deciding effort ,
    never deciding winners .
 7. defeated claims are preserved with status + reason class , never
    deleted . unverified-with-rationale IS the archive ; no new state .
 8. every loss records its reason class - calibration needs to know WHAT
    broke , not just that something did .
 9. contest record (untested / tested / battle_tested) is a tiebreaker
    only . history ranks survivors of the gate , never rescues gate failures .
10. rebuttal vs undercut . "A is false" and "A's evidence is unreliable"
    are different attacks with different resolutions .
11. pairwise resolution is valid only while pairs do not overlap . pairs
    sharing a claim are PARKED and flagged for graph reasoning - resolving
    them in arbitrary order would let order change outcomes .
12. the argument graph , when needed , is DERIVED from provenance on
    demand - never materialized as stored edges (read model , not storage
    model ; generalize on the second instance) .
13. burden of proof is asymmetric : when everything else ties , the claim
    that was promoted higher carries the heavier burden and yields .
    confidence is expensive to build , so it is expensive to keep .

the adjudicator PROPOSES , the kernel PERMITS , the workspace RECORDS -
same posture as the evaluator . every transition goes through
update_belief_of_claim , so the kernel re-checks legality (contested can
only exit to supported or unverified ; verified is never reachable from a
fight - trust rebuilds gradually , per LEGAL_MOVES) .
'''

import json
from dataclasses import dataclass, field

from llm.client import ask_llm, ask_llm_voted, vote_split_tier
from workspace import Workspace   # NOTE: reconcile flat vs graph.workspace layout before orchestrator wiring
# (layout reconciled : workspace stays a root-level module , see orchestrator.py header)
from core.calibration import log_adjudication_outcome, log_contradiction_outcome, log_vote_split


# ===========================================================================
# PARAMETRES (fuses and vocabularies - code-owned , invisible to any prompt)
# ===========================================================================

# the fuse and the shared orderings moved to parametres.py - one file owns
# every constant , and no prompt can ever see them
from parametres import (
    MAX_PAIRS_PER_ITERATION,
    BELIEF_ORDER,
    # contested is absent on purpose : flag state , not a rung , comparable to nothing
    EVIDENCE_TYPE_ORDER,
    VOTING_N,
)

# contest record tiers . a label ordering , not a count-as-score
CONTEST_RECORD_ORDER = {"untested": 0, "tested": 1, "battle_tested": 2}

# conflict kinds the classification call may return (principle 10)
VALID_CONFLICT_KINDS = (
    "rebuttal",                # the claims directly contradict each other
    "undercut_a_attacks_b",    # claim a attacks claim b's EVIDENCE , not claim b
    "undercut_b_attacks_a",    # claim b attacks claim a's EVIDENCE , not claim a
    "not_a_contradiction",     # the detector saw a ghost (paraphrase , scope)
)

# reason classes recorded on every loss (principle 8) . each names its own
# reopening condition implicitly : new evidence , re-validated evidence , etc.
REASON_CLASSES = (
    "lost_on_entailment",           # its evidence did not state/entail it
    "lost_on_evidence_discredited", # its supporting evidence was invalidated
    "lost_on_evidence_type",        # tied at the gate , weaker evidence type
    "lost_on_contest_record",       # tied on type , weaker challenge history
    "lost_on_burden",               # tied on everything , carried the heavier burden
    "neither_stands",               # both sides failed the gate
)


class AdjudicatorFuseTripped(Exception):
    # raised to HALT the run and show the human . never caught silently
    pass


# ===========================================================================
# the case file - the trial packet for ONE side of ONE fight .
# a MESSAGE like EvaluationContext : built , consumed , gone
# ===========================================================================

@dataclass
class CaseFile:
    claim_id: int
    statement: str
    pre_contest_belief: str        # the rung this claim held BEFORE the fight
    contest_record: str            # "untested" | "tested" | "battle_tested"
    evidence_artifacts: list       # list of (artifact_id, content) tuples
    # filled by the entailment gate :
    gate_passed: bool = False
    gate_gaps: list = field(default_factory=list)
    evidence_type: str = "empirical"   # judged from the evidence content , code-validated


# ===========================================================================
# provenance readers (pure code , no llm) . the adjudicator is the first
# component that actually CONSUMES provenance - the audit trail gets a reader
# ===========================================================================

def _read_pre_contest_belief(workspace, claim_id):
    # the last transition INTO contested records where the claim came from ,
    # e.g. action "belief:supported->contested" . we walk provenance newest
    # to oldest and take the origin of the most recent contest
    entries = workspace.get_provenance("claim", claim_id)
    for entry in reversed(entries):
        if entry.action.startswith("belief:") and entry.action.endswith("->contested"):
            transition = entry.action[len("belief:"):]
            origin = transition.split("->")[0]
            return origin
    # never contested in provenance ? defensive default - treat as unverified ,
    # the least entrenched rung , so a bookkeeping gap can never grant standing
    return "unverified"


def _read_contest_record(workspace, claim_id):
    # principle 9 : count past fights SURVIVED , as a label tier .
    # a survival is a contested->supported transition (the claim came back) .
    # the CURRENT fight's entry into contested is excluded - being in a fight
    # is not surviving one
    entries = workspace.get_provenance("claim", claim_id)

    survivals = 0
    for entry in entries:
        if entry.action == "belief:contested->supported":
            survivals += 1

    if survivals == 0:
        return "untested"
    if survivals == 1:
        return "tested"
    return "battle_tested"


def _build_case_file(workspace, claim_id):
    snapshot = workspace.snapshot()

    statement = ""
    for claim in snapshot["claims"]:
        if claim.id == claim_id:
            statement = claim.statement
            break

    evidence_artifacts = []
    for artifact in workspace.evidence_for_claim(claim_id):
        evidence_artifacts.append((artifact.id, artifact.content))

    return CaseFile(
        claim_id=claim_id,
        statement=statement,
        pre_contest_belief=_read_pre_contest_belief(workspace, claim_id),
        contest_record=_read_contest_record(workspace, claim_id),
        evidence_artifacts=evidence_artifacts,
    )


# ===========================================================================
# overlap detection (principle 11) . pairs sharing a claim are parked -
# resolving overlapping fights in arbitrary order lets order change
# outcomes . the successor is grounded semantics over a graph DERIVED from
# provenance (principle 12) - a read model , built when the second instance
# of overlap demands it , never a stored schema
# ===========================================================================

def _split_overlapping_pairs(contested_pairs):
    # count how many pairs each claim appears in
    appearances = {}
    for pair in contested_pairs:
        for claim_id in (pair["claim_id_a"], pair["claim_id_b"]):
            if claim_id not in appearances:
                appearances[claim_id] = 0
            appearances[claim_id] += 1

    resolvable = []
    parked = []
    for pair in contested_pairs:
        a_overlaps = appearances[pair["claim_id_a"]] > 1
        b_overlaps = appearances[pair["claim_id_b"]] > 1
        if a_overlaps or b_overlaps:
            parked.append(pair)
        else:
            resolvable.append(pair)

    return resolvable, parked


# ===========================================================================
# conflict classification (llm call #1 per pair , code-validated) .
# principle 10 : a rebuttal and an undercut are different attacks needing
# different resolutions . the classifier is a detector-adjacent judgment ,
# so code validates its label like everyone else's
# ===========================================================================

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "conflict_kind": {
            "type": "string",
            "description": "One of: rebuttal, undercut_a_attacks_b, "
                           "undercut_b_attacks_a, not_a_contradiction",
        },
        "reason": {"type": "string"},
    },
    "required": ["conflict_kind", "reason"],
}


def _classify_conflict(case_a, case_b):
    prompt = (
        "Two claims were flagged as conflicting. Classify the conflict.\n\n"
        f"CLAIM A (id {case_a.claim_id}): {case_a.statement}\n"
        f"EVIDENCE FOR A:\n{_render_evidence_block(case_a)}\n\n"
        f"CLAIM B (id {case_b.claim_id}): {case_b.statement}\n"
        f"EVIDENCE FOR B:\n{_render_evidence_block(case_b)}\n\n"
        "KINDS (operational definitions - apply the one that fits)\n"
        "- rebuttal: the two claim statements cannot both be true.\n"
        "- undercut_a_attacks_b: claim A's content states that claim B's "
        "EVIDENCE is flawed, unreliable, or invalid - without denying B itself.\n"
        "- undercut_b_attacks_a: the mirror case.\n"
        "- not_a_contradiction: the claims differ in emphasis, scope, or "
        "wording but can both be true.\n"
    )
    result = json.loads(ask_llm(prompt, CLASSIFY_SCHEMA))
    return result


def _render_evidence_block(case_file):
    lines = []
    for artifact_id, content in case_file.evidence_artifacts:
        lines.append(f"[artifact {artifact_id}]\n{content}")
    return "\n---\n".join(lines) or "(no evidence artifacts)"


# ===========================================================================
# the entailment gate (llm call #2 , one per side , BLIND) . principle 2 :
# this call sees ONE claim and ITS evidence - not the opponent , not any
# history , not any belief label . the gap-list form (logic inversion) :
# an empty gap list IS a pass ; a padded gap item names something checkable .
# the same call classifies the evidence type it read - judged from content ,
# validated by code
# ===========================================================================

GATE_SCHEMA = {
    "type": "object",
    "properties": {
        "gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every part of the claim that the evidence does "
                           "not state or directly entail. Empty if none.",
        },
        "evidence_type": {
            "type": "string",
            "description": "One of: empirical, deductive, testimonial",
        },
        "reason": {"type": "string"},
    },
    "required": ["gaps", "evidence_type", "reason"],
}


def _extract_gate_label(response_text):
    # the label a vote is counted on : an empty gap list IS a pass .
    # labels only - the voting machinery never reads the gap content
    result = json.loads(response_text)
    if result["gaps"]:
        return "fail"
    return "pass"


def _run_entailment_gate(case_file):
    prompt = (
        "Read the evidence, then the claim. List every part of the claim "
        "that the evidence does not state or directly entail. Judge only "
        "what is written - not plausibility, not your own knowledge.\n"
        "If the evidence fully states or entails the claim, return an "
        "empty list.\n\n"
        "Also classify the evidence: 'empirical' if it reports observations "
        "or search results; 'deductive' if it derives the claim by "
        "calculation or logic; 'testimonial' if it relays what a source or "
        "authority states.\n\n"
        f"EVIDENCE:\n{_render_evidence_block(case_file)}\n\n"
        f"CLAIM: {case_file.statement}\n"
    )
    # self-consistency voting at the gate : n blind samples , majority on
    # the pass/fail label . a non-unanimous vote is itself calibration
    # data - logged as a label tier , never a ratio
    majority_label, split, winning_response = ask_llm_voted(
        prompt, GATE_SCHEMA, _extract_gate_label, VOTING_N,
    )
    tier = vote_split_tier(split)
    if tier != "unanimous":
        log_vote_split("adjudicator_entailment_gate", tier, majority_label)

    result = json.loads(winning_response)

    case_file.gate_gaps = result["gaps"]
    case_file.gate_passed = len(result["gaps"]) == 0

    evidence_type = result["evidence_type"].strip().lower()
    if evidence_type not in EVIDENCE_TYPE_ORDER:
        # code validates the llm's enums , always . an unrecognized type
        # defaults to the WEAKEST tier - a bad label can never grant strength
        print(f"[adjudicator] bad evidence_type {evidence_type!r} on claim "
              f"{case_file.claim_id} - defaulting to testimonial")
        evidence_type = "testimonial"
    case_file.evidence_type = evidence_type


# ===========================================================================
# the lexicographic comparison (pure code) . principles 3 , 9 , 13 .
# runs ONLY between two claims that both passed the gate . ordered
# tiebreakers , each a label comparison , each naming itself in the ruling :
#   tier 1 - evidence type   (stronger type wins)
#   tier 2 - contest record  (battle-tested beats untested)
#   tier 3 - burden of proof (everything tied : the HIGHER pre-contest rung
#            yields . principle 13 - confidence expensive to keep . this is
#            the one deliberately inverted comparison , and it runs LAST so
#            entrenchment protects a claim through tiers 1-2 and burdens it
#            only at a dead tie)
#   tier 4 - both_stand      (no stated rule separates them ; per-case
#            discretion is not a rule , so neither loses)
# ===========================================================================

@dataclass
class Ruling:
    outcome: str          # "a_prevails" | "b_prevails" | "both_stand" | "neither_stands"
    deciding_tier: str    # which stated rule decided it - the auditable part
    loser_reason_class: str = ""


def _compare_survivors(case_a, case_b):
    # tier 1 - evidence type
    type_a = EVIDENCE_TYPE_ORDER[case_a.evidence_type]
    type_b = EVIDENCE_TYPE_ORDER[case_b.evidence_type]
    if type_a > type_b:
        return Ruling("a_prevails", "evidence_type", "lost_on_evidence_type")
    if type_b > type_a:
        return Ruling("b_prevails", "evidence_type", "lost_on_evidence_type")

    # tier 2 - contest record
    record_a = CONTEST_RECORD_ORDER[case_a.contest_record]
    record_b = CONTEST_RECORD_ORDER[case_b.contest_record]
    if record_a > record_b:
        return Ruling("a_prevails", "contest_record", "lost_on_contest_record")
    if record_b > record_a:
        return Ruling("b_prevails", "contest_record", "lost_on_contest_record")

    # tier 3 - burden of proof (inverted : higher rung yields)
    rung_a = BELIEF_ORDER.get(case_a.pre_contest_belief, 0)
    rung_b = BELIEF_ORDER.get(case_b.pre_contest_belief, 0)
    if rung_a > rung_b:
        return Ruling("b_prevails", "burden_of_proof", "lost_on_burden")
    if rung_b > rung_a:
        return Ruling("a_prevails", "burden_of_proof", "lost_on_burden")

    # tier 4 - no stated rule separates them
    return Ruling("both_stand", "exhausted_tiers")


# ===========================================================================
# outcome application (pure code , single door) . every transition goes
# through update_belief_of_claim - the kernel re-checks legality , the
# workspace logs provenance . the verdict rationale carries the reason
# class AND the deciding tier , so every loss names its own reopening
# condition (principle 6 , 7 , 8)
# ===========================================================================

def _apply_transition(workspace, claim_id, new_belief, rationale):
    adjudication_verdict = {
        "claim_id": claim_id,
        "proposed_belief": new_belief,
        "evidence_ids": [],
        "evidence_type": "deductive",   # the ruling itself is a derivation
        "is_negative": False,
        "rationale": rationale,
    }
    try:
        workspace.update_belief_of_claim(
            claim_id, new_belief, adjudication_verdict, actor="adjudicator",
        )
        return True
    except ValueError as error:
        # kernel said no - surface it , never swallow it silently
        print(f"[adjudicator] kernel rejected transition on claim "
              f"{claim_id}: {error}")
        return False


def _log_if_was_promoted(case_file, overturned, reason):
    # principle 8 meets calibration : log_adjudication_outcome is the
    # numerator that pairs with promotion_applied . only claims that had
    # actually been PROMOTED before the fight generate an entry - a fresh
    # unverified claim losing a fight says nothing about the evaluator's
    # promotion judgment
    if case_file.pre_contest_belief in ("supported", "verified"):
        log_adjudication_outcome(
            case_file.claim_id,
            case_file.evidence_type,
            overturned,
            reason,
        )


def _resolve_rebuttal(workspace, case_a, case_b, propagation_flags):
    # principle 2 made literal : the gate first , history later

    _run_entailment_gate(case_a)
    _run_entailment_gate(case_b)

    # gate-mixed : one side's evidence holds , the other's does not .
    # decided at tier zero - history never enters
    if case_a.gate_passed and not case_b.gate_passed:
        ruling = Ruling("a_prevails", "entailment_gate", "lost_on_entailment")
    elif case_b.gate_passed and not case_a.gate_passed:
        ruling = Ruling("b_prevails", "entailment_gate", "lost_on_entailment")
    elif not case_a.gate_passed and not case_b.gate_passed:
        ruling = Ruling("neither_stands", "entailment_gate", "neither_stands")
    else:
        # both passed - only NOW does entrenchment get a voice
        ruling = _compare_survivors(case_a, case_b)

    # map ruling to transitions (contested exits to supported or unverified
    # only - the kernel enforces this even if we get it wrong here)
    if ruling.outcome in ("a_prevails", "b_prevails"):
        if ruling.outcome == "a_prevails":
            winner, loser = case_a, case_b
        else:
            winner, loser = case_b, case_a

        _apply_transition(
            workspace, winner.claim_id, "supported",
            f"won adjudication vs claim {loser.claim_id} "
            f"(decided on: {ruling.deciding_tier})",
        )
        _apply_transition(
            workspace, loser.claim_id, "unverified",
            f"[{ruling.loser_reason_class}] lost adjudication vs claim "
            f"{winner.claim_id} (decided on: {ruling.deciding_tier}) . "
            f"gaps: {'; '.join(loser.gate_gaps) or 'none'} . "
            f"reopens on: new evidence for this claim",
        )
        _log_if_was_promoted(winner, overturned=False,
                             reason=f"survived adjudication ({ruling.deciding_tier})")
        _log_if_was_promoted(loser, overturned=True,
                             reason=f"{ruling.loser_reason_class} ({ruling.deciding_tier})")

        # principle 10 (propagation) : the loser's other dependents are now
        # suspect . flagged for the next iteration , never auto-re-evaluated
        propagation_flags.append({
            "claim_id": loser.claim_id,
            "cause": "claim_demoted_in_adjudication",
        })

    elif ruling.outcome == "both_stand":
        for case_file in (case_a, case_b):
            _apply_transition(
                workspace, case_file.claim_id, "supported",
                f"adjudication: both stand (opponent claim "
                f"{case_b.claim_id if case_file is case_a else case_a.claim_id} , "
                "no stated rule separates them)",
            )
            _log_if_was_promoted(case_file, overturned=False,
                                 reason="both_stand after full comparison")

    else:  # neither_stands
        for case_file in (case_a, case_b):
            _apply_transition(
                workspace, case_file.claim_id, "unverified",
                f"[neither_stands] adjudication: evidence failed the "
                f"entailment gate . gaps: {'; '.join(case_file.gate_gaps) or 'none'} . "
                "reopens on: new evidence for this claim",
            )
            _log_if_was_promoted(case_file, overturned=True,
                                 reason="neither_stands: evidence failed entailment")
            propagation_flags.append({
                "claim_id": case_file.claim_id,
                "cause": "claim_demoted_in_adjudication",
            })

    return ruling


def _resolve_undercut(workspace, attacker, target, propagation_flags):
    # principles 4 , 5 , 10 : the attack is on the target's EVIDENCE , so
    # the minimal component to retract is an evidence LINK , not the claim .
    # the target's belief then changes as a CONSEQUENCE of losing support ,
    # never as a decree .
    #
    # the attacker must still pass its own gate - an unsupported accusation
    # discredits nothing (fairness : the identical procedure , both sides)

    _run_entailment_gate(attacker)

    if not attacker.gate_passed:
        # the undercut itself is unsupported . the attacker loses ; the
        # target survives the fight untouched (its evidence was never
        # legitimately impeached)
        _apply_transition(
            workspace, attacker.claim_id, "unverified",
            f"[lost_on_entailment] undercut of claim {target.claim_id}'s "
            f"evidence was itself unsupported . gaps: "
            f"{'; '.join(attacker.gate_gaps)} . reopens on: new evidence",
        )
        _apply_transition(
            workspace, target.claim_id, "supported",
            f"survived undercut from claim {attacker.claim_id} "
            "(the attack failed its own entailment gate)",
        )
        _log_if_was_promoted(attacker, overturned=True,
                             reason="undercut attempt failed entailment")
        _log_if_was_promoted(target, overturned=False,
                             reason="survived an unsupported undercut")
        return Ruling("target_survives", "entailment_gate", "lost_on_entailment")

    # the undercut holds : discredit the target's evidence links
    # (removal-with-receipt via unlink_evidence - the link's existence and
    # removal both live in provenance forever) , then RECOMPUTE the target
    discredited_ids = []
    for artifact_id, _content in target.evidence_artifacts:
        workspace.unlink_evidence(target.claim_id, artifact_id, actor="adjudicator")
        discredited_ids.append(artifact_id)

        # principle 10 (propagation) : every OTHER claim leaning on this
        # artifact is now suspect - flagged , not auto-demoted
        propagation_flags.append({
            "artifact_id": artifact_id,
            "cause": "evidence_discredited_in_adjudication",
        })

    # recompute : contested exits DOWN because its support is gone .
    # v1 note : the undercut targets the evidence that was in the fight ,
    # which for a v1 claim is typically all of it . if partial discrediting
    # arrives later (some links survive) , the recompute becomes : remaining
    # evidence -> supported , none -> unverified
    _apply_transition(
        workspace, target.claim_id, "unverified",
        f"[lost_on_evidence_discredited] supporting evidence (artifacts "
        f"{discredited_ids}) discredited by claim {attacker.claim_id} . "
        "reopens on: the discredited evidence being re-validated , or new "
        "independent evidence",
    )
    _apply_transition(
        workspace, attacker.claim_id, "supported",
        f"won adjudication: undercut of claim {target.claim_id}'s evidence "
        "held (decided on: entailment_gate)",
    )
    _log_if_was_promoted(target, overturned=True,
                         reason="supporting evidence discredited")
    _log_if_was_promoted(attacker, overturned=False,
                         reason="undercut held")

    return Ruling("attacker_prevails", "entailment_gate",
                  "lost_on_evidence_discredited")


def _resolve_ghost(workspace, case_a, case_b):
    # the detector saw a ghost (principle 7 meets LEGAL_MOVES) : both claims
    # return to supported , NOT to their pre-contest rung . a formerly
    # verified claim does not snap back - trust once questioned rebuilds
    # gradually , even when the question was mistaken . (DESIGN.md decision)
    for case_file, opponent in ((case_a, case_b), (case_b, case_a)):
        _apply_transition(
            workspace, case_file.claim_id, "supported",
            f"adjudication: flagged conflict with claim {opponent.claim_id} "
            "was not a real contradiction (emphasis/scope/paraphrase)",
        )


# ===========================================================================
# the contested decision function (principle 6 - fan-out decides EFFORT ,
# never winners) . v1 STUB : probe everything . the seam for the real CDF :
#   fan-out = 0  -> park (stay contested , report shows both sides)
#   probe        -> resolve with the calls above
#   diverge      -> spawn a contradiction-triggered verify task - the ONLY
#                   legal birthplace of kind="verify" , depth cap 1
# fan-out will be computed from graph data (dependents per claim) that only
# becomes meaningful after the evaluator has run for a while - deferred
# from evidence , not imagination
# ===========================================================================

def _decide_effort(pair):
    return "probe"   # v1 : every fight gets resolved


# ===========================================================================
# the one public function . the orchestrator calls this after
# evaluate_iteration returns contested_pairs
# ===========================================================================

def adjudicate_contested_pairs(workspace, contested_pairs):
    if not contested_pairs:
        return {"resolved": [], "parked": [], "propagation_flags": []}

    # fuse : runaway fighting
    if len(contested_pairs) > MAX_PAIRS_PER_ITERATION:
        raise AdjudicatorFuseTripped(
            f"received {len(contested_pairs)} contested pairs in one "
            f"iteration (fuse: {MAX_PAIRS_PER_ITERATION}). Abnormal - "
            "halting for human review."
        )

    # principle 11 : overlapping pairs are parked , never guessed at
    resolvable, parked = _split_overlapping_pairs(contested_pairs)
    for pair in parked:
        print(f"[adjudicator] parked pair ({pair['claim_id_a']}, "
              f"{pair['claim_id_b']}): shares a claim with another pair - "
              "needs graph reasoning (derived from provenance when built)")

    resolved = []
    propagation_flags = []

    for pair in resolvable:

        if _decide_effort(pair) != "probe":
            continue   # v1 : unreachable ; the seam for the real CDF

        case_a = _build_case_file(workspace, pair["claim_id_a"])
        case_b = _build_case_file(workspace, pair["claim_id_b"])

        # llm call #1 : classify the conflict . code validates the label
        classification = _classify_conflict(case_a, case_b)
        conflict_kind = classification["conflict_kind"].strip().lower()
        if conflict_kind not in VALID_CONFLICT_KINDS:
            print(f"[adjudicator] bad conflict_kind {conflict_kind!r} on "
                  f"pair ({case_a.claim_id}, {case_b.claim_id}) - parking it")
            parked.append(pair)
            continue

        # the detector's report card : was this flag a real fight ?
        was_real = conflict_kind != "not_a_contradiction"
        log_contradiction_outcome(
            case_a.claim_id, case_b.claim_id, was_real,
            classification["reason"],
        )

        if conflict_kind == "not_a_contradiction":
            _resolve_ghost(workspace, case_a, case_b)
            ruling = Ruling("dismissed", "classification")
        elif conflict_kind == "undercut_a_attacks_b":
            ruling = _resolve_undercut(workspace, attacker=case_a,
                                       target=case_b,
                                       propagation_flags=propagation_flags)
        elif conflict_kind == "undercut_b_attacks_a":
            ruling = _resolve_undercut(workspace, attacker=case_b,
                                       target=case_a,
                                       propagation_flags=propagation_flags)
        else:  # rebuttal
            ruling = _resolve_rebuttal(workspace, case_a, case_b,
                                       propagation_flags)

        resolved.append({
            "claim_id_a": case_a.claim_id,
            "claim_id_b": case_b.claim_id,
            "conflict_kind": conflict_kind,
            "outcome": ruling.outcome,
            "deciding_tier": ruling.deciding_tier,
        })

    return {
        "resolved": resolved,
        "parked": parked,
        "propagation_flags": propagation_flags,
    }