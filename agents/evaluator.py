'''
now that the executor integerate the artifcates as unverified , the evaluator will be responsible to:
-compare the delievered artifacts with the done_when in the task
- belief proposal : an example of that is claim 5 deserves to be supported based on artifact 3 and 4
    we introduce a new metric here for the evidence , that is the evidence_type since not all evidence are of the same weight
    for example a theorem is not of the same weight as a saying .
- finally contradiction detection between claims , the resolution of contradiction will be handled by the adjucator (detector vs judge analogy)
'''
'''
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
import re
from dataclasses import dataclass, field

from llm.client import ask_llm, ask_llm_voted, vote_split_tier
from workspace import Workspace   # NOTE: reconcile flat vs graph.workspace layout before orchestrator wiring
# (layout reconciled : workspace stays a root-level module , see orchestrator.py header)
from model.verdict import Verdict  # fixed typo: was model.veridict
from core.calibration import log_dual_pass_disagreement, log_promotion_applied, log_vote_split


# ===========================================================================
# PARAMETRES
# ===========================================================================

# fuses and shared vocabularies moved to parametres.py - one file owns
# every constant , and no prompt can ever see them
from parametres import (
    MAX_VERDICTS_PER_ITERATION,
    BELIEF_ORDER,
    VALID_BELIEF_LABELS,
    VALID_EVIDENCE_TYPES,
    VOTING_N,
    MIN_SOURCE_TASKS_FOR_SUPPORTED,
    TERMINAL_TASK_KINDS,
    EXTERNAL_CONFIRMATION_MARKERS,
    FORMULA_MARKERS,
    UNOBSERVED_MEASUREMENT_MARKERS,
)


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
    linked_claims: dict   # claim_id -> {"belief": ..., "statement": ...}
    # one dict , one entry per claim . two parallel dicts keyed by the same
    # ids was the synced-lists disease again - nothing stopped them drifting


def _build_evaluation_contexts(workspace, new_artifact_ids=None):
    # one context per completed task . pure code , no llm
    # FIX: without a filter , every completed task from every PAST iteration
    # was re-judged each time - burning the verdict fuse for nothing and
    # re-litigating settled claims . the orchestrator passes the ids of the
    # artifacts integrated THIS iteration ; None still means "everything"
    # so the function keeps working standalone
    snapshot = workspace.snapshot()
    contexts = []

    for task in snapshot["tasks"]:
        if task.status != "completed":
            continue

        for artifact in snapshot["artifacts"]:
            if artifact.task_id != task.id:
                continue
            if new_artifact_ids is not None and artifact.id not in new_artifact_ids:
                continue

            # claims linked to this artifact (via the synced evidence lists)
            linked_claims = {}
            for claim in snapshot["claims"]:
                if artifact.id in claim.evidence_ids:
                    linked_claims[claim.id] = {
                        "belief": claim.belief,
                        "statement": claim.statement,
                    }

            contexts.append(EvaluationContext(
                task_id=task.id,
                done_when=task.done_when,
                artifact_id=artifact.id,
                artifact_content=artifact.content,
                linked_claims=linked_claims,
            ))

    return contexts


def _format_claim_line(claim_id, belief, statement):
    # the ONE place the claim-line format lives . the belief table and the
    # per-context listing share presentation , never data
    return f"- [{claim_id}][{belief}] {statement}"


#basically this is a table of claims with their ids and belief labels , for awareness and for contradiction detection
def _render_belief_table(workspace):
    # iteration-level view : ALL claims with ids . used by contradiction
    # detection only now - _propose_verdicts no longer sees it (an evaluator
    # shown unrelated claims starts judging by coherence with them , which
    # is plausibility leaking back in)
    snapshot = workspace.snapshot()
    belief_lines = []
    for claim in snapshot["claims"]:
        belief_lines.append(_format_claim_line(claim.id, claim.belief, claim.statement))
    return "\n".join(belief_lines) or "(no claims yet)"


# ===========================================================================
# verdict proposal (llm call #1 , one per context)
# the prompt never sees the gauntlet rules (for bias prevention) , the kernel thresholds or the
# fuses . graders' rubrics stay invisible to generators
#
# prompt style rule : operational definitions , never dispositions . we do
# not tell the model to "be conservative" (it would perform conservatism) ;
# we define the labels so tightly that conservatism is the only fit
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
                        "description": "One or two sentences: which part of the "
                                       "artifact establishes which part of the claim.",
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


def _propose_verdicts(context):
    claim_lines = []
    for claim_id in context.linked_claims:
        entry = context.linked_claims[claim_id]
        claim_lines.append(_format_claim_line(claim_id, entry["belief"], entry["statement"]))
    claims_listing = "\n".join(claim_lines) or "(this artifact asserted no claims)"

    prompt = (
        "You are an evaluation component. Judge whether ONE task's output "
        "entitles its claims to a change of belief label. Your judgment is "
        "about what the artifact's text states - not what is plausible, and "
        "not what you know from elsewhere.\n\n"
        f"It counts as done when: {context.done_when}\n\n"
        f"THE ARTIFACT (id {context.artifact_id}) THIS TASK PRODUCED:\n"
        f"{context.artifact_content}\n\n"
        f"CLAIMS LINKED TO THIS ARTIFACT:\n{claims_listing}\n\n"
        "LABELS (operational definitions - apply the one that fits)\n"
        "- unverified: the artifact does not state the claim and the claim "
        "does not follow from it.\n"
        "- supported: the artifact states something that backs the claim, "
        "but parts of the claim go beyond what the artifact establishes.\n"
        "- verified: the artifact explicitly states the claim, or the claim "
        "follows from the artifact by a derivation you can trace step by step.\n"
        "- contested: the artifact states something incompatible with the claim.\n\n"
        "RULES\n"
        "- Judge each linked claim exactly once, by its id.\n"
        "- evidence_type: 'empirical' if the artifact reports observations "
        "or search results; 'deductive' if it derives the claim by "
        "calculation or logic; 'testimonial' if it relays what a source "
        "or authority states.\n"
        "- is_negative: true ONLY when the evidence is of the form "
        "'searched and found nothing'.\n"
        "- Judge what the artifact states, not how much text there is or "
        "how it is phrased.\n"
        "- Also answer whether the artifact satisfies the done-when "
        "condition, with a one-line reason.\n"
    )
    # SEAM: verdict proposal stays single-call for now . it can opt into
    # ask_llm_voted later by extracting a label per (claim_id ,
    # proposed_belief) pair - the swap happens HERE and only here
    result = json.loads(ask_llm(prompt, VERDICT_SCHEMA))
    return result


# ===========================================================================
# proposed verdicts evaluation (code-only)
#
# the gauntlet is decomposed one-function-per-check so each gate is
# auditable at a glance and unit-testable in isolation . the driver below
# (_run_gauntlet) is just the gates in order .
#
# GAUNTLET vs KERNEL : the kernel holds LAWS of the ladder (what the labels
# mean , forever) ; the gauntlet holds this evaluator's POLICY (tunable
# quality standards , crude v1 heuristics) . relaxing a gauntlet check
# lowers standards ; relaxing a kernel rule changes what "verified" means .
# ===========================================================================

@dataclass
class CheckResult:
    verdict_alive: bool
    proposed_belief: str      # possibly capped by the check
    note: str = ""            # drop reason or cap annotation


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


def _check_label_sanity(proposed, evidence_type):
    # code validates the llm's enums , always
    if proposed not in VALID_BELIEF_LABELS:
        return CheckResult(False, proposed, f"bad label {proposed!r}")
    if evidence_type not in VALID_EVIDENCE_TYPES:
        return CheckResult(False, proposed, f"bad evidence_type {evidence_type!r}")
    return CheckResult(True, proposed)


def _check_evidence_linked(proposed, claim, artifact_id):
    # the cited evidence must actually be linked to the claim
    if artifact_id not in claim.evidence_ids:
        return CheckResult(False, proposed,
                           f"artifact {artifact_id} is not linked to it")
    return CheckResult(True, proposed)


def _check_tempo(proposed, current_belief, claim_id, promoted_this_iteration):
    # tempo : one rung at a time , once per iteration .
    # contested and downward moves always allowed (doubt is free)
    if proposed in BELIEF_ORDER and current_belief in BELIEF_ORDER:
        step = BELIEF_ORDER[proposed] - BELIEF_ORDER[current_belief]
        if step > 1:
            return CheckResult(False, proposed,
                               f"{current_belief} -> {proposed} skips a rung")
        if step == 1 and claim_id in promoted_this_iteration:
            return CheckResult(False, proposed,
                               "already promoted this iteration (tempo rule)")
    return CheckResult(True, proposed)


def _cap_rung_skip(proposed, current_belief):
    # FIX: run evidence (32 of 32 verdicts in a single run) showed the
    # proposer almost always answers "verified" for a claim whose artifact
    # states it - a self-asserted claim trivially fits "the artifact
    # explicitly states the claim" . DROPPING those verdicts starved the
    # ladder : nothing ever reached supported , so no dual-pass , no
    # contradiction had anything to work on . the gauntlet's own principle
    # is that caps beat drops ("the information survives , the overreach
    # does not") , so a rung-skip is now capped to exactly ONE rung above
    # the current belief . the dual-pass still gates the capped promotion
    if proposed in BELIEF_ORDER and current_belief in BELIEF_ORDER:
        step = BELIEF_ORDER[proposed] - BELIEF_ORDER[current_belief]
        if step > 1:
            next_rung = proposed
            for label in BELIEF_ORDER:
                if BELIEF_ORDER[label] == BELIEF_ORDER[current_belief] + 1:
                    next_rung = label
                    break
            return CheckResult(True, next_rung,
                               " [capped: one rung at a time]")
    return CheckResult(True, proposed)


def _check_testimonial_cap(proposed, evidence_type):
    # testimonial evidence never verifies . pointers alone
    # cannot make a claim reliable
    if proposed == "verified" and evidence_type == "testimonial":
        return CheckResult(True, "supported",
                           " [capped: testimonial evidence cannot verify]")
    return CheckResult(True, proposed)


def _check_negative_scope(proposed, is_negative, artifact_content):
    # negative evidence must state its search scope or it
    # supports nothing (absence of evidence != evidence of absence)
    if is_negative and proposed in ("supported", "verified"):
        if not _artifact_states_scope(artifact_content):
            return CheckResult(False, proposed,
                               "negative evidence with no stated search scope")
    return CheckResult(True, proposed)


def _check_quantitative_cap(proposed, evidence_type, claim_statement):
    # quantitative trap : numbers without methodology cap at supported
    # (sample size / base rate checking is judgment ; the cap is code) .
    # applies to EMPIRICAL/TESTIMONIAL only - a deductive claim with numbers
    # ("2^10 = 1024") is checked by re-derivation , not by source-counting ;
    # without this guard , math-heavy domains could never reach verified
    if proposed == "verified" and evidence_type != "deductive" \
            and _looks_quantitative(claim_statement):
        return CheckResult(True, "supported",
                           " [capped: quantitative claim, single-source]")
    return CheckResult(True, proposed)


def _check_framing_independence(proposed, claim, artifacts_by_id):
    # framing independence : verified needs evidence from at
    # least 2 DIFFERENT tasks (same method = same blind spots)
    if proposed == "verified":
        source_task_ids = set()
        for linked_artifact_id in claim.evidence_ids:
            linked_artifact = artifacts_by_id.get(linked_artifact_id)
            if linked_artifact is not None:
                source_task_ids.add(linked_artifact.task_id)
        if len(source_task_ids) < 2:
            return CheckResult(True, "supported",
                               " [capped: verified needs evidence from 2+ tasks]")
    return CheckResult(True, proposed)


# ---------------------------------------------------------------------------
# integrity gates added after the live-run audit . all four apply to
# PROMOTIONS only - doubt stays free . all four are drop gates : the
# fail-safe direction is under-confidence
# ---------------------------------------------------------------------------

_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")


def _extract_numbers(text):
    # every numeric token in a text , normalized : commas stripped , and a
    # percentage also recorded as its fraction ("95%" pins both 95 and
    # 0.95 , so downstream restatements in either form still match)
    found = set()
    for index, match in enumerate(_NUMBER_PATTERN.finditer(text)):
        token = match.group(0).replace(",", "")
        try:
            value = float(token)
        except ValueError:
            continue
        found.add(value)
        end = match.end()
        if end < len(text) and text[end] == "%":
            found.add(value / 100.0)
    return found


def _pinned_numbers_from_spec(spec):
    # ground-truth pinning : every number the USER supplied , collected
    # from every string in the ratified spec (prompt , scope , anchors ,
    # criteria - the walk is recursive so the spec's shape never matters) .
    # pure code , runs once per evaluation round
    pinned = set()

    def walk(node):
        if isinstance(node, str):
            pinned.update(_extract_numbers(node))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(spec)
    return pinned


def _check_external_confirmation(proposed, current_belief, statement):
    # claims asserting sign-offs / approvals / verifications by people are
    # unpromotable from run-internal text : no artifact this system
    # produces is evidence that a human actually approved something .
    # the honest label is unverified-pending , not a styled endorsement
    is_promotion = (proposed in BELIEF_ORDER and current_belief in BELIEF_ORDER
                    and BELIEF_ORDER[proposed] > BELIEF_ORDER[current_belief])
    if is_promotion:
        lowered = statement.lower()
        for marker in EXTERNAL_CONFIRMATION_MARKERS:
            if marker in lowered:
                return CheckResult(False, proposed,
                                   f"asserts an external confirmation ({marker!r}) - "
                                   "unpromotable without out-of-run evidence ; "
                                   "correct status is pending , not believed")
        # v1 has no observation tools , so a claim reporting a benchmark
        # or measurement is always invented . unpromotable until tools
        # exist (a tool-call citation becomes the exemption then)
        for marker in UNOBSERVED_MEASUREMENT_MARKERS:
            if marker in lowered:
                return CheckResult(False, proposed,
                                   f"reports a measurement ({marker!r}) this run "
                                   "cannot have performed - no observation tools "
                                   "exist ; the number is invented")
    return CheckResult(True, proposed)


def _check_terminal_evidence(proposed, current_belief, evidence_ids,
                             artifacts_by_id, tasks_by_id):
    # no circular evidence : an artifact produced by a synthesis/deliverable
    # task ('produce' kind) is TERMINAL - it is a conclusion , and a
    # conclusion cited as evidence for further claims is the run agreeing
    # with itself
    is_promotion = (proposed in BELIEF_ORDER and current_belief in BELIEF_ORDER
                    and BELIEF_ORDER[proposed] > BELIEF_ORDER[current_belief])
    if is_promotion:
        for artifact_id in evidence_ids:
            artifact = artifacts_by_id.get(artifact_id)
            if artifact is None:
                continue
            producing_task = tasks_by_id.get(artifact.task_id)
            if producing_task is not None and producing_task.kind in TERMINAL_TASK_KINDS:
                return CheckResult(False, proposed,
                                   f"artifact {artifact_id} is a synthesis output "
                                   f"({producing_task.kind}-kind task) - terminal , "
                                   "never evidence")
    return CheckResult(True, proposed)


def _check_source_independence(proposed, evidence_type, is_negative,
                               claim, artifacts_by_id):
    # one source echoed n times is one source . leaving unverified needs
    # evidence from 2+ different tasks , with one exemption : positive
    # deductive evidence (a derivation is re-traced , not corroborated) .
    # negative claims never get the exemption - a single search that found
    # nothing closes no question
    if proposed == "supported":
        needs_two = is_negative or evidence_type != "deductive"
        if needs_two:
            source_task_ids = set()
            for artifact_id in claim.evidence_ids:
                artifact = artifacts_by_id.get(artifact_id)
                if artifact is not None:
                    source_task_ids.add(artifact.task_id)
            if len(source_task_ids) < MIN_SOURCE_TASKS_FOR_SUPPORTED:
                return CheckResult(False, proposed,
                                   f"supported needs evidence from "
                                   f"{MIN_SOURCE_TASKS_FOR_SUPPORTED}+ independent tasks "
                                   f"(has {len(source_task_ids)})")
    return CheckResult(True, proposed)


def _check_pinned_numbers(proposed, current_belief, statement,
                          artifact_content, pinned_numbers):
    # auditable arithmetic : a promotable quantitative claim must either
    # restate the user's own numbers (the pinned set) or cite an artifact
    # that shows its working (formula markers) . a number that matches
    # neither appeared from nowhere - the exact substitution genre the
    # audit caught (recall=0.92 invented over the user's 0.95)
    is_promotion = (proposed in BELIEF_ORDER and current_belief in BELIEF_ORDER
                    and BELIEF_ORDER[proposed] > BELIEF_ORDER[current_belief])
    if is_promotion and _looks_quantitative(statement):
        claim_numbers = _extract_numbers(statement)
        unpinned = claim_numbers - pinned_numbers
        if unpinned:
            lowered = artifact_content.lower()
            shows_working = False
            for marker in FORMULA_MARKERS:
                if marker in lowered:
                    shows_working = True
                    break
            if not shows_working:
                return CheckResult(False, proposed,
                                   f"numbers {sorted(unpinned)} match no user-pinned "
                                   "value and the artifact shows no formula - "
                                   "untraceable arithmetic")
    return CheckResult(True, proposed)


def _run_gauntlet(raw_verdicts, context, workspace, promoted_this_iteration,
                  pinned_numbers):
    # the driver : gates in order , cheap existence checks first ,
    # judgment-adjacent last . a dropped verdict is logged , never fatal .
    # caps DEMOTE instead of drop - the information survives , the
    # overreach does not
    snapshot = workspace.snapshot()

    claims_by_id = {}
    for claim in snapshot["claims"]:
        claims_by_id[claim.id] = claim

    artifacts_by_id = {}
    for artifact in snapshot["artifacts"]:
        artifacts_by_id[artifact.id] = artifact

    tasks_by_id = {}
    for task in snapshot["tasks"]:
        tasks_by_id[task.id] = task

    survivors = []

    for raw in raw_verdicts:

        claim = claims_by_id.get(raw["claim_id"])

        # check 1 - the claim must exist
        if claim is None:
            print(f"[evaluator] dropped verdict: claim {raw['claim_id']} does not exist")
            continue

        proposed = raw["proposed_belief"].strip().lower()
        evidence_type = raw["evidence_type"].strip().lower()

        # the verdict's evidence is the artifact under evaluation - code
        # decides what counts as evidence , the model only judges
        evidence_ids = [context.artifact_id]

        # check 4 - no-op transitions carry no information (kept inline :
        # it is a skip , not a drop - nothing to log)
        if proposed == claim.belief:
            continue

        # FIX: the rung-skip cap runs BEFORE the drop gates , so the tempo
        # check below judges the capped (one-rung) proposal - its
        # once-per-iteration drop still applies , and its own skip-a-rung
        # drop remains as belt-and-braces behind this cap
        rung_cap = _cap_rung_skip(proposed, claim.belief)
        if rung_cap.proposed_belief != proposed:
            raw["rationale"] += rung_cap.note
            proposed = rung_cap.proposed_belief

        # the drop gates - any failure kills this one verdict
        drop_gates = [
            _check_label_sanity(proposed, evidence_type),                              # check 2
            _check_evidence_linked(proposed, claim, context.artifact_id),              # check 3
            _check_tempo(proposed, claim.belief, claim.id, promoted_this_iteration),   # check 5
            _check_negative_scope(proposed, raw["is_negative"], context.artifact_content),  # check 7
            # integrity gates (live-run audit) - promotions only , all fail-safe
            _check_external_confirmation(proposed, claim.belief, claim.statement),     # check 10
            _check_terminal_evidence(proposed, claim.belief, evidence_ids,
                                     artifacts_by_id, tasks_by_id),                    # check 11
            _check_source_independence(proposed, evidence_type, raw["is_negative"],
                                       claim, artifacts_by_id),                        # check 12
            _check_pinned_numbers(proposed, claim.belief, claim.statement,
                                  context.artifact_content, pinned_numbers),           # check 13
        ]
        dropped = False
        for result in drop_gates:
            if not result.verdict_alive:
                print(f"[evaluator] dropped verdict on claim {claim.id}: {result.note}")
                dropped = True
                break
        if dropped:
            continue

        # the cap gates - never drop , demote and annotate . order matters :
        # each cap sees the (possibly already-capped) proposed belief
        cap_gates = [
            lambda p: _check_testimonial_cap(p, evidence_type),                        # check 6
            lambda p: _check_quantitative_cap(p, evidence_type, claim.statement),      # check 8
            lambda p: _check_framing_independence(p, claim, artifacts_by_id),          # check 9
        ]
        for gate in cap_gates:
            result = gate(proposed)
            if result.proposed_belief != proposed:
                raw["rationale"] += result.note
                proposed = result.proposed_belief

        # FIX: a verified verdict that cites only the artifact under
        # evaluation can never satisfy the kernel's 2-evidence law , even
        # though the claim itself holds evidence from 2+ tasks (the framing
        # independence cap guarantees that) . for verified , code cites the
        # claim's full linked evidence - still artifact ids only , still
        # assigned by code , never asked from the llm
        if proposed == "verified":
            evidence_ids = list(claim.evidence_ids)

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
# first pass's rationale , not the proposed label .
#
# logic-inversion refinement : instead of a yes/no ("does the evidence
# support the claim?" - where yes is the agreeable answer and the gate
# leaks) , we ask for the GAP LIST : every part of the claim the evidence
# does not state or entail . an empty list IS confirmation . the model's
# eagerness to produce content now does the strict work , and a padded gap
# item names something specific - checkable , unlike a padded "no"
# ===========================================================================

DUAL_PASS_SCHEMA = {
    "type": "object",
    "properties": {
        "gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every part of the claim that the evidence does "
                           "not state or directly entail. Empty if none.",
        },
        "reason": {"type": "string"},
    },
    "required": ["gaps", "reason"],
}


def _extract_dual_pass_label(response_text):
    # the label a vote is counted on : an empty gap list IS a pass .
    # labels only - the voting machinery never reads the gap content
    result = json.loads(response_text)
    if result["gaps"]:
        return "fail"
    return "pass"


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
        "Read the evidence, then the claim. List every part of the claim "
        "that the evidence does not state or directly entail. Judge only "
        "what is written - not plausibility, not your own knowledge.\n"
        "If the evidence fully states or entails the claim, return an "
        "empty list.\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        f"CLAIM: {claim_statement}\n"
    )
    # self-consistency voting at the promotion gate : n blind samples ,
    # majority on the pass/fail label . a non-unanimous vote is itself
    # calibration data - logged as a label tier , never a ratio
    majority_label, split, winning_response = ask_llm_voted(
        prompt, DUAL_PASS_SCHEMA, _extract_dual_pass_label, VOTING_N,
    )
    tier = vote_split_tier(split)
    if tier != "unanimous":
        log_vote_split("evaluator_dual_pass", tier, majority_label)

    result = json.loads(winning_response)

    gaps = result["gaps"]
    if gaps:
        # the gap list itself is the reason - better calibration data than
        # a free-text verdict , each item names disputable missing content
        return False, "gaps: " + "; ".join(gaps)
    return True, result["reason"]


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
        "- A pair qualifies only if asserting both leads to a logical or "
        "factual impossibility. Different emphasis, different scope, or "
        "paraphrase is NOT a contradiction.\n"
        "- An empty list is a valid and common answer.\n\n"
        f"CLAIMS:\n{belief_table}\n"
    )
    # SEAM: contradiction detection stays single-call for now . it can opt
    # into ask_llm_voted later by voting per-pair on "contradiction /
    # not-contradiction" labels - the swap happens HERE and only here
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

def evaluate_iteration(spec, workspace, new_artifact_ids=None):
    contexts = _build_evaluation_contexts(workspace, new_artifact_ids)

    if not contexts:
        print("[evaluator] nothing to evaluate - no completed-task artifacts")
        return {"applied": [], "contested_pairs": [], "done_when_failures": []}

    applied = []
    done_when_failures = []
    promoted_this_iteration = set()
    total_raw_verdicts = 0

    # ground-truth pinning : the user's own numbers , extracted from the
    # ratified spec once per round . code reads the spec ; no prompt does
    pinned_numbers = _pinned_numbers_from_spec(spec)

    for context in contexts:

        proposal = _propose_verdicts(context)
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
            proposal["verdicts"], context, workspace, promoted_this_iteration,
            pinned_numbers,
        )

        for verdict in survivors:

            # dual-pass gate on promotions only - demotions and contested
            # always pass through (doubt is free)
            linked_entry = context.linked_claims.get(verdict.claim_id, {})
            old_belief = linked_entry.get("belief", "")

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
                # FIX: log_promotion_applied was never called - the
                # denominator of every calibration rate was missing .
                # logged RIGHT AFTER the single-door write succeeds
                log_promotion_applied(verdict)
            applied.append(verdict_record)

    # contradiction pass runs AFTER verdicts so it sees the updated table
    contested_pairs = handle_contradictions(workspace)

    return {
        "applied": applied,
        "contested_pairs": contested_pairs,
        "done_when_failures": done_when_failures,
    }