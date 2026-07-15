'''
This is a calibration log for the evaluator module , since the evaluator can make mistakes
we need to keep track of the decisions made by the evaluator and the reasoning behind them.
This will help us to identify and correct any errors in the evaluation process, and to improve the overall quality of the evaluation.
(NOT provenance - provenance records what happened to STATE , calibration records
what happened to JUDGMENT . different objects , different lifetimes)
'''
'''
calibration.py - the evaluator's report card , kept over time (lives in core/)

the evaluator will be wrong sometimes . calibration is the append-only log
that records HOW its judgments turned out :
- a promotion was applied -> the denominator . every rate needs one
- a dual-pass blocked a promotion -> the two passes disagreed , someone was wrong
- the adjudicator overturned a promoted claim -> the evaluator over-promoted
- the adjudicator dismissed a flagged contradiction -> the detector saw a ghost

three consumers :
1. the final report : "14 supported claims ; historically , promotions of this
   type are overturned X% of the time" - calibrated confidence , demonstrated
   not claimed . this is the product's core promise made measurable .
   X% = adjudicator_overturned / promotion_applied , per evidence_type .
2. the human tuning the system : if the log shows testimonial promotions keep
   getting overturned , that is a concrete prompt fix with data behind it
3. future adaptive behavior (v3+) : auto-require dual-pass on evidence types
   with bad track records . the log is what lets the system learn about its
   own judgment without retraining anything

design :
- append-only . entries are never edited or deleted (same rule as provenance)
- entries are plain dicts written as json lines to disk - readable with any
  text editor , greppable , survives crashes , no database needed
- the log is NOT part of the workspace : it outlives runs . the workspace is
  one problem's state ; calibration is the evaluator's history across problems
'''

import json
import time
import os

# the log path lives in parametres.py now (still anchored to the project
# root , never the current working directory)
from parametres import CALIBRATION_LOG_PATH


# the outcome vocabulary . labels not numbers , same as everywhere else
OUTCOME_KINDS = (
    "promotion_applied",          # a promotion passed every gate and was written .
                                  # the DENOMINATOR - without it , overturn rates
                                  # have a numerator and nothing to divide by
    "dual_pass_disagreement",     # second pass refused the first pass's promotion
    "adjudicator_overturned",     # a promoted claim lost its fight
    "adjudicator_confirmed",      # a promoted claim survived its fight
    "contradiction_dismissed",    # flagged pair judged not a real contradiction
    "contradiction_confirmed",    # flagged pair was a real fight
    "vote_split",                 # a self-consistency vote was not unanimous -
                                  # the judgment site was genuinely uncertain .
                                  # tier is a label ("majority" / "split") ,
                                  # never a ratio - no arithmetic downstream
)


def _append_entry(entry):
    # append-only , one json object per line . never rewrites the file .
    # the vocabulary is ENFORCED here , not just documented - adding a new
    # logger without extending OUTCOME_KINDS is a crash , not silent drift
    if entry["kind"] not in OUTCOME_KINDS:
        raise ValueError(f"unknown calibration kind: {entry['kind']!r}")

    # copy before stamping - never mutate the caller's dict
    record = dict(entry)
    record["timestamp"] = time.time()
    with open(CALIBRATION_LOG_PATH, "a") as log_file:
        log_file.write(json.dumps(record) + "\n")


def log_promotion_applied(verdict):
    # called by the evaluator RIGHT AFTER update_belief_of_claim succeeds on
    # a promotion . this is the healthy , common case - and the denominator
    # every calibration rate is computed against
    _append_entry({
        "kind": "promotion_applied",
        "claim_id": verdict.claim_id,
        "new_belief": verdict.proposed_belief,
        "evidence_type": verdict.evidence_type,
        "is_negative": verdict.is_negative,
    })


def log_dual_pass_disagreement(verdict, reason):
    # called by the evaluator when the blind second pass refuses a promotion
    _append_entry({
        "kind": "dual_pass_disagreement",
        "claim_id": verdict.claim_id,
        "proposed_belief": verdict.proposed_belief,
        "evidence_type": verdict.evidence_type,
        "is_negative": verdict.is_negative,
        "reason": reason,
    })


def log_adjudication_outcome(claim_id, evidence_type, overturned, reason):
    # called by the adjudicator (tomorrow's file) after resolving a fight :
    # overturned=True -> the evaluator's earlier promotion did not survive
    if overturned:
        kind = "adjudicator_overturned"
    else:
        kind = "adjudicator_confirmed"
    _append_entry({
        "kind": kind,
        "claim_id": claim_id,
        "evidence_type": evidence_type,
        "reason": reason,
    })


def log_vote_split(site, tier, majority_label):
    # called by a voted judgment site (dual-pass , entailment gate) when
    # the vote was NOT unanimous . site names WHERE the disagreement
    # happened ; tier is the label form of the split ("majority" or
    # "split") ; majority_label is what the vote settled on anyway
    _append_entry({
        "kind": "vote_split",
        "site": site,
        "tier": tier,
        "majority_label": majority_label,
    })


def log_contradiction_outcome(claim_id_a, claim_id_b, was_real, reason):
    # called by the adjudicator after examining a flagged pair :
    # was_real=False -> the detector saw a ghost (paraphrase , scope difference)
    if was_real:
        kind = "contradiction_confirmed"
    else:
        kind = "contradiction_dismissed"
    _append_entry({
        "kind": kind,
        "claim_id_a": claim_id_a,
        "claim_id_b": claim_id_b,
        "reason": reason,
    })


def read_calibration_summary():
    # counts per kind , per evidence_type where present . this is what the
    # final report consumes . pure read , no llm .
    # a half-written last line (process died mid-append) is skipped with a
    # warning , never a crash - the read side is as crash-tolerant as the
    # write side claims to be
    if not os.path.exists(CALIBRATION_LOG_PATH):
        return {"total_entries": 0, "by_kind": {}, "by_evidence_type": {},
                "vote_splits_by_tier": {}}

    by_kind = {}
    by_evidence_type = {}
    vote_splits_by_tier = {}
    total = 0

    with open(CALIBRATION_LOG_PATH, "r") as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print("[calibration] skipped a malformed log line (interrupted write?)")
                continue
            total += 1

            kind = entry["kind"]
            if kind not in by_kind:
                by_kind[kind] = 0
            by_kind[kind] += 1

            evidence_type = entry.get("evidence_type")
            if evidence_type is not None:
                if evidence_type not in by_evidence_type:
                    by_evidence_type[evidence_type] = {}
                if kind not in by_evidence_type[evidence_type]:
                    by_evidence_type[evidence_type][kind] = 0
                by_evidence_type[evidence_type][kind] += 1

            # vote splits also count per tier - label tiers only , the
            # report and checkpoint never do arithmetic on these
            if kind == "vote_split":
                tier = entry.get("tier", "unknown")
                if tier not in vote_splits_by_tier:
                    vote_splits_by_tier[tier] = 0
                vote_splits_by_tier[tier] += 1

    return {
        "total_entries": total,
        "by_kind": by_kind,
        "by_evidence_type": by_evidence_type,
        "vote_splits_by_tier": vote_splits_by_tier,
    }