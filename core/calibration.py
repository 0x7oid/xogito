'''
This is a provenance log for the evaluator module , since the evaluator can make mistakes
we need to keep track of the decisions made by the evaluator and the reasoning behind them.
This will help us to identify and correct any errors in the evaluation process, and to improve the overall quality of the evaluation.
'''
'''
calibration.py - the evaluator's report card , kept over time (lives in core/)

the evaluator will be wrong sometimes . calibration is the append-only log
that records HOW its judgments turned out :
- a dual-pass blocked a promotion -> the two passes disagreed , someone was wrong
- the adjudicator overturned a promoted claim -> the evaluator over-promoted
- the adjudicator dismissed a flagged contradiction -> the detector saw a ghost
- a verified claim still standing at the end -> the evaluator was right

three consumers :
1. the final report : "14 supported claims ; historically , promotions of this
   type are overturned X% of the time" - calibrated confidence , demonstrated
   not claimed . this is the product's core promise made measurable .
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


# one file , json-lines format : one entry per line , append and forget
CALIBRATION_LOG_PATH = "calibration_log.jsonl"


# the outcome vocabulary . labels not numbers , same as everywhere else
OUTCOME_KINDS = (
    "dual_pass_disagreement",     # second pass refused the first pass's promotion
    "adjudicator_overturned",     # a promoted claim lost its fight
    "adjudicator_confirmed",      # a promoted claim survived its fight
    "contradiction_dismissed",    # flagged pair judged not a real contradiction
    "contradiction_confirmed",    # flagged pair was a real fight
)


def _append_entry(entry):
    # append-only , one json object per line . never rewrites the file
    entry["timestamp"] = time.time()
    with open(CALIBRATION_LOG_PATH, "a") as log_file:
        log_file.write(json.dumps(entry) + "\n")


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
    # final report consumes . pure read , no llm
    if not os.path.exists(CALIBRATION_LOG_PATH):
        return {"total_entries": 0, "by_kind": {}, "by_evidence_type": {}}

    by_kind = {}
    by_evidence_type = {}
    total = 0

    with open(CALIBRATION_LOG_PATH, "r") as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
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

    return {
        "total_entries": total,
        "by_kind": by_kind,
        "by_evidence_type": by_evidence_type,
    }