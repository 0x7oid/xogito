'''
compressor.py - turns the full run trace into the SHORT , human-first
summary at the top of the report . this is a first-class pipeline stage ,
not a formatting afterthought : the summary is the only part most readers
will actually read , so it gets the same posture as everything else -
structured state passes through VERBATIM , only narrative is compressed .

what it produces :
- the answer : a real , named result if the established claims support
  one . if they do not , it says so plainly instead of dressing a thin
  table up as a conclusion .
- the summary : what was checked , which arguments survived scrutiny ,
  which were floated and dropped - in plain language .
- the statistics : real counts pulled from the run (sources , claims per
  label , iterations , fights) . counts , never scores .

honesty rule : compression that loses the specific facts is not
compression , it is data loss . the synthesis llm call sees ONLY the
established claims and the goal - it cannot leak the audit trail into
the summary , and everything it says sits directly above the verbatim
claims it was built from , so a reader can check it in one glance .
when no synthesis is possible (llm unreachable , or nothing established)
the compressor falls back to a code-built extractive summary and SAYS
that it did - never a silent guess .
'''

import json

from llm.client import ask_llm
from parametres import BELIEF_ORDER


SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The specific, named result the goal asks for, "
                           "stated in one or two sentences. If the claims "
                           "below do not establish one, say exactly that "
                           "in plain words.",
        },
        "summary": {
            "type": "string",
            "description": "One short paragraph, plain language: what was "
                           "checked and which findings held up.",
        },
    },
    "required": ["answer", "summary"],
}


# ===========================================================================
# statistics (pure code) . counts , never scores
# ===========================================================================

def collect_statistics(workspace, run_log):
    snapshot = workspace.snapshot()

    tasks_completed = 0
    tasks_failed = 0
    tasks_rejected = 0
    for task in snapshot["tasks"]:
        if task.status == "completed":
            tasks_completed += 1
        elif task.status == "failed":
            tasks_failed += 1
        elif task.status == "rejected":
            tasks_rejected += 1

    claims_by_belief = {"verified": 0, "supported": 0,
                        "unverified": 0, "contested": 0}
    for claim in snapshot["claims"]:
        if claim.belief in claims_by_belief:
            claims_by_belief[claim.belief] += 1

    contradictions_detected = 0
    adjudications_resolved = 0
    for record in run_log:
        contradictions_detected += len(record["evaluator"]["contested_pairs"])
        adjudications_resolved += len(record["adjudicator"]["resolved"])

    return {
        "iterations_run": len(run_log),
        "sources_produced": len(snapshot["artifacts"]),
        "tasks_completed": tasks_completed,
        "tasks_failed": tasks_failed,
        "tasks_rejected": tasks_rejected,
        "claims_by_belief": claims_by_belief,
        "contradictions_detected": contradictions_detected,
        "adjudications_resolved": adjudications_resolved,
    }


# ===========================================================================
# survivors vs dropped (pure code) . "which arguments survived scrutiny"
# is a state question , not a judgment - the belief table already holds
# the answer , this only reads it out
# ===========================================================================

def split_survivors(workspace, run_log):
    snapshot = workspace.snapshot()

    # claims that were promoted at some point during the run
    ever_promoted_ids = set()
    for record in run_log:
        for verdict in record["evaluator"]["applied"]:
            if verdict["proposed_belief"] in ("supported", "verified"):
                ever_promoted_ids.add(verdict["claim_id"])

    surviving = []
    dropped = []
    for claim in snapshot["claims"]:
        entry = {"claim_id": claim.id, "statement": claim.statement,
                 "belief": claim.belief}
        if claim.belief in ("supported", "verified"):
            surviving.append(entry)
        elif claim.id in ever_promoted_ids:
            # floated , held a rung for a while , and lost it - the reader
            # deserves to know it was considered and did not survive
            entry["what_happened"] = "was supported at some point but did not survive scrutiny"
            dropped.append(entry)
        elif claim.belief == "contested":
            entry["what_happened"] = "is still under dispute"
            dropped.append(entry)

    # strongest first : verified beats supported , then lower ids (earlier
    # claims) first - a stated ordering , not a score
    def strength_key(entry):
        return (-BELIEF_ORDER.get(entry["belief"], 0), entry["claim_id"])
    surviving.sort(key=strength_key)

    return surviving, dropped


# ===========================================================================
# the synthesis call . sees ONLY the goal and the established claims -
# never the trace , never the belief machinery . falls back to an
# extractive , code-built summary when synthesis is unavailable
# ===========================================================================

def _synthesize(goal, surviving, statistics):
    claim_lines = []
    for entry in surviving:
        claim_lines.append(f"- {entry['statement']}")

    prompt = (
        "Write the top of a report for a non-technical reader.\n"
        "Use ONLY the established findings listed below - no outside "
        "knowledge, no speculation. If the findings do not add up to a "
        "specific answer to the goal, your 'answer' must say exactly "
        "that, plainly.\n\n"
        f"GOAL: {goal}\n\n"
        f"ESTABLISHED FINDINGS:\n" + "\n".join(claim_lines) + "\n\n"
        "For the summary: say what was checked and which findings held "
        "up, in plain everyday words. Do not mention internal component "
        "names or labels.\n"
    )
    result = json.loads(ask_llm(prompt, SYNTHESIS_SCHEMA))
    return result["answer"].strip(), result["summary"].strip()


def _fallback_summary(statistics, surviving, dropped):
    # extractive , code-built , honest about being the fallback
    beliefs = statistics["claims_by_belief"]
    sentences = []
    sentences.append(
        f"The run used {statistics['iterations_run']} iteration(s), "
        f"completed {statistics['tasks_completed']} task(s) and produced "
        f"{statistics['sources_produced']} source document(s)."
    )
    sentences.append(
        f"{beliefs['verified'] + beliefs['supported']} claim(s) ended up "
        f"backed by evidence, {beliefs['contested']} remained under "
        f"dispute, and {beliefs['unverified']} stayed unsupported."
    )
    if statistics["contradictions_detected"] > 0:
        sentences.append(
            f"{statistics['contradictions_detected']} conflict(s) between "
            f"claims were found; {statistics['adjudications_resolved']} "
            "were resolved by comparing the evidence on both sides."
        )
    if dropped:
        sentences.append(
            f"{len(dropped)} claim(s) were considered and did not survive scrutiny."
        )
    return " ".join(sentences)


def compress_run(spec, workspace, run_log, halted_by_fuse=""):
    statistics = collect_statistics(workspace, run_log)
    surviving, dropped = split_survivors(workspace, run_log)

    finished = False
    if not halted_by_fuse and run_log:
        last_checkpoint = run_log[-1]["checkpoint"]
        if last_checkpoint is not None and last_checkpoint["decision"] == "stop_success":
            finished = True

    answer = ""
    summary = ""
    note = ""

    if surviving:
        goal = spec["problem_specification"]["goal"]
        try:
            answer, summary = _synthesize(goal, surviving, statistics)
        except Exception as error:
            # never a silent guess : fall back to extractive and SAY so
            note = (f"synthesis was unavailable ({type(error).__name__}) - "
                    "showing the strongest established claim verbatim instead")
            answer = surviving[0]["statement"]
            summary = _fallback_summary(statistics, surviving, dropped)
    else:
        note = "no claim ended up backed by evidence - there is no answer to show"
        summary = _fallback_summary(statistics, surviving, dropped)

    return {
        "finished": finished,
        "answer": answer,
        "answer_note": note,
        "summary": summary,
        "statistics": statistics,
        "surviving_claims": surviving,
        "dropped_claims": dropped,
    }
