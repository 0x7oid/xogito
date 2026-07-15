'''
report.py - the final output , in two stages : first a JSON report object
(the data , machine-readable , the audit trail) , then a single
self-contained HTML file rendered from it with jinja2 (the human view ,
Arial , readable offline , no external assets) .

the report never shows a numeric confidence score anywhere - labels and
counts only . quantitative claims are phrased as what they are ("two
independent sources assert X") , never as statistical conclusions the
system did not perform .

THE RUN LOG STRUCTURE (built by the orchestrator , one dict per iteration):
{
    "iteration": int,                      # 1-based
    "tasks_created_by_planner": int,       # accepted AND rejected
    "accepted_task_ids": [int, ...],
    "frontier_empty_reason": str,          # "" unless the frontier was empty
    "dispatched_task_ids": [int, ...],
    "integrated_artifact_ids": [int, ...],
    "evaluator": {                         # evaluate_iteration's return
        "applied": [verdict_record, ...],
        "contested_pairs": [pair, ...],
        "done_when_failures": [{"task_id", "reason"}, ...],
    },
    "adjudicator": {                       # adjudicate_contested_pairs' return
        "resolved": [ruling_record, ...],
        "parked": [pair, ...],
        "propagation_flags": [flag, ...],
    },
    "checkpoint": {"decision", "reason", "warnings"},
}
'''

import json
import os
import time

from jinja2 import Environment, FileSystemLoader

from core.calibration import read_calibration_summary
# the scope markers the evaluator's gauntlet checks - reused here so the
# report can SHOW the stated search scope of negative-evidence claims
from agents.evaluator import SCOPE_MARKERS
from parametres import PROJECT_ROOT


# reports are timestamped , never overwritten - no deletion applies to
# the audit trail too
REPORTS_DIRECTORY = os.path.join(PROJECT_ROOT, "reports")

TEMPLATE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FILENAME = "report_template.html"


# ===========================================================================
# section builders (pure code , no llm) . each returns plain dicts/lists
# so the whole report object is json-serializable as-is
# ===========================================================================

def _build_problem_section(spec):
    ps = spec["problem_specification"]
    return {
        "original_prompt": spec.get("problem", "(not recorded)"),
        "goal": ps["goal"],
        "constraints": ps["constraints"],
        "success_criteria": ps["success_criteria"],
        "scope": ps["scope"],
        # labeled explicitly as user-declared : the model never wrote these
        "user_declared_anchors": ps["contextual_anchors"],
        "assumptions": ps["assumptions"],
        "formalization": _describe_formalization(spec),
    }


def _describe_formalization(spec):
    # honesty about framing : which structure was chosen , what survived
    # the translation and what was dropped or altered
    chosen_id = spec["structure_id"]
    for candidate in spec["candidates"]:
        if candidate["structure_id"] == chosen_id:
            return {
                "selected_structure": chosen_id,
                "formal_statement": candidate["formal_statement"],
                "preserved_from_original": candidate["preserved_from_original"],
                "dropped_or_altered": candidate["dropped_or_altered"],
            }
    return {
        "selected_structure": chosen_id,
        "formal_statement": "(candidate record missing)",
        "preserved_from_original": "",
        "dropped_or_altered": "",
    }


def _build_journey_section(run_log, workspace):
    # per-iteration timeline . task ids are sequential and only the
    # planner creates tasks (v1) , so each iteration's created ids are the
    # contiguous range after the previous iteration's total
    snapshot = workspace.snapshot()
    tasks_by_id = {}
    for task in snapshot["tasks"]:
        tasks_by_id[task.id] = task

    journey = []
    next_created_id = 0

    for record in run_log:

        planned = []
        rejected = []
        for created_id in range(next_created_id,
                                next_created_id + record["tasks_created_by_planner"]):
            task = tasks_by_id.get(created_id)
            if task is None:
                continue
            entry = {
                "task_id": task.id,
                "kind": task.kind,
                "description": task.description,
                "done_when": task.done_when,
            }
            if task.rejection_reason:
                entry["rejection_reason"] = task.rejection_reason
                rejected.append(entry)
            else:
                planned.append(entry)
        next_created_id += record["tasks_created_by_planner"]

        executed = []
        for task_id in record["dispatched_task_ids"]:
            task = tasks_by_id.get(task_id)
            if task is None:
                continue
            executed.append({
                "task_id": task.id,
                "description": task.description,
                "status": task.status,
            })

        transitions = []
        for verdict in record["evaluator"]["applied"]:
            transitions.append({
                "claim_id": verdict["claim_id"],
                "new_belief": verdict["proposed_belief"],
                "evidence_type": verdict["evidence_type"],
                "rationale": verdict["rationale"],
            })

        adjudications = []
        for ruling in record["adjudicator"]["resolved"]:
            adjudications.append(ruling)

        journey.append({
            "iteration": record["iteration"],
            "frontier_empty_reason": record["frontier_empty_reason"],
            "tasks_planned": planned,
            "tasks_rejected": rejected,
            "tasks_executed": executed,
            "belief_transitions": transitions,
            "contradictions_detected": record["evaluator"]["contested_pairs"],
            "adjudications": adjudications,
            "checkpoint": record["checkpoint"],
        })

    return journey


def _find_scope_statement(artifact_content):
    # the first line of the artifact that states where/how it searched -
    # shown verbatim so the reader judges the scope , not us
    for line in artifact_content.split("\n"):
        lowered = line.lower()
        for marker in SCOPE_MARKERS:
            if marker in lowered:
                return line.strip()
    return "(no scope statement found in the evidence text)"


def _build_belief_table_section(workspace, run_log):
    snapshot = workspace.snapshot()

    artifacts_by_id = {}
    for artifact in snapshot["artifacts"]:
        artifacts_by_id[artifact.id] = artifact

    # honesty flags gathered from the applied verdicts across the run
    testimonial_capped_claim_ids = set()
    negative_evidence_claim_ids = set()
    for record in run_log:
        for verdict in record["evaluator"]["applied"]:
            if "[capped: testimonial" in verdict["rationale"]:
                testimonial_capped_claim_ids.add(verdict["claim_id"])
            if verdict["is_negative"]:
                negative_evidence_claim_ids.add(verdict["claim_id"])

    table = []
    for claim in snapshot["claims"]:

        evidence_chain = []
        source_task_ids = set()
        for artifact_id in claim.evidence_ids:
            artifact = artifacts_by_id.get(artifact_id)
            if artifact is None:
                continue
            evidence_chain.append({
                "artifact_id": artifact.id,
                "summary": artifact.summary or "(no summary)",
            })
            source_task_ids.add(artifact.task_id)

        flags = []
        if len(source_task_ids) < 2:
            flags.append("single-source")
        if claim.id in testimonial_capped_claim_ids:
            flags.append("testimonial-capped")

        search_scope = ""
        if claim.id in negative_evidence_claim_ids:
            flags.append("negative-evidence")
            scope_lines = []
            for artifact_id in claim.evidence_ids:
                artifact = artifacts_by_id.get(artifact_id)
                if artifact is not None:
                    scope_lines.append(_find_scope_statement(artifact.content))
            search_scope = " | ".join(scope_lines)

        table.append({
            "claim_id": claim.id,
            "statement": claim.statement,
            "belief": claim.belief,
            "evidence_chain": evidence_chain,
            "honesty_flags": flags,
            "stated_search_scope": search_scope,
        })

    return table


def _build_contested_section(workspace, run_log):
    # contested pairs still open are shown with BOTH sides and their
    # evidence - never coin-flipped , never hidden
    snapshot = workspace.snapshot()

    claims_by_id = {}
    for claim in snapshot["claims"]:
        claims_by_id[claim.id] = claim

    artifacts_by_id = {}
    for artifact in snapshot["artifacts"]:
        artifacts_by_id[artifact.id] = artifact

    def describe_side(claim_id):
        claim = claims_by_id.get(claim_id)
        if claim is None:
            return {"claim_id": claim_id, "statement": "(claim not found)",
                    "belief": "", "evidence": []}
        evidence = []
        for artifact_id in claim.evidence_ids:
            artifact = artifacts_by_id.get(artifact_id)
            if artifact is not None:
                evidence.append({
                    "artifact_id": artifact.id,
                    "summary": artifact.summary or "(no summary)",
                })
        return {
            "claim_id": claim.id,
            "statement": claim.statement,
            "belief": claim.belief,
            "evidence": evidence,
        }

    # parked pairs across the run , each with its parking reason
    parked = []
    for record in run_log:
        for pair in record["adjudicator"]["parked"]:
            parked.append({
                "side_a": describe_side(pair["claim_id_a"]),
                "side_b": describe_side(pair["claim_id_b"]),
                "detection_reason": pair.get("reason", ""),
                "parking_reason": "shares a claim with another pair - needs "
                                  "graph reasoning, or its conflict kind "
                                  "could not be classified",
            })

    # claims still contested at the end , shown individually too (a claim
    # can be contested without an open pair if its opponent moved on)
    still_contested = []
    for claim in snapshot["claims"]:
        if claim.belief == "contested":
            still_contested.append(describe_side(claim.id))

    return {"parked_pairs": parked, "still_contested_claims": still_contested}


def _build_calibration_section():
    # phrased carefully by the template : this is the HISTORICAL judgment
    # record across runs - rates contextualize confidence , they never
    # prove it . counts and label tiers only
    summary = read_calibration_summary()
    return {
        "total_entries": summary["total_entries"],
        "counts_by_kind": summary["by_kind"],
        "counts_by_evidence_type": summary["by_evidence_type"],
        "vote_splits_by_tier": summary["vote_splits_by_tier"],
    }


def _build_run_health_section(run_log, halted_by_fuse):
    done_when_failures = []
    propagation_flags = []
    for record in run_log:
        for failure in record["evaluator"]["done_when_failures"]:
            done_when_failures.append(failure)
        for flag in record["adjudicator"]["propagation_flags"]:
            propagation_flags.append(flag)

    if halted_by_fuse:
        stop_reason = f"HALTED BY FUSE - {halted_by_fuse}"
        warnings = []
        if run_log and run_log[-1]["checkpoint"] is not None:
            warnings = run_log[-1]["checkpoint"]["warnings"]
    elif run_log and run_log[-1]["checkpoint"] is not None:
        checkpoint = run_log[-1]["checkpoint"]
        stop_reason = f"{checkpoint['decision']}: {checkpoint['reason']}"
        warnings = checkpoint["warnings"]
    else:
        stop_reason = "(the loop never completed an iteration)"
        warnings = []

    return {
        "halted_by_fuse": halted_by_fuse,
        "stop_reason": stop_reason,
        "iterations_run": len(run_log),
        "done_when_failures": done_when_failures,
        "unaddressed_propagation_flags": propagation_flags,
        "warnings": warnings,
    }


def _pick_language_register(spec):
    # v1 heuristic : a directly measurable problem reads as technical ,
    # everything else gets plain prose . the richer version would classify
    # the domain from the spec text (an llm call the report deliberately
    # does not make - the report only presents , it never judges)
    characteristics = spec.get("characteristics", {})
    if characteristics.get("measurability") == "directly_measurable":
        return "technical"
    return "plain"


# ===========================================================================
# the one public function
# ===========================================================================

def generate_report(workspace, spec, run_log, halted_by_fuse=""):
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "register": _pick_language_register(spec),
        "problem": _build_problem_section(spec),
        "journey": _build_journey_section(run_log, workspace),
        "belief_table": _build_belief_table_section(workspace, run_log),
        "contested_and_unresolved": _build_contested_section(workspace, run_log),
        "calibration_context": _build_calibration_section(),
        "run_health": _build_run_health_section(run_log, halted_by_fuse),
    }

    os.makedirs(REPORTS_DIRECTORY, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(REPORTS_DIRECTORY, f"report_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)

    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIRECTORY),
        autoescape=True,
    )
    template = environment.get_template(TEMPLATE_FILENAME)
    html_text = template.render(report=report)

    html_path = os.path.join(REPORTS_DIRECTORY, f"report_{stamp}.html")
    with open(html_path, "w", encoding="utf-8") as html_file:
        html_file.write(html_text)

    return json_path, html_path
