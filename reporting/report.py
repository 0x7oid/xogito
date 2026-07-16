'''
report.py - the final output , in two stages : first a JSON report object
(the data , machine-readable , the audit trail) , then a single
self-contained HTML file rendered from it with jinja2 (the human view ,
Arial , readable offline , no external assets) .

the report is HUMAN-FIRST : the answer , a plain-language summary , and
anything that affects whether the reader should trust the result (dropped
anchors , load-bearing guesses) sit at the top , always visible . the
audit trail - the per-iteration trace , the belief table , calibration ,
run health - lives in independent collapsible panels below , each with a
one-line teaser so opening one is never a gamble . an answer with
receipts , not a dump of receipts .

the report never shows a numeric confidence score anywhere - labels and
counts only . quantitative claims are phrased as what they are ("two
independent sources assert X") , never as statistical conclusions the
system did not perform . internal vocabulary (fuse names , done_when ,
belief machinery) is translated to plain language in the visible layer ;
the technical terms stay available inside the toggled panels .

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
from core.compressor import compress_run
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
# plain-language vocabulary . closed maps in code - the visible layer of
# the report speaks these , the toggled panels may keep the internal terms
# ===========================================================================

PLAIN_BELIEF = {
    "verified": "well-supported (multiple independent sources)",
    "supported": "backed by evidence",
    "unverified": "not yet backed by evidence",
    "contested": "under dispute",
}

PLAIN_DECISION = {
    "continue": "kept going - there was still work to do",
    "stop_success": "finished - the success conditions were met",
    "stop_stall": "stopped early - it was no longer making progress",
    "stop_budget": "stopped - it used up its allowed number of rounds",
}

PLAIN_FUSE = {
    "PlannerFuseTripped": "the planning step behaved abnormally, so the "
                          "run was stopped as a safety measure",
    "ExecutorFuseTripped": "the run used up its time budget while doing "
                           "the work, so it was stopped",
    "EvaluatorFuseTripped": "the checking step behaved abnormally, so the "
                            "run was stopped as a safety measure",
    "AdjudicatorFuseTripped": "too many conflicting claims appeared at "
                              "once, so the run was stopped for a human to look at",
}


def _plain_fuse(halted_by_fuse):
    # halted_by_fuse looks like "ExecutorFuseTripped: batch exceeded ..."
    for fuse_name in PLAIN_FUSE:
        if halted_by_fuse.startswith(fuse_name):
            return PLAIN_FUSE[fuse_name]
    return "the run hit an internal safety stop before finishing"


def _plain_status(headline, halted_by_fuse, run_log):
    if halted_by_fuse:
        return ("This run did not finish - " + _plain_fuse(halted_by_fuse)
                + ". Here is as far as it got.")
    if headline["finished"]:
        return "This run finished."
    if run_log and run_log[-1]["checkpoint"] is not None:
        decision = run_log[-1]["checkpoint"]["decision"]
        return "This run " + PLAIN_DECISION.get(decision, "stopped") + \
               ". Here is as far as it got."
    return "This run never completed a full round of work."


# ===========================================================================
# section builders (pure code) . each returns plain dicts/lists so the
# whole report object is json-serializable as-is
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
        "verification_checklist": ps.get("verification_checklist", []),
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


def _build_trust_section(spec):
    # the two things that materially affect whether the reader should
    # trust the result : anchors the framing dropped , and load-bearing
    # guesses . NEVER behind a toggle
    anchor_trace = spec.get("anchor_trace", {"carried": [], "dropped": []})

    load_bearing = []
    peripheral = []
    for entry in spec.get("assumption_review", []):
        if entry["impact"] == "load_bearing":
            load_bearing.append(entry)
        else:
            peripheral.append(entry)

    return {
        "dropped_anchors": anchor_trace["dropped"],
        "carried_anchors": anchor_trace["carried"],
        "load_bearing_assumptions": load_bearing,
        "peripheral_assumptions": peripheral,
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
                "new_belief_plain": PLAIN_BELIEF.get(verdict["proposed_belief"],
                                                     verdict["proposed_belief"]),
                "evidence_type": verdict["evidence_type"],
                "rationale": verdict["rationale"],
            })

        checkpoint = record["checkpoint"]
        checkpoint_plain = ""
        if checkpoint is not None:
            checkpoint_plain = PLAIN_DECISION.get(checkpoint["decision"],
                                                  checkpoint["decision"])

        journey.append({
            "iteration": record["iteration"],
            "frontier_empty_reason": record["frontier_empty_reason"],
            "tasks_planned": planned,
            "tasks_rejected": rejected,
            "tasks_executed": executed,
            "belief_transitions": transitions,
            "contradictions_detected": record["evaluator"]["contested_pairs"],
            "adjudications": record["adjudicator"]["resolved"],
            "checkpoint": checkpoint,
            "checkpoint_plain": checkpoint_plain,
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
            "belief_plain": PLAIN_BELIEF.get(claim.belief, claim.belief),
            "evidence_chain": evidence_chain,
            "honesty_flags": flags,
            "stated_search_scope": search_scope,
        })

    return table


# ===========================================================================
# the decision map : the original question at the root , each task as a
# line of inquiry beneath it , with source counts and a supporting /
# conflicting split . rendered as a nested collapsible tree - shallow by
# default , expand a branch to see its sources and claims
# ===========================================================================

def _build_decision_map(spec, workspace):
    snapshot = workspace.snapshot()

    artifacts_by_task = {}
    for artifact in snapshot["artifacts"]:
        if artifact.task_id not in artifacts_by_task:
            artifacts_by_task[artifact.task_id] = []
        artifacts_by_task[artifact.task_id].append(artifact)

    branches = []
    for task in snapshot["tasks"]:
        if task.status == "rejected":
            continue   # rejected proposals live in the process panel

        task_artifacts = artifacts_by_task.get(task.id, [])
        artifact_ids = set()
        for artifact in task_artifacts:
            artifact_ids.add(artifact.id)

        supporting = 0
        conflicting = 0
        unresolved = 0
        claim_entries = []
        for claim in snapshot["claims"]:
            touches_this_task = False
            for evidence_id in claim.evidence_ids:
                if evidence_id in artifact_ids:
                    touches_this_task = True
                    break
            if not touches_this_task:
                continue
            if claim.belief in ("supported", "verified"):
                supporting += 1
            elif claim.belief == "contested":
                conflicting += 1
            else:
                unresolved += 1
            claim_entries.append({
                "claim_id": claim.id,
                "statement": claim.statement,
                "belief": claim.belief,
                "belief_plain": PLAIN_BELIEF.get(claim.belief, claim.belief),
            })

        if task.status == "completed" and conflicting == 0:
            state = "resolved"
        elif task.status == "failed":
            state = "failed"
        else:
            state = "open"

        sources = []
        for artifact in task_artifacts:
            sources.append({
                "artifact_id": artifact.id,
                "summary": artifact.summary or "(no summary)",
            })

        branches.append({
            "task_id": task.id,
            "question": task.description,
            "state": state,
            "sources_checked": len(task_artifacts),
            "supporting": supporting,
            "conflicting": conflicting,
            "unresolved": unresolved,
            "sources": sources,
            "claims": claim_entries,
        })

    resolved_count = 0
    for branch in branches:
        if branch["state"] == "resolved":
            resolved_count += 1

    return {
        "root_question": spec.get("problem", "(not recorded)"),
        "branches": branches,
        "resolved_count": resolved_count,
    }


# ===========================================================================
# the dependency map : which claim rests on which source , which claims
# share a source . progressive disclosure : one summary line per claim by
# default , the full picture (including a coded svg diagram) on demand -
# traceability , not a puzzle
# ===========================================================================

def _build_dependency_map(workspace):
    snapshot = workspace.snapshot()

    artifacts_by_id = {}
    for artifact in snapshot["artifacts"]:
        artifacts_by_id[artifact.id] = artifact

    claims = []
    for claim in snapshot["claims"]:
        artifact_entries = []
        shared_with = set()
        for artifact_id in claim.evidence_ids:
            artifact = artifacts_by_id.get(artifact_id)
            if artifact is None:
                continue
            for other_claim_id in artifact.claim_ids:
                if other_claim_id != claim.id:
                    shared_with.add(other_claim_id)
            artifact_entries.append({
                "artifact_id": artifact.id,
                "summary": artifact.summary or "(no summary)",
                "task_id": artifact.task_id,
            })
        claims.append({
            "claim_id": claim.id,
            "statement": claim.statement,
            "belief": claim.belief,
            "artifact_count": len(artifact_entries),
            "artifacts": artifact_entries,
            "shared_with_claims": sorted(shared_with),
        })

    return {
        "claims": claims,
        "svg": _render_dependency_svg(snapshot),
    }


def _shorten(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit - 1] + "…"


def _svg_escape(text):
    # the svg block is inserted into the html unescaped (|safe) , so any
    # claim/source text drawn inside it is escaped HERE - llm text never
    # reaches the page raw
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_dependency_svg(snapshot):
    # a coded diagram , not ascii : claims on the left , sources on the
    # right , a line per evidence link . computed here in plain code so
    # the html stays self-contained (inline svg , no libraries)
    claims = snapshot["claims"]
    artifacts = snapshot["artifacts"]
    if not claims or not artifacts:
        return ""

    row_height = 44
    rows = max(len(claims), len(artifacts))
    height = rows * row_height + 40
    width = 860

    belief_colors = {"verified": "#2c7a2c", "supported": "#3b5bb0",
                     "unverified": "#888888", "contested": "#c07a1a"}

    claim_y = {}
    lines = []
    lines.append(f'<svg viewBox="0 0 {width} {height}" role="img" '
                 'style="width:100%;height:auto;font-family:Arial,sans-serif;">')
    lines.append(f'<text x="20" y="20" font-size="12" fill="#666">Claims</text>')
    lines.append(f'<text x="620" y="20" font-size="12" fill="#666">Sources they rest on</text>')

    for position in range(len(claims)):
        claim = claims[position]
        y = 40 + position * row_height + row_height // 2
        claim_y[claim.id] = y
        color = belief_colors.get(claim.belief, "#888888")
        lines.append(f'<circle cx="20" cy="{y}" r="6" fill="{color}"/>')
        label = _svg_escape(_shorten(f"claim {claim.id}: {claim.statement}", 55))
        lines.append(f'<text x="32" y="{y + 4}" font-size="12" fill="#222">{label}</text>')

    artifact_y = {}
    for position in range(len(artifacts)):
        artifact = artifacts[position]
        y = 40 + position * row_height + row_height // 2
        artifact_y[artifact.id] = y
        lines.append(f'<rect x="610" y="{y - 7}" width="14" height="14" '
                     'fill="#e8e8e4" stroke="#888"/>')
        label = _svg_escape(_shorten(f"source {artifact.id}: {artifact.summary or ''}", 38))
        lines.append(f'<text x="632" y="{y + 4}" font-size="12" fill="#222">{label}</text>')

    for claim in claims:
        for evidence_id in claim.evidence_ids:
            if claim.id in claim_y and evidence_id in artifact_y:
                lines.append(f'<line x1="420" y1="{claim_y[claim.id]}" '
                             f'x2="608" y2="{artifact_y[evidence_id]}" '
                             'stroke="#b9b9b2" stroke-width="1.5"/>')

    lines.append("</svg>")
    return "\n".join(lines)


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
                "parking_reason": "this dispute overlaps with another one, "
                                  "so deciding it in isolation could give "
                                  "the wrong result - it was set aside for "
                                  "a human to look at",
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
        stop_reason_plain = _plain_fuse(halted_by_fuse)
        warnings = []
        if run_log and run_log[-1]["checkpoint"] is not None:
            warnings = run_log[-1]["checkpoint"]["warnings"]
    elif run_log and run_log[-1]["checkpoint"] is not None:
        checkpoint = run_log[-1]["checkpoint"]
        stop_reason = f"{checkpoint['decision']}: {checkpoint['reason']}"
        stop_reason_plain = PLAIN_DECISION.get(checkpoint["decision"],
                                               checkpoint["decision"])
        warnings = checkpoint["warnings"]
    else:
        stop_reason = "(the loop never completed an iteration)"
        stop_reason_plain = "the run never completed a full round of work"
        warnings = []

    return {
        "halted_by_fuse": halted_by_fuse,
        "halted_by_fuse_plain": _plain_fuse(halted_by_fuse) if halted_by_fuse else "",
        "stop_reason": stop_reason,
        "stop_reason_plain": stop_reason_plain,
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
# panel teasers . a collapsed panel always says what is inside , so
# opening one is never a gamble
# ===========================================================================

def _build_teasers(report):
    beliefs = report["headline"]["statistics"]["claims_by_belief"]
    trust = report["trust"]
    decision_map = report["decision_map"]
    contested = report["contested_and_unresolved"]
    health = report["run_health"]

    total_claims = 0
    for label in beliefs:
        total_claims += beliefs[label]

    open_branches = len(decision_map["branches"]) - decision_map["resolved_count"]

    return {
        "assumptions": (f"{len(trust['carried_anchors'])} of your facts carried "
                        f"through, {len(trust['dropped_anchors'])} dropped, "
                        f"{len(trust['load_bearing_assumptions'])} important "
                        "guess(es) made"),
        "process": (f"{health['iterations_run']} round(s) of work, "
                    f"{report['headline']['statistics']['tasks_completed']} "
                    "task(s) completed"),
        "decision_map": (f"{len(decision_map['branches'])} line(s) of inquiry - "
                         f"{decision_map['resolved_count']} settled, "
                         f"{open_branches} open"),
        "belief_table": (f"{total_claims} claim(s) - "
                         f"{beliefs['verified'] + beliefs['supported']} backed "
                         f"by evidence, {beliefs['contested']} under dispute, "
                         f"{beliefs['unverified']} unsupported"),
        "dependency_map": (f"{len(report['dependency_map']['claims'])} claim(s) "
                           "traced to the sources they rest on"),
        "contested": (f"{len(contested['still_contested_claims'])} open "
                      f"dispute(s), {len(contested['parked_pairs'])} set aside"),
        "calibration": (f"{report['calibration_context']['total_entries']} "
                        "past judgment(s) on record across runs"),
        "run_health": health["stop_reason_plain"],
        "problem_framing": "how your question was framed, what was kept and what was dropped",
    }


# ===========================================================================
# the one public function
# ===========================================================================

def generate_report(workspace, spec, run_log, halted_by_fuse=""):
    # compression first - the summary is a first-class output and the
    # rest of the report is the receipts behind it
    headline = compress_run(spec, workspace, run_log, halted_by_fuse)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "register": _pick_language_register(spec),
        "headline": headline,
        "trust": _build_trust_section(spec),
        "problem": _build_problem_section(spec),
        "journey": _build_journey_section(run_log, workspace),
        "decision_map": _build_decision_map(spec, workspace),
        "dependency_map": _build_dependency_map(workspace),
        "belief_table": _build_belief_table_section(workspace, run_log),
        "contested_and_unresolved": _build_contested_section(workspace, run_log),
        "calibration_context": _build_calibration_section(),
        "run_health": _build_run_health_section(run_log, halted_by_fuse),
    }
    report["status_plain"] = _plain_status(headline, halted_by_fuse, run_log)
    report["teasers"] = _build_teasers(report)

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
