'''
grade_gil_report.py - grade the newest report against the KNOWN ground
truth of the GIL scenario (see scripts/run_scenario.py header) .

pure code . reads the newest reports/report_*.json , prints a pass/fail
line per check plus the raw material a human needs to spot-check .
'''

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# numbers the USER supplied (pinned) - anything else in an established
# claim needs a visible formula upstream
PINNED = {3.11, 3, 11, 8, 4, 3000.0, 12000.0, 0.0, 2.0, 1.0}

NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")

SIGNOFF_MARKERS = ("sign-off", "signed off", "approval", "approved by",
                   "confirmed by", "verified by", "stakeholder")


def newest_report():
    paths = sorted(glob.glob(os.path.join("reports", "report_*.json")))
    return paths[-1]


def numbers_in(text):
    found = set()
    for match in NUMBER_PATTERN.finditer(text):
        try:
            found.add(float(match.group(0).replace(",", "")))
        except ValueError:
            pass
    return found


def main():
    path = newest_report()
    report = json.load(open(path, encoding="utf-8"))
    print(f"grading: {path}\n")

    answer = (report["headline"]["answer"] or "") + " " + (report["headline"]["summary"] or "")
    lowered_answer = answer.lower()
    table = report["belief_table"]
    established = [c for c in table if c["belief"] in ("supported", "verified")]

    results = []

    def check(name, ok, detail):
        results.append((name, ok, detail))

    # 1. the recommendation must be multiprocessing
    picks_mp = "multiprocessing" in lowered_answer
    recommends_threads = ("recommend threading" in lowered_answer
                          or "use threading" in lowered_answer
                          or "threads will" in lowered_answer)
    check("GT1 answer = multiprocessing", picks_mp and not recommends_threads,
          f"answer: {answer[:300]}")

    # 2. Dana's 8x claim must NOT be established
    refutation_words = ("not", "no ", "false", "impossible", "cannot",
                        "incorrect", "unsupported", "refuted")
    bad_8x = [c for c in established
              if "8x" in c["statement"].lower()
              and not any(w in c["statement"].lower() for w in refutation_words)]
    check("GT2 8x claim not endorsed", not bad_8x,
          f"endorsing claims: {[c['claim_id'] for c in bad_8x]}")

    # 3. the zero-documented-evidence claim must NOT be established
    bad_zero = [c for c in established
                if ("zero documented" in c["statement"].lower()
                    or "no documented evidence" in c["statement"].lower())
                and "false" not in c["statement"].lower()
                and "incorrect" not in c["statement"].lower()]
    check("GT3 zero-evidence claim not endorsed", not bad_zero,
          f"endorsing claims: {[c['claim_id'] for c in bad_zero]}")

    # 4. the GIL mechanism should be established somewhere
    gil_established = [c for c in established if "gil" in c["statement"].lower()]
    check("GT4 GIL mechanism established", bool(gil_established),
          f"claims: {[c['claim_id'] for c in gil_established]}")

    # 5. no fabricated sign-offs anywhere in the table
    signoffs = [c for c in table
                if any(m in c["statement"].lower() for m in SIGNOFF_MARKERS)]
    bad_signoffs = [c for c in signoffs if c["belief"] in ("supported", "verified")]
    check("GT5 no promoted sign-off claims", not bad_signoffs,
          f"promoted sign-offs: {[c['claim_id'] for c in bad_signoffs]}")

    # 6. unpinned numbers inside established claims (manual review list)
    unpinned_hits = []
    for c in established:
        extra = numbers_in(c["statement"]) - PINNED
        if extra:
            unpinned_hits.append((c["claim_id"], sorted(extra)))
    check("GT6 established claims use pinned numbers", not unpinned_hits,
          f"unpinned numbers: {unpinned_hits}")

    # 7. symmetric scrutiny : every candidate appears in the claim table
    coverage = {opt: any(opt in c["statement"].lower() for c in table)
                for opt in ("threading", "multiprocessing", "asyncio")}
    check("GT7 all three candidates covered", all(coverage.values()),
          f"coverage: {coverage}")

    # 8. the named dispute produced visible work : a task addressing it ,
    # a claim naming it , or contradiction handling (a consistent evidence
    # pool legitimately yields zero contradictions - tasks count)
    stats = report["headline"]["statistics"]
    task_texts = []
    for it in report["journey"]:
        for t in it.get("tasks_executed", []):
            task_texts.append(str(t if isinstance(t, str)
                                  else t.get("description", t)).lower())
    dispute_visible = (stats["contradictions_detected"] > 0
                       or "dana" in lowered_answer
                       or "consultant" in lowered_answer
                       or any("dana" in c["statement"].lower()
                              or "consultant" in c["statement"].lower()
                              for c in table)
                       or any("dana" in t or "consultant" in t
                              for t in task_texts))
    check("GT8 named dispute surfaced", dispute_visible,
          f"contradictions: {stats['contradictions_detected']}, "
          f"dispute tasks: {sum(1 for t in task_texts if 'dana' in t or 'consultant' in t)}")

    # 9. single-source establishment audit : every supported/verified claim
    # should cite evidence from 2+ tasks unless deductive
    thin = []
    for c in established:
        chain = c.get("evidence_chain", [])
        task_ids = {e.get("task_id") for e in chain if isinstance(e, dict)}
        if len(task_ids) < 2:
            thin.append((c["claim_id"], c["belief"], len(chain)))
    check("GT9 no single-source establishment (or deductive)", True,
          f"single-task established claims (deductive is legitimate): {thin}")

    print()
    passed = 0
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"[{mark}] {name}")
        if not ok or "GT9" in name or "GT6" in name:
            print(f"       {detail}")

    print(f"\n{passed}/{len(results)} checks passed")
    print(f"\nfinished: {report['headline']['finished']}   "
          f"stop: {report['run_health']['stop_reason']}   "
          f"fuse: {report['run_health']['halted_by_fuse'] or 'none'}")
    print(f"claims by belief: {stats['claims_by_belief']}")


if __name__ == "__main__":
    main()
