'''
test_framework.py - offline test suite . no api key , no llm calls , no
network : every test exercises PURE CODE (gates , predicates , matchers ,
parsers) , which is exactly where this framework keeps its guarantees .

run :  python tests/test_framework.py        (plain asserts , exit 0 = pass)
   or :  pytest tests/test_framework.py

each test names the failure genre it guards against . most were added
after a live stress run produced the failure for real - see
docs/STRESS_TEST.md for the run-level evidence .
'''

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.evaluator import (
    _extract_numbers,
    _pinned_numbers_from_spec,
    _check_external_confirmation,
    _check_source_independence,
    _check_pinned_numbers,
    _check_terminal_evidence,
    _check_testimonial_cap,
    _check_tempo,
)
from agents.executor import _find_corroborated_claim, _claim_profile
from agents.checkpoint import _all_rejected_as_duplicates
from intake.catalog import CHARACTERISTIC_VALUES, STRUCTURE_CATALOG
from intake.formalization import filter_structures
from core.kernel import assert_spec_ratified
from llm.client import _server_retry_delay_seconds, vote_split_tier


# ---------------------------------------------------------------------------
# tiny stand-ins for workspace entities (dataclass shape only)
# ---------------------------------------------------------------------------

class FakeArtifact:
    def __init__(self, id, task_id):
        self.id = id
        self.task_id = task_id


class FakeClaim:
    def __init__(self, evidence_ids):
        self.evidence_ids = evidence_ids


class FakeTask:
    def __init__(self, kind, status="completed", rejection_reason=""):
        self.kind = kind
        self.status = status
        self.rejection_reason = rejection_reason


PASSED = []


def test(name):
    def wrap(fn):
        fn()
        PASSED.append(name)
    return wrap


# ===========================================================================
# 1. ground-truth pinning - claims may not invent numbers
# ===========================================================================

@test("pinning: user numbers extracted from any spec shape")
def _():
    spec = {"goal": "fraud triage", "anchors": ["recall is 95%", "cost $8"],
            "nested": {"sla": "2 hours", "deep": ["prevalence 0.5%"]}}
    pinned = _pinned_numbers_from_spec(spec)
    for expected in (95.0, 0.95, 8.0, 2.0, 0.5, 0.005):
        assert expected in pinned, (expected, pinned)


@test("pinning: invented number without a formula is dropped")
def _():
    pinned = {95.0, 0.95, 8.0}
    r = _check_pinned_numbers("supported", "unverified",
                              "recall of 0.92 was achieved",
                              "plain text, no working shown", pinned)
    assert not r.verdict_alive


@test("pinning: user's own number promotes ; formula-backed number promotes")
def _():
    pinned = {95.0, 0.95, 3000.0, 4.0, 12000.0}
    r = _check_pinned_numbers("supported", "unverified",
                              "recall is 95%", "artifact text", pinned)
    assert r.verdict_alive
    r = _check_pinned_numbers("supported", "unverified",
                              "throughput reaches 12000/hour",
                              "throughput = 3000 x 4 = 12000", pinned)
    assert r.verdict_alive


@test("pinning: demotions are never blocked (doubt is free)")
def _():
    r = _check_pinned_numbers("unverified", "supported",
                              "recall of 0.92", "no working", {95.0})
    assert r.verdict_alive


# ===========================================================================
# 2. non-fabrication - approvals and measurements are unpromotable
# ===========================================================================

@test("fabrication: sign-off claims cannot be promoted")
def _():
    r = _check_external_confirmation(
        "supported", "unverified",
        "The CFO provided formal sign-off on the cost figures")
    assert not r.verdict_alive


@test("fabrication: measurement claims cannot be promoted (no tools exist)")
def _():
    r = _check_external_confirmation(
        "supported", "unverified",
        "The 4-process benchmark achieved a throughput of 11,850 images per hour")
    assert not r.verdict_alive


@test("fabrication: ordinary factual claims still promote")
def _():
    r = _check_external_confirmation(
        "supported", "unverified",
        "The GIL restricts bytecode execution to one thread at a time")
    assert r.verdict_alive


# ===========================================================================
# 3. evidentiary independence - one echoed source is one source
# ===========================================================================

@test("independence: one task echoed twice cannot make supported")
def _():
    artifacts = {1: FakeArtifact(1, 10), 2: FakeArtifact(2, 10)}
    r = _check_source_independence("supported", "empirical", False,
                                   FakeClaim([1, 2]), artifacts)
    assert not r.verdict_alive


@test("independence: two distinct tasks make supported")
def _():
    artifacts = {1: FakeArtifact(1, 10), 3: FakeArtifact(3, 11)}
    r = _check_source_independence("supported", "empirical", False,
                                   FakeClaim([1, 3]), artifacts)
    assert r.verdict_alive


@test("independence: deductive single-source is exempt , negative never is")
def _():
    artifacts = {1: FakeArtifact(1, 10)}
    r = _check_source_independence("supported", "deductive", False,
                                   FakeClaim([1]), artifacts)
    assert r.verdict_alive
    r = _check_source_independence("supported", "deductive", True,
                                   FakeClaim([1]), artifacts)
    assert not r.verdict_alive


# ===========================================================================
# 4. no circular evidence - synthesis outputs are terminal
# ===========================================================================

@test("circularity: a produce-kind artifact cannot serve as evidence")
def _():
    artifacts = {1: FakeArtifact(1, 10), 3: FakeArtifact(3, 11)}
    tasks = {10: FakeTask("produce"), 11: FakeTask("investigate")}
    r = _check_terminal_evidence("supported", "unverified", [1], artifacts, tasks)
    assert not r.verdict_alive
    r = _check_terminal_evidence("supported", "unverified", [3], artifacts, tasks)
    assert r.verdict_alive


# ===========================================================================
# 5. claim corroboration - twins merge , negation flips never do
# ===========================================================================

@test("corroboration: near-duplicate statements corroborate the original")
def _():
    known = {2: _claim_profile(
        "The multiprocessing module bypasses the GIL by utilizing "
        "separate memory spaces and interpreters for each process.")}
    match = _find_corroborated_claim(
        "The multiprocessing module bypasses the GIL by utilizing "
        "separate memory spaces for each process.", known)
    assert match == 2


@test("corroboration: a negation flip NEVER merges")
def _():
    known = {1: _claim_profile(
        "Asyncio does not provide performance improvements for "
        "CPU-bound tasks in CPython.")}
    match = _find_corroborated_claim(
        "Asyncio does provide performance improvements for "
        "CPU-bound tasks in CPython.", known)
    assert match is None


@test("corroboration: unrelated statements do not merge")
def _():
    known = {1: _claim_profile("RabbitMQ quorum queues require three nodes.")}
    match = _find_corroborated_claim(
        "The GIL serializes bytecode execution.", known)
    assert match is None


# ===========================================================================
# 6. ladder discipline - caps and tempo
# ===========================================================================

@test("ladder: testimonial evidence caps at supported")
def _():
    r = _check_testimonial_cap("verified", "testimonial")
    assert r.verdict_alive and r.proposed_belief == "supported"


@test("ladder: tempo forbids a second promotion in one iteration")
def _():
    r = _check_tempo("supported", "unverified", claim_id=7,
                     promoted_this_iteration={7})
    assert not r.verdict_alive


# ===========================================================================
# 7. checkpoint - all-duplicates is exhaustion , anything else is a stall
# ===========================================================================

@test("checkpoint: all proposals rejected as duplicates = frontier exhausted")
def _():
    record = {"tasks_created_by_planner": 2, "accepted_task_ids": []}
    snapshot = {"tasks": [
        FakeTask("investigate", "rejected", "duplicate: already covered"),
        FakeTask("investigate", "rejected", "duplicate: same as task 3"),
    ]}
    assert _all_rejected_as_duplicates(record, snapshot)


@test("checkpoint: a vague rejection keeps it a stall")
def _():
    record = {"tasks_created_by_planner": 2, "accepted_task_ids": []}
    snapshot = {"tasks": [
        FakeTask("investigate", "rejected", "duplicate: already covered"),
        FakeTask("investigate", "rejected", "vague: no checkable done_when"),
    ]}
    assert not _all_rejected_as_duplicates(record, snapshot)


# ===========================================================================
# 8. catalog - coverage and reachability
# ===========================================================================

@test("catalog: every structure is reachable by some characteristics combo")
def _():
    import itertools
    reachable = set()
    keys = list(CHARACTERISTIC_VALUES)
    for combo in itertools.product(*CHARACTERISTIC_VALUES.values()):
        characteristics = dict(zip(keys, combo))
        characteristics["uncertainty_present"] = True
        for s in filter_structures(characteristics):
            reachable.add(s["id"])
    missing = {s["id"] for s in STRUCTURE_CATALOG} - reachable
    assert not missing, missing


@test("catalog: every primary_output surfaces at least one candidate")
def _():
    base = {"cardinality": "single_decision", "stakeholders": "single",
            "temporal_structure": "one_shot", "success_type": "graded",
            "solution_uniqueness": "any_qualifying_ok",
            "measurability": "proxy_only", "uncertainty_present": True}
    for output in CHARACTERISTIC_VALUES["primary_output"]:
        characteristics = dict(base, primary_output=output)
        assert filter_structures(characteristics), output


@test("catalog: ids unique , every entry carries the required fields")
def _():
    ids = [s["id"] for s in STRUCTURE_CATALOG]
    assert len(ids) == len(set(ids))
    for s in STRUCTURE_CATALOG:
        for field in ("id", "name", "definition", "applies_if"):
            assert field in s, (s.get("id"), field)
        assert callable(s["applies_if"])


# ===========================================================================
# 9. kernel + client plumbing
# ===========================================================================

@test("kernel: unratified spec is refused at every door")
def _():
    try:
        assert_spec_ratified({"ratified": False})
        assert False, "should have raised"
    except Exception:
        pass
    assert_spec_ratified({"ratified": True})   # must not raise


@test("client: server-stated 429 retry delay is honored")
def _():
    class E:
        details = {"error": {"details": [
            {"@type": "type.googleapis.com/google.rpc.RetryInfo",
             "retryDelay": "37s"}]}}
    assert _server_retry_delay_seconds(E()) == 37.0

    class NoDetails:
        details = None
    assert _server_retry_delay_seconds(NoDetails()) is None


@test("client: vote splits reduce to label tiers , never ratios")
def _():
    assert vote_split_tier((3,)) == "unanimous"
    assert vote_split_tier((2, 1)) == "majority"
    assert vote_split_tier((1, 1, 1)) == "split"


# ===========================================================================

@test("numbers: percent tokens pin both forms")
def _():
    numbers = _extract_numbers("recall is 95% at a 0.5% prevalence")
    for expected in (95.0, 0.95, 0.5, 0.005):
        assert expected in numbers


if __name__ == "__main__":
    print(f"{len(PASSED)}/{len(PASSED)} tests passed:")
    for name in PASSED:
        print(f"  [PASS] {name}")
