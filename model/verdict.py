from dataclasses import dataclass
from workspace import belief_ladder
from typing import Literal

EVIDENCE_TYPES = Literal["empirical", "deductive", "testimonial"]
# emprical postivie and empirical negative exist (also called negative evidence)
@dataclass
class Verdict:
    "The evaluation of a claim , used for belief transition"
    claim_id: int
    proposed_belief: belief_ladder
    evidence_ids: list[int]
    evidence_type: EVIDENCE_TYPES
    # FIX: this field was named is_empirical_negative but every consumer
    # (evaluator constructs Verdict(is_negative=...) , calibration reads
    # verdict.is_negative , the verdict dicts use "is_negative") already
    # agreed on is_negative - renamed to match the majority and the schema
    is_negative: bool
    rationale : str

# the veridict is consumed by both the update belief function in the workspace and the kernel for the transition verification