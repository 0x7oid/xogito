def assert_spec_ratified(spec):
    if spec.get("decision", {}).get("contested") or spec.get("contested"):
        raise ValueError("spec is contested - cannot proceed unratified")

    if not spec.get("ratified"):
        raise ValueError("spec was never ratified by the user")

    return True

# lets define the legal transitions of the state machine for the belief laddeer
legal_transitions = {
    "unverified": ["supported", "contested"],
    "supported":  ["verified", "unverified", "contested"],
    "verified":   ["contested"],
    "contested":  ["supported", "unverified"],
}
# notice that contested is reachable from everwhere , and that unverified < supported < verified . contested is a special state
#notice also that contested is not a trapping state

def assert_legal_transition(old_belief: str, new_belief: str , veridict: dict):
    # the veridict is sent by the evaluator
    # this function will verify if the belief update is legal according to the legal_transitions defined above
    # first we need to check if the transition is legal
    if new_belief not in legal_transitions.get(old_belief, []):
        raise ValueError(f"Illegal belief transition from {old_belief} to {new_belief}")
    # moving up costs evidence , moving down is cheap
    if old_belief == "unverified" and new_belief == "supported":
        if not veridict.get("evidence"):
            raise ValueError(f"Cannot move from {old_belief} to {new_belief} without evidence")
    if new_belief == "verified":
        evidence = veridict.get("evidence_ids", [])
        if len(evidence) < 2:
            raise ValueError(f"Cannot move from {old_belief} to {new_belief} without at least 2 pieces of evidence")
        
