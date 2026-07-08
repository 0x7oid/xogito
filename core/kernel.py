# kernel.py — no print, no input, just a check
def assert_spec_ratified(spec):
    if spec.get("contested"):
        raise ValueError("spec is contested — cannot proceed unratified")
    if not spec.get("ratified"):
        raise ValueError("spec was never ratified by the user")