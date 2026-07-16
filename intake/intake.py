"""
intake.py - User query schema + intake for the reasoning orchestrator.

Design rules encoded here (from the framework spec):
  * The prompt is the ONLY required field. Everything else is optional
    but recommended, skippable with a plain Enter.
  * Every question is asked in natural language a non-expert understands,
    including the CONSEQUENCE of the field (e.g. anchors are never questioned).
  * MAX_ITERATIONS is a framework constant. The user cannot tweak it -
    it is not domain knowledge they possess, so it is not a question we ask.
  * Raw user text is stored verbatim (provenance). Normalization into
    structured policy/anchors happens later, in Formalisation, and is
    ratified by the user there - intake only collects.

BUGFIX (this revision), two real defects found by an actual test run:

1. MULTI-LINE INPUT. A user's real answer is often more than one line -
   a paragraph, several sentences, a pasted block. input("> ") reads
   exactly ONE line. Anything pasted with embedded newlines gets sliced
   across whatever input() call runs next - scope leaks into anchors,
   anchors leak into the next question, etc, silently. _read_multiline
   fixes this: it keeps reading lines until a blank line is submitted
   (or the stream hits EOF, for piped/redirected input), so one field
   safely absorbs an entire pasted paragraph as a single string.

2. RawField LEAKING PAST INTAKE. formalization.py (and possibly other
   downstream code) was written against plain strings and crashed calling
   len() on a RawField object. The fix is at the boundary: RawField is
   now a thin subclass of str instead of a bare pydantic BaseModel, so
   it behaves like a string everywhere a string would be used (len(),
   slicing, f-strings, .split(), string concatenation, etc) with zero
   changes needed at any downstream call site. `.text` and `.provided_at`
   remain available for code that wants the wrapper explicitly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Final, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Framework constants - protected, not user input
# ---------------------------------------------------------------------------

MAX_ITERATIONS: Final[int] = 5  # kernel-owned; never exposed to intake


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class RawField(str):
    """
    A user-provided text field, kept verbatim for provenance.

    Subclasses str on purpose: every place in the codebase that expects
    plain text (len(), slicing, string formatting, .split(), .strip(),
    membership checks) keeps working with zero changes, because a
    RawField IS a str. `.text` is kept as an alias for readability at
    call sites that want to be explicit they're reading a user-provided
    field; `.provided_at` carries the provenance timestamp a bare str
    could never hold.

    Construct with RawField(text) - same call shape a plain str would have.
    """

    def __new__(cls, text: str, provided_at: Optional[datetime] = None):
        instance = super().__new__(cls, text)
        instance.provided_at = provided_at or datetime.now(timezone.utc)
        return instance

    @property
    def text(self) -> str:
        # explicit alias - same value as the string itself
        return str(self)


class UserQuery(BaseModel):
    """
    Everything the user hands the orchestrator, exactly as they said it.

    `prompt` is required. `scope`, `policy_base`, and `contextual_anchors`
    are optional: None means the user skipped them, and Formalisation must
    then work from the prompt alone (and may ask targeted repair questions).
    """
    model_config = {"arbitrary_types_allowed": True}

    prompt: RawField
    scope: Optional[RawField] = None
    policy_base: Optional[RawField] = None
    contextual_anchors: Optional[RawField] = None

    # Not user-settable; recorded on the query so every run is self-describing.
    max_iterations: int = Field(default=MAX_ITERATIONS, frozen=True)

    @field_validator("max_iterations")
    @classmethod
    def _enforce_kernel_iterations(cls, v: int) -> int:
        if v != MAX_ITERATIONS:
            raise ValueError(
                f"max_iterations is kernel-owned and fixed at {MAX_ITERATIONS}."
            )
        return v

    def summary(self) -> str:
        """One-screen echo of what was collected (pre-Formalisation)."""
        skip = "- skipped -"
        return (
            "\n=== Your request, as I received it ===\n"
            f"Problem:\n  {self.prompt}\n\n"
            f"Scope:\n  {self.scope if self.scope else skip}\n\n"
            f"Fixed facts (anchors):\n"
            f"  {self.contextual_anchors if self.contextual_anchors else skip}\n\n"
            f"Reasoning rules (policy):\n"
            f"  {self.policy_base if self.policy_base else skip}\n"
            "======================================\n"
        )


# ---------------------------------------------------------------------------
# Intake helpers
# ---------------------------------------------------------------------------

def _read_multiline() -> str:
    # keeps reading lines until a blank line is submitted, or the input
    # stream ends (EOFError - piped/redirected stdin running out). this
    # is what lets one field absorb a whole pasted paragraph instead of
    # scattering across whatever input() call happens to run next.
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _ask_required(header: str, explanation: str) -> RawField:
    """Ask until we get a non-empty answer. Accepts multi-line paste."""
    print(f"\n{header}\n{explanation}")
    print("(paste as much as you need, then press Enter on a blank line to finish)")
    while True:
        text = _read_multiline()
        if text:
            return RawField(text)
        print("This one I do need - please describe your problem in a sentence or two.")


def _ask_optional(header: str, explanation: str) -> Optional[RawField]:
    """Ask once; empty input (blank line immediately) means skip. Accepts multi-line paste."""
    print(f"\n{header}\n{explanation}")
    print("(press Enter on a blank line to skip, or paste your answer then press "
          "Enter on a blank line to finish)")
    text = _read_multiline()
    return RawField(text) if text else None


# ---------------------------------------------------------------------------
# The four intake functions
# ---------------------------------------------------------------------------

def get_user_prompt() -> RawField:
    return _ask_required(
        "WHAT IS YOUR PROBLEM OR QUESTION?  (required)",
        "Describe what you want figured out or decided, in your own words.\n"
        "Example: \"Should my two-person team build our app with Django or\n"
        "Laravel, given we need to launch in three months?\"",
    )


def get_user_scope() -> Optional[RawField]:
    return _ask_optional(
        "SCOPE - what should I include, and what should I leave out?  (optional, recommended)",
        "The boundaries of the investigation. Telling me what is IN and what is\n"
        "OUT keeps me from wasting effort - and from answering the wrong question.\n"
        "Examples: \"only free or open-source options\", \"Algerian market only\",\n"
        "\"ignore cost, I care about speed\", \"don't consider hiring more people\".",
    )


def get_user_contextual_anchors() -> Optional[RawField]:
    return _ask_optional(
        "FIXED FACTS - things I should treat as given and never question  (optional, recommended)",
        "Facts about your situation that are simply true for you. I will build\n"
        "on these without gathering evidence for or against them - so only put\n"
        "something here if you do NOT want me to double-check it.\n"
        "Examples: \"my budget is $10k\", \"the team is 2 people\",\n"
        "\"the deadline is March\", \"we already committed to Python\".",
    )


def get_user_policy_base() -> Optional[RawField]:
    return _ask_optional(
        "REASONING RULES - how do you want me to think and settle disagreements?  (optional)",
        "Any rules for HOW I should reason, not what about. This can include:\n"
        "  - preferences: \"prefer recent sources\", \"be conservative with claims\"\n"
        "  - conflict handling: \"when sources disagree, flag it instead of picking\"\n"
        "  - rules of thumb you trust: \"official statistics beat blog posts\"\n"
        "Note: rules that would break my integrity checks (e.g. \"skip verification\")\n"
        "will be declined and I will tell you so.",
    )


# ---------------------------------------------------------------------------
# Orchestrator entry point for intake
# ---------------------------------------------------------------------------

def collect_user_query() -> UserQuery:
    """Run the full intake dialogue and return a validated UserQuery."""
    print(
        "I'll ask you four things. Only the first is required -\n"
        "for the rest, just press Enter on a blank line to skip. The more you\n"
        "give me, the sharper the result."
    )
    query = UserQuery(
        prompt=get_user_prompt(),
        scope=get_user_scope(),
        contextual_anchors=get_user_contextual_anchors(),
        policy_base=get_user_policy_base(),
    )
    print(query.summary())
    return query


if __name__ == "__main__":
    q = collect_user_query()
    # Next stage: hand `q` to Formalisation (extraction + validation gate).
    # q.model_dump_json() is what you persist for provenance.
    print(f"[intake complete - {MAX_ITERATIONS} iterations budgeted, kernel-fixed]")