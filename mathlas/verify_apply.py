"""Tiered applicability VERIFICATION -- does a retrieved result actually apply?

Retrieval finds a *candidate* existing result; MAP argues it applies. This
module is the discriminator that separates a real hit from a confident
hallucination -- the piece TheoremSearch (and every retrieval-only tool) lacks.
Three tiers, cheapest-first (the AlphaEvolve cheap->expensive cascade):

  NUMERIC  (airtight)  -- when the claim reduces to a numeric identity, re-check
                          it independently at high precision. Reuses verify.py.
  FORMAL   (Lean)      -- when a Lean term/check is available, kernel-check it.
                          Correct-by-construction but narrow; stubbed pending a
                          Lean toolchain (interface fixed so it slots in).
  INFORMAL (the moat)  -- structured-adversarial LLM self-verification for the
                          general NL case. This is where the 2025-2026 SOTA lives
                          and where retrieval-only systems give up.

Informal-tier design (grounded in the latest verification research)
-------------------------------------------------------------------
The honest finding from 2025-2026 NL-proof grading is that a single LLM-as-judge
is unreliable -- sensitive to model/prompt, and a bare rubric does NOT reliably
help (Petrov et al., "ProofGrader," arXiv:2510.13888, ICLR 2026; "Scaling
Generative Verifiers," arXiv:2511.13027). What *does* work is **separating
generation from verification and decomposing the check into atomic, individually
falsifiable conditions** -- the generator-verifier split of DeepSeekMath-V2
(arXiv:2511.22570): an adversarial verifier hunts for the single condition that
fails, rather than rendering a holistic score.

So our informal verifier does NOT ask "is this right? 0-7". It:
  1. EXTRACTS the candidate's hypotheses as a checklist of atomic preconditions
     (from the result's own statement -- a problem-specific marking scheme, not a
     generic rubric).
  2. ADVERSARIALLY checks each: a skeptic pass tries to find a precondition the
     problem does NOT establish, or a mismatch between what the problem needs and
     what the result delivers. Any single unmet precondition => does-not-apply.
  3. AGGREGATES: applies only if every precondition is satisfied AND need-meets-
     guarantee, with the failing condition surfaced when not.

We also honour the two hard-won lessons: (a) do NOT prepend retrieved math
blindly into a strong model -- verification is a *gated* step, run only on the
specific candidate MAP proposed; (b) "typechecks / sounds plausible" != "solves
the problem" -- we check need-vs-guarantee fit, not mere coherence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .llm import LLM
from .map import Mapping, _extract_json


class Tier(str, Enum):
    NUMERIC = "numeric"
    FORMAL = "formal"
    INFORMAL = "informal"


@dataclass(frozen=True)
class Condition:
    text: str
    satisfied: Optional[bool]      # None = undetermined
    evidence: str = ""


@dataclass(frozen=True)
class ApplyVerdict:
    tier: Tier
    applies: bool                  # verifier's decision (may downgrade MAP's)
    confidence: float
    conditions: List[Condition] = field(default_factory=list)
    failure: Optional[str] = None  # the single condition that sank it, if any
    note: str = ""

    @property
    def unmet(self) -> List[Condition]:
        return [c for c in self.conditions if c.satisfied is False]


# --------------------------------------------------------------------------- #
# NUMERIC tier -- airtight, reuses the existing independent re-evaluation.
# --------------------------------------------------------------------------- #
def verify_numeric_claim(value, candidate_expr: str, *,
                         dps_verify: int = 50, min_digits: int = 20) -> ApplyVerdict:
    """Applicability == an exact numeric identity. Delegates to the airtight
    independent re-evaluation in verify.py (search-low / verify-high / different
    library). Use when MAP's connection reduces to 'these two quantities are
    equal to high precision'."""
    import mpmath
    from .verify import verify_closed_form
    v = value if isinstance(value, mpmath.mpf) else mpmath.mpf(value)
    vr = verify_closed_form(v, candidate_expr, dps_verify=dps_verify,
                            min_digits=min_digits)
    cond = Condition(
        text=f"{candidate_expr} == query to >= {min_digits} digits",
        satisfied=vr.ok,
        evidence=f"agreed {vr.digits_agreed} digits"
                 + (f"; {vr.error}" if vr.error else ""),
    )
    return ApplyVerdict(
        tier=Tier.NUMERIC, applies=vr.ok,
        confidence=1.0 if vr.ok else 0.0, conditions=[cond],
        failure=None if vr.ok else "numeric identity not confirmed",
        note="independent high-precision re-evaluation",
    )


# --------------------------------------------------------------------------- #
# FORMAL tier -- Lean kernel check (stub; interface fixed for a real toolchain).
# --------------------------------------------------------------------------- #
def verify_formal(lean_snippet: Optional[str]) -> ApplyVerdict:
    """Kernel-check a Lean term establishing applicability. Stubbed: returns an
    undetermined verdict unless a checker is wired in. Kept so the cascade has a
    real formal slot (build on LeanDojo/Loogle when a toolchain is present).
    Remember 'typecheck != correctness': a real impl must check that the term
    proves the *applicability claim*, not merely that something compiles."""
    if not lean_snippet:
        return ApplyVerdict(tier=Tier.FORMAL, applies=False, confidence=0.0,
                            note="no Lean snippet provided; formal tier skipped")
    return ApplyVerdict(
        tier=Tier.FORMAL, applies=False, confidence=0.0,
        note="Lean checker not installed in this environment; snippet not "
             "kernel-checked (interface ready for LeanDojo/Loogle).",
    )


# --------------------------------------------------------------------------- #
# INFORMAL tier -- structured-adversarial self-verification (the moat).
# --------------------------------------------------------------------------- #
_EXTRACT_SYS = (
    "You are a meticulous mathematical referee. Extract the EXACT hypotheses / "
    "preconditions a stated result requires before it can be applied. List them "
    "atomically -- one checkable condition each. Do not judge the problem yet. "
    "Respond with ONLY the JSON asked for."
)
_EXTRACT_PROMPT = """\
KNOWN RESULT:
  name: {name}
  statement: {statement}

List the result's hypotheses as atomic, individually-checkable preconditions
(the things that must hold for it to apply). Respond with ONLY:
{{"preconditions": ["<condition 1>", "<condition 2>", ...]}}"""

_CHECK_SYS = (
    "You are an ADVERSARIAL mathematical verifier. A colleague claims a known "
    "result applies to a problem. Your job is to FALSIFY that claim if you can: "
    "find ONE precondition the problem fails to establish, or a gap between what "
    "the problem NEEDS and what the result DELIVERS. Be skeptical and concrete. "
    "Soundness over politeness. A result applies ONLY if every precondition is "
    "met AND its guarantee supplies the need. Respond with ONLY the JSON asked for."
)
_CHECK_PROMPT = """\
PROBLEM:
{problem}

KNOWN RESULT:
  name: {name}
  statement: {statement}

CLAIMED CONNECTION (from the mapping step):
  need:        {need}
  guarantee:   {guarantee}
  reduction:   {reduction}

PRECONDITIONS the result requires:
{precond_block}

For EACH precondition, decide whether the PROBLEM establishes it. Then judge
overall applicability. A single unmet precondition, or a need/guarantee mismatch,
means it does NOT apply. Respond with ONLY:
{{
  "conditions": [
    {{"text": "<precondition>", "satisfied": true|false|null, "evidence": "<why>"}}
  ],
  "need_met_by_guarantee": true|false,
  "applies": true|false,
  "confidence": 0.0-1.0,
  "failure": "<the single condition or mismatch that breaks it, or null>"
}}"""


def _extract_preconditions(mapping: Mapping, llm: LLM) -> List[str]:
    c = mapping.candidate
    raw = llm.complete(
        _EXTRACT_PROMPT.format(name=c.name or "(unnamed)", statement=c.statement),
        system=_EXTRACT_SYS, temperature=0.0,
    )
    try:
        obj = json.loads(_extract_json(raw))
        pre = [str(p) for p in obj.get("preconditions", []) if str(p).strip()]
    except (json.JSONDecodeError, ValueError):
        pre = []
    # Fall back to the preconditions MAP already proposed, if extraction is empty.
    return pre or list(mapping.preconditions)


def verify_informal(mapping: Mapping, llm: LLM, *,
                    passes: int = 1) -> ApplyVerdict:
    """Structured-adversarial applicability check for the general NL case.

    Decomposes the candidate into atomic preconditions, then runs an adversarial
    skeptic that must either confirm every one against the problem or name the
    one that fails. ``passes`` > 1 runs the skeptic independently several times
    and takes the WORST verdict (any pass that finds a failure wins) -- a cheap
    self-consistency-for-rejection that counters single-judge variance.
    """
    preconds = _extract_preconditions(mapping, llm)
    block = "\n".join(f"  - {p}" for p in preconds) or "  (none stated; infer them)"
    c = mapping.candidate

    verdicts: List[ApplyVerdict] = []
    for _ in range(max(1, passes)):
        raw = llm.complete(
            _CHECK_PROMPT.format(
                problem=_problem_text(mapping),
                name=c.name or "(unnamed)", statement=c.statement,
                need=mapping.need or "(unspecified)",
                guarantee=mapping.guarantee or "(unspecified)",
                reduction=mapping.reduction or "(none)",
                precond_block=block,
            ),
            system=_CHECK_SYS, temperature=0.0,
        )
        verdicts.append(_parse_informal(raw))

    # Worst-case aggregation: a single pass finding non-application rejects.
    rejecting = [v for v in verdicts if not v.applies]
    chosen = (min(rejecting, key=lambda v: v.confidence) if rejecting
              else max(verdicts, key=lambda v: v.confidence))
    return chosen


def _problem_text(mapping: Mapping) -> str:
    """The informal verifier needs the original problem. MAP threads it onto the
    Mapping (``problem`` field); fall back to need+connection if absent."""
    return mapping.problem or (
        f"(need) {mapping.need}\n(context) {mapping.connection}")


def _parse_informal(raw: str) -> ApplyVerdict:
    try:
        obj = json.loads(_extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        return ApplyVerdict(tier=Tier.INFORMAL, applies=False, confidence=0.0,
                            note="verifier reply unparseable -> reject")
    conds = []
    for c in obj.get("conditions", []) or []:
        sat = c.get("satisfied")
        conds.append(Condition(
            text=str(c.get("text", "")),
            satisfied=(None if sat is None else bool(sat)),
            evidence=str(c.get("evidence", "")),
        ))
    applies = bool(obj.get("applies", False)) and bool(
        obj.get("need_met_by_guarantee", True))
    return ApplyVerdict(
        tier=Tier.INFORMAL,
        applies=applies,
        confidence=float(obj.get("confidence", 0.0) or 0.0),
        conditions=conds,
        failure=(obj.get("failure") or None) if not applies else None,
        note="structured-adversarial precondition check",
    )


__all__ = [
    "Tier", "Condition", "ApplyVerdict",
    "verify_numeric_claim", "verify_formal", "verify_informal",
]
