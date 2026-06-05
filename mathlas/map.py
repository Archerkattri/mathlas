"""MAP — the needs<->guarantees engine (the genuinely novel core).

Given a PROBLEM and CANDIDATE existing results, decide which candidates apply
and HOW. This is the step every retrieval tool (TheoremSearch included) punts to
the human: not keyword similarity, but matching what the problem NEEDS to what a
known result GUARANTEES — directly, or after a reduction/analogy.

Two-stage structural mapping (abduction -> deduction)
-----------------------------------------------------
The 2024-2026 analogy-reasoning literature finds LLMs do analogy by *emergent
pattern recognition* and lack a formal mechanism to keep a source->target
correspondence consistent; a two-stage scheme -- first ABSTRACT the structure,
then APPLY it -- beats one-shot mapping on hard cases (Webb et al. on LLM
analogy; structural-mapping work, e.g. arXiv:2603.29997). So MAP runs:

  1. ABDUCTION (once per problem): extract a structure-level *requirement
     signature* -- the objects, the property/conclusion sought, and the
     hypotheses the problem already provides -- independent of any candidate.
  2. DEDUCTION (per candidate): match the candidate's GUARANTEE against that
     fixed signature -- direct or via a reduction -- rather than re-reading the
     problem fresh each time (which lets the model drift toward feature/keyword
     similarity). This is matching a *requirement* to a *guarantee*, the step
     the unit-distance disproof showed is the valuable one.

The reasoning is done by a pluggable LLM (see llm.py). Every step parses a
strict JSON verdict, so each mapping is auditable. The engine NEVER claims
novelty — it only connects a problem to existing results. The result of MAP is
then handed to the tiered VERIFY (verify_apply.py) for an adversarial check.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from .llm import LLM
from .retrieve import Candidate

_SYSTEM = (
    "You are a mathematical reasoning engine. Your ONE job: decide whether a "
    "KNOWN result can be APPLIED to a given problem, by matching what the problem "
    "NEEDS to what the result GUARANTEES. You never invent new mathematics; you "
    "only connect the problem to existing results. Be skeptical -- if the result "
    "does not actually apply, say applies=false. Respond with ONLY the JSON asked for."
)

# --- Stage 1: ABDUCTION -- the problem's structure-level requirement signature.
_SIG_SYS = (
    "You are a mathematical reasoning engine. Extract the STRUCTURE of a problem: "
    "what objects it involves, the property or conclusion it seeks, and the "
    "hypotheses it already provides. Stay at the level of mathematical structure, "
    "not surface keywords. Respond with ONLY the JSON asked for."
)
_SIG_PROMPT = """\
PROBLEM:
{problem}

Extract the problem's requirement signature -- the abstract structure a solving
result must match, independent of any specific theorem. Respond with ONLY:
{{
  "objects": ["<the mathematical objects in play>"],
  "need": "<the property/object/conclusion the problem seeks, one line>",
  "given": ["<hypotheses the problem already provides>"],
  "field_hints": ["<sub-fields whose results might apply>"]
}}"""

# --- Stage 2: DEDUCTION -- match one candidate's guarantee to the signature.
_PROMPT = """\
PROBLEM (verbatim):
{problem}

PROBLEM REQUIREMENT SIGNATURE (extracted):
  objects: {objects}
  need:    {sig_need}
  given:   {given}

CANDIDATE KNOWN RESULT:
  name: {name}
  statement: {statement}
  source: {source}

Decide whether this known result can be applied, matching the SIGNATURE's need
to the result's GUARANTEE:
- What does the RESULT guarantee (its conclusion, under its hypotheses)?
- Does that guarantee supply the signature's need -- directly, or after a
  reduction/transform? Match requirement-to-guarantee, not keyword similarity.
- What hypotheses/preconditions must hold for the result to apply HERE, and does
  the problem's `given` establish them?

Respond with ONLY this JSON object:
{{
  "applies": true|false,
  "confidence": 0.0-1.0,
  "need": "<what the problem needs, one line>",
  "guarantee": "<what the result guarantees, one line>",
  "connection": "<how the result supplies the need, or why it does not>",
  "preconditions": ["<hypotheses that must hold here>"],
  "reduction": "<the transform/reduction linking them, or null>"
}}"""


@dataclass(frozen=True)
class Signature:
    """The problem's structure-level requirement signature (abduction stage)."""
    objects: List[str]
    need: str
    given: List[str]
    field_hints: List[str]


@dataclass(frozen=True)
class Mapping:
    candidate: Candidate
    applies: bool
    confidence: float
    need: str
    guarantee: str
    connection: str
    preconditions: List[str]
    reduction: Optional[str]
    problem: str = ""              # original problem text (threaded to VERIFY)


def extract_signature(problem: str, llm: LLM) -> Signature:
    """ABDUCTION: extract the problem's requirement signature once, up front.
    On a parse failure, fall back to a minimal signature (problem as its own
    need) so the deduction stage still runs."""
    raw = llm.complete(_SIG_PROMPT.format(problem=problem),
                       system=_SIG_SYS, temperature=0.0)
    try:
        obj = json.loads(_extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        return Signature(objects=[], need=problem, given=[], field_hints=[])
    return Signature(
        objects=list(obj.get("objects", []) or []),
        need=str(obj.get("need", "") or problem),
        given=list(obj.get("given", []) or []),
        field_hints=list(obj.get("field_hints", []) or []),
    )


def map_candidate(problem: str, candidate: Candidate, llm: LLM,
                  signature: Optional[Signature] = None) -> Optional[Mapping]:
    """DEDUCTION: needs<->guarantees mapping for one candidate against the
    (precomputed) signature. Returns None if the reply can't be parsed."""
    sig = signature or extract_signature(problem, llm)
    prompt = _PROMPT.format(
        problem=problem,
        objects=", ".join(sig.objects) or "(unspecified)",
        sig_need=sig.need,
        given=", ".join(sig.given) or "(none stated)",
        name=candidate.name or "(unnamed)",
        statement=candidate.statement,
        source=candidate.source or "(unknown)",
    )
    raw = llm.complete(prompt, system=_SYSTEM, temperature=0.0)
    try:
        obj = json.loads(_extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        return None
    return Mapping(
        candidate=candidate,
        applies=bool(obj.get("applies", False)),
        confidence=float(obj.get("confidence", 0.0) or 0.0),
        need=str(obj.get("need", "")),
        guarantee=str(obj.get("guarantee", "")),
        connection=str(obj.get("connection", "")),
        preconditions=list(obj.get("preconditions", []) or []),
        reduction=obj.get("reduction") or None,
        problem=problem,
    )


def map_candidates(problem: str, candidates: List[Candidate], llm: LLM,
                   min_confidence: float = 0.5,
                   signature: Optional[Signature] = None) -> List[Mapping]:
    """Map each candidate against ONE shared signature (extracted once), keeping
    those that apply at/above the confidence threshold, best first."""
    sig = signature or extract_signature(problem, llm)
    mapped = (map_candidate(problem, c, llm, signature=sig) for c in candidates)
    keep = [m for m in mapped if m and m.applies and m.confidence >= min_confidence]
    keep.sort(key=lambda m: m.confidence, reverse=True)
    return keep


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of an LLM reply (handles ``` fences + prose)."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1] if s.count("```") >= 2 else s.strip("`")
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found")
    return s[start:end + 1]
